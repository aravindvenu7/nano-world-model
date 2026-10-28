"""
BuilderBench adapter: runs the vendored JAX environments (src/builderbench) and hands torch tensors to the agent.

JAX is forced onto the CPU (the physics is MuJoCo's C++ rollout either way), so it never competes with torch for
GPU memory. Both protocols are supported:
  - single task ("cube-N-taskK"): the env's target covers the task's K cubes, placed in the first K goal slots.
  - play ("cube-N-play"): each episode carries a random task target + goal_mask, used only for evaluation.

Observation layout for N cubes (from build_block*.get_obs):
  gripper pos 3 | gripper quat 4 | gripper linvel 3 | cube pos 3N | cube quat 4N | cube linvel 3N | cube angvel 3N | fingers 2
Tokens: gripper = [pos, quat, linvel, fingers] (12, padded to 13); cube = [pos, quat, linvel, angvel] (13).
Positions are standardised with the env's own workspace bounds (`_workspace_bounds`, the box outside which a cube
ends the episode) as a uniform distribution's mean / std; velocities are divided by fixed scales.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch

from builderbench.env_utils import make_env
from builderbench.wrapper import wrap_env

TOKEN_DIM = 13
GRIPPER_FEATS = 12


class BuilderBenchEnv:
    def __init__(self, env_id, num_envs, num_threads, seed, device, early_termination=True,
                 permutation_invariant_reward=True, vel_scale=1.0, angvel_scale=10.0):
        env_cls, cfg = make_env(SimpleNamespace(env_id=env_id, env_early_termination=early_termination,
                                                permutation_invariant_reward=permutation_invariant_reward))
        self.play = env_id.endswith("-play")
        self.num_cubes, self.episode_length, self.num_envs = cfg.num_cubes, cfg.episode_length, num_envs
        self.device, self.vel_scale, self.angvel_scale = device, vel_scale, angvel_scale
        self.env = wrap_env(env_cls(num_envs=num_envs, num_threads=num_threads, config=cfg), cfg.episode_length)
        self.action_size = self.env.action_size
        self._key = jax.random.PRNGKey(seed)

        def _step(state, action):
            state = self.env.pre_step(state, action)
            physics_state, sensor_data = self.env.step(state, action)
            return self.env.post_step(state, physics_state, sensor_data)
        self._reset, self._step = jax.jit(self.env.reset), jax.jit(_step)
        # Workspace bounds from the env itself: x [-0.05, 0.45], y [-0.35, 0.35], z [0, 0.5] (m) in BuilderBench.
        self.pos_lo, self.pos_hi = np.asarray(self.env._workspace_bounds, dtype=np.float64)
        self._pos_mean = torch.tensor((self.pos_lo + self.pos_hi) / 2, dtype=torch.float32, device=device)
        self._pos_std = torch.tensor((self.pos_hi - self.pos_lo) / np.sqrt(12), dtype=torch.float32, device=device)

    def _t(self, x, dtype=torch.float32):
        return torch.from_numpy(np.asarray(x)).to(self.device, dtype)

    def _view(self, state):
        """Torch view of a JAX state: obs, done, success metrics, and the task target in N goal slots + mask."""
        n = self.num_cubes
        target = self._t(state.info["target_goal"]).view(self.num_envs, -1, 3)
        if self.play:
            mask = self._t(state.info["goal_mask"], torch.bool)
        else:  # single task: K task cubes -> first K slots
            k = target.shape[1]
            target = torch.cat([target, target.new_zeros(self.num_envs, n - k, 3)], dim=1)
            mask = torch.arange(n, device=self.device)[None].expand(self.num_envs, n) < k
        out = {"obs": self._t(state.obs), "done": self._t(state.done) > 0, "target": target, "target_mask": mask,
               "success": self._t(state.metrics["success"]), "easy_success": self._t(state.metrics["easy_success"])}
        if self.play:
            out["task_id"] = self._t(state.info["task_id"], torch.long)
        return out

    def reset(self):
        self._key, key = jax.random.split(self._key)
        self._state = self._reset(jax.random.split(key, self.num_envs))
        return self._view(self._state)

    def step(self, action):
        self._state = self._step(self._state, jnp.asarray(action.detach().float().cpu().numpy()))
        return self._view(self._state)

    # ── tokenization ──────────────────────────────────────────────────────────

    def _slices(self):
        n = self.num_cubes
        return dict(g_pos=slice(0, 3), g_quat=slice(3, 7), g_vel=slice(7, 10), c_pos=slice(10, 10 + 3 * n),
                    c_quat=slice(10 + 3 * n, 10 + 7 * n), c_vel=slice(10 + 7 * n, 10 + 10 * n),
                    c_ang=slice(10 + 10 * n, 10 + 13 * n), fingers=slice(10 + 13 * n, 12 + 13 * n))

    def norm_pos(self, pos):
        return (pos - self._pos_mean) / self._pos_std

    def state_tokens(self, obs):
        """obs (B, obs_dim) -> (B, 1 + N, 13): gripper token then one token per cube."""
        s, b, n = self._slices(), obs.shape[0], self.num_cubes
        gripper = torch.cat([self.norm_pos(obs[:, s["g_pos"]]), obs[:, s["g_quat"]], obs[:, s["g_vel"]] / self.vel_scale,
                             obs[:, s["fingers"]], obs.new_zeros(b, TOKEN_DIM - GRIPPER_FEATS)], dim=-1)
        cubes = torch.cat([self.norm_pos(obs[:, s["c_pos"]].view(b, n, 3)), obs[:, s["c_quat"]].view(b, n, 4),
                           obs[:, s["c_vel"]].view(b, n, 3) / self.vel_scale, obs[:, s["c_ang"]].view(b, n, 3) / self.angvel_scale], dim=-1)
        return torch.cat([gripper[:, None], cubes], dim=1)

    def cube_positions(self, obs):
        return obs[:, self._slices()["c_pos"]].view(obs.shape[0], self.num_cubes, 3)


class GoalSpace:
    """What the GTSE density lives on. 'pos': one 3-dim token per cube (normalised position). 'full': the state
    tokens themselves (gripper + cubes, 13 dims). A goal is a set of tokens plus a mask saying which count."""

    def __init__(self, kind, env: BuilderBenchEnv):
        if kind not in ("pos", "full"):
            raise ValueError(f"goal_space must be 'pos' or 'full', got {kind}")
        self.kind, self.env, n, dev = kind, env, env.num_cubes, env.device
        self.has_gripper = kind == "full"
        self.dim = 3 if kind == "pos" else TOKEN_DIM
        self.num_tokens = n + int(self.has_gripper)
        self.types = torch.ones(self.num_tokens, dtype=torch.long, device=dev)
        feat = torch.ones(self.num_tokens, self.dim, device=dev)
        if self.has_gripper:
            self.types[0] = 0
            feat[0, GRIPPER_FEATS:] = 0
        self.feat_mask = feat  # (Ng, dim): 1 = a real feature

    def from_obs(self, obs):
        """Achieved goal of an observation: (B, Ng, dim)."""
        if self.kind == "pos":
            return self.env.norm_pos(self.env.cube_positions(obs))
        return self.env.state_tokens(obs)

    def from_target(self, target, target_mask):
        """Task target (B, N, 3) cube positions + (B, N) mask -> goal tokens + goal mask. In the full space the cube
        tokens get the target position, identity orientation and zero velocity, and the gripper is masked out."""
        pos = self.env.norm_pos(target)
        if self.kind == "pos":
            return pos, target_mask
        b, n = target.shape[:2]
        quat = torch.zeros(b, n, 4, device=target.device)
        quat[..., 0] = 1
        cubes = torch.cat([pos, quat, torch.zeros(b, n, 6, device=target.device)], dim=-1)
        tokens = torch.cat([torch.zeros(b, 1, TOKEN_DIM, device=target.device), cubes], dim=1)
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=target.device), target_mask], dim=1)
        return tokens, mask

    def random_mask(self, b, num_cubes=None):
        """Random goal subsets for training / exploration: each cube kept w.p. 1/2 (at least one), or exactly
        `num_cubes` cubes; in the full space the gripper token is kept w.p. 1/2."""
        n, dev = self.env.num_cubes, self.env.device
        if num_cubes is None:
            cubes = torch.rand(b, n, device=dev) < 0.5
            cubes[torch.arange(b, device=dev), torch.randint(n, (b,), device=dev)] = True
        else:
            cubes = torch.rand(b, n, device=dev).argsort(1) < num_cubes
        if not self.has_gripper:
            return cubes
        return torch.cat([torch.rand(b, 1, device=dev) < 0.5, cubes], dim=1)
