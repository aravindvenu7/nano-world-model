"""
Critic snapshots for online BuilderBench runs, after dre/ogbench/toy_bgrw_online/diagnostics.py.

BuilderBench's state is 12 + 13N numbers, so each figure is a 2D slice: a grid over the workspace (x, y), with
everything else held at a pinned anchor state (a start configuration of the eval env, fixed at the first
snapshot along with its task goal, so snapshots are comparable across training).

Figure 1, cube slice (`step_XXXXXXXX_cube.png`):
  row 1, state axis: goal fixed, cube `diagnostics.cube` moved to each (x, y) at table height, critic Q(s, a, g)
         at the planner's action there. Visitation = where any cube has been (replay buffer).
  row 2, goal axis:  (s, a) fixed at the anchor, goal = "a cube at (x, y)" (one-cube goal), critic value.
         log M = cdr + log N(g) adds back the Gaussian reference the time score is measured against.
Figure 2, gripper slice (`step_XXXXXXXX_gripper.png`): state axis with the gripper moved to each (x, y) at its
         anchor height, goal fixed. Visitation = where the gripper has been.
Each row ends with the mean critic over grid points the buffer has / has not visited; their difference
(unvisited - visited) is the optimism statistic of the toy runs, logged as diag/*_gap.

The critic is the one the agent acts on: Q = sum_k dt * score(t_k) (cdr, no marginal term).
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch



class Snapshotter:
    def __init__(self, cfg, env, eval_env, gs, agent, run_dir):
        d = cfg.diagnostics
        self.env, self.eval_env, self.gs, self.agent = env, eval_env, gs, agent
        self.cube, self.table_z, self.n, self.chunk = d.cube, d.table_z, d.grid, d.chunk
        self.out_dir = os.path.join(run_dir, "diagnostics")
        lo, hi = env.pos_lo, env.pos_hi
        xs = torch.linspace(lo[0], hi[0], self.n, device=env.device)
        ys = torch.linspace(lo[1], hi[1], self.n, device=env.device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        self.xy = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (n*n, 2)
        self.extent = [lo[0], hi[0], lo[1], hi[1]]
        self.anchor = None

    def _pin(self):
        """Anchor state + task goal from one eval-env reset, fixed for the whole run."""
        view = self.eval_env.reset()
        self.anchor = view["obs"][:1].clone()
        self.goal, self.goal_mask = self.gs.from_target(view["target"][:1], view["target_mask"][:1])
        self.goal_xyz = view["target"][0][view["target_mask"][0]].cpu().numpy()

    # ── critic queries (chunked) ─────────────────────────────────────────────

    def _q(self, obs, goal, mask, actions):
        """Q(s, a, g) for P queries; goal/mask may be one goal (1, ...) broadcast to all."""
        out = []
        for i in range(0, len(obs), self.chunk):
            o = obs[i:i + self.chunk]
            g = goal.expand(len(o), -1, -1) if goal.shape[0] == 1 else goal[i:i + self.chunk]
            m = mask.expand(len(o), -1) if mask.shape[0] == 1 else mask[i:i + self.chunk]
            with torch.no_grad():
                G = self.agent.goal_tensor(g, m)
                out.append(self.agent.q_values(G, self.env.state_tokens(o), actions[i:i + self.chunk, None])[:, 0])
        return torch.cat(out)

    def _policy_actions(self, obs, goal, mask):
        out = []
        for i in range(0, len(obs), self.chunk):
            o = obs[i:i + self.chunk]
            out.append(self.agent.act(self.env.state_tokens(o), goal.expand(len(o), -1, -1), mask.expand(len(o), -1), epsilon=0.0))
        return torch.cat(out)

    def _moved(self, which):
        """Anchor observation with the swept object moved to every grid point."""
        s = self.env._slices()
        obs = self.anchor.repeat(len(self.xy), 1)
        if which == "cube":
            start = s["c_pos"].start + 3 * self.cube
            obs[:, start:start + 2] = self.xy
            obs[:, start + 2] = self.table_z
        else:
            obs[:, s["g_pos"].start:s["g_pos"].start + 2] = self.xy
        return obs

    def _state_field(self, which):
        obs = self._moved(which)
        return self._q(obs, self.goal, self.goal_mask, self._policy_actions(obs, self.goal, self.goal_mask))

    def _goal_field(self):
        """(s, a) at the anchor; goal = cube `self.cube`'s token at each grid point, only that cube counted."""
        p = len(self.xy)
        target = torch.zeros(p, self.env.num_cubes, 3, device=self.env.device)
        target[:, self.cube, :2], target[:, self.cube, 2] = self.xy, self.table_z
        tmask = torch.zeros(p, self.env.num_cubes, dtype=torch.bool, device=self.env.device)
        tmask[:, self.cube] = True
        goal, mask = self.gs.from_target(target, tmask)
        action = self._policy_actions(self.anchor, self.goal, self.goal_mask)
        cdr = self._q(self.anchor.repeat(p, 1), goal, mask, action.repeat(p, 1))
        fm = self.gs.feat_mask[None] * mask[..., None].float()  # log N(g) over the counted goal dims
        base = (-0.5 * goal ** 2 - 0.5 * math.log(2 * math.pi)).mul(fm).flatten(1).sum(1)
        return cdr, cdr + base

    # ── visitation ───────────────────────────────────────────────────────────

    def _visits(self, buffer, which, max_rows=200_000):
        filled = min(buffer.head, buffer.cap)
        obs = buffer.obs[:filled].reshape(-1, buffer.obs.shape[-1])
        if len(obs) > max_rows:
            obs = obs[torch.randint(len(obs), (max_rows,), device=obs.device)]
        s = self.env._slices()
        if which == "cube":
            xy = obs[:, s["c_pos"]].view(len(obs), -1, 3)[..., :2].reshape(-1, 2)
        else:
            xy = obs[:, s["g_pos"]][:, :2]
        xy = xy.cpu().numpy()
        counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=self.n, range=[self.extent[:2], self.extent[2:]])
        return counts

    # ── figures ──────────────────────────────────────────────────────────────

    def __call__(self, buffer, env_steps):
        if self.anchor is None:
            self._pin()
        stats = {}
        cube_visits, grip_visits = self._visits(buffer, "cube"), self._visits(buffer, "gripper")
        s_cube = self._state_field("cube").cpu().numpy()
        g_cdr, g_logm = (f.cpu().numpy() for f in self._goal_field())
        s_grip = self._state_field("gripper").cpu().numpy()
        stats.update(_gap(s_cube, cube_visits, "diag/cube_state_"))
        stats.update(_gap(g_cdr, cube_visits, "diag/cube_goal_"))
        stats.update(_gap(s_grip, grip_visits, "diag/gripper_state_"))
        peak = self.xy[int(np.argmax(g_cdr))].cpu().numpy()
        stats["diag/goal_peak_x"], stats["diag/goal_peak_y"] = float(peak[0]), float(peak[1])

        s = self.env._slices()
        a = self.anchor[0].cpu().numpy()
        cubes_xy = a[s["c_pos"]].reshape(-1, 3)[:, :2]
        grip_xy = a[s["g_pos"]][:2]
        tag = f"step_{env_steps:09d}"
        title = (f"{'GTSE-b' if self.agent.model.phi_time else 'GTSE-r'} @ {env_steps} env steps | goal: "
                 + ", ".join(f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in self.goal_xyz))
        os.makedirs(self.out_dir, exist_ok=True)

        fig, ax = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"cube slice (cube {self.cube}, z = {self.table_z}) | {title}", fontsize=12)
        self._visit_panel(ax[0, 0], cube_visits, "cube visitation (all cubes, replay)", cubes_xy, grip_xy)
        self._field_panel(ax[0, 1], s_cube, f"state axis: Q(s with cube {self.cube} at (x, y), a*, goal)", cubes_xy, grip_xy)
        _bar_panel(ax[0, 2], stats, "diag/cube_state_", "state axis: visited vs unvisited")
        self._field_panel(ax[1, 0], g_logm, "goal axis: log M = cdr + log N(g), (s, a) at anchor", cubes_xy, grip_xy, peak)
        self._field_panel(ax[1, 1], g_cdr, f"goal axis: cdr, a cube at (x, y)\npeak ({peak[0]:.2f}, {peak[1]:.2f})", cubes_xy, grip_xy, peak)
        _bar_panel(ax[1, 2], stats, "diag/cube_goal_", "goal axis: visited vs unvisited goals")
        _save(fig, os.path.join(self.out_dir, f"{tag}_cube.png"))

        fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
        fig.suptitle(f"gripper slice (z = {a[s['g_pos']][2]:.2f}) | {title}", fontsize=12)
        self._visit_panel(ax[0], grip_visits, "gripper visitation (replay)", cubes_xy, grip_xy)
        self._field_panel(ax[1], s_grip, "state axis: Q(s with gripper at (x, y), a*, goal)", cubes_xy, grip_xy)
        _bar_panel(ax[2], stats, "diag/gripper_state_", "state axis: visited vs unvisited")
        _save(fig, os.path.join(self.out_dir, f"{tag}_gripper.png"))
        return stats

    def _markers(self, axis, cubes_xy, grip_xy, peak=None):
        axis.scatter(self.goal_xyz[:, 0], self.goal_xyz[:, 1], color="black", marker="*", s=200, zorder=5, label="goal")
        axis.scatter(cubes_xy[:, 0], cubes_xy[:, 1], color="red", marker="s", s=50, zorder=5, label="anchor cubes")
        axis.scatter([cubes_xy[self.cube, 0]], [cubes_xy[self.cube, 1]], facecolors="none", edgecolors="red", marker="o", s=160, zorder=5)
        axis.scatter([grip_xy[0]], [grip_xy[1]], color="lime", marker="^", s=90, zorder=5, label="anchor gripper")
        if peak is not None:
            axis.scatter([peak[0]], [peak[1]], color="magenta", marker="D", s=70, zorder=5, label="peak")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.legend(fontsize=7, loc="upper right")

    def _visit_panel(self, axis, counts, title, cubes_xy, grip_xy):
        im = axis.imshow(np.log10(counts.T + 1), origin="lower", extent=self.extent, cmap="magma", aspect="auto")
        plt.colorbar(im, ax=axis, label="log10(1 + visits)")
        axis.set_title(f"{title}\ncoverage {(counts > 0).mean():.3f} of grid cells", fontsize=10)
        self._markers(axis, cubes_xy, grip_xy)

    def _field_panel(self, axis, field, title, cubes_xy, grip_xy, peak=None):
        im = axis.imshow(field.reshape(self.n, self.n).T, origin="lower", extent=self.extent, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=axis)
        axis.set_title(title, fontsize=10)
        self._markers(axis, cubes_xy, grip_xy, peak)


def _gap(field, counts, prefix):
    visited = counts.flatten() > 0
    v = float(field[visited].mean()) if visited.any() else float("nan")
    u = float(field[~visited].mean()) if (~visited).any() else float("nan")
    return {f"{prefix}visited": v, f"{prefix}unvisited": u, f"{prefix}gap": u - v}


def _bar_panel(axis, stats, prefix, title):
    v, u = stats[f"{prefix}visited"], stats[f"{prefix}unvisited"]
    axis.bar(["visited", "unvisited"], [v, u], color=["tab:blue", "tab:red"])
    axis.axhline(0, color="gray", lw=1)
    axis.set_ylabel("mean critic value over grid points")
    axis.set_title(f"{title}\ngap (unvisited - visited) = {u - v:+.3f}", fontsize=10)
    axis.grid(True, alpha=0.3, axis="y")


def _save(fig, path):
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
