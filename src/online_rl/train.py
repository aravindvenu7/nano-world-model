"""
Online RL from scratch on BuilderBench with GTSE-r / GTSE-b (entry point: src/main_builderbench.py).

Loop: collect `rollout_length` steps from `num_envs` envs (uniform actions during warmup, then gradient ascent on
the critic with epsilon-greedy), add them to the replay buffer, take `grad_steps_per_iter` GTSE updates.
Goals the agent pursues while collecting:
  - single task (cube-N-taskK): the task's target, from the env.
  - play (cube-N-play): self-chosen exploration goals, resampled when an episode ends, from the replay buffer:
    `buffer` = a random stored configuration, `mega` = the lowest-density one among `mega_candidates` under a
    Gaussian KDE of stored configurations (MEGA, as BuilderBench's play_ppo_mega). A random subset of cubes counts.
Evaluation (both protocols) runs full episodes on the env's task targets with epsilon = 0 and reports the
BuilderBench success rates (per task for play).
"""

import json
import os
import time

import torch
from omegaconf import OmegaConf

from .builderbench_env import TOKEN_DIM, BuilderBenchEnv, GoalSpace
from .diagnostics import Snapshotter
from .gtse_agent import GTSEAgent
from .replay import ReplayBuffer


def _exploration_goals(n, buffer, gs, cfg):
    """Goal tokens + mask for n envs from stored configurations (play protocol)."""
    ex = cfg.exploration
    if ex.goal_source == "buffer":
        goal = gs.from_obs(buffer.sample_obs(n))
    elif ex.goal_source == "mega":
        cand = buffer.sample_obs(n * ex.mega_candidates)
        ref = gs.env.norm_pos(gs.env.cube_positions(buffer.sample_obs(ex.mega_kde_size))).flatten(1)
        x = gs.env.norm_pos(gs.env.cube_positions(cand)).flatten(1)
        density = torch.exp(-torch.cdist(x, ref) ** 2 / (2 * ex.mega_bandwidth ** 2)).mean(1)
        pick = density.view(n, ex.mega_candidates).argmin(1)
        goal = gs.from_obs(cand.view(n, ex.mega_candidates, -1)[torch.arange(n, device=cand.device), pick])
    else:
        raise ValueError(f"exploration.goal_source must be 'buffer' or 'mega', got {ex.goal_source}")
    mask = gs.random_mask(n)
    if gs.has_gripper:  # exploration goals, like tasks, are cube configurations
        mask[:, 0] = False
    return goal, mask


def evaluate(agent, env, gs):
    """One episode per eval env on the env's task targets, greedy (epsilon = 0)."""
    view = env.reset()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(active)
    easy = torch.zeros_like(active)
    task_id = view.get("task_id")
    for _ in range(env.episode_length):
        goal, mask = gs.from_target(view["target"], view["target_mask"])
        view = env.step(agent.act(env.state_tokens(view["obs"]), goal, mask, epsilon=0.0))
        success |= active & (view["success"] > 0)
        easy |= active & (view["easy_success"] > 0)
        active &= ~view["done"]
        if not active.any():
            break
    out = {"eval/success_rate": success.float().mean().item(), "eval/easy_success_rate": easy.float().mean().item()}
    if task_id is not None:
        for tid in task_id.unique().tolist():
            sel = task_id == tid
            out[f"eval/task_{tid + 1}_success_rate"] = success[sel].float().mean().item()
            out[f"eval/task_{tid + 1}_easy_success_rate"] = easy[sel].float().mean().item()
    return out


def run(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    e = cfg.env
    env = BuilderBenchEnv(cfg.env_id, cfg.num_envs, cfg.num_threads, cfg.seed, device, e.early_termination,
                          e.permutation_invariant_reward, e.vel_scale, e.angvel_scale)
    eval_env = BuilderBenchEnv(cfg.env_id, cfg.num_eval_envs, cfg.num_threads, cfg.seed + 1, device, e.early_termination,
                               e.permutation_invariant_reward, e.vel_scale, e.angvel_scale)
    gs = GoalSpace(cfg.goal_space, env)
    agent = GTSEAgent(cfg, gs, TOKEN_DIM, env.action_size, device)
    view = env.reset()
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.num_envs, view["obs"].shape[1], env.action_size, device, env.episode_length)
    # Single task: the loss uses subsets of exactly as many cubes as the task has; play: random subsets.
    loss_mask_cubes = None if env.play else int(view["target_mask"][0].sum())

    run_dir = os.getcwd()  # Hydra run dir
    writer = None
    if cfg.logging.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(run_dir, "tb"))
    log_file = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    n_params = sum(p.numel() for p in agent.model.parameters())
    print(f"[builderbench] {cfg.env_id} ({'play' if env.play else 'single task'}), {env.num_cubes} cubes, episode {env.episode_length} "
          f"steps | agent {cfg.agent} (phi_time={agent.model.phi_time}, psi_time={agent.model.psi_time}), {n_params / 1e6:.2f}M params "
          f"| goal space {cfg.goal_space} | device {device} | run dir {run_dir}", flush=True)

    snap_every = cfg.diagnostics.every_env_steps
    snapshot = Snapshotter(cfg, env, eval_env, gs, agent, run_dir) if snap_every else None
    next_snap = 0

    ep_id = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    goal = mask = None
    env_steps, it, t0, last_eval = 0, 0, time.time(), -1
    train_info, ep_success = {}, []
    while env_steps < cfg.total_env_steps:
        for _ in range(cfg.rollout_length):
            obs = view["obs"]
            if env_steps < cfg.warmup_env_steps:
                action = torch.rand(cfg.num_envs, env.action_size, device=device) * 2 - 1
            else:
                if env.play:
                    if goal is None:
                        goal, mask = _exploration_goals(cfg.num_envs, buffer, gs, cfg)
                else:
                    goal, mask = gs.from_target(view["target"], view["target_mask"])
                action = agent.act(env.state_tokens(obs), goal, mask, cfg.planning.epsilon)
            buffer.add(obs, action, ep_id)
            view = env.step(action)
            env_steps += cfg.num_envs
            done = view["done"]
            if done.any():
                ep_id = ep_id + done.long()
                if not env.play:  # play episodes carry a random task the agent is not pursuing
                    ep_success += (view["success"][done] > 0).float().tolist()
                if env.play and goal is not None:  # new exploration goals for the envs that just reset
                    g, m = _exploration_goals(int(done.sum()), buffer, gs, cfg)
                    goal[done], mask[done] = g, m
        it += 1

        # Critic snapshots: the first one right after warmup, before any update (the untrained baseline).
        if snapshot is not None and env_steps >= cfg.warmup_env_steps and env_steps >= next_snap:
            _log({**snapshot(buffer, env_steps), "env_steps": env_steps}, env_steps, writer, log_file)
            next_snap = (env_steps // snap_every + 1) * snap_every

        if env_steps >= cfg.warmup_env_steps and buffer.head > 2:
            for _ in range(cfg.grad_steps_per_iter):
                train_info = agent.update(buffer.sample(cfg.batch_size, cfg.gtse.discount), env, loss_mask_cubes)

        if it % cfg.logging.log_every == 0:
            info = {f"train/{k}": v for k, v in train_info.items()}
            if ep_success:
                info["train/success_at_episode_end"] = sum(ep_success) / len(ep_success)
                ep_success = []
            info.update({"env_steps": env_steps, "sps": env_steps / (time.time() - t0)})
            _log(info, env_steps, writer, log_file)
        if cfg.logging.eval_every and it % cfg.logging.eval_every == 0 and env_steps >= cfg.warmup_env_steps:
            _log({**evaluate(agent, eval_env, gs), "env_steps": env_steps}, env_steps, writer, log_file)
            last_eval = env_steps
        if cfg.logging.save_every and it % cfg.logging.save_every == 0:
            torch.save({**agent.state_dict(), "cfg": OmegaConf.to_container(cfg), "env_steps": env_steps}, os.path.join(run_dir, "checkpoint.pt"))

    if last_eval != env_steps:
        _log({**evaluate(agent, eval_env, gs), "env_steps": env_steps}, env_steps, writer, log_file)
    torch.save({**agent.state_dict(), "cfg": OmegaConf.to_container(cfg), "env_steps": env_steps}, os.path.join(run_dir, "checkpoint.pt"))
    log_file.close()


def _log(info, step, writer, log_file):
    print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in info.items()), flush=True)
    log_file.write(json.dumps(info) + "\n")
    log_file.flush()
    if writer is not None:
        for k, v in info.items():
            if k != "env_steps":
                writer.add_scalar(k, v, step)
