"""
GTSE rewards from a trained NanoWM GTSE checkpoint (src/main_gtse.py).

Reward = the conditional term of dre_iclr/ogbench/impls/agents/gtse.py `_compute_critic` (cdr[-1]):

    r(s, a, g) = sum_{k=0}^{nfe-1} dt * sum_dims score_theta(g, t_k; s, a, task),   t_k = k * dt,  dt = t1_val / nfe

which approximates log p(g | s, a) - log N(g; 0, I). The marginal term (mdr) is not computed: with one goal per
trajectory it is a constant and does not change the per-trajectory ranking (VOC).

Writes the same rewards.csv / summary.json layout as extract_rewards_occupancy.py, so compute_voc_occupancy.py
and the other scripts read it unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from extract_rewards_occupancy import (
    build_windows,
    choose_trajectories,
    create_dataset,
    encode_video,
    load_checkpoint,
    load_config,
    load_trajectory,
    parse_indices,
    resolve_checkpoint,
    resolve_run_dir,
    state_columns,
)
from experiments.train_experiment_gtse import NanoWMGTSETrainingModule


@torch.no_grad()
def estimate_gtse_rewards(pl_module, contexts, action_windows, task_id, goal, t1_val, nfe, window_batch_size, device, use_target_model):
    model = pl_module.target_model if use_target_model else pl_module._online_model()
    model.eval()

    dt = t1_val / nfe
    t = torch.arange(nfe, device=device, dtype=torch.float32) * dt  # = linspace(0, t1_val - dt, nfe)

    all_rewards = []
    for start in tqdm(range(0, contexts.shape[0], window_batch_size), desc="reward windows", leave=False):
        context_batch = contexts[start : start + window_batch_size].to(device)
        bsz = context_batch.shape[0]
        action_batch = None if action_windows is None else action_windows[start : start + window_batch_size].to(device)
        task_batch = torch.full((bsz,), task_id, dtype=torch.long, device=device)

        rep_context = torch.repeat_interleave(context_batch, nfe, dim=0)
        rep_action = None if action_batch is None else torch.repeat_interleave(action_batch, nfe, dim=0)
        rep_task = torch.repeat_interleave(task_batch, nfe, dim=0)
        rep_goal = goal.to(device).expand(bsz * nfe, 1, *goal.shape[1:])
        rep_t = t.repeat(bsz)

        score = model(rep_goal, pl_module._tau_to_model_t(rep_t), context=rep_context, action=rep_action, task=rep_task, use_fp16=False)
        score = score.float().flatten(1).sum(dim=1)  # scalar time score
        all_rewards.append((dt * score.view(bsz, nfe)).sum(dim=1).cpu())

    return torch.cat(all_rewards, dim=0).numpy()


def write_plot(output_dir: Path, traj_idx: int, frame_indices, rewards, gt_rewards) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib is not installed; skipped PNG plots.")
        return
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(frame_indices, rewards, marker="o", linewidth=1.5, markersize=3, label="GTSE reward")
    ax.set_xlabel("frame")
    ax.set_ylabel("GTSE reward")
    ax.set_title(f"trajectory {traj_idx}")
    handles, labels = ax.get_legend_handles_labels()
    if np.isfinite(gt_rewards).any():
        ax_gt = ax.twinx()
        ax_gt.step(frame_indices, gt_rewards, where="post", color="tab:orange", linewidth=1.25, label="GT reward")
        ax_gt.set_ylabel("GT reward")
        gt_handles, gt_labels = ax_gt.get_legend_handles_labels()
        handles, labels = handles + gt_handles, labels + gt_labels
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"traj_{traj_idx:04d}_rewards.png", dpi=160)
    plt.close(fig)


def run(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = resolve_run_dir(args.run_dir)
    cfg, config_path = load_config(run_dir, args.config_path)
    ckpt_path = resolve_checkpoint(run_dir, args.ckpt_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / f"gtse_rewards_t1={args.t1_val}_nfe={args.nfe}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] run_dir={run_dir}")
    print(f"[setup] config={config_path}")
    print(f"[setup] checkpoint={ckpt_path}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] device={device} t1_val={args.t1_val} nfe={args.nfe}")

    print("[model] Loading GTSE module and checkpoint...")
    pl_module = NanoWMGTSETrainingModule(cfg)
    load_checkpoint(pl_module, ckpt_path)
    pl_module.to(device)
    pl_module.eval()
    pl_module.target_model.eval()
    pl_module.latent_codec.eval()

    print("[data] Loading dataset metadata/source...")
    dataset = create_dataset(cfg, args.split)
    traj_indices = choose_trajectories(dataset, parse_indices(args.trajectory_indices), args.num_trajectories)

    n_context = int(cfg.model.n_context_frames)
    rows = []
    summary = {
        "run_dir": str(run_dir), "config_path": str(config_path), "checkpoint_path": str(ckpt_path), "split": args.split,
        "t1_val": args.t1_val, "nfe": args.nfe, "window_batch_size": args.window_batch_size, "window_stride": args.window_stride,
        "use_target_model": args.use_target_model, "trajectories": [],
    }

    for traj_idx in traj_indices:
        print(f"[trajectory] {traj_idx}")
        video, actions, states, gt_rewards_all, gt_successes_all, task_id = load_trajectory(dataset, traj_idx, args.max_frames)
        z = encode_video(pl_module, video, device, args.encode_batch_size)

        goal_step = z.shape[0] - 1 if args.goal_step is None else int(args.goal_step)
        if not (0 <= goal_step < z.shape[0]):
            raise ValueError(f"goal_step={goal_step} is outside [0, {z.shape[0]}) for trajectory {traj_idx}.")
        goal = z[goal_step : goal_step + 1]

        starts, context_end_steps, contexts, action_windows = build_windows(z, actions, n_context=n_context, window_stride=args.window_stride)
        rewards = estimate_gtse_rewards(
            pl_module, contexts, action_windows if bool(cfg.model.use_action) else None, task_id, goal,
            args.t1_val, args.nfe, args.window_batch_size, device, args.use_target_model,
        )

        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards) + 1e-8)
        frame_indices = context_end_steps * int(dataset.frame_interval)
        gt_rewards = gt_rewards_all[context_end_steps]
        gt_successes = gt_successes_all[context_end_steps]
        if not args.no_plots:
            write_plot(output_dir, traj_idx, frame_indices, rewards, gt_rewards)

        for i, (start_step, context_end_step, frame_idx, reward, gt_reward, gt_success) in enumerate(
            zip(starts, context_end_steps, frame_indices, rewards, gt_rewards, gt_successes)
        ):
            row = {
                "traj_idx": traj_idx, "task_id": task_id, "window_index": i, "window_start_step": start_step,
                "context_end_step": int(context_end_step), "frame_idx": int(frame_idx), "goal_step": int(goal_step),
                "goal_frame_idx": int(goal_step * int(dataset.frame_interval)), "reward": float(reward),
                "reward_zscore": float((reward - reward_mean) / reward_std), "gt_reward": float(gt_reward), "gt_success": float(gt_success),
            }
            row.update(state_columns(states, int(context_end_step)))
            rows.append(row)

        summary["trajectories"].append({
            "traj_idx": traj_idx, "task_id": task_id, "num_frames": int(z.shape[0]), "num_windows": int(len(rewards)),
            "goal_step": int(goal_step), "reward_min": float(np.min(rewards)), "reward_max": float(np.max(rewards)), "reward_mean": reward_mean,
        })

    csv_path = output_dir / "rewards.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {len(rows)} reward rows to {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=None, help="Training run directory. Defaults to newest results/*-mt50.")
    parser.add_argument("--config-path", default=None, help="Run config YAML. Defaults to <run-dir>/config.yaml.")
    parser.add_argument("--ckpt-path", default=None, help="Checkpoint path. Defaults to newest checkpoint under run-dir.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <run-dir>/gtse_rewards_t1=<t1>_nfe=<nfe>.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--trajectory-indices", default=None, help="Comma-separated episode indices to inspect.")
    parser.add_argument("--num-trajectories", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames loaded per trajectory.")
    parser.add_argument("--goal-step", type=int, default=None, help="Goal frame in processed trajectory coordinates. Default: last.")
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--t1-val", type=float, default=0.95, help="Upper end of the time integral (gtse.py t1_val).")
    parser.add_argument("--nfe", type=int, default=30, help="Number of integration steps (gtse.py nfe).")
    parser.add_argument("--window-batch-size", type=int, default=8, help="Windows per forward pass; each pass is window_batch_size * nfe items.")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-target-model", action="store_true", help="Use target_model instead of online model.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
