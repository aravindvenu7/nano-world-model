"""
Inspect ORS rewards from a trained NanoWM occupancy checkpoint.

This script intentionally does not train a reward network. It directly computes
the Monte Carlo estimate from ORS Eq. 4:

    r(s, a, g) = -E_{x0,t} || v_theta(t, s, a, x_t) - (g - x0) ||^2

where x_t = (1 - t) x0 + t g.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm

sys.path.append(os.path.split(sys.path[0])[0])

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from experiments.train_experiment_occupancy import NanoWMOccupancyTrainingModule
from wm_datasets.world_model_dataset_occupancy import create_world_model_dataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cfg_select(cfg, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir is not None:
        return Path(run_dir).expanduser().resolve()

    candidates = sorted(
        Path("results").glob("*-mt50"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No run_dir supplied and no results/*-mt50 runs were found.")
    return candidates[0].resolve()


def resolve_checkpoint(run_dir: Path, ckpt_path: str | None) -> Path:
    if ckpt_path is not None:
        return Path(ckpt_path).expanduser().resolve()

    latest = sorted(
        (run_dir / "checkpoints" / "latest").glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if latest:
        return latest[0].resolve()

    across = sorted(
        (run_dir / "checkpoints" / "across_timesteps").glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if across:
        return across[0].resolve()

    raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints.")


def load_config(run_dir: Path, config_path: str | None):
    path = Path(config_path).expanduser().resolve() if config_path else run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.infra.compile = False
    return cfg, path


def load_checkpoint(pl_module: NanoWMOccupancyTrainingModule, ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    has_prefix = any(k.startswith("model.") or k.startswith("target_model.") for k in state_dict)
    if not has_prefix:
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    ckpt_has_orig_mod = any("model._orig_mod." in k for k in state_dict)
    model_compiled = hasattr(pl_module.model, "_orig_mod")
    if ckpt_has_orig_mod and not model_compiled:
        state_dict = {
            k.replace("model._orig_mod.", "model.", 1) if k.startswith("model._orig_mod.") else k: v
            for k, v in state_dict.items()
        }
    elif model_compiled and not ckpt_has_orig_mod:
        state_dict = {
            k.replace("model.", "model._orig_mod.", 1)
            if k.startswith("model.") and not k.startswith("model._orig_mod.")
            else k: v
            for k, v in state_dict.items()
        }

    missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict, strict=False)
    model_missing = [k for k in missing_keys if k.startswith("model.")]
    if model_missing:
        raise RuntimeError(f"Checkpoint loading failed; missing model keys: {model_missing}")
    if unexpected_keys:
        print(f"[checkpoint] Ignored unexpected keys: {unexpected_keys}")
    if any(k.startswith("target_model.") for k in missing_keys):
        pl_module.target_model.load_state_dict(pl_module._online_model().state_dict())
        print("[checkpoint] No target_model in checkpoint; copied online model into target_model.")


def create_dataset(cfg, split: str):
    slice_mode = (
        cfg_select(cfg, "dataset.loader.val_slice_mode", "exhaustive")
        if split == "val"
        else cfg_select(cfg, "dataset.loader.train_slice_mode", "random")
    )
    return create_world_model_dataset(
        dataset_name=cfg.dataset.name,
        data_path=cfg_select(cfg, "dataset.loader.data_path"),
        num_frames=int(cfg.model.num_frames),
        frame_interval=int(cfg.dataset.frame_interval),
        image_size=(int(cfg.model.image_size), int(cfg.model.image_size)),
        split=split,
        split_ratio=float(cfg_select(cfg, "dataset.loader.split_ratio", 0.9)),
        normalize_action=bool(cfg_select(cfg, "dataset.loader.normalize_action", True)),
        normalize_state=bool(cfg_select(cfg, "dataset.loader.normalize_state", False)),
        normalize_pixel=True,
        n_rollout=cfg_select(cfg, "dataset.loader.n_rollout"),
        random_seed=int(cfg_select(cfg, "dataset.loader.random_seed", 42)),
        slice_mode=slice_mode,
        stride=int(cfg_select(cfg, "dataset.loader.stride", 1)),
        resize_mode=cfg_select(cfg, "dataset.loader.resize_mode", "stretch"),
        root=cfg_select(cfg, "dataset.loader.root"),
        image_key=cfg_select(cfg, "dataset.loader.image_key", "observation.image"),
        explicit_episodes=cfg_select(cfg, "dataset.loader.explicit_episodes"),
        num_tasks=int(cfg_select(cfg, "model.num_tasks", 50)),
    )


def parse_indices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def choose_trajectories(dataset, requested: list[int] | None, count: int) -> list[int]:
    if requested is not None:
        return requested
    traj_indices = getattr(dataset, "traj_indices", list(range(dataset.num_trajectories)))
    return [int(idx) for idx in traj_indices[:count]]


def preprocess_video(dataset, video: torch.Tensor) -> torch.Tensor:
    if video.shape[-2:] != dataset.image_size:
        if dataset.resize_mode == "stretch":
            video = F.interpolate(video, size=dataset.image_size, mode="bilinear", align_corners=False)
        elif dataset.resize_mode == "pad":
            t, c, h, w = video.shape
            target_h, target_w = dataset.image_size
            scale = min(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            video = F.pad(video, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
        else:
            raise ValueError(f"Unknown resize_mode: {dataset.resize_mode}")

    if dataset.normalize_pixel:
        video = video * 2.0 - 1.0
    return video


def nested_item_value(item, keys: tuple[str, ...]):
    for key in keys:
        if key in item:
            return item[key]

        cur = item
        found = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                found = False
                break
        if found:
            return cur
    return None


def scalar_or_nan(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def load_ground_truth_signals(dataset, traj_idx: int, usable_frames: int) -> tuple[np.ndarray, np.ndarray]:
    data_source = dataset.data_source
    if not hasattr(data_source, "dataset") or not hasattr(data_source, "_episode_global_range"):
        missing = np.full((usable_frames,), np.nan, dtype=np.float32)
        return missing, missing.copy()

    abs_start, _ = data_source._episode_global_range(traj_idx)
    rewards = []
    successes = []
    for rel_idx in range(0, usable_frames * int(dataset.frame_interval), int(dataset.frame_interval)):
        item = data_source.dataset[abs_start + rel_idx]
        rewards.append(scalar_or_nan(nested_item_value(item, ("next.reward", "reward"))))
        successes.append(scalar_or_nan(nested_item_value(item, ("next.success", "success"))))

    return np.asarray(rewards, dtype=np.float32), np.asarray(successes, dtype=np.float32)


def load_trajectory(dataset, traj_idx: int, max_frames: int | None):
    traj = dataset.data_source.load_trajectory(traj_idx)
    frame_interval = int(dataset.frame_interval)
    usable_frames = traj.seq_length // frame_interval
    if max_frames is not None:
        usable_frames = min(usable_frames, max_frames)
    if usable_frames < 2:
        raise ValueError(f"Trajectory {traj_idx} is too short after frame_interval={frame_interval}.")

    raw_end = usable_frames * frame_interval
    video = dataset.data_source.load_visual_frames(traj_idx, 0, raw_end, step=frame_interval)
    video = preprocess_video(dataset, video[:usable_frames])

    actions_full = traj.actions[:raw_end]
    if dataset.normalize_action:
        actions_full = (actions_full - dataset._raw_action_mean) / dataset._raw_action_std
    if frame_interval > 1:
        actions = actions_full.reshape(usable_frames, frame_interval, -1).reshape(usable_frames, -1)
    else:
        actions = actions_full[:usable_frames]

    states = traj.states[:raw_end:frame_interval][:usable_frames] if dataset.state_dim > 0 else None
    gt_rewards, gt_successes = load_ground_truth_signals(dataset, traj_idx, usable_frames)
    task_id = int(traj.meta["task_index"]) if traj.meta and "task_index" in traj.meta else 0
    return video, actions, states, gt_rewards, gt_successes, task_id


@torch.no_grad()
def encode_video(pl_module: NanoWMOccupancyTrainingModule, video: torch.Tensor, device: str, batch_size: int):
    latents = []
    pl_module.latent_codec.to(device)
    for start in range(0, video.shape[0], batch_size):
        chunk = video[start : start + batch_size].to(device)
        latents.append(pl_module._vae_encode(chunk).detach().cpu())
    return torch.cat(latents, dim=0)


def build_windows(z: torch.Tensor, actions: torch.Tensor | None, n_context: int, window_stride: int):
    starts = list(range(0, z.shape[0] - n_context + 1, window_stride))
    contexts = torch.stack([z[start : start + n_context] for start in starts], dim=0)
    if actions is None:
        action_windows = None
    else:
        action_windows = torch.stack([actions[start : start + n_context] for start in starts], dim=0)
    context_end_steps = np.array([start + n_context - 1 for start in starts], dtype=np.int64)
    return starts, context_end_steps, contexts, action_windows


@torch.no_grad()
def estimate_ors_rewards(
    pl_module: NanoWMOccupancyTrainingModule,
    contexts: torch.Tensor,
    action_windows: torch.Tensor | None,
    task_id: int,
    goal: torch.Tensor,
    mc_samples: int,
    mc_chunk_size: int,
    window_batch_size: int,
    device: str,
    use_target_model: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model = pl_module.target_model if use_target_model else pl_module._online_model()
    model.eval()

    all_costs = []
    for start in tqdm(range(0, contexts.shape[0], window_batch_size), desc="reward windows", leave=False):
        context_batch = contexts[start : start + window_batch_size].to(device)
        bsz = context_batch.shape[0]
        action_batch = None if action_windows is None else action_windows[start : start + window_batch_size].to(device)
        task_batch = torch.full((bsz,), task_id, dtype=torch.long, device=device)
        goal_batch = goal.to(device).expand(bsz, 1, *goal.shape[1:])

        cost_sum = torch.zeros(bsz, device=device)
        sample_count = 0
        while sample_count < mc_samples:
            cur_samples = min(mc_chunk_size, mc_samples - sample_count)
            rep_context = torch.repeat_interleave(context_batch, cur_samples, dim=0)
            rep_action = None if action_batch is None else torch.repeat_interleave(action_batch, cur_samples, dim=0)
            rep_task = torch.repeat_interleave(task_batch, cur_samples, dim=0)
            rep_goal = torch.repeat_interleave(goal_batch, cur_samples, dim=0)

            noise = torch.randn_like(rep_goal)
            tau = torch.rand(rep_goal.shape[0], device=device, dtype=rep_goal.dtype)
            tau = (0.4 * tau) + 0.3
            #tau = torch.sigmoid(0.3 * torch.randn(rep_goal.shape[0], device=device, dtype=rep_goal.dtype))
            x_t = pl_module._interpolate(noise, rep_goal, tau)
            model_t = pl_module._tau_to_model_t(tau)
            target_velocity = rep_goal - noise

            pred_velocity = model(
                x_t,
                model_t,
                context=rep_context,
                action=rep_action,
                task=rep_task,
                use_fp16=False,
            )
            costs = (pred_velocity - target_velocity).square().flatten(1).mean(dim=1)
            costs = costs.view(bsz, cur_samples).sum(dim=1)
            cost_sum += costs
            sample_count += cur_samples

        all_costs.append((cost_sum / mc_samples).detach().cpu())

    costs_np = torch.cat(all_costs, dim=0).numpy()
    return -costs_np, costs_np


def write_plot(
    output_dir: Path,
    traj_idx: int,
    frame_indices: np.ndarray,
    rewards: np.ndarray,
    gt_rewards: np.ndarray,
    gt_successes: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib is not installed; skipped PNG plots.")
        return

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(frame_indices, rewards, marker="o", linewidth=1.5, markersize=3, label="ORS reward")
    ax.set_xlabel("frame")
    ax.set_ylabel("ORS reward")
    ax.set_title(f"trajectory {traj_idx}")

    overlay_handles, overlay_labels = ax.get_legend_handles_labels()
    if np.isfinite(gt_rewards).any():
        ax_gt = ax.twinx()
        ax_gt.step(frame_indices, gt_rewards, where="post", color="tab:orange", linewidth=1.25, label="GT reward")
        ax_gt.set_ylabel("GT reward")
        gt_handles, gt_labels = ax_gt.get_legend_handles_labels()
        overlay_handles += gt_handles
        overlay_labels += gt_labels
        if np.isfinite(gt_successes).any():
            success_mask = np.isfinite(gt_successes) & (gt_successes > 0)
            if success_mask.any():
                ymin, ymax = ax.get_ylim()
                ax.scatter(
                    frame_indices[success_mask],
                    np.full(success_mask.sum(), ymax),
                    color="tab:green",
                    marker="|",
                    s=80,
                    label="GT success",
                )
                ax.set_ylim(ymin, ymax)
                overlay_handles, overlay_labels = ax.get_legend_handles_labels()
                overlay_handles += gt_handles
                overlay_labels += gt_labels

    if overlay_handles:
        ax.legend(overlay_handles, overlay_labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"traj_{traj_idx:04d}_rewards.png", dpi=160)
    plt.close(fig)


def state_columns(states: torch.Tensor | None, step: int) -> dict[str, float]:
    if states is None:
        return {}
    values = states[step].detach().cpu().numpy().astype(float)
    return {f"state_{i}": float(v) for i, v in enumerate(values)}


def run(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = resolve_run_dir(args.run_dir)
    cfg, config_path = load_config(run_dir, args.config_path)
    ckpt_path = resolve_checkpoint(run_dir, args.ckpt_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "ors_rewards_occupancy"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] run_dir={run_dir}")
    print(f"[setup] config={config_path}")
    print(f"[setup] checkpoint={ckpt_path}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] device={device}")

    print("[model] Loading occupancy module and checkpoint...")
    pl_module = NanoWMOccupancyTrainingModule(cfg)
    load_checkpoint(pl_module, ckpt_path)
    pl_module.to(device)
    pl_module.eval()
    pl_module.target_model.eval()
    pl_module.latent_codec.eval()

    print("[data] Loading dataset metadata/source...")
    dataset = create_dataset(cfg, args.split)
    traj_indices = choose_trajectories(dataset, parse_indices(args.trajectory_indices), args.num_trajectories)

    n_context = int(cfg.model.n_context_frames)
    csv_path = output_dir / "rewards.csv"
    rows = []
    summary = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "split": args.split,
        "mc_samples": args.mc_samples,
        "mc_chunk_size": args.mc_chunk_size,
        "window_batch_size": args.window_batch_size,
        "window_stride": args.window_stride,
        "use_target_model": args.use_target_model,
        "trajectories": [],
    }

    for traj_idx in traj_indices:
        print(f"[trajectory] {traj_idx}")
        video, actions, states, gt_rewards_all, gt_successes_all, task_id = load_trajectory(
            dataset,
            traj_idx,
            args.max_frames,
        )
        z = encode_video(pl_module, video, device, args.encode_batch_size)

        goal_step = z.shape[0] - 1 if args.goal_step is None else int(args.goal_step)
        if not (0 <= goal_step < z.shape[0]):
            raise ValueError(f"goal_step={goal_step} is outside [0, {z.shape[0]}) for trajectory {traj_idx}.")
        goal = z[goal_step : goal_step + 1]

        starts, context_end_steps, contexts, action_windows = build_windows(
            z,
            actions,
            n_context=n_context,
            window_stride=args.window_stride,
        )
        rewards, costs = estimate_ors_rewards(
            pl_module=pl_module,
            contexts=contexts,
            action_windows=action_windows if bool(cfg.model.use_action) else None,
            task_id=task_id,
            goal=goal,
            mc_samples=args.mc_samples,
            mc_chunk_size=args.mc_chunk_size,
            window_batch_size=args.window_batch_size,
            device=device,
            use_target_model=args.use_target_model,
        )

        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards) + 1e-8)
        frame_indices = context_end_steps * int(dataset.frame_interval)
        gt_rewards = gt_rewards_all[context_end_steps]
        gt_successes = gt_successes_all[context_end_steps]

        # np.savez(
        #     output_dir / f"traj_{traj_idx:04d}_rewards.npz",
        #     rewards=rewards,
        #     costs=costs,
        #     gt_rewards=gt_rewards,
        #     gt_successes=gt_successes,
        #     gt_rewards_all=gt_rewards_all,
        #     gt_successes_all=gt_successes_all,
        #     context_end_steps=context_end_steps,
        #     frame_indices=frame_indices,
        #     goal_step=np.array(goal_step, dtype=np.int64),
        #     task_id=np.array(task_id, dtype=np.int64),
        #     states=np.empty((0,), dtype=np.float32) if states is None else states.detach().cpu().numpy(),
        # )
        if not args.no_plots:
            write_plot(output_dir, traj_idx, frame_indices, rewards, gt_rewards, gt_successes)

        for i, (start_step, context_end_step, frame_idx, reward, cost, gt_reward, gt_success) in enumerate(
            zip(starts, context_end_steps, frame_indices, rewards, costs, gt_rewards, gt_successes)
        ):
            row = {
                "traj_idx": traj_idx,
                "task_id": task_id,
                "window_index": i,
                "window_start_step": start_step,
                "context_end_step": int(context_end_step),
                "frame_idx": int(frame_idx),
                "goal_step": int(goal_step),
                "goal_frame_idx": int(goal_step * int(dataset.frame_interval)),
                "reward": float(reward),
                "cost": float(cost),
                "reward_zscore": float((reward - reward_mean) / reward_std),
                "gt_reward": float(gt_reward),
                "gt_success": float(gt_success),
            }
            row.update(state_columns(states, int(context_end_step)))
            rows.append(row)

        finite_gt_rewards = gt_rewards[np.isfinite(gt_rewards)]
        finite_gt_successes = gt_successes[np.isfinite(gt_successes)]
        summary["trajectories"].append(
            {
                "traj_idx": traj_idx,
                "task_id": task_id,
                "num_frames": int(z.shape[0]),
                "num_windows": int(len(rewards)),
                "goal_step": int(goal_step),
                "reward_min": float(np.min(rewards)),
                "reward_max": float(np.max(rewards)),
                "reward_mean": reward_mean,
                "gt_reward_sum": float(np.sum(finite_gt_rewards)) if finite_gt_rewards.size else None,
                "gt_reward_max": float(np.max(finite_gt_rewards)) if finite_gt_rewards.size else None,
                "gt_success_count": int(np.sum(finite_gt_successes > 0)) if finite_gt_successes.size else None,
            }
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote {len(rows)} reward rows to {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Training run directory. Defaults to newest results/*-mt50.")
    parser.add_argument("--config-path", default=None, help="Run config YAML. Defaults to <run-dir>/config.yaml.")
    parser.add_argument("--ckpt-path", default=None, help="Checkpoint path. Defaults to newest checkpoint under run-dir.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <run-dir>/ors_rewards_occupancy.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--trajectory-indices", default=None, help="Comma-separated episode indices to inspect.")
    parser.add_argument("--num-trajectories", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames loaded per trajectory.")
    parser.add_argument("--goal-step", type=int, default=None, help="Goal frame in processed trajectory coordinates. Default: last.")
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--mc-chunk-size", type=int, default=10)
    parser.add_argument("--window-batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-target-model", action="store_true", help="Use target_model instead of online model.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
