"""
Render dataset-frame videos with occupancy reward graphs attached.

The input is a reward extraction rewards.csv file. The script infers the run
config from a parent directory, loads raw frames from the dataset, and writes
one MP4 per trajectory to a folder next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0])

from wm_datasets.world_model_dataset_occupancy import create_world_model_dataset


def cfg_select(cfg, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def resolve_rewards_csv(rewards_csv: str) -> Path:
    path = Path(rewards_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"rewards.csv not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a CSV file, got: {path}")
    return path


def resolve_config_path(rewards_csv: Path, config_path: str | None) -> Path:
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    for parent in rewards_csv.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not infer config.yaml from parents of {rewards_csv}. "
        "Pass --config-path explicitly."
    )


def resolve_output_dir(rewards_csv: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return rewards_csv.parent / "reward_videos"


def load_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def create_dataset(cfg, split: str, dataset_root: str | None):
    root = dataset_root if dataset_root is not None else cfg_select(cfg, "dataset.loader.root")
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
        normalize_pixel=False,
        n_rollout=cfg_select(cfg, "dataset.loader.n_rollout"),
        random_seed=int(cfg_select(cfg, "dataset.loader.random_seed", 42)),
        slice_mode="random",
        stride=int(cfg_select(cfg, "dataset.loader.stride", 1)),
        resize_mode=cfg_select(cfg, "dataset.loader.resize_mode", "stretch"),
        root=root,
        image_key=cfg_select(cfg, "dataset.loader.image_key", "observation.image"),
        explicit_episodes=cfg_select(cfg, "dataset.loader.explicit_episodes"),
        num_tasks=int(cfg_select(cfg, "model.num_tasks", 50)),
    )


def parse_float(raw: str | None) -> float:
    if raw is None or raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def parse_int(raw: str | None) -> int:
    value = parse_float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Expected an integer-like value, got {raw!r}")
    return int(value)


def parse_indices(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def read_reward_rows(
    rewards_csv: Path,
    *,
    task_column: str,
    trajectory_column: str,
    frame_column: str,
    reward_column: str,
    gt_reward_column: str,
) -> dict[int, list[dict[str, float | int]]]:
    groups: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    with rewards_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {task_column, trajectory_column, frame_column, reward_column, gt_reward_column}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"{rewards_csv} is missing required columns: {missing}. "
                f"Available columns: {sorted(fieldnames)}"
            )

        for row in reader:
            traj_idx = parse_int(row.get(trajectory_column))
            groups[traj_idx].append(
                {
                    "task_id": parse_int(row.get(task_column)),
                    "traj_idx": traj_idx,
                    "frame_idx": parse_int(row.get(frame_column)),
                    "reward": parse_float(row.get(reward_column)),
                    "gt_reward": parse_float(row.get(gt_reward_column)),
                }
            )

    return {
        traj_idx: sorted(rows, key=lambda row: int(row["frame_idx"]))
        for traj_idx, rows in sorted(groups.items())
    }


def filter_trajectories(
    trajectory_rows: dict[int, list[dict[str, float | int]]],
    trajectory_indices: set[int] | None,
    max_trajectories: int | None,
) -> dict[int, list[dict[str, float | int]]]:
    selected = {
        traj_idx: rows
        for traj_idx, rows in trajectory_rows.items()
        if trajectory_indices is None or traj_idx in trajectory_indices
    }
    if max_trajectories is not None:
        selected = dict(list(selected.items())[:max_trajectories])
    if not selected:
        raise ValueError("No trajectories selected for rendering.")
    return selected


def tensor_frame_to_uint8(frame: torch.Tensor, resize_to: tuple[int, int] | None) -> np.ndarray:
    frame = frame.detach().cpu()
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor [C,H,W], got shape {tuple(frame.shape)}")
    if frame.shape[0] in (1, 3, 4):
        frame = frame.permute(1, 2, 0)

    arr = frame.numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)

    if resize_to is not None:
        image = Image.fromarray(arr)
        image = image.resize(resize_to, Image.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8)
    return arr


def finite_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def canvas_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def set_vertical_line(line, x_value: float) -> None:
    line.set_xdata([x_value, x_value])


def render_trajectory_video(
    dataset,
    traj_idx: int,
    rows: list[dict[str, float | int]],
    output_dir: Path,
    *,
    fps: int,
    dpi: int,
    frame_width: int,
    max_video_frames: int | None,
    video_ext: str,
) -> Path:
    if max_video_frames is not None:
        rows = rows[:max_video_frames]
    if not rows:
        raise ValueError(f"Trajectory {traj_idx} has no reward rows to render.")

    frame_indices = np.asarray([int(row["frame_idx"]) for row in rows], dtype=np.int64)
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    gt_rewards = np.asarray([float(row["gt_reward"]) for row in rows], dtype=np.float64)
    task_id = int(rows[0]["task_id"])

    raw_end = int(np.max(frame_indices)) + 1
    video = dataset.data_source.load_visual_frames(traj_idx, 0, raw_end, step=1)
    resized_frames = [
        tensor_frame_to_uint8(video[int(frame_idx)], resize_to=(frame_width, frame_width))
        for frame_idx in frame_indices
    ]

    reward_ylim = finite_limits(rewards)
    gt_ylim = finite_limits(gt_rewards)
    x_min = float(np.min(frame_indices))
    x_max = float(np.max(frame_indices))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    fig, (ax_image, ax_reward) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        dpi=dpi,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    image_artist = ax_image.imshow(resized_frames[0])
    ax_image.set_axis_off()
    ax_image.set_title(f"task {task_id}, trajectory {traj_idx}")

    reward_line, = ax_reward.plot(
        frame_indices,
        rewards,
        color="tab:blue",
        linewidth=1.7,
        label="model reward",
    )
    reward_marker, = ax_reward.plot([], [], "o", color="tab:blue", markersize=5)
    current_line = ax_reward.axvline(frame_indices[0], color="black", linewidth=1.1, alpha=0.75)
    ax_reward.set_xlabel("frame")
    ax_reward.set_ylabel("model reward")
    ax_reward.set_xlim(x_min, x_max)
    ax_reward.set_ylim(*reward_ylim)
    ax_reward.grid(True, alpha=0.25)

    ax_gt = ax_reward.twinx()
    gt_line = None
    gt_marker = None
    if np.isfinite(gt_rewards).any():
        gt_line, = ax_gt.plot(
            frame_indices,
            gt_rewards,
            color="tab:orange",
            linewidth=1.4,
            alpha=0.9,
            label="gt_reward",
        )
        gt_marker, = ax_gt.plot([], [], "o", color="tab:orange", markersize=5)
    ax_gt.set_ylabel("gt_reward")
    ax_gt.set_ylim(*gt_ylim)

    handles = [reward_line]
    labels = ["model reward"]
    if gt_line is not None:
        handles.append(gt_line)
        labels.append("gt_reward")
    ax_reward.legend(handles, labels, loc="lower right")

    info_text = ax_reward.text(
        0.02,
        0.98,
        "",
        transform=ax_reward.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
    )

    fig.tight_layout()
    output_path = output_dir / f"traj_{traj_idx:04d}_reward_video{video_ext}"
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame, frame_idx, reward, gt_reward in zip(resized_frames, frame_indices, rewards, gt_rewards):
            image_artist.set_data(frame)
            set_vertical_line(current_line, float(frame_idx))
            reward_marker.set_data([float(frame_idx)], [float(reward)])
            if gt_marker is not None and math.isfinite(float(gt_reward)):
                gt_marker.set_data([float(frame_idx)], [float(gt_reward)])
            elif gt_marker is not None:
                gt_marker.set_data([], [])

            gt_text = f"{gt_reward:.4g}" if math.isfinite(float(gt_reward)) else "nan"
            info_text.set_text(f"frame {frame_idx}\nreward {reward:.4g}\ngt_reward {gt_text}")
            writer.append_data(canvas_to_rgb(fig))

    plt.close(fig)
    return output_path


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive, got {args.dpi}")
    if args.frame_width <= 0:
        raise ValueError(f"--frame-width must be positive, got {args.frame_width}")
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise ValueError(f"--max-trajectories must be positive, got {args.max_trajectories}")
    if args.max_video_frames is not None and args.max_video_frames <= 0:
        raise ValueError(f"--max-video-frames must be positive, got {args.max_video_frames}")
    if not args.video_ext.startswith("."):
        raise ValueError(f"--video-ext must start with '.', got {args.video_ext!r}")


def run(args: argparse.Namespace) -> None:
    validate_args(args)

    rewards_csv = resolve_rewards_csv(args.rewards_csv)
    config_path = resolve_config_path(rewards_csv, args.config_path)
    output_dir = resolve_output_dir(rewards_csv, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] rewards_csv={rewards_csv}")
    print(f"[setup] config={config_path}")
    print(f"[setup] output_dir={output_dir}")

    cfg = load_config(config_path)
    trajectory_rows = read_reward_rows(
        rewards_csv,
        task_column=args.task_column,
        trajectory_column=args.trajectory_column,
        frame_column=args.frame_column,
        reward_column=args.reward_column,
        gt_reward_column=args.gt_reward_column,
    )
    selected = filter_trajectories(
        trajectory_rows,
        parse_indices(args.trajectory_indices),
        args.max_trajectories,
    )

    print(f"[data] Loading dataset split={args.split}...")
    dataset = create_dataset(cfg, args.split, args.dataset_root)
    print(f"[data] Rendering {len(selected)} trajectories...")

    written = []
    for traj_idx, rows in selected.items():
        print(f"[trajectory] {traj_idx}")
        video_path = render_trajectory_video(
            dataset,
            traj_idx,
            rows,
            output_dir,
            fps=args.fps,
            dpi=args.dpi,
            frame_width=args.frame_width,
            max_video_frames=args.max_video_frames,
            video_ext=args.video_ext,
        )
        written.append(video_path)
        print(f"[trajectory] wrote {video_path}")

    print(f"[done] wrote {len(written)} videos to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewards_csv", help="Path to a reward extraction rewards.csv file.")
    parser.add_argument("--config-path", default=None, help="Run config YAML. Defaults to nearest parent config.yaml.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset.loader.root / MT50_DATA_ROOT.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <directory containing rewards.csv>/reward_videos.",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--trajectory-indices", default=None, help="Comma-separated trajectory indices to render.")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Render at most this many trajectories.")
    parser.add_argument("--max-video-frames", type=int, default=None, help="Render at most this many frames per video.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--frame-width", type=int, default=360, help="Displayed dataset frame width and height.")
    parser.add_argument("--video-ext", default=".mp4")
    parser.add_argument("--reward-column", default="reward", help="Model reward column.")
    parser.add_argument("--gt-reward-column", default="gt_reward", help="Ground-truth reward column.")
    parser.add_argument("--frame-column", default="frame_idx", help="Frame index column.")
    parser.add_argument("--trajectory-column", default="traj_idx", help="Trajectory id column.")
    parser.add_argument("--task-column", default="task_id", help="Task id column.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
