"""
Compute value order correlation (VOC) for extracted occupancy rewards.

VOC is computed per trajectory as Spearman's rank correlation between the model
reward sequence and frame index. The per-trajectory VOC values are then averaged
by task_id and overall.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def resolve_rewards_csv(rewards_csv: str) -> Path:
    path = Path(rewards_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"rewards.csv not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a CSV file, got: {path}")
    return path


def safe_alpha_name(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def resolve_output_dir(rewards_csv: Path, output_dir: str | None, alpha: float) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return rewards_csv.parent / f"voc_alpha_{safe_alpha_name(alpha)}"


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


def read_rows(
    rewards_csv: Path,
    *,
    task_column: str,
    trajectory_column: str,
    frame_column: str,
    reward_column: str,
) -> dict[tuple[int, int], list[dict[str, float | int]]]:
    groups: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)
    with rewards_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {task_column, trajectory_column, frame_column, reward_column}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"{rewards_csv} is missing required columns: {missing}. "
                f"Available columns: {sorted(fieldnames)}"
            )

        for row in reader:
            task_id = parse_int(row.get(task_column))
            traj_idx = parse_int(row.get(trajectory_column))
            frame_idx = parse_float(row.get(frame_column))
            reward = parse_float(row.get(reward_column))
            groups[(task_id, traj_idx)].append(
                {
                    "task_id": task_id,
                    "traj_idx": traj_idx,
                    "frame_idx": frame_idx,
                    "reward": reward,
                }
            )
    return groups


def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values.copy()

    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, values.size):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end

    return ranks


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray, min_points: int) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < min_points:
        return math.nan

    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_std = float(np.std(x_rank))
    y_std = float(np.std(y_rank))
    if x_std == 0.0 or y_std == 0.0:
        return math.nan

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr))


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(np.std(arr))


def compute_voc(
    groups: dict[tuple[int, int], list[dict[str, float | int]]],
    *,
    alpha: float,
    min_points: int,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[str, float | int | str | None]]:
    trajectory_rows: list[dict[str, float | int]] = []

    for (task_id, traj_idx), rows in sorted(groups.items()):
        sorted_rows = sorted(rows, key=lambda row: (float(row["frame_idx"]), int(row["traj_idx"])))
        frames = np.asarray([float(row["frame_idx"]) for row in sorted_rows], dtype=np.float64)
        rewards = np.asarray([float(row["reward"]) for row in sorted_rows], dtype=np.float64)

        model_mask = np.isfinite(frames) & np.isfinite(rewards)
        model_frames = frames[model_mask]
        model_rewards = rewards[model_mask]
        model_ema_rewards = exponential_moving_average(model_rewards, alpha)
        model_voc = spearman_rank_correlation(model_ema_rewards, model_frames, min_points=min_points)

        trajectory_rows.append(
            {
                "task_id": int(task_id),
                "traj_idx": int(traj_idx),
                "num_model_points": int(model_frames.size),
                "frame_start": float(frames[0]) if frames.size else math.nan,
                "frame_end": float(frames[-1]) if frames.size else math.nan,
                "model_voc": model_voc,
            }
        )

    task_groups: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in trajectory_rows:
        task_groups[int(row["task_id"])].append(row)

    task_rows: list[dict[str, float | int]] = []
    for task_id, rows in sorted(task_groups.items()):
        model_voc_values = [float(row["model_voc"]) for row in rows]
        valid_model_values = [value for value in model_voc_values if math.isfinite(value)]
        task_rows.append(
            {
                "task_id": int(task_id),
                "num_trajectories": int(len(rows)),
                "num_valid_model_trajectories": int(len(valid_model_values)),
                "model_voc_mean": finite_mean(model_voc_values),
                "model_voc_std": finite_std(model_voc_values),
            }
        )

    trajectory_model_vocs = [float(row["model_voc"]) for row in trajectory_rows]
    task_model_voc_means = [float(row["model_voc_mean"]) for row in task_rows]
    summary: dict[str, float | int | str | None] = {
        "alpha": float(alpha),
        "num_tasks": int(len(task_rows)),
        "num_trajectories": int(len(trajectory_rows)),
        "num_valid_model_trajectories": int(sum(math.isfinite(value) for value in trajectory_model_vocs)),
        "overall_model_voc_mean": finite_mean(trajectory_model_vocs),
        "overall_model_voc_std": finite_std(trajectory_model_vocs),
        "task_macro_model_voc_mean": finite_mean(task_model_voc_means),
        "task_macro_model_voc_std": finite_std(task_model_voc_means),
    }

    return trajectory_rows, task_rows, summary


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.alpha <= 1.0):
        raise ValueError(f"--alpha must be in (0, 1], got {args.alpha}")
    if args.min_points < 2:
        raise ValueError(f"--min-points must be at least 2, got {args.min_points}")


def run(args: argparse.Namespace) -> None:
    validate_args(args)

    rewards_csv = resolve_rewards_csv(args.rewards_csv)
    output_dir = resolve_output_dir(rewards_csv, args.output_dir, args.alpha)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] rewards_csv={rewards_csv}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] alpha={args.alpha}")

    groups = read_rows(
        rewards_csv,
        task_column=args.task_column,
        trajectory_column=args.trajectory_column,
        frame_column=args.frame_column,
        reward_column=args.reward_column,
    )
    trajectory_rows, task_rows, summary = compute_voc(
        groups,
        alpha=args.alpha,
        min_points=args.min_points,
    )
    summary["rewards_csv"] = str(rewards_csv)
    summary["reward_column"] = args.reward_column
    summary["frame_column"] = args.frame_column
    summary["trajectory_column"] = args.trajectory_column
    summary["task_column"] = args.task_column

    write_csv(output_dir / "voc_by_trajectory.csv", trajectory_rows)
    write_csv(output_dir / "voc_by_task.csv", task_rows)
    with (output_dir / "voc_summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    print(f"[done] wrote {len(trajectory_rows)} trajectory VOC rows")
    print(f"[done] wrote {len(task_rows)} task VOC rows")
    print(f"[summary] overall_model_voc_mean={summary['overall_model_voc_mean']}")
    print(f"[summary] task_macro_model_voc_mean={summary['task_macro_model_voc_mean']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewards_csv", help="Path to a reward extraction rewards.csv file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <directory containing rewards.csv>/voc_alpha_<alpha>.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="EMA update weight. ema[t] = alpha * reward[t] + (1 - alpha) * ema[t-1]. alpha=1 disables smoothing.",
    )
    parser.add_argument("--min-points", type=int, default=2, help="Minimum finite points needed for a trajectory VOC.")
    parser.add_argument("--reward-column", default="reward", help="Reward column to score.")
    parser.add_argument("--frame-column", default="frame_idx", help="Frame index column.")
    parser.add_argument("--trajectory-column", default="traj_idx", help="Trajectory id column.")
    parser.add_argument("--task-column", default="task_id", help="Task id column.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
