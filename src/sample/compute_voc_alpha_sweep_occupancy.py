"""
Sweep EMA alpha values for occupancy reward value order correlation (VOC).

For each trajectory, this script computes Spearman's rank correlation between
EMA-smoothed model rewards and frame index for alpha values:

    1.0, 0.9, ..., 0.1, 0.09, 0.08, ..., 0.01

It writes the full alpha sweep table, plots VOC vs alpha with all trajectories
on one graph, and reports the trajectory-weighted average of each trajectory's
maximum VOC over the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_voc_occupancy import (
    exponential_moving_average,
    finite_mean,
    finite_std,
    json_safe,
    read_rows,
    resolve_rewards_csv,
    spearman_rank_correlation,
    write_csv,
)


def default_alpha_values() -> list[float]:
    coarse = [round(i / 10.0, 2) for i in range(10, 0, -1)]
    fine = [round(i / 100.0, 2) for i in range(9, 0, -1)]
    return coarse + fine


def resolve_output_dir(rewards_csv: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return rewards_csv.parent / "voc_alpha_sweep"


def trajectory_arrays(
    groups: dict[tuple[int, int], list[dict[str, float | int]]],
) -> dict[tuple[int, int], dict[str, np.ndarray | int | float]]:
    trajectories: dict[tuple[int, int], dict[str, np.ndarray | int | float]] = {}
    for (task_id, traj_idx), rows in sorted(groups.items()):
        sorted_rows = sorted(rows, key=lambda row: (float(row["frame_idx"]), int(row["traj_idx"])))
        frames = np.asarray([float(row["frame_idx"]) for row in sorted_rows], dtype=np.float64)
        rewards = np.asarray([float(row["reward"]) for row in sorted_rows], dtype=np.float64)
        mask = np.isfinite(frames) & np.isfinite(rewards)
        frames = frames[mask]
        rewards = rewards[mask]
        trajectories[(task_id, traj_idx)] = {
            "task_id": int(task_id),
            "traj_idx": int(traj_idx),
            "frames": frames,
            "rewards": rewards,
            "frame_start": float(frames[0]) if frames.size else math.nan,
            "frame_end": float(frames[-1]) if frames.size else math.nan,
        }
    return trajectories


def compute_sweep_rows(
    trajectories: dict[tuple[int, int], dict[str, np.ndarray | int | float]],
    alpha_values: list[float],
    min_points: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for (task_id, traj_idx), trajectory in sorted(trajectories.items()):
        frames = trajectory["frames"]
        rewards = trajectory["rewards"]
        assert isinstance(frames, np.ndarray)
        assert isinstance(rewards, np.ndarray)

        for alpha in alpha_values:
            smoothed_rewards = exponential_moving_average(rewards, alpha)
            voc = spearman_rank_correlation(smoothed_rewards, frames, min_points=min_points)
            rows.append(
                {
                    "alpha": float(alpha),
                    "task_id": int(task_id),
                    "traj_idx": int(traj_idx),
                    "num_model_points": int(frames.size),
                    "frame_start": float(trajectory["frame_start"]),
                    "frame_end": float(trajectory["frame_end"]),
                    "model_voc": voc,
                }
            )
    return rows


def summarize_by_alpha(sweep_rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    by_alpha: dict[float, list[float]] = defaultdict(list)
    for row in sweep_rows:
        by_alpha[float(row["alpha"])].append(float(row["model_voc"]))

    rows: list[dict[str, float | int]] = []
    for alpha in sorted(by_alpha.keys(), reverse=True):
        vocs = by_alpha[alpha]
        valid_vocs = [value for value in vocs if math.isfinite(value)]
        rows.append(
            {
                "alpha": float(alpha),
                "num_trajectories": int(len(vocs)),
                "num_valid_trajectories": int(len(valid_vocs)),
                "overall_model_voc_mean": finite_mean(vocs),
                "overall_model_voc_std": finite_std(vocs),
            }
        )
    return rows


def summarize_best_by_trajectory(
    sweep_rows: list[dict[str, float | int]],
) -> tuple[list[dict[str, float | int]], dict[str, float | int | list[float] | str | None]]:
    by_trajectory: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)
    for row in sweep_rows:
        by_trajectory[(int(row["task_id"]), int(row["traj_idx"]))].append(row)

    best_rows: list[dict[str, float | int]] = []
    for (task_id, traj_idx), rows in sorted(by_trajectory.items()):
        valid_rows = [row for row in rows if math.isfinite(float(row["model_voc"]))]
        if valid_rows:
            best_row = max(valid_rows, key=lambda row: (float(row["model_voc"]), float(row["alpha"])))
            best_alpha = float(best_row["alpha"])
            max_voc = float(best_row["model_voc"])
        else:
            best_alpha = math.nan
            max_voc = math.nan

        best_rows.append(
            {
                "task_id": int(task_id),
                "traj_idx": int(traj_idx),
                "num_alphas": int(len(rows)),
                "num_valid_alphas": int(len(valid_rows)),
                "best_alpha": best_alpha,
                "max_model_voc": max_voc,
            }
        )

    max_vocs = [float(row["max_model_voc"]) for row in best_rows]
    summary: dict[str, float | int | list[float] | str | None] = {
        "num_trajectories": int(len(best_rows)),
        "num_valid_trajectories": int(sum(math.isfinite(value) for value in max_vocs)),
        "average_max_model_voc": finite_mean(max_vocs),
        "std_max_model_voc": finite_std(max_vocs),
    }
    return best_rows, summary


def write_voc_plot(
    output_dir: Path,
    sweep_rows: list[dict[str, float | int]],
    *,
    max_labeled_trajectories: int,
) -> Path:
    by_trajectory: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)
    for row in sweep_rows:
        by_trajectory[(int(row["task_id"]), int(row["traj_idx"]))].append(row)

    fig, ax = plt.subplots(figsize=(11, 7))
    should_label = len(by_trajectory) <= max_labeled_trajectories
    cmap = plt.get_cmap("tab20")

    for i, ((task_id, traj_idx), rows) in enumerate(sorted(by_trajectory.items())):
        sorted_rows = sorted(rows, key=lambda row: float(row["alpha"]))
        alphas = [float(row["alpha"]) for row in sorted_rows]
        vocs = [float(row["model_voc"]) for row in sorted_rows]
        label = f"task {task_id}, traj {traj_idx}" if should_label else None
        ax.plot(
            alphas,
            vocs,
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            alpha=0.75,
            color=cmap(i % cmap.N),
            label=label,
        )

    ax.set_xlabel("EMA alpha")
    ax.set_ylabel("VOC")
    ax.set_title("VOC vs EMA alpha by trajectory")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    if should_label:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.tight_layout()
    plot_path = output_dir / "voc_vs_alpha_by_trajectory.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def validate_args(args: argparse.Namespace) -> None:
    if args.min_points < 2:
        raise ValueError(f"--min-points must be at least 2, got {args.min_points}")
    if args.max_labeled_trajectories < 0:
        raise ValueError(
            f"--max-labeled-trajectories must be nonnegative, got {args.max_labeled_trajectories}"
        )


def run(args: argparse.Namespace) -> None:
    validate_args(args)

    rewards_csv = resolve_rewards_csv(args.rewards_csv)
    output_dir = resolve_output_dir(rewards_csv, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = default_alpha_values()
    print(f"[setup] rewards_csv={rewards_csv}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] alpha_values={alpha_values}")

    groups = read_rows(
        rewards_csv,
        task_column=args.task_column,
        trajectory_column=args.trajectory_column,
        frame_column=args.frame_column,
        reward_column=args.reward_column,
    )
    trajectories = trajectory_arrays(groups)
    sweep_rows = compute_sweep_rows(trajectories, alpha_values, args.min_points)
    alpha_rows = summarize_by_alpha(sweep_rows)
    best_rows, best_summary = summarize_best_by_trajectory(sweep_rows)
    plot_path = write_voc_plot(
        output_dir,
        sweep_rows,
        max_labeled_trajectories=args.max_labeled_trajectories,
    )

    summary = {
        **best_summary,
        "alpha_values": alpha_values,
        "num_alphas": int(len(alpha_values)),
        "rewards_csv": str(rewards_csv),
        "reward_column": args.reward_column,
        "frame_column": args.frame_column,
        "trajectory_column": args.trajectory_column,
        "task_column": args.task_column,
        "plot_path": str(plot_path),
    }

    write_csv(output_dir / "voc_alpha_sweep_by_trajectory.csv", sweep_rows)
    write_csv(output_dir / "voc_alpha_sweep_summary_by_alpha.csv", alpha_rows)
    write_csv(output_dir / "voc_alpha_sweep_best_by_trajectory.csv", best_rows)
    with (output_dir / "voc_alpha_sweep_summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    print(f"[done] wrote {len(sweep_rows)} alpha/trajectory VOC rows")
    print(f"[done] wrote {len(alpha_rows)} alpha summary rows")
    print(f"[done] wrote {len(best_rows)} best trajectory rows")
    print(f"[done] wrote plot to {plot_path}")
    print(f"[summary] average_max_model_voc={summary['average_max_model_voc']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewards_csv", help="Path to a reward extraction rewards.csv file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <directory containing rewards.csv>/voc_alpha_sweep.",
    )
    parser.add_argument("--min-points", type=int, default=2, help="Minimum finite points needed for VOC.")
    parser.add_argument("--reward-column", default="reward", help="Reward column to score.")
    parser.add_argument("--frame-column", default="frame_idx", help="Frame index column.")
    parser.add_argument("--trajectory-column", default="traj_idx", help="Trajectory id column.")
    parser.add_argument("--task-column", default="task_id", help="Task id column.")
    parser.add_argument(
        "--max-labeled-trajectories",
        type=int,
        default=30,
        help="Only draw a legend if the number of trajectories is at most this value.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
