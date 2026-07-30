"""
Count parameters in a NanoWM occupancy model.

Examples:
    python src/sample/count_params_occupancy.py dataset=mt50/mt50 model=nanowm_b2
    python src/sample/count_params_occupancy.py --config results/<run>/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.train_experiment_occupancy import build_occupancy_model


def _format_count(value: int) -> str:
    return f"{value:,} ({value / 1_000_000:.3f}M)"


def _load_config(config_path: str | None, overrides: list[str]):
    if config_path:
        cfg = OmegaConf.load(Path(config_path).expanduser())
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
    else:
        with initialize_config_dir(config_dir=str(SRC_DIR / "configs"), version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

    OmegaConf.set_struct(cfg, False)
    return cfg


def _top_level_parameter_counts(model) -> OrderedDict[str, int]:
    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        top_name = name.split(".", 1)[0] if name else "<root>"
        counts[top_name] = counts.get(top_name, 0) + param.numel()
    return OrderedDict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _count_parameters(model) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "buffers": buffers,
        "top_level": _top_level_parameter_counts(model),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional resolved run config.yaml. If omitted, compose src/configs/config.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print counts as JSON.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides, e.g. dataset=mt50/mt50 model=nanowm_b2.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = _load_config(args.config, args.overrides)
    model = build_occupancy_model(cfg)
    counts = _count_parameters(model)

    payload = {
        "arch": cfg.model.arch,
        "model_name": cfg.model.name,
        "dataset": cfg.dataset.name,
        "num_frames": int(cfg.model.num_frames),
        "image_size": int(cfg.model.image_size),
        "latent_shape": [
            int(cfg.model.latent_channels),
            int(cfg.model.latent_size),
            int(cfg.model.latent_size),
        ],
        "action_dim": int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval),
        "parameters": {
            "online_model": {
                "total": counts["total"],
                "trainable": counts["trainable"],
                "frozen": counts["frozen"],
            },
            "target_model_duplicate": {
                "total": counts["total"],
                "note": "Training keeps a frozen EMA target_model with the same parameter count.",
            },
            "online_plus_target": {
                "total": counts["total"] * 2,
            },
            "buffers": counts["buffers"],
            "top_level": dict(counts["top_level"]),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("NanoWM occupancy parameter count")
    print(f"  arch: {payload['arch']}")
    print(f"  dataset: {payload['dataset']}")
    print(f"  num_frames: {payload['num_frames']}")
    print(f"  latent_shape: {tuple(payload['latent_shape'])}")
    print(f"  action_dim: {payload['action_dim']}")
    print()
    print(f"Online model total:      {_format_count(counts['total'])}")
    print(f"Online model trainable:  {_format_count(counts['trainable'])}")
    print(f"Online model frozen:     {_format_count(counts['frozen'])}")
    print(f"Buffers:                 {_format_count(counts['buffers'])}")
    print()
    print(f"Training online+target:  {_format_count(counts['total'] * 2)}")
    print("  Note: this excludes the latent codec/VAE and optimizer state.")
    print()
    print("Top-level parameter groups:")
    for name, value in counts["top_level"].items():
        print(f"  {name:<28} {_format_count(value)}")


if __name__ == "__main__":
    main()
