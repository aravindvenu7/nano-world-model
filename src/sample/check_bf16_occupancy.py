"""
Measure how much bf16 autocast perturbs the occupancy losses of a checkpoint (or a random init if none).

Runs `_occupancy_losses` on the same batches with identical RNG under three precisions:
  fp32  - full fp32 matmuls (reference)
  tf32  - fp32 with TF32 matmuls (what training with mixed_precision=false uses; main sets matmul precision 'high')
  bf16  - bf16 autocast (what training with mixed_precision=true uses)
Latents are encoded once in fp32 and shared across modes, so only the flow model's precision differs.
Forward hooks capture the network outputs so each mode can be compared elementwise against fp32.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

os.environ.setdefault("HF_HOME", "/zfsauton/scratch/avenugo2/.cache/huggingface")
os.environ.setdefault("MT50_DATA_ROOT", "/zfsauton/scratch/avenugo2/mt50")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.split(sys.path[0])[0])

from extract_rewards_occupancy import create_dataset, load_checkpoint, load_config, resolve_checkpoint, resolve_run_dir
from experiments.train_experiment_occupancy import NanoWMOccupancyTrainingModule

MODES = ("fp32", "tf32", "bf16")


def set_mode(mode: str):
    tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def capture_outputs(module, store):
    return module.register_forward_hook(lambda _m, _i, out: store.append(out.detach().float()))


@torch.no_grad()
def run_mode(pl_module, z, action, task, mode: str, seed: int):
    online, target = [], []
    hooks = [capture_outputs(pl_module.model, online), capture_outputs(pl_module.target_model, target)]
    torch.manual_seed(seed)
    with set_mode(mode):
        losses = pl_module._occupancy_losses(z, action, task)
    for h in hooks:
        h.remove()
    # online calls: [transition pred, bootstrap pred]; target calls: [flow_steps Euler velocities..., bootstrap target velocity]
    return {k: float(v) for k, v in losses.items()}, {"trans_pred": online[0], "boot_pred": online[1], "boot_target": target[-1]}


def mse(a, b):
    return float((a - b).square().mean())


def compose_config(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = os.path.join(os.path.split(sys.path[0])[0], "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=occupancy", "dataset=mt50/mt50", "model=nanowm_b2", *overrides])
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.infra.compile = False
    return cfg


def find_checkpoint(run_dir, ckpt_path):
    if ckpt_path is not None:
        return resolve_checkpoint(run_dir, ckpt_path)
    if run_dir is None:
        return None
    try:
        return resolve_checkpoint(run_dir, None)
    except FileNotFoundError:
        print(f"[setup] no checkpoint under {run_dir}/checkpoints; using random init")
        return None


def main(args):
    torch.set_float32_matmul_precision("highest")
    if args.run_dir:
        run_dir = resolve_run_dir(args.run_dir)
        cfg, _ = load_config(run_dir, None)
    else:
        run_dir = None
        cfg = compose_config(args.overrides)
    ckpt_path = find_checkpoint(run_dir, args.ckpt_path)
    print(f"[setup] run_dir={run_dir}\n[setup] checkpoint={ckpt_path or 'none (random init, target_model = copy of model)'}")

    pl_module = NanoWMOccupancyTrainingModule(cfg)
    if ckpt_path is not None:
        load_checkpoint(pl_module, ckpt_path)
    pl_module.to("cuda").eval()

    dataset = create_dataset(cfg, args.split)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.batch_size * args.num_batches, len(dataset)), replace=False).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    losses = {m: [] for m in MODES}
    diffs = {m: {"trans_pred": [], "boot_pred": [], "boot_target": [], "boot_residual": []} for m in MODES if m != "fp32"}
    for b, batch in enumerate(loader):
        set_mode("fp32")  # latents encoded once in full fp32, shared by all modes
        z, action, task = pl_module._encode_batch(batch)
        outs = {}
        for mode in MODES:
            l, outs[mode] = run_mode(pl_module, z, action, task, mode, seed=args.seed + b)
            losses[mode].append(l)
        ref = outs["fp32"]
        for mode in diffs:
            o = outs[mode]
            for key in ("trans_pred", "boot_pred", "boot_target"):
                diffs[mode][key].append(mse(o[key], ref[key]))
            # change in the bootstrap residual (pred - target) itself, i.e. the quantity the loss squares
            diffs[mode]["boot_residual"].append(mse(o["boot_pred"] - o["boot_target"], ref["boot_pred"] - ref["boot_target"]))
        print(f"[batch {b}] " + "  ".join(f"{m}: boot={losses[m][-1]['bootstrap_loss']:.3e} trans={losses[m][-1]['transition_loss']:.3e}" for m in MODES))

    mean = lambda xs: float(np.mean(xs))
    print("\n=== losses (mean over batches) ===")
    for m in MODES:
        print(f"{m:>5}: bootstrap={mean([l['bootstrap_loss'] for l in losses[m]]):.4e}  transition={mean([l['transition_loss'] for l in losses[m]]):.4e}")
    ref_boot = mean([l["bootstrap_loss"] for l in losses["fp32"]])
    print("\n=== deviation from fp32 (elementwise MSE; compare to fp32 bootstrap loss) ===")
    for m, d in diffs.items():
        boot_gap = mean([abs(a["bootstrap_loss"] - r["bootstrap_loss"]) for a, r in zip(losses[m], losses["fp32"])])
        print(f"{m:>5}: |bootstrap loss gap|={boot_gap:.3e} ({100 * boot_gap / ref_boot:.1f}% of fp32 bootstrap loss)")
        print(f"       residual MSE vs fp32={mean(d['boot_residual']):.3e} ({100 * mean(d['boot_residual']) / ref_boot:.1f}% of fp32 bootstrap loss)")
        print(f"       boot_pred={mean(d['boot_pred']):.3e}  boot_target={mean(d['boot_target']):.3e}  trans_pred={mean(d['trans_pred']):.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="Use this run's config.yaml. Omit to compose experiment=occupancy dataset=mt50/mt50 model=nanowm_b2.")
    p.add_argument("--ckpt-path", default=None, help="Defaults to newest checkpoint under run-dir; random init if none.")
    p.add_argument("overrides", nargs="*", help="Extra Hydra overrides when --run-dir is omitted, e.g. model.decoder_depth=6")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
