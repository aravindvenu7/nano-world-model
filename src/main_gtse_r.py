import signal
import os
os.environ["PL_DISABLE_SUBPROCESS_LOGGING"] = "1"
# Keep all caches/logs on scratch, not home. Must be set before huggingface_hub/wandb/torch are imported; existing exports take precedence.
_SCRATCH = "/zfsauton/scratch/avenugo2"
for _k, _v in {"HF_HOME": f"{_SCRATCH}/.cache/huggingface", "XDG_CACHE_HOME": f"{_SCRATCH}/.cache", "TORCH_HOME": f"{_SCRATCH}/.cache/torch",
               "TRITON_CACHE_DIR": f"{_SCRATCH}/.cache/triton", "TORCHINDUCTOR_CACHE_DIR": f"{_SCRATCH}/.cache/torchinductor", "MPLCONFIGDIR": f"{_SCRATCH}/.config/matplotlib",
               "WANDB_DIR": f"{_SCRATCH}/wandb", "WANDB_CACHE_DIR": f"{_SCRATCH}/.cache/wandb", "WANDB_CONFIG_DIR": f"{_SCRATCH}/.config/wandb", "WANDB_DATA_DIR": f"{_SCRATCH}/.local/share/wandb"}.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault("MT50_DATA_ROOT", "/zfsauton/scratch/avenugo2/mt50")
os.environ.setdefault("RESULTS_DIR", "/zfsauton/scratch/avenugo2/nanowm_results")
import hydra
import torch
from omegaconf import DictConfig

from experiments.train_experiment_gtse_r import TrainGTSERExperiment
def signal_handler(sig, frame):
    print(f"\nInterrupt received. Force quitting.")
    os._exit(0)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    torch.set_float32_matmul_precision('high')
    experiment = TrainGTSERExperiment(cfg)
    experiment.exec()


if __name__ == "__main__":
    main()
