import os
# Keep all caches/logs on scratch, not home; JAX (BuilderBench env) on the CPU. Existing exports take precedence.
_SCRATCH = "/zfsauton/scratch/avenugo2"
for _k, _v in {"XDG_CACHE_HOME": f"{_SCRATCH}/.cache", "TORCH_HOME": f"{_SCRATCH}/.cache/torch", "MPLCONFIGDIR": f"{_SCRATCH}/.config/matplotlib",
               "RESULTS_DIR": f"{_SCRATCH}/nanowm_results", "JAX_PLATFORMS": "cpu"}.items():
    os.environ.setdefault(_k, _v)
import hydra
from omegaconf import DictConfig

from online_rl.train import run


@hydra.main(config_path="configs", config_name="rl_builderbench", version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
