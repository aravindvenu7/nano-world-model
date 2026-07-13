import signal
import os
os.environ["PL_DISABLE_SUBPROCESS_LOGGING"] = "1"
import hydra
import torch
from omegaconf import DictConfig

from experiments.train_experiment_occupancy import TrainOccupancyExperiment
def signal_handler(sig, frame):
    print(f"\nInterrupt received. Force quitting.")
    os._exit(0)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    torch.set_float32_matmul_precision('high')
    experiment = TrainOccupancyExperiment(cfg)
    experiment.exec()


if __name__ == "__main__":
    main()
