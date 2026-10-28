"""
GTSE-b (bilinear time-score critic) on the NanoWM bilinear model.

Port of dre_iclr/ogbench/impls/agents/gtse_legendre.py with full=False (ContrastiveTimeScoreFn critic
phi(s, a, t) · psi(x_t, t)), adapted the same way as train_experiment_gtse.py: same data, windows, geometric
future frames, t sampling, TD loss, Huber, target network and time embedding. What changes:
  - Model: models/nanowm_bilinear.py. Its per-dimension output is summed to one scalar per sample, which is
    still bilinear in (phi, psi).
  - Target: gtse.py's scalar `epsilon_target` (full=False), summed over all latent dimensions, with temp_coeff.
  - experiment.gtse.phi_time / psi_time choose which tower sees t (both on here, as in ContrastiveTimeScoreFn).
Legendre quadrature only changes inference in gtse_legendre.py, so it is not part of training. Only the
conditional network is trained (no marginal network, no actor), as in train_experiment_gtse.py.
"""

import torch

from models.nanowm_bilinear import NanoWM_models as BILINEAR_MODELS
from utils.gtse_targets import epsilon_target

from .train_experiment_gtse import NanoWMGTSETrainingModule, TrainGTSEExperiment
from .train_experiment_occupancy import _cfg_select, build_occupancy_model


class NanoWMGTSEBilinearTrainingModule(NanoWMGTSETrainingModule):
    phi_time_default = True
    psi_time_default = True

    def __init__(self, args):
        super().__init__(args)
        self.temp_coeff = float(_cfg_select(args, "experiment.gtse.temp_coeff", 1.0))
        phi_time, psi_time = self._tower_times(args)
        print(f"[Init] {type(self).__name__}: phi_time={phi_time}, psi_time={psi_time}, temp_coeff={self.temp_coeff}", flush=True)

    def _tower_times(self, args):
        return (bool(_cfg_select(args, "experiment.gtse.phi_time", self.phi_time_default)),
                bool(_cfg_select(args, "experiment.gtse.psi_time", self.psi_time_default)))

    def _build_model(self, args):
        phi_time, psi_time = self._tower_times(args)
        return build_occupancy_model(args, registry=BILINEAR_MODELS, phi_time=phi_time, psi_time=psi_time, **self._model_kwargs(args))

    def _score_target(self, epsilon, x1, t):
        return epsilon_target(epsilon, x1, t, self.temp_coeff)

    def _score_output(self, out):
        # Sum in fp32: under bf16 autocast a 4096-term sum rounded to bf16 loses ~3 significant digits.
        return out.float().flatten(1).sum(1)


class TrainGTSEBilinearExperiment(TrainGTSEExperiment):
    module_cls = NanoWMGTSEBilinearTrainingModule
