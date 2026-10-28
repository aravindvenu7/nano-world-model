"""
GTSE-r on the NanoWM bilinear model.

Port of dre_iclr/ogbench/impls/agents/gtse_r.py (TimeGCBilinearValue critic phi^T psi / sqrt(d)), adapted the
same way as train_experiment_gtse.py. Its loss is the same scalar GTSE loss as GTSE-b; the difference is the
critic's time conditioning, set by experiment.gtse.phi_time / psi_time. Defaults follow the GTSE-r command in
dre_iclr's README: psi sees t (psi_time=True), phi does not (phi_time=False, gtse_r.py's default).
"""

from .train_experiment_gtse import TrainGTSEExperiment
from .train_experiment_gtse_bilinear import NanoWMGTSEBilinearTrainingModule


class NanoWMGTSERTrainingModule(NanoWMGTSEBilinearTrainingModule):
    phi_time_default = False
    psi_time_default = True


class TrainGTSERExperiment(TrainGTSEExperiment):
    module_cls = NanoWMGTSERTrainingModule
