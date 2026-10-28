"""
GTSE (time-score occupancy) training on the NanoWM occupancy encoder/decoder.

Port of the conditional GTSE TD loss from dre_iclr/ogbench/impls/agents/gtse.py (`full=True`): the decoder
output (same shape as the latent) is the per-dimension time score of p(s+ | s, a, task) against N(0, I), and
its sum over dimensions is the scalar time score. Only the conditional network is trained (no marginal
network, no actor).

Interpolant: x_t = sqrt(1 - t^2) * x0 + t * x1 with x0 ~ N(0, I), t ~ U[0, 1). How t reaches the model is set
by experiment.gtse.time_embedding (see `_tau_to_model_t`): fourier (default, as in gtse.py) or linear (same as ORS).

Loss = (1 - discount) * loss_s_ + discount * loss_f:
  - loss_s_: score matching at the next frame s' given (s, a).
  - loss_f: TD bootstrap at a geometric future frame sf of the next state s' (same trajectory), regressing the
    online score at (s, a) onto the target network's score at (s', a').
"""

import logging

import torch
from einops import rearrange
from torch.utils.data import Subset

from .train_experiment_occupancy import (
    NanoWMOccupancyTrainingModule,
    TrainOccupancyExperiment,
    _cfg_select,
    build_occupancy_model,
    _mean_flat,
)


def full_epsilon_target(epsilon, x1, t):
    """Per-dimension GTSE score target and its scale lambda_t (gtse.py `full_epsilon_target`, factor=1)."""
    var = 1 - t ** 2
    temp = torch.sqrt(1 + t ** 2)
    return var / temp, (t - t * epsilon ** 2 + torch.sqrt(var) * epsilon * x1) / temp


TIME_EMBEDDINGS = ("linear", "fourier")


def _time_embedding(args):
    mode = _cfg_select(args, "experiment.gtse.time_embedding", "fourier")
    if mode not in TIME_EMBEDDINGS:
        raise ValueError(f"experiment.gtse.time_embedding must be one of {TIME_EMBEDDINGS}, got {mode}")
    return mode


class NanoWMGTSETrainingModule(NanoWMOccupancyTrainingModule):
    def __init__(self, args):
        super().__init__(args)
        self.time_embedding = _time_embedding(args)
        self.gamma = float(_cfg_select(args, "experiment.gtse.discount", 0.99))
        self.target_tau = float(_cfg_select(args, "experiment.gtse.target_tau", 0.005))
        self.goal_noise_std = float(_cfg_select(args, "experiment.gtse.goal_noise_std", 1e-4))
        self.huber = bool(_cfg_select(args, "experiment.gtse.huber", True))
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"experiment.gtse.discount must be in [0, 1), got {self.gamma}")
        print(
            f"[Init] GTSE config: discount={self.gamma}, target_tau={self.target_tau}, "
            f"goal_noise_std={self.goal_noise_std}, huber={self.huber}, time_embedding={self.time_embedding}",
            flush=True,
        )

    def _model_kwargs(self, args):
        if _time_embedding(args) == "fourier":
            return dict(timestep_embedding="fourier", time_dim=int(_cfg_select(args, "experiment.gtse.time_dim", 256)))
        return {}

    def _build_model(self, args):
        return build_occupancy_model(args, **self._model_kwargs(args))

    def _score_target(self, epsilon, x1, t):
        """(lambda_t, target) for the score regression; per-dimension (gtse.py full=True)."""
        return full_epsilon_target(epsilon, x1, t)

    def _score_output(self, out):
        """Model output -> score compared against `_score_target`; per-dimension here."""
        return out

    def _tau_to_model_t(self, tau):
        """Map GTSE time t to the timestep-embedder input.

        linear:  (1 - t) * 999 into the DiT sinusoidal embedder (same as ORS).
        fourier: -log(1 - t) into the GTSE Fourier embedder (frequencies 1 .. 100).
        """
        if self.time_embedding == "linear":
            return super()._tau_to_model_t(tau)
        return -torch.log(torch.clamp(1.0 - tau.float(), 1e-5, 1.0))

    def _encode_frames(self, x):
        b = x.shape[0]
        with torch.no_grad():
            z = self._vae_encode(rearrange(x.to(self.device), "b f c h w -> (b f) c h w").contiguous())
        return rearrange(z, "(b f) c h w -> b f c h w", b=b).contiguous()

    def _future_windows(self, z_future):
        """Geometric future of each window's NEXT state s' (the frame after the last context frame), in
        `_make_windows` order: sf = s_{t+1+k}, k >= 1, geometric renormalized to the steps left in the trajectory."""
        n_context = int(self.args.model.n_context_frames)
        num_windows = z_future.shape[1] - n_context
        sf = z_future[:, n_context:n_context + num_windows]
        return rearrange(sf, "b k c h w -> (b k) 1 c h w")

    def _add_goal_noise(self, x):
        return x + self.goal_noise_std * torch.randn_like(x)

    def _gtse_losses(self, batch):
        z, action, task = self._encode_batch(batch)
        z_future = self._encode_frames(batch["future_video"])
        context, x_next, next_context, current_action, next_action, task = self._make_windows(z, action, task)
        sf = self._add_goal_noise(self._future_windows(z_future))

        n = x_next.shape[0]
        # t ~ U[0, 1 - 1e-5), shared by both branches as in gtse.py.
        t = (torch.rand(n, device=x_next.device) * (1.0 - 1e-5)).view(n, 1, 1, 1, 1)
        model_t = self._tau_to_model_t(t.view(n))

        # Immediate branch: score at s' given (s, a).
        s_ = self._add_goal_noise(x_next)
        x0_s = torch.randn_like(s_)
        xt_s = x0_s * torch.sqrt(1 - t ** 2) + t * s_
        lambda_s, targets_s = self._score_target(x0_s, s_, t)
        preds_s = self._score_output(self.model(xt_s, model_t, context=context, action=current_action, task=task, use_fp16=False))
        loss_s = _mean_flat((targets_s - lambda_s * preds_s) ** 2).mean()

        # TD branch: online score at (s, a) vs. target score at (s', a'), both at x_t built from sf.
        x0_f = torch.randn_like(sf)
        xt_f = x0_f * torch.sqrt(1 - t ** 2) + t * sf
        lambda_f, _ = self._score_target(x0_f, sf, t)
        with torch.no_grad():
            self.target_model.eval()
            targets_f = self._score_output(self.target_model(xt_f, model_t, context=next_context, action=next_action, task=task, use_fp16=False))
        preds_f = self._score_output(self.model(xt_f, model_t, context=context, action=current_action, task=task, use_fp16=False))
        residual = lambda_f * (targets_f.detach() - preds_f)
        if self.huber:
            delta = 1.0 / torch.sqrt(torch.clamp(1.0 - t, 1e-5, 1.0))
            delta = delta.reshape(-1, *([1] * (residual.ndim - 1)))  # broadcast against full or scalar scores
            per_elem = torch.where(residual.abs() <= delta, 0.5 * residual ** 2, delta * (residual.abs() - 0.5 * delta))
        else:
            per_elem = residual ** 2
        loss_f = _mean_flat(per_elem).mean()

        loss = (1.0 - self.gamma) * loss_s + self.gamma * loss_f
        return {"loss": loss, "loss_s_": loss_s, "loss_f": loss_f}, z.shape[0]

    def training_step(self, batch, batch_idx):
        loss_dict, bsz = self._gtse_losses(batch)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, prog_bar=(k == "loss"), logger=True, sync_dist=True, batch_size=bsz)

        if (self.global_step + 1) % self.args.experiment.training.log_every == 0:
            self.logger_instance.info(
                f"(step={self.global_step+1:07d}/epoch={self.current_epoch:04d}) "
                f"Train Loss: {loss_dict['loss']:.4f}, loss_s_: {loss_dict['loss_s_']:.4f}, loss_f: {loss_dict['loss_f']:.4f}"
            )
            for handler in self.logger_instance.handlers:
                handler.flush()
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, bsz = self._gtse_losses(batch)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "loss"), logger=True, sync_dist=True, batch_size=bsz)
        return loss_dict["loss"]


class TrainGTSEExperiment(TrainOccupancyExperiment):
    module_cls = NanoWMGTSETrainingModule

    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)

    def _setup_common(self, need_train: bool):
        common = super()._setup_common(need_train)
        discount = float(_cfg_select(self.cfg, "experiment.gtse.discount", 0.99))
        for ds in (common["train_dataset"], common["val_dataset"]):
            while isinstance(ds, Subset):
                ds = ds.dataset
            if ds is not None:
                ds.future_discount = discount
        return common
