import copy
import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from callbacks import CUDACallback
from latent_codecs import (
    build_latent_codec,
    get_model_latent_channels,
    get_model_latent_size,
    resolve_latent_codec_config,
)
from models.nanowm_occupancy import NanoWM_models
from utils.distributed_utils import is_rank_zero
from utils.logger_utils import create_tensorboard_logger, create_wandb_logger
from utils.nanowm_utils import cleanup, clip_grad_norm_
from wm_datasets.world_model_dataset_occupancy import (
    create_eval_only_dataset,
    create_train_val_datasets,
)

from .base import BaseExperiment


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _trainer_precision(args):
    return "bf16" if args.experiment.infra.mixed_precision else "32"


def _cfg_select(cfg, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).mean(dim=1)


def build_occupancy_model(args):
    if "NanoWM" not in args.model.arch:
        raise ValueError(f"{args.model.arch} Model Not Supported!")

    action_dim = args.dataset.spec.action_dim * args.dataset.frame_interval
    return NanoWM_models[args.model.arch](
        input_size=get_model_latent_size(args),
        in_channels=get_model_latent_channels(args),
        num_classes=args.model.num_classes,
        num_frames=args.model.num_frames,
        extras=args.model.extras,
        use_action=args.model.use_action,
        action_dim=action_dim,
        action_injection_type=args.model.action_injection.type,
        causal=args.model.causal,
        num_tasks=int(_cfg_select(args, "model.num_tasks", 50)),
        decoder_depth=_cfg_select(args, "model.decoder_depth", None),
    )


class NanoWMOccupancyTrainingModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger_instance = logging.getLogger(__name__)

        self.num_tasks = int(_cfg_select(args, "model.num_tasks", 50))
        self.gamma = float(
            _cfg_select(
                args,
                "experiment.occupancy.gamma",
                _cfg_select(args, "experiment.occupancy.discount", 0.99),
            )
        )
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"experiment.occupancy.gamma must be in [0, 1], got {self.gamma}")
        self.flow_steps = int(_cfg_select(args, "experiment.occupancy.flow_steps", 10))
        self.target_tau = float(_cfg_select(args, "experiment.occupancy.target_tau", 0.005))
        self.num_bootstrap_samples = int(
            _cfg_select(args, "experiment.occupancy.num_bootstrap_samples", 1)
        )
        self.flow_timestep_scale = int(_cfg_select(args, "experiment.diffusion.diffusion_steps", 1000))

        if self.flow_steps <= 0:
            raise ValueError("experiment.occupancy.flow_steps must be positive")
        if self.num_bootstrap_samples <= 0:
            raise ValueError("experiment.occupancy.num_bootstrap_samples must be positive")

        if args.model.extras != 1:
            raise ValueError("Occupancy training currently supports model.extras == 1 only")

        print("[Init] NanoWMOccupancyTrainingModule: building model", flush=True)
        model = build_occupancy_model(args)
        if args.experiment.pretrained:
            self._load_pretrained_parameters(args, model)

        self.target_model = copy.deepcopy(model)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        self.model = model
        if getattr(args.experiment.infra, "compile", False):
            print("[Init] torch.compile(self.model) -- first step will pay JIT cost", flush=True)
            self.model = torch.compile(self.model)

        self.logger_instance.info(
            f"Model Parameters: {sum(p.numel() for p in self._online_model().parameters()):,}"
        )

        self.latent_codec_config = resolve_latent_codec_config(args)
        print(
            f"[Init] Loading latent codec={self.latent_codec_config.kind} "
            f"from: {self.latent_codec_config.model_path}",
            flush=True,
        )
        self.latent_codec = build_latent_codec(args)
        self.vae = getattr(self.latent_codec, "vae", None)
        self.vae_scale_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", None)
        self._vae_precision = self.latent_codec_config.precision
        print(
            f"[Init] Latent codec loaded, shape={self.latent_codec_config.latent_shape.as_tuple()}, "
            f"scaling_factor={self.vae_scale_factor}, vae_precision={self._vae_precision}",
            flush=True,
        )
        self._sanity_check_latent_codec(args)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.experiment.training.optimizer.lr,
            weight_decay=args.experiment.training.optimizer.weight_decay,
        )
        self.lr_scheduler = None

        self.latent_codec.requires_grad_(False)
        self.latent_codec.eval()
        self.model.train()
        print(
            "[Init] Occupancy flow config: "
            f"gamma={self.gamma}, flow_steps={self.flow_steps}, "
            f"target_tau={self.target_tau}, bootstrap_samples={self.num_bootstrap_samples}",
            flush=True,
        )

    def _online_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _sanity_check_latent_codec(self, args):
        H = args.model.image_size
        probe = torch.randn(1, 3, H, H)
        with torch.no_grad():
            z = self.latent_codec.encode(probe)
            recon = self.latent_codec.decode(z) if self.latent_codec.has_decoder else None
        expected_shape = self.latent_codec.latent_shape.as_tuple()
        actual_shape = tuple(z.shape[1:])
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Latent codec produced shape {actual_shape}, expected {expected_shape}. "
                "Check model.latent_size/model.latent_channels against the codec."
            )
        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"Latent codec encode produced NaN/Inf at init "
                f"(precision={self._vae_precision}, path={self.latent_codec_config.model_path})."
            )
        if recon is not None and not torch.isfinite(recon).all():
            raise RuntimeError(
                f"Latent codec decode produced NaN/Inf at init "
                f"(precision={self._vae_precision}, path={self.latent_codec_config.model_path})."
            )
        print(f"[Init] Latent codec sanity: clean under precision={self._vae_precision}", flush=True)

    def _vae_encode(self, x):
        self.latent_codec.to(x.device)
        return self.latent_codec.encode(x)

    def _load_pretrained_parameters(self, args, model):
        pretrained_path = args.experiment.pretrained
        self.logger_instance.info(f"Loading pretrained weights from {pretrained_path}")

        if str(pretrained_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            raw_sd = load_file(str(pretrained_path))
        else:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            raw_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        has_model_prefix = any(k.startswith("model.") for k in raw_sd)
        clean = {}
        for k, v in raw_sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if has_model_prefix:
                if not k.startswith("model."):
                    continue
                k = k[len("model."):]
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            clean[k] = v

        model_sd = model.state_dict()
        matched = {
            k: v for k, v in clean.items()
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
        }
        skipped = len(clean) - len(matched)
        if not matched:
            raise RuntimeError(f"No matching tensors found in pretrained checkpoint: {pretrained_path}")

        missing, unexpected = model.load_state_dict(matched, strict=False)
        self.logger_instance.info(
            f"Loaded {len(matched)} tensors from {pretrained_path}; "
            f"skipped {skipped} unmatched tensors. "
            f"New/unloaded occupancy tensors: {len(missing)}, unexpected: {len(unexpected)}"
        )

    def _task_from_batch(self, batch, batch_size: int) -> torch.Tensor:
        task = batch.get("task")
        if task is None:
            metadata = batch.get("metadata")
            if isinstance(metadata, dict) and "task_index" in metadata:
                task = metadata["task_index"]
        if task is None:
            raise ValueError(
                "Occupancy training requires batch['task'] or metadata['task_index']; "
                "no task id was found for this batch."
            )
        if not isinstance(task, torch.Tensor):
            task = torch.as_tensor(task)
        task = task.to(self.device)
        if task.ndim == 0:
            task = task.repeat(batch_size)
        if task.ndim == 1:
            task = task.long()
            if torch.any((task < 0) | (task >= self.num_tasks)):
                raise ValueError(
                    f"Task ids must be in [0, {self.num_tasks}); "
                    f"got min={task.min().item()}, max={task.max().item()}."
                )
            return task
        return task

    def _encode_batch(self, batch):
        x = batch["video"].to(self.device)
        action = batch["action"].to(self.device) if self.args.model.use_action else None
        task = self._task_from_batch(batch, x.shape[0])

        with torch.no_grad():
            b = x.shape[0]
            x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
            z = self._vae_encode(x)
            z = rearrange(z, "(b f) c h w -> b f c h w", b=b).contiguous()
        return z, action, task

    def _make_windows(self, z, action, task):
        n_context = int(self.args.model.n_context_frames)
        frames = z.shape[1]
        if frames < n_context + 1:
            raise ValueError(
                f"Occupancy flow needs at least n_context_frames + 1 frames; "
                f"got frames={frames}, n_context_frames={n_context}"
            )

        num_windows = frames - n_context
        contexts = []
        targets = []
        next_contexts = []
        current_actions = [] if action is not None else None
        next_actions = [] if action is not None else None

        for start in range(num_windows):
            contexts.append(z[:, start:start + n_context])
            targets.append(z[:, start + n_context:start + n_context + 1])
            next_contexts.append(z[:, start + 1:start + n_context + 1])
            if action is not None:
                current_actions.append(action[:, start:start + n_context])
                next_actions.append(action[:, start + 1:start + n_context + 1])

        context = rearrange(torch.stack(contexts, dim=1), "b k f c h w -> (b k) f c h w")
        target = rearrange(torch.stack(targets, dim=1), "b k f c h w -> (b k) f c h w")
        next_context = rearrange(torch.stack(next_contexts, dim=1), "b k f c h w -> (b k) f c h w")

        if action is not None:
            current_action = rearrange(torch.stack(current_actions, dim=1), "b k f d -> (b k) f d")
            next_action = rearrange(torch.stack(next_actions, dim=1), "b k f d -> (b k) f d")
        else:
            current_action = None
            next_action = None

        if task.ndim == 1:
            task = task[:, None].expand(-1, num_windows).reshape(-1)
        else:
            task = task[:, None].expand(-1, num_windows, -1).reshape(-1, task.shape[-1])

        return context, target, next_context, current_action, next_action, task

    def _tau_to_model_t(self, tau: torch.Tensor) -> torch.Tensor:
        max_t = max(float(self.flow_timestep_scale - 1), 1.0)
        return ((1.0 - tau.float()) * max_t).clamp(0.0, max_t)

    @staticmethod
    def _interpolate(noise: torch.Tensor, data: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        view_shape = (tau.shape[0],) + (1,) * (data.ndim - 1)
        tau_view = tau.view(view_shape)
        return (1.0 - tau_view) * noise + tau_view * data

    @staticmethod
    def _repeat_first_dim(value, repeats: int):
        if value is None or repeats == 1:
            return value
        return torch.repeat_interleave(value, repeats, dim=0)

    @torch.no_grad()
    def _sample_target_occupancy(self, noise, context, action, task):
        self.target_model.eval()
        sample = noise
        for i in range(self.flow_steps):
            tau = torch.full(
                (sample.shape[0],),
                i / self.flow_steps,
                device=sample.device,
                dtype=sample.dtype,
            )
            model_t = self._tau_to_model_t(tau)
            velocity = self.target_model(
                sample,
                model_t,
                context=context,
                action=action,
                task=task,
                use_fp16=False,
            )
            sample = sample + velocity / self.flow_steps
        return sample.detach()

    def _occupancy_losses(self, z, action, task):
        context, x_next, next_context, current_action, next_action, task = self._make_windows(
            z, action, task
        )

        batch_size = x_next.shape[0]

        noise = torch.randn_like(x_next)
        #tau = torch.rand(batch_size, device=x_next.device, dtype=x_next.dtype)
        tau = torch.sigmoid(torch.randn(batch_size, device=x_next.device, dtype=x_next.dtype))
        model_t = self._tau_to_model_t(tau)
        x_t = self._interpolate(noise, x_next, tau)
        target_velocity = x_next - noise

        pred_velocity = self.model(
            x_t,
            model_t,
            context=context,
            action=current_action,
            task=task,
            use_fp16=False,
        )
        transition_loss = _mean_flat((pred_velocity - target_velocity) ** 2).mean()

        repeats = self.num_bootstrap_samples
        boot_context = self._repeat_first_dim(context, repeats)
        boot_next_context = self._repeat_first_dim(next_context, repeats)
        boot_current_action = self._repeat_first_dim(current_action, repeats)
        boot_next_action = self._repeat_first_dim(next_action, repeats)
        boot_task = self._repeat_first_dim(task, repeats)

        boot_noise = torch.randn_like(self._repeat_first_dim(x_next, repeats))
        with torch.no_grad():
            boot_target = self._sample_target_occupancy(
                boot_noise,
                boot_next_context,
                boot_next_action,
                boot_task,
            )

        #boot_tau = torch.rand(boot_noise.shape[0], device=boot_noise.device, dtype=boot_noise.dtype)
        boot_tau = torch.sigmoid(torch.randn(boot_noise.shape[0], device=boot_noise.device, dtype=boot_noise.dtype))
        boot_model_t = self._tau_to_model_t(boot_tau)
        boot_x_t = self._interpolate(boot_noise, boot_target, boot_tau)

        with torch.no_grad():
            boot_target_velocity = self.target_model(
                boot_x_t,
                boot_model_t,
                context=boot_next_context,
                action=boot_next_action,
                task=boot_task,
                use_fp16=False,
            ).detach()

        boot_pred_velocity = self.model(
            boot_x_t,
            boot_model_t,
            context=boot_context,
            action=boot_current_action,
            task=boot_task,
            use_fp16=False,
        )
        bootstrap_loss = _mean_flat((boot_pred_velocity - boot_target_velocity) ** 2).mean()

        loss = (1.0 - self.gamma) * transition_loss + self.gamma * bootstrap_loss
        return {
            "loss": loss,
            "transition_loss": transition_loss,
            "bootstrap_loss": bootstrap_loss,
        }

    def training_step(self, batch, batch_idx):
        z, action, task = self._encode_batch(batch)
        loss_dict = self._occupancy_losses(z, action, task)
        loss = loss_dict["loss"]

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True, batch_size=z.shape[0])
        self.log(
            "train_transition_loss",
            loss_dict["transition_loss"],
            logger=True,
            sync_dist=True,
            batch_size=z.shape[0],
        )
        self.log(
            "train_bootstrap_loss",
            loss_dict["bootstrap_loss"],
            logger=True,
            sync_dist=True,
            batch_size=z.shape[0],
        )

        if (self.global_step + 1) % self.args.experiment.training.log_every == 0:
            self.logger_instance.info(
                f"(step={self.global_step+1:07d}/epoch={self.current_epoch:04d}) "
                f"Train Loss: {loss:.4f}, "
                f"Transition: {loss_dict['transition_loss']:.4f}, "
                f"Bootstrap: {loss_dict['bootstrap_loss']:.4f}"
            )
            for handler in self.logger_instance.handlers:
                handler.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        z, action, task = self._encode_batch(batch)
        loss_dict = self._occupancy_losses(z, action, task)
        loss = loss_dict["loss"]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=z.shape[0])
        self.log(
            "val_transition_loss",
            loss_dict["transition_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=z.shape[0],
        )
        self.log(
            "val_bootstrap_loss",
            loss_dict["bootstrap_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=z.shape[0],
        )
        return loss

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx=None,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        should_clip = self.global_step >= self.args.experiment.training.gradient_clip_start_step
        gradient_norm = clip_grad_norm_(
            self.model.parameters(),
            self.args.experiment.training.gradient_clip_norm,
            clip_grad=should_clip,
        )
        self.log("gradient_norm", gradient_norm, logger=True, sync_dist=True)

    @torch.no_grad()
    def _update_target_model(self):
        online_state = self._online_model().state_dict()
        target_state = self.target_model.state_dict()
        for key, target_value in target_state.items():
            online_value = online_state[key].detach()
            if torch.is_floating_point(target_value):
                target_value.mul_(1.0 - self.target_tau).add_(online_value, alpha=self.target_tau)
            else:
                target_value.copy_(online_value)

    def on_before_zero_grad(self, optimizer):
        self._update_target_model()
        self.target_model.eval()

    def configure_optimizers(self):
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.opt,
            num_warmup_steps=self.args.experiment.training.optimizer.lr_warmup_steps,
            num_training_steps=self.args.experiment.training.max_steps,
        )
        return [self.opt], [self.lr_scheduler]

    def on_train_start(self):
        self.target_model.eval()
        rank = self.global_rank
        experiment_dir = os.getcwd()

        rank_logger = logging.getLogger(f"rank_{rank}")
        rank_logger.setLevel(logging.INFO)
        rank_logger.propagate = False

        log_file = os.path.join(experiment_dir, f"rank_{rank}.log")
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s][RANK %(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        rank_logger.addHandler(fh)

        if rank == 0:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(message)s"))
            rank_logger.addHandler(ch)

        self.logger_instance = rank_logger
        self.logger_instance.info(f"***** Rank {rank} logging initialized at {log_file} *****")


class TrainOccupancyExperiment(BaseExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)

    def _create_experiment_directory(self):
        experiment_dir = HydraConfig.get().runtime.output_dir
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "across_timesteps"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "latest"), exist_ok=True)
        if self.cfg.logger is not None and self.cfg.logger.name == "tensorboard":
            os.makedirs(os.path.join(experiment_dir, "tb"), exist_ok=True)

        if is_rank_zero:
            OmegaConf.save(self.cfg, os.path.join(experiment_dir, "config.yaml"))

        return experiment_dir, checkpoint_dir

    def _create_loggers(self, experiment_dir):
        loggers = []

        if self.cfg.logger.name == "tensorboard":
            tb_logger = create_tensorboard_logger(
                experiment_dir=experiment_dir,
                name=self.cfg.logger.logger_name,
            )
            loggers.append(tb_logger)

        if self.cfg.wandb.enabled:
            wandb_logger = create_wandb_logger(
                project=self.cfg.wandb.project,
                name=f"{self.cfg.model.name}-occupancy-F{self.cfg.model.num_frames}-{self.cfg.dataset.name}",
                experiment_dir=experiment_dir,
                entity=self.cfg.wandb.entity,
                mode=self.cfg.wandb.mode,
            )
            if wandb_logger is not None:
                loggers.append(wandb_logger)

        if is_rank_zero and len(loggers) > 0:
            flat_cfg = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=False)
            for logger in loggers:
                try:
                    logger.log_hyperparams(flat_cfg)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        f"log_hyperparams failed on {type(logger).__name__}: {exc}"
                    )

        return loggers

    def _apply_fixed_validation_subset(self, val_dataset, experiment_dir):
        loader_cfg = self.cfg.dataset.loader
        subset_path = loader_cfg.validation_fixed_subset_path
        subset_size = loader_cfg.validation_fixed_subset_size
        subset_seed = loader_cfg.validation_fixed_subset_seed

        if subset_path is None and subset_size is None:
            return val_dataset

        if val_dataset.slice_mode != "exhaustive":
            raise ValueError("Fixed validation subset requires slice_mode='exhaustive'")

        if subset_path is None:
            subset_path = os.path.join(experiment_dir, "validation_subset.json")
        elif not os.path.isabs(subset_path):
            subset_path = os.path.join(get_original_cwd(), subset_path)

        if os.path.exists(subset_path):
            with open(subset_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            slice_specs = payload.get("slices", [])
            if len(slice_specs) == 0:
                raise ValueError(f"Empty validation subset file: {subset_path}")
            if subset_size is not None and len(slice_specs) != subset_size:
                raise ValueError(
                    f"Fixed subset size mismatch: requested {subset_size}, "
                    f"but file has {len(slice_specs)} slices ({subset_path})"
                )
            val_dataset.set_fixed_slices(slice_specs)
            if is_rank_zero:
                self.logger.info(f"Loaded fixed validation subset: {len(val_dataset)} slices from {subset_path}")
            return val_dataset

        if subset_size is None:
            raise ValueError("validation_fixed_subset_size is required to generate a new fixed subset")
        if subset_seed is None:
            raise ValueError("validation_fixed_subset_seed is required to generate a new fixed subset")

        slice_specs = val_dataset.sample_fixed_slice_specs(subset_size, seed=subset_seed)
        payload = {
            "dataset": {
                "name": self.cfg.dataset.name,
                "num_frames": self.cfg.model.num_frames,
                "frame_interval": self.cfg.dataset.frame_interval,
                "split_ratio": loader_cfg.split_ratio,
                "val_slice_mode": loader_cfg.val_slice_mode,
                "stride": loader_cfg.stride,
                "random_seed": loader_cfg.random_seed,
            },
            "slices": slice_specs,
        }
        os.makedirs(os.path.dirname(subset_path), exist_ok=True)
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        val_dataset.set_fixed_slices(slice_specs)
        if is_rank_zero:
            self.logger.info(f"Saved fixed validation subset: {len(val_dataset)} slices to {subset_path}")
        return val_dataset

    def _load_checkpoint(self, args, pl_module):
        if not args.experiment.resume_from_checkpoint:
            return

        checkpoint_path = self._resolve_checkpoint_path(args.experiment.resume_from_checkpoint)
        self.logger.info(f"Attempting to load checkpoint weights from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        has_prefix = any(k.startswith("model.") or k.startswith("target_model.") for k in state_dict)
        if not has_prefix:
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        ckpt_has_orig_mod = any("model._orig_mod." in k for k in state_dict)
        model_compiled = hasattr(pl_module.model, "_orig_mod")
        if ckpt_has_orig_mod and not model_compiled:
            state_dict = {
                k.replace("model._orig_mod.", "model.", 1) if k.startswith("model._orig_mod.") else k: v
                for k, v in state_dict.items()
            }
        elif model_compiled and not ckpt_has_orig_mod:
            state_dict = {
                k.replace("model.", "model._orig_mod.", 1)
                if k.startswith("model.") and not k.startswith("model._orig_mod.")
                else k: v
                for k, v in state_dict.items()
            }

        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict, strict=False)
        model_missing = [k for k in missing_keys if k.startswith("model.")]
        if model_missing:
            raise RuntimeError(f"Checkpoint loading failed: missing model keys: {model_missing}")
        if unexpected_keys:
            self.logger.warning(f"Ignored unexpected checkpoint keys: {unexpected_keys}")
        if any(k.startswith("target_model.") for k in missing_keys):
            pl_module.target_model.load_state_dict(pl_module._online_model().state_dict())
            self.logger.info("Checkpoint did not include target_model; copied online model into target_model.")

        self.logger.info("Successfully loaded occupancy checkpoint weights.")

    def _resolve_checkpoint_path(self, checkpoint_path):
        if not checkpoint_path:
            return None
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if os.path.isabs(checkpoint_path):
            return checkpoint_path

        candidates = [
            checkpoint_path,
            os.path.join(get_original_cwd(), checkpoint_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return checkpoint_path

    def _setup_common(self, need_train: bool):
        args = self.cfg
        seed = args.experiment.infra.seed
        _seed_everything(seed)
        self._seed = seed

        experiment_dir, checkpoint_dir = self._create_experiment_directory()
        loggers = self._create_loggers(experiment_dir)

        if not hasattr(args, "dataset"):
            raise ValueError("Missing dataset in config. Define configs/dataset/...")

        dataset_cfg = OmegaConf.to_container(args.dataset, resolve=True)
        if "name" not in dataset_cfg:
            raise ValueError("Missing required config: dataset.name")
        if "loader" not in dataset_cfg:
            raise ValueError("Missing required config: dataset.loader")

        dataset_name = dataset_cfg["name"]
        loader_cfg = dataset_cfg["loader"]

        used_fast_path = False
        if not need_train:
            subset_path = loader_cfg.get("validation_fixed_subset_path")
            if subset_path and os.path.exists(subset_path):
                with open(subset_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                slice_specs = payload.get("slices", [])
                if slice_specs:
                    val_dataset = create_eval_only_dataset(
                        dataset_name=dataset_name,
                        num_frames=args.model.num_frames,
                        frame_interval=args.dataset.frame_interval,
                        image_size=args.model.image_size,
                        precomputed_slices=slice_specs,
                        **loader_cfg,
                    )
                    train_dataset = None
                    used_fast_path = True
                    if is_rank_zero:
                        print(
                            f"[Data] Fast path: loaded {len(slice_specs)} precomputed slices, "
                            "skipped full indexing",
                            flush=True,
                        )

        if not used_fast_path:
            train_dataset, val_dataset = create_train_val_datasets(
                dataset_name=dataset_name,
                num_frames=args.model.num_frames,
                frame_interval=args.dataset.frame_interval,
                image_size=args.model.image_size,
                **loader_cfg,
            )
            if is_rank_zero:
                print("[Data] Occupancy datasets created", flush=True)
            val_dataset = self._apply_fixed_validation_subset(val_dataset, experiment_dir)

        if (
            args.dataset.loader.validation_size is not None
            and args.dataset.loader.validation_fixed_subset_path is None
            and args.dataset.loader.validation_fixed_subset_size is None
        ):
            indices = list(range(min(len(val_dataset), args.dataset.loader.validation_size)))
            val_dataset = Subset(val_dataset, indices)
            if is_rank_zero:
                self.logger.info(f"Using {len(val_dataset)} validation samples")

        pl_module = NanoWMOccupancyTrainingModule(args)
        if args.experiment.resume_from_checkpoint and not need_train:
            self._load_checkpoint(args, pl_module)

        callbacks_list = [CUDACallback()]
        if is_rank_zero and len(loggers) > 0:
            callbacks_list.append(LearningRateMonitor())

        eval_metrics_cfg = args.experiment.evaluation.metrics
        if eval_metrics_cfg.evaluate and is_rank_zero:
            self.logger.info(
                "Occupancy trainer logs flow losses only; diffusion video metrics are disabled."
            )

        return {
            "experiment_dir": experiment_dir,
            "checkpoint_dir": checkpoint_dir,
            "loggers": loggers,
            "train_dataset": train_dataset if need_train else None,
            "val_dataset": val_dataset,
            "pl_module": pl_module,
            "eval_callbacks": callbacks_list,
        }

    def _make_val_loader(self):
        args = self.cfg
        eval_batch_size = args.experiment.evaluation.get("batch_size", None)
        return dict(
            batch_size=eval_batch_size or args.experiment.training.batch_size,
            shuffle=False,
            num_workers=args.experiment.infra.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_seed_worker if args.experiment.infra.num_workers > 0 else None,
        )

    def training(self):
        args = self.cfg
        common = self._setup_common(need_train=True)
        checkpoint_dir = common["checkpoint_dir"]
        loggers = common["loggers"]
        train_dataset = common["train_dataset"]
        val_dataset = common["val_dataset"]
        pl_module = common["pl_module"]
        callbacks_list = common["eval_callbacks"]

        loader_seed_gen = torch.Generator()
        loader_seed_gen.manual_seed(self._seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.experiment.training.batch_size,
            shuffle=True,
            num_workers=args.experiment.infra.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.experiment.infra.num_workers > 0,
            prefetch_factor=4 if args.experiment.infra.num_workers > 0 else None,
            worker_init_fn=_seed_worker if args.experiment.infra.num_workers > 0 else None,
            generator=loader_seed_gen,
        )
        val_loader = DataLoader(
            val_dataset,
            persistent_workers=args.experiment.infra.num_workers > 0,
            prefetch_factor=4 if args.experiment.infra.num_workers > 0 else None,
            **self._make_val_loader(),
        )
        if is_rank_zero:
            print("[Data] DataLoaders created", flush=True)
            self.logger.info(f"Training dataset contains {len(train_dataset)} videos")
            self.logger.info(f"Validation dataset contains {len(val_dataset)} videos")
            self.logger.info(f"One epoch iteration {math.ceil(len(train_loader))} steps")

        ckpt_cfg = args.experiment.training.checkpointing
        across_timesteps_cfg = ckpt_cfg.across_timesteps
        callbacks_list.append(ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, "across_timesteps"),
            filename=across_timesteps_cfg.filename,
            save_top_k=across_timesteps_cfg.save_top_k,
            every_n_train_steps=across_timesteps_cfg.every_n_train_steps,
            save_on_train_epoch_end=across_timesteps_cfg.save_on_train_epoch_end,
            save_weights_only=across_timesteps_cfg.save_weights_only,
        ))

        latest_cfg = ckpt_cfg.latest
        callbacks_list.append(ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, "latest"),
            filename=latest_cfg.filename,
            save_top_k=latest_cfg.save_top_k,
            every_n_train_steps=latest_cfg.every_n_train_steps,
            save_on_train_epoch_end=latest_cfg.save_on_train_epoch_end,
            save_weights_only=latest_cfg.save_weights_only,
        ))

        num_gpus = torch.cuda.device_count()
        num_nodes = args.experiment.infra.get("num_nodes", 1)
        trainer = Trainer(
            accelerator="gpu",
            devices=num_gpus,
            strategy=("ddp_find_unused_parameters_false" if num_gpus > 1 else "auto"),
            num_nodes=num_nodes,
            enable_checkpointing=True,
            max_steps=args.experiment.training.max_steps,
            logger=loggers,
            callbacks=callbacks_list,
            log_every_n_steps=args.experiment.training.log_every,
            val_check_interval=args.experiment.training.val_every_n_steps,
            check_val_every_n_epoch=None,
            accumulate_grad_batches=args.experiment.training.gradient_accumulation,
            precision=_trainer_precision(args),
            num_sanity_val_steps=0,
        )

        ckpt_path = self._resolve_checkpoint_path(args.experiment.resume_from_checkpoint)
        if ckpt_path:
            self.logger.info(f"Resuming full Lightning trainer state from: {ckpt_path}")
        trainer.fit(
            pl_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

        pl_module.model.eval()
        self.logger.info("Done!")
        cleanup()

    def evaluate(self):
        args = self.cfg
        if not args.experiment.resume_from_checkpoint:
            raise ValueError(
                "resume_from_checkpoint is required for evaluation task. "
                "Please specify a checkpoint path."
            )

        common = self._setup_common(need_train=False)
        loggers = common["loggers"]
        val_dataset = common["val_dataset"]
        pl_module = common["pl_module"]
        callbacks_list = common["eval_callbacks"]

        val_loader = DataLoader(val_dataset, **self._make_val_loader())

        num_gpus = torch.cuda.device_count()
        num_nodes = args.experiment.infra.get("num_nodes", 1)
        trainer = Trainer(
            accelerator="gpu",
            devices=num_gpus,
            strategy=("ddp_find_unused_parameters_false" if num_gpus > 1 else "auto"),
            num_nodes=num_nodes,
            enable_checkpointing=False,
            logger=loggers,
            callbacks=callbacks_list,
            precision=_trainer_precision(args),
            num_sanity_val_steps=0,
        )

        self.logger.info("***** Running Occupancy Evaluation *****")
        trainer.validate(pl_module, dataloaders=val_loader)

        pl_module.model.eval()
        self.logger.info("Evaluation Done!")
        cleanup()
