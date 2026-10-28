"""GTSE score targets shared by the MetaWorld trainers and the online BuilderBench agent (torch only)."""

import torch


def epsilon_target(epsilon, x1, t, temp_coeff=1.0, mask=None):
    """Scalar GTSE score target and lambda_t (gtse.py `epsilon_target`, factor=1); sums run over all latent dims,
    or only over the dims where `mask` (same shape as epsilon, 1 = counted) is set."""
    t = t.reshape(-1)
    if mask is None:
        d = epsilon[0].numel()
    else:
        mask = mask.to(epsilon.dtype)
        d = mask.flatten(1).sum(1)
        epsilon, x1 = epsilon * mask, x1 * mask
    var = 1 - t ** 2
    temp = torch.sqrt(1 + t ** 2)
    eps_sq = epsilon.pow(2).flatten(1).sum(1)
    eps_x1 = (epsilon * x1).flatten(1).sum(1)
    return var / temp, (temp_coeff * (d * t - t * eps_sq) + torch.sqrt(var) * eps_x1) / temp
