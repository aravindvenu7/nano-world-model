"""
Bilinear NanoWM occupancy model: score = W_out · M(s, a, task, t) · psi(x_t, t), per goal patch.

Two towers that never see each other's inputs, joined once at the end:
  - Context tower phi: the occupancy NanoWM encoder over the history frames, their actions and the task.
    Unlike the joint model, frame k is conditioned on its own action a_k (no shift), so the current
    action a_t reaches the context side.
  - Goal tower psi: the decoder with cross-attention removed, i.e. spatial self-attention blocks over the
    noised goal x_t, conditioned on t only.
  - Readout: attention without the softmax. out_i = sum_j (Q_i · K_j) V_j / (M sqrt(d)) = M(s, a)^T Q_i, with
    Q from psi and K, V from phi. Linear in psi for a fixed context, and nothing nonlinear follows it (no
    LayerNorm, no adaLN, no bias), so a goal that matches no context token scores ~0, as in the bilinear
    ContrastiveTimeScoreFn critic. The output layer is zero-initialised, so the score is 0 everywhere at init.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .nanowm_occupancy import NanoWM, TransformerBlock


def bilinear_memory(kv, heads):
    """Context memory M(s, a) = K^T V / (M sqrt(hd)) per head, from projected context tokens kv: (B, M, 2D)."""
    B, M, D2 = kv.shape
    hd = D2 // 2 // heads
    k, v = kv.view(B, M, 2, heads, hd).permute(2, 0, 3, 1, 4)
    return torch.einsum('bhmk,bhmv->bhkv', k, v) / (M * hd ** 0.5)  # (B, H, hd, hd)


def bilinear_apply(q, memory):
    """Readout Q_i^T M for projected goal tokens q: (B, N, D) and memory (B, H, hd, hd) -> (B, N, D)."""
    B, N, D = q.shape
    H, hd = memory.shape[1], memory.shape[2]
    out = torch.einsum('bhnk,bhkv->bhnv', q.view(B, N, H, hd).transpose(1, 2), memory)
    return out.transpose(1, 2).reshape(B, N, D)


class NanoWMBilinear(NanoWM):
    def __init__(self, *args, readout_heads=1, phi_time=True, psi_time=True, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.hidden_size
        if hidden_size % readout_heads:
            raise ValueError(f"hidden_size {hidden_size} not divisible by readout_heads {readout_heads}")
        self.readout_heads = readout_heads
        # Whether each tower sees t (TimeGCBilinearValue's phi_time / psi_time). Off = the tower's timestep
        # input is fixed at 0, so its conditioning is a learned constant.
        self.phi_time = phi_time
        self.psi_time = psi_time

        # Goal tower: self-attention + MLP only. 'additive' adds no action-specific parameters, and the
        # goal tower never receives an action, so every parameter is used (DDP find_unused_parameters=False).
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)  # NanoWM's default; the size constructors don't set it
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, self.num_heads, mlp_ratio=mlp_ratio, action_injection_type='additive')
            for _ in range(self.decoder_depth)
        ])
        del self.final_layer  # replaced by the bilinear readout; its LayerNorm/adaLN would break bilinearity

        self.readout_goal_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.readout_ctx_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.readout_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.readout_out = nn.Linear(hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=False)
        self._init_bilinear()

    def _init_bilinear(self):
        """Initialise only the modules created here; the rest was initialised by NanoWM.__init__."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for module in (self.decoder_blocks, self.readout_q, self.readout_kv):
            module.apply(_basic_init)
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.readout_out.weight, 0)

    def _prepare_action_embeddings(self, action, batches, context_frames, target_frames, use_fp16=False):
        """Context frame k gets its own action a_k (no shift); nothing goes to the goal tower."""
        if not self.use_action or action is None:
            return None, None
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.shape[0] != batches:
            raise ValueError(f"action batch ({action.shape[0]}) must match x batch ({batches})")
        if action.shape[1] < context_frames:
            raise ValueError(f"need an action for each of the {context_frames} context frames, got {action.shape[1]}")
        action_emb = self.action_embedder(action[:, :context_frames].to(dtype=self.action_embedder.mlp[0].weight.dtype))
        if use_fp16:
            action_emb = action_emb.to(dtype=torch.float16)
        return action_emb, None

    def _encode_context(self, context, context_t, **kwargs):
        if not self.phi_time:
            context_t = torch.zeros_like(context_t)
        return super()._encode_context(context, context_t, **kwargs)

    def _bilinear_readout(self, goal, context):
        """out_i = sum_j (Q_i · K_j) V_j / (M sqrt(d)), computed as Q_i^T (K^T V) so the context memory is explicit."""
        memory = bilinear_memory(self.readout_kv(self.readout_ctx_norm(context)), self.readout_heads)
        return bilinear_apply(self.readout_q(self.readout_goal_norm(goal)), memory)

    def _decode_prediction(self, x, t, encoder_context, task_emb=None, action_emb=None, use_fp16=False):
        batches, frames = x.shape[:2]
        if frames != 1:
            raise ValueError(f"Bilinear decoder expects exactly one prediction frame, got {frames}")

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed
        if t.ndim == 2:
            if t.shape[0] != batches:
                raise ValueError(f"t batch ({t.shape[0]}) must match x batch ({batches})")
            t = t[:, -1]
        # Goal tower is conditioned on t only: psi(x_t, t), or psi(x_t) with psi_time=False.
        c = self.t_embedder(t if self.psi_time else torch.zeros_like(t), use_fp16=use_fp16)
        for block in self.decoder_blocks:
            x = self._run_block(block, x, c, rope_freqs=self.spatial_rope_freqs)

        x = self.readout_out(self._bilinear_readout(x, encoder_context))
        x = self.unpatchify(x)
        return rearrange(x, '(b f) c h w -> b f c h w', b=batches, f=frames)


_SIZES = {
    'XL': dict(depth=28, hidden_size=1152, num_heads=16),
    'L': dict(depth=24, hidden_size=1024, num_heads=16),
    'B': dict(depth=12, hidden_size=768, num_heads=12),
    'S': dict(depth=12, hidden_size=384, num_heads=6),
}
_PATCHES = {'XL': (2, 4, 8), 'L': (1, 2, 4, 8), 'B': (1, 2, 4, 8), 'S': (1, 2, 4, 8)}


def _make(size, patch_size):
    return lambda **kwargs: NanoWMBilinear(patch_size=patch_size, **_SIZES[size], **kwargs)


# Same names as nanowm_occupancy.NanoWM_models, so model.arch works unchanged.
NanoWM_models = {f'NanoWM-{s}/{p}': _make(s, p) for s in _SIZES for p in _PATCHES[s]}
