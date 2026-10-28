"""
Object-token bilinear time-score model for state-based environments (BuilderBench).

The same two-tower bilinear design as models/nanowm_bilinear.py, on token sets instead of image patches:
  - Context tower phi(s, a[, t]): one token per object in the state (gripper, cubes) plus one action token,
    NanoWM TransformerBlocks (adaLN-Zero, SwiGLU), conditioned on t only when phi_time is set.
  - Goal tower psi(x_t[, t]): one token per goal object (cubes, plus the gripper in the full-observation goal
    space), with a key mask so any subset of objects can be the goal. Conditioned on t when psi_time is set.
  - Readout: out_i = W_out M(s, a)^T Q_i per goal token (attention without the softmax, bilinear_memory /
    bilinear_apply), no bias and nothing nonlinear after it; W_out starts at zero, so the score starts at 0.
No positional embeddings or RoPE: tokens are a set, and the model is equivariant to permuting the cubes, which
matches BuilderBench's permutation-invariant success criterion.
"""

import torch
import torch.nn as nn

from .nanowm_bilinear import bilinear_apply, bilinear_memory
from .nanowm_occupancy import TimestepEmbedder, TransformerBlock


class ObjectBilinearTimeScore(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_size=256, depth=4, num_heads=4, mlp_ratio=4.0,
                 readout_heads=1, time_embedding="fourier", time_dim=256, phi_time=True, psi_time=True):
        super().__init__()
        if time_embedding not in ("fourier", "linear"):
            raise ValueError(f"time_embedding must be 'fourier' or 'linear', got {time_embedding}")
        if hidden_size % readout_heads:
            raise ValueError(f"hidden_size {hidden_size} not divisible by readout_heads {readout_heads}")
        self.goal_dim, self.readout_heads = goal_dim, readout_heads
        self.time_embedding, self.phi_time, self.psi_time = time_embedding, phi_time, psi_time

        # Token embeddings: a linear map per stream plus a learned type embedding (0 = gripper, 1 = cube).
        self.state_in = nn.Linear(state_dim, hidden_size)
        self.action_in = nn.Linear(action_dim, hidden_size)
        self.goal_in = nn.Linear(goal_dim, hidden_size)
        self.state_type = nn.Parameter(torch.zeros(2, hidden_size))
        self.goal_type = nn.Parameter(torch.zeros(2, hidden_size))
        self.action_type = nn.Parameter(torch.zeros(1, hidden_size))

        emb_type = "fourier" if time_embedding == "fourier" else "sinusoidal"
        self.phi_t = TimestepEmbedder(hidden_size, embedding_type=emb_type, fourier_dim=time_dim)
        self.psi_t = TimestepEmbedder(hidden_size, embedding_type=emb_type, fourier_dim=time_dim)
        self.phi_blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.psi_blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])

        self.readout_goal_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.readout_ctx_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.readout_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.readout_out = nn.Linear(hidden_size, goal_dim, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for p in (self.state_type, self.goal_type, self.action_type):
            nn.init.normal_(p, std=0.02)
        for emb in (self.phi_t, self.psi_t):
            nn.init.normal_(emb.mlp[0].weight, std=0.02)
            nn.init.normal_(emb.mlp[2].weight, std=0.02)
        for block in list(self.phi_blocks) + list(self.psi_blocks):  # adaLN-Zero: every block starts as the identity
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.readout_out.weight, 0)

    def _time_input(self, t, use_time):
        """GTSE time t in [0, 1) -> timestep-embedder input; a constant 0 when that tower does not see t."""
        t = t.float() if use_time else torch.zeros_like(t, dtype=torch.float32)
        if self.time_embedding == "fourier":
            return -torch.log(torch.clamp(1.0 - t, 1e-5, 1.0))
        return (1.0 - t) * 999.0

    def _context_tokens(self, state, action, t):
        types = torch.ones(state.shape[1], dtype=torch.long, device=state.device)
        types[0] = 0
        x = self.state_in(state) + self.state_type[types]
        x = torch.cat([x, (self.action_in(action) + self.action_type)[:, None]], dim=1)
        c = self.phi_t(self._time_input(t, self.phi_time))
        for block in self.phi_blocks:
            x = block(x, c)
        return self.readout_kv(self.readout_ctx_norm(x))

    def context_memory(self, state, action, t):
        """state: (B, Ns, state_dim), token 0 = gripper; action: (B, action_dim); t: (B,) -> memory (B, H, hd, hd)."""
        return bilinear_memory(self._context_tokens(state, action, t), self.readout_heads)

    def context_kv(self, state, action, t):
        """Keys and values of the context tokens, (B, M, H, hd) each, with the key scaled so that
        memory = sum_j k_j v_j^T equals `context_memory` (lets callers skip building the hd x hd memory)."""
        kv = self._context_tokens(state, action, t)
        b, m, d2 = kv.shape
        hd = d2 // 2 // self.readout_heads
        k, v = kv.view(b, m, 2, self.readout_heads, hd).unbind(2)
        return k / (m * hd ** 0.5), v

    def goal_queries(self, x_t, t, goal_mask, goal_types):
        """x_t: (B, Ng, goal_dim) noised goal tokens; goal_mask: (B, Ng) bool; goal_types: (Ng,) long -> (B, Ng, D)."""
        x = self.goal_in(x_t) + self.goal_type[goal_types]
        c = self.psi_t(self._time_input(t, self.psi_time))
        attn_mask = goal_mask[:, None, None, :]
        for block in self.psi_blocks:
            x = block(x, c, attn_mask=attn_mask)
        return self.readout_q(self.readout_goal_norm(x))

    def score_from(self, q, memory):
        """Per-dimension time score of each goal token, (B, Ng, goal_dim)."""
        return self.readout_out(bilinear_apply(q, memory))

    def forward(self, state, action, x_t, t, goal_mask, goal_types):
        return self.score_from(self.goal_queries(x_t, t, goal_mask, goal_types), self.context_memory(state, action, t))


def scalar_score(score, goal_mask, feat_mask):
    """Sum of the per-dimension score over the valid dims of the valid goal tokens -> (B,). Still bilinear."""
    return (score.float() * feat_mask * goal_mask[..., None]).flatten(1).sum(1)
