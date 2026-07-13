# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

def modulate(x, shift, scale):
    # Support both 2D (batch-wise) and 3D (token-wise) shift/scale
    if shift.ndim == 2:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale) + shift

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (for QK-Norm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: replaces standard MLP with gated activation."""
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.w3(nn.functional.silu(self.w1(x)) * self.w2(x)))


def get_2d_rotary_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D rotary position embeddings (frequencies)."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack([grid[0].flatten(), grid[1].flatten()], dim=-1)  # [H*W, 2]
    half_dim = embed_dim // 4
    freqs = 1.0 / (10000.0 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
    # [H*W, half_dim] for each spatial dim
    freqs_h = torch.outer(grid[:, 0], freqs)  # [H*W, half_dim]
    freqs_w = torch.outer(grid[:, 1], freqs)
    freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # [H*W, embed_dim//2]
    return freqs


def get_1d_rotary_pos_embed(embed_dim, length):
    """Generate 1D rotary position embeddings (frequencies)."""
    half_dim = embed_dim // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
    positions = torch.arange(length, dtype=torch.float32)
    freqs = torch.outer(positions, freqs)  # [length, embed_dim//2]
    return freqs


def apply_rotary_pos_embed(x, freqs):
    """Apply rotary position embeddings to tensor x.
    x: [B, H, N, D] (attention head format)
    freqs: [N, D//2]
    """
    d = x.shape[-1]
    freqs = freqs[:x.shape[-2], :d // 2].to(x.device)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, N, D//2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Self-attention over PyTorch SDPA.

    SDPA auto-selects FlashAttention-2 under bf16/fp16 on sm80+, falls back to
    memory-efficient or math kernels otherwise. There is no longer a knob to
    pick a backend explicitly — the previous `attention_mode` plumbing was a
    debug leftover and never reached anything from config.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.attn_drop_p = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, rope_freqs=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_freqs is not None:
            q = apply_rotary_pos_embed(q, rope_freqs)
            k = apply_rotary_pos_embed(k, rope_freqs)

        # SDPA output is [B, H, N, D]; transpose H/N before fusing heads.
        dropout_p = self.attn_drop_p if self.training else 0.0
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=dropout_p,
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CrossAttention(nn.Module):
    """Cross-attention: Q from x, K/V from context (action embeddings)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop_p = attn_drop
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, is_causal=False):
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        if is_causal:
            assert N == M, f"Causal cross-attention requires N==M (got N={N}, M={M})"
        dropout_p = self.attn_drop_p if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=dropout_p,
        )
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ActionEmbedder(nn.Module):
    """
    Embeds action vectors into hidden representations for frame-wise conditioning.
    """
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # Initialize to zero so action has no effect at the beginning
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, action):
        """
        action: (B, T, action_dim) tensor of actions
        Returns: (B, T, hidden_size) tensor of action embeddings
        """
        return self.mlp(action)


class TaskEmbedder(nn.Module):
    """
    Embeds one-hot task vectors into hidden representations for global conditioning.
    """
    def __init__(self, num_tasks, hidden_size):
        super().__init__()
        self.num_tasks = num_tasks
        self.mlp = nn.Sequential(
            nn.Linear(num_tasks, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, task):
        """
        task: (B, num_tasks) one-hot tensor, or (B,) integer task ids
        Returns: (B, hidden_size) tensor of task embeddings
        """
        if task.ndim == 1:
            task = nn.functional.one_hot(task.long(), num_classes=self.num_tasks)
        task = task.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(task)


#################################################################################
#                                 Core NanoWM Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A NanoWM transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Supports multiple action injection methods: additive, adaln_fuse, adaln, film, cross_attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 action_injection_type='additive'):
        super().__init__()
        self.action_injection_type = action_injection_type
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if action_injection_type == 'adaln':
            self.action_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        elif action_injection_type == 'film':
            self.film_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        elif action_injection_type == 'cross_attention':
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
            self.cross_gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, c, action_emb=None, rope_freqs=None, is_causal=False):
        # c: [B', D] (batch-wise, spatial) or [B', N', D] (token-wise, temporal)
        # x: [B', N', D]
        # is_causal: only set True by caller for temporal blocks when model.causal=True
        if c.ndim == 3:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Action injection: adaln — add action's shift/scale/gate to timestep's
        if self.action_injection_type == 'adaln' and action_emb is not None:
            if c.ndim == 2:
                # Spatial: action_emb is [B*F, 1, D], squeeze to [B*F, D]
                a_params = self.action_adaLN_modulation(action_emb.squeeze(1)).chunk(6, dim=1)
            else:
                # Temporal: action_emb is [B*P, F, D]
                a_params = self.action_adaLN_modulation(action_emb).chunk(6, dim=2)
            shift_msa = shift_msa + a_params[0]
            scale_msa = scale_msa + a_params[1]
            gate_msa = gate_msa + a_params[2]
            shift_mlp = shift_mlp + a_params[3]
            scale_mlp = scale_mlp + a_params[4]
            gate_mlp = gate_mlp + a_params[5]

        # Action injection: additive — add action_emb directly to x
        if self.action_injection_type == 'additive' and action_emb is not None:
            x = x + action_emb

        # Self-attention + MLP with adaLN
        if c.ndim == 3:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope_freqs=rope_freqs, is_causal=is_causal)
            # Action injection: film — modulate after self-attn, before MLP
            if self.action_injection_type == 'film' and action_emb is not None:
                gamma, beta = self.film_modulation(action_emb).chunk(2, dim=2)
                x = (1 + gamma) * x + beta
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope_freqs=rope_freqs, is_causal=is_causal)
            # Action injection: film — modulate after self-attn, before MLP
            if self.action_injection_type == 'film' and action_emb is not None:
                gamma, beta = self.film_modulation(action_emb.squeeze(1)).chunk(2, dim=1)
                x = (1 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        # Action injection: cross_attention — after self-attn+MLP
        if self.action_injection_type == 'cross_attention' and action_emb is not None:
            x = x + self.cross_gate * self.cross_attn(self.norm_cross(x), action_emb, is_causal=is_causal)

        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block: self-attention over prediction-frame patches,
    standard cross-attention to encoder tokens, then MLP.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 action_injection_type='additive'):
        super().__init__()
        self.action_injection_type = action_injection_type
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.encoder_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if action_injection_type == 'adaln':
            self.action_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        elif action_injection_type == 'film':
            self.film_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        elif action_injection_type == 'cross_attention':
            self.norm_action_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.action_cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
            self.action_cross_gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, c, encoder_context, action_emb=None, rope_freqs=None):
        if c.ndim == 3:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        if self.action_injection_type == 'adaln' and action_emb is not None:
            if c.ndim == 2:
                a_params = self.action_adaLN_modulation(action_emb.squeeze(1)).chunk(6, dim=1)
            else:
                a_params = self.action_adaLN_modulation(action_emb).chunk(6, dim=2)
            shift_msa = shift_msa + a_params[0]
            scale_msa = scale_msa + a_params[1]
            gate_msa = gate_msa + a_params[2]
            shift_mlp = shift_mlp + a_params[3]
            scale_mlp = scale_mlp + a_params[4]
            gate_mlp = gate_mlp + a_params[5]

        if self.action_injection_type == 'additive' and action_emb is not None:
            x = x + action_emb

        if c.ndim == 3:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope_freqs=rope_freqs)
            x = x + self.encoder_attn(self.norm_cross(x), encoder_context)
            if self.action_injection_type == 'film' and action_emb is not None:
                gamma, beta = self.film_modulation(action_emb).chunk(2, dim=2)
                x = (1 + gamma) * x + beta
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope_freqs=rope_freqs)
            x = x + self.encoder_attn(self.norm_cross(x), encoder_context)
            if self.action_injection_type == 'film' and action_emb is not None:
                gamma, beta = self.film_modulation(action_emb.squeeze(1)).chunk(2, dim=1)
                x = (1 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        if self.action_injection_type == 'cross_attention' and action_emb is not None:
            x = x + self.action_cross_gate * self.action_cross_attn(self.norm_action_cross(x), action_emb)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of NanoWM.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class NanoWM(nn.Module):
    """
    Occupancy diffusion model with a NanoWM encoder and cross-attention decoder.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        extras=1,
        use_action=False,
        action_dim=None,
        action_injection_type='additive',
        causal=False,
        num_tasks=50,
        decoder_depth=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames
        self.action_injection_type = action_injection_type
        self.causal = causal
        self.num_tasks = num_tasks
        self.decoder_depth = depth if decoder_depth is None else decoder_depth

        # Action conditioning
        self.use_action = use_action
        self.action_dim = action_dim
        if self.use_action:
            assert action_dim is not None, "action_dim must be specified when use_action=True"
            self.action_embedder = ActionEmbedder(action_dim, hidden_size)
        self.task_embedder = TaskEmbedder(num_tasks, hidden_size)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches
        head_dim = hidden_size // num_heads
        grid_size = int(num_patches ** 0.5)
        # RoPE frequencies (not learned parameters, computed on-the-fly)
        self.register_buffer("spatial_rope_freqs", get_2d_rotary_pos_embed(head_dim, grid_size), persistent=False)
        self.register_buffer("temporal_rope_freqs", get_1d_rotary_pos_embed(head_dim, num_frames), persistent=False)
        # Keep pos_embed for backward compatibility (patch embedding offset), but make it learnable
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size))
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                             action_injection_type=action_injection_type) for _ in range(depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                         action_injection_type=action_injection_type) for _ in range(self.decoder_depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # pos_embed and temp_embed are learned offsets (RoPE handles positional info)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.temp_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize task conditioning as a no-op, matching adaLN-fuse's additive path.
        nn.init.constant_(self.task_embedder.mlp[-1].weight, 0)
        nn.init.constant_(self.task_embedder.mlp[-1].bias, 0)

        # Zero-out adaLN modulation layers in encoder/decoder blocks:
        for block in list(self.blocks) + list(self.decoder_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # Zero-out action-specific layers
            if hasattr(block, 'action_adaLN_modulation'):
                nn.init.constant_(block.action_adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.action_adaLN_modulation[-1].bias, 0)
            if hasattr(block, 'film_modulation'):
                nn.init.constant_(block.film_modulation[-1].weight, 0)
                nn.init.constant_(block.film_modulation[-1].bias, 0)
            if hasattr(block, 'cross_attn'):
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)
            if hasattr(block, 'action_cross_attn'):
                nn.init.constant_(block.action_cross_attn.proj.weight, 0)
                nn.init.constant_(block.action_cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _get_task_embedding(self, task, batches, device, use_fp16=False):
        if task is None:
            return None
        task = task.to(device)
        task_emb = self.task_embedder(task)
        if task_emb.shape[0] != batches:
            raise ValueError(f"task batch ({task_emb.shape[0]}) must match x batch ({batches})")
        if use_fp16:
            task_emb = task_emb.to(dtype=torch.float16)
        return task_emb

    def _get_timestep_conditioning(self, t, batches, frames, use_fp16=False):
        num_patches = self.pos_embed.shape[1]
        if t.ndim == 2:
            if t.shape != (batches, frames):
                raise ValueError(f"t.shape must be {(batches, frames)} for {frames} frames, got {tuple(t.shape)}")
            t_flat = t.flatten()
            t_emb_flat = self.t_embedder(t_flat, use_fp16=use_fp16)
            t_emb = t_emb_flat.reshape(batches, frames, -1)
            timestep_spatial = t_emb_flat
            timestep_temp = repeat(t_emb, 'b f d -> (b p) f d', p=num_patches)
        else:
            if t.shape[0] != batches:
                raise ValueError(f"t batch ({t.shape[0]}) must match x batch ({batches})")
            t_emb = self.t_embedder(t, use_fp16=use_fp16)
            timestep_spatial = repeat(t_emb, 'b d -> (b f) d', f=frames)
            timestep_temp = repeat(t_emb, 'b d -> (b p) d', p=num_patches)
        return timestep_spatial, timestep_temp

    def _shift_action_embeddings(self, action_emb):
        if action_emb.shape[1] == 0:
            return action_emb
        return torch.cat([
            torch.zeros_like(action_emb[:, :1, :]),
            action_emb[:, :-1, :],
        ], dim=1)

    def _prepare_action_embeddings(self, action, batches, context_frames, target_frames, use_fp16=False):
        if not self.use_action or action is None:
            return None, None
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.shape[0] != batches:
            raise ValueError(f"action batch ({action.shape[0]}) must match x batch ({batches})")

        action = action.to(dtype=self.action_embedder.mlp[0].weight.dtype)
        action_emb_all = self.action_embedder(action)
        if use_fp16:
            action_emb_all = action_emb_all.to(dtype=torch.float16)

        total_frames = context_frames + target_frames
        action_frames = action_emb_all.shape[1]
        if action_frames >= total_frames:
            shifted = self._shift_action_embeddings(action_emb_all[:, :total_frames])
            return shifted[:, :context_frames], shifted[:, context_frames:total_frames]

        if action_frames == context_frames and context_frames > 0:
            context_action = self._shift_action_embeddings(action_emb_all)
            target_action = action_emb_all[:, -1:, :].repeat(1, target_frames, 1)
            return context_action, target_action

        if action_frames < total_frames:
            pad = torch.zeros(
                batches,
                total_frames - action_frames,
                action_emb_all.shape[-1],
                device=action_emb_all.device,
                dtype=action_emb_all.dtype,
            )
            action_emb_all = torch.cat([action_emb_all, pad], dim=1)

        shifted = self._shift_action_embeddings(action_emb_all[:, :total_frames])
        return shifted[:, :context_frames], shifted[:, context_frames:total_frames]

    def _encode_context(self, context, context_t, task_emb=None, action_emb=None,
                        y=None, text_embedding=None, use_fp16=False):
        batches, frames = context.shape[:2]
        if frames < 1:
            raise ValueError("Occupancy encoder requires at least one context frame")
        if frames > self.temp_embed.shape[1]:
            raise ValueError(
                f"context has {frames} frames, but model was initialized for {self.temp_embed.shape[1]}"
            )

        action_emb_spatial_in = None
        action_emb_temp_in = None
        if action_emb is not None:
            num_patches = self.pos_embed.shape[1]
            action_emb_spatial_in = rearrange(action_emb, 'b f d -> (b f) 1 d')
            action_emb_temp_in = repeat(action_emb, 'b f d -> (b p) f d', p=num_patches)

        x = rearrange(context, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed
        timestep_spatial, timestep_temp = self._get_timestep_conditioning(
            context_t, batches, frames, use_fp16=use_fp16,
        )

        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=frames)
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'n d -> (n c) d', c=frames)

        task_spatial = None
        task_temp = None
        if task_emb is not None:
            task_spatial = repeat(task_emb, 'b d -> (b f) d', f=frames)
            task_temp = repeat(task_emb, 'b d -> (b p) f d', p=self.pos_embed.shape[1], f=frames)

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            else:
                c = timestep_spatial
            if task_spatial is not None:
                c = c + task_spatial

            spatial_action = None
            if action_emb is not None:
                if self.action_injection_type == 'adaln_fuse':
                    c = c + action_emb_spatial_in.squeeze(1)
                else:
                    spatial_action = action_emb_spatial_in

            x = spatial_block(x, c, action_emb=spatial_action, rope_freqs=self.spatial_rope_freqs, is_causal=False)
            x = rearrange(x, '(b f) p d -> (b p) f d', b=batches)

            if i == 0:
                x = x + self.temp_embed[:, :frames]

            if self.extras == 2:
                raise ValueError("extras == 2 is not supported for the occupancy encoder")
            elif self.extras == 78:
                raise ValueError("extras == 78 is not supported for the occupancy encoder")
            else:
                c = timestep_temp
            if task_temp is not None:
                c = c + task_temp

            temporal_action = None
            if action_emb is not None:
                if self.action_injection_type == 'adaln_fuse':
                    if c.ndim == 2:
                        c = c.unsqueeze(1) + action_emb_temp_in
                    else:
                        c = c + action_emb_temp_in
                else:
                    temporal_action = action_emb_temp_in

            x = temp_block(x, c, action_emb=temporal_action, rope_freqs=self.temporal_rope_freqs, is_causal=self.causal)
            x = rearrange(x, '(b p) f d -> (b f) p d', b=batches)

        return rearrange(x, '(b f) p d -> b (f p) d', b=batches, f=frames)

    def _decode_prediction(self, x, t, encoder_context, task_emb=None, action_emb=None, use_fp16=False):
        batches, frames = x.shape[:2]
        if frames != 1:
            raise ValueError(f"Occupancy decoder expects exactly one prediction frame, got {frames}")

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed

        if t.ndim == 2:
            if t.shape[0] != batches:
                raise ValueError(f"t batch ({t.shape[0]}) must match x batch ({batches})")
            t = t[:, -1]
        timestep = self.t_embedder(t, use_fp16=use_fp16)

        decoder_action = None
        for block in self.decoder_blocks:
            c = timestep
            if task_emb is not None:
                c = c + task_emb
            if action_emb is not None:
                if self.action_injection_type == 'adaln_fuse':
                    c = c + action_emb.squeeze(1)
                else:
                    decoder_action = action_emb
            x = block(
                x,
                c,
                encoder_context=encoder_context,
                action_emb=decoder_action,
                rope_freqs=self.spatial_rope_freqs,
            )

        c_final = timestep
        if task_emb is not None:
            c_final = c_final + task_emb
        x = self.final_layer(x, c_final)
        x = self.unpatchify(x)
        return rearrange(x, '(b f) c h w -> b f c h w', b=batches, f=frames)

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self,
                x,
                t,
                y=None,
                text_embedding=None,
                use_fp16=False,
                action=None,
                context=None,
                context_t=None,
                task=None):
        """
        Forward pass of NanoWM occupancy.
        x: (N, 1, C, H, W) noised prediction frame, or (N, C, H, W)
        context: (N, F_ctx, C, H, W) history frames for the encoder
        t: (N,) or (N, 1) regular diffusion timesteps for the prediction frame;
           reused for context unless context_t is explicitly supplied
        task: (N, 50) one-hot task vector, or (N,) integer task ids
        action: (N, F_ctx + 1, action_dim) preferred, or (N, F_ctx, action_dim)
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)

        if context is None:
            if x.shape[1] < 2:
                raise ValueError(
                    "context must be provided when x contains only the noised prediction frame"
                )
            context = x[:, :-1].contiguous()
            x = x[:, -1:].contiguous()
            if t.ndim == 2 and t.shape[1] >= context.shape[1] + 1:
                t = t[:, -1:].contiguous()
        elif context.ndim == 4:
            context = context.unsqueeze(1)

        if x.shape[1] != 1:
            raise ValueError(f"x must contain exactly one prediction frame, got {x.shape[1]}")
        if context.shape[0] != x.shape[0]:
            raise ValueError(f"context batch ({context.shape[0]}) must match x batch ({x.shape[0]})")

        if use_fp16:
            x = x.to(dtype=torch.float16)
            context = context.to(dtype=torch.float16)

        batches = x.shape[0]
        context_frames = context.shape[1]
        target_frames = x.shape[1]
        if context_t is None:
            if t.ndim == 2:
                context_t = t[:, -1:].repeat(1, context_frames)
            else:
                context_t = t.unsqueeze(1).repeat(1, context_frames)

        task_emb = self._get_task_embedding(task, batches, x.device, use_fp16=use_fp16)
        context_action_emb, target_action_emb = self._prepare_action_embeddings(
            action,
            batches,
            context_frames,
            target_frames,
            use_fp16=use_fp16,
        )

        encoder_context = self._encode_context(
            context,
            context_t,
            task_emb=task_emb,
            action_emb=context_action_emb,
            y=y,
            text_embedding=text_embedding,
            use_fp16=use_fp16,
        )
        return self._decode_prediction(
            x,
            t,
            encoder_context,
            task_emb=task_emb,
            action_emb=target_action_emb,
            use_fp16=use_fp16,
        )

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False,
                         text_embedding=None, action=None, context=None,
                         context_t=None, task=None):
        """
        Forward pass of NanoWM, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)

        def _dup_first_half(value):
            if value is None:
                return None
            value_half = value[: len(value) // 2]
            return torch.cat([value_half, value_half], dim=0)
        
        t_combined = _dup_first_half(t) if t.shape[0] == x.shape[0] else t
        action_combined = _dup_first_half(action)
        context_combined = _dup_first_half(context)
        context_t_combined = _dup_first_half(context_t)
        task_combined = _dup_first_half(task)
        
        model_out = self.forward(
            combined,
            t_combined,
            y=y,
            use_fp16=use_fp16,
            text_embedding=text_embedding,
            action=action_combined,
            context=context_combined,
            context_t=context_t_combined,
            task=task_combined,
        )
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   NanoWM Configs                                  #
#################################################################################

def NanoWM_XL_2(**kwargs):
    return NanoWM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def NanoWM_XL_4(**kwargs):
    return NanoWM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def NanoWM_XL_8(**kwargs):
    return NanoWM(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def NanoWM_L_2(**kwargs):
    return NanoWM(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def NanoWM_L_1(**kwargs):
    return NanoWM(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def NanoWM_L_4(**kwargs):
    return NanoWM(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def NanoWM_L_8(**kwargs):
    return NanoWM(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def NanoWM_B_2(**kwargs):
    return NanoWM(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def NanoWM_B_1(**kwargs):
    return NanoWM(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def NanoWM_B_4(**kwargs):
    return NanoWM(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def NanoWM_B_8(**kwargs):
    return NanoWM(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def NanoWM_S_2(**kwargs):
    return NanoWM(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def NanoWM_S_1(**kwargs):
    return NanoWM(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def NanoWM_S_4(**kwargs):
    return NanoWM(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def NanoWM_S_8(**kwargs):
    return NanoWM(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


NanoWM_models = {
    'NanoWM-XL/2': NanoWM_XL_2,  'NanoWM-XL/4': NanoWM_XL_4,  'NanoWM-XL/8': NanoWM_XL_8,
    'NanoWM-L/1':  NanoWM_L_1,   'NanoWM-L/2':  NanoWM_L_2,   'NanoWM-L/4':  NanoWM_L_4,   'NanoWM-L/8':  NanoWM_L_8,
    'NanoWM-B/1':  NanoWM_B_1,   'NanoWM-B/2':  NanoWM_B_2,   'NanoWM-B/4':  NanoWM_B_4,   'NanoWM-B/8':  NanoWM_B_8,
    'NanoWM-S/1':  NanoWM_S_1,   'NanoWM-S/2':  NanoWM_S_2,   'NanoWM-S/4':  NanoWM_S_4,   'NanoWM-S/8':  NanoWM_S_8,
}

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    network = NanoWM_XL_2().to(device)
    from thop import profile 
    flops, params = profile(network, inputs=(img, t))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    # out = network(img, t, y)
    # print(out.shape)
