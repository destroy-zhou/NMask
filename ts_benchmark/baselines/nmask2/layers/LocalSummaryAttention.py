"""Target-to-covariate attention over local patches and pooled summaries."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .Embed import RotaryEmbedding


def validate_channel_attention(attention_type, window, summaries):
    if attention_type not in ("full", "local_summary"):
        raise ValueError("channel_attn_type must be full or local_summary")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0 or window % 2 == 0:
        raise ValueError("channel_window must be a positive odd integer")
    if isinstance(summaries, bool) or not isinstance(summaries, int) or summaries < 0:
        raise ValueError("channel_summaries must be a nonnegative integer")


class LocalSummaryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_channels, target_channels,
                 window=5, summaries=4, mode="rope", dropout=0.0, head_dim=None):
        super().__init__()
        validate_channel_attention("local_summary", window, summaries)
        if mode not in ("rope", "none", "embedding"):
            raise ValueError("channel_attn_mode must be rope, none or embedding")
        if not 0 < target_channels < n_channels:
            raise ValueError("local_summary requires at least one target and one covariate")
        self.window, self.summaries = window, summaries
        self.n_channels, self.target_channels = n_channels, target_channels
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        if mode == "rope" and self.head_dim % 2:
            self.head_dim += 1
        self.mode = mode
        width = n_heads * self.head_dim
        self.query_projection = nn.Linear(d_model, width)
        self.key_projection = nn.Linear(d_model, width)
        self.value_projection = nn.Linear(d_model, width)
        self.out_projection = nn.Linear(width, d_model)
        self.gate_query = nn.Linear(d_model, width)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if mode == "rope" else None
        self.channel_embedding = None
        if mode == "embedding":
            self.channel_embedding = nn.Parameter(torch.empty(1, n_channels, 1, d_model))
            nn.init.normal_(self.channel_embedding, std=0.02)

    def _rotate_channels(self, tensor):
        # [B, C, P, H, d] -> rotate along C, never along local memory slots.
        b, c, p, h, d = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1, 4).reshape(b * p * h, c, d)
        tensor = self.rope(tensor)
        return tensor.reshape(b, p, h, c, d).permute(0, 3, 1, 2, 4)

    def _local_indices(self, length, device):
        offsets = torch.arange(-(self.window // 2), self.window // 2 + 1, device=device)
        indices = torch.arange(length, device=device)[:, None] + offsets
        valid = (indices >= 0) & (indices < length)
        return indices.clamp(0, length - 1), valid

    @staticmethod
    def _pool(tensor, count):
        # [B, E, H, P, d] -> ordered, per-variable temporal summaries.
        b, e, h, p, d = tensor.shape
        flat = tensor.permute(0, 1, 2, 4, 3).reshape(b * e * h, d, p)
        return F.adaptive_avg_pool1d(flat, count).reshape(b, e, h, d, count).transpose(-1, -2)

    def forward(self, x):
        # x: [B, C, P, D], target channels first, then covariates.
        b, c, p, _ = x.shape
        if c != self.n_channels or p == 0:
            raise ValueError("Unexpected channel count or empty patch sequence")
        s, h, d = self.target_channels, self.n_heads, self.head_dim
        qk = x if self.channel_embedding is None else x + self.channel_embedding
        q = self.query_projection(qk).reshape(b, c, p, h, d)
        k = self.key_projection(qk).reshape(b, c, p, h, d)
        v = self.value_projection(x).reshape(b, c, p, h, d)
        if self.rope is not None:
            q, k = self._rotate_channels(q), self._rotate_channels(k)
        q = q[:, :s].permute(0, 1, 3, 2, 4)
        k, v = (t[:, s:].permute(0, 1, 3, 2, 4) for t in (k, v))

        indices, valid = self._local_indices(p, x.device)
        local_k, local_v = k[:, :, :, indices, :], v[:, :, :, indices, :]
        scores = torch.einsum("bshpd,behpwd->bshepw", q, local_k) / math.sqrt(d)
        scores = scores.masked_fill(~valid, float("-inf"))
        count = min(self.summaries, p)
        if count:
            summary_k, summary_v = self._pool(k, count), self._pool(v, count)
            global_scores = torch.einsum("bshpd,behrd->bshepr", q, summary_k) / math.sqrt(d)
            scores = torch.cat([scores, global_scores], dim=-1)
        # Normalize within each variable first, then fuse variables dynamically.
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.einsum("bshepw,behpwd->bshepd", weights[..., :self.window], local_v)
        if count:
            context = context + torch.einsum("bshepr,behrd->bshepd", weights[..., self.window:], summary_v)
        context = context.permute(0, 1, 4, 3, 2, 5).reshape(b, s, p, c - s, h * d)
        gate_query = self.gate_query(qk[:, :s]).unsqueeze(-2)
        gate_keys = context
        if self.channel_embedding is not None:
            # A constant per-variable K offset cancels in the within-variable
            # softmax. Include identity in variable selection as well, without
            # adding it to the values being fused.
            identities = F.linear(self.channel_embedding[:, s:, 0], self.key_projection.weight)
            gate_keys = context + identities[:, None, None]
        gate = torch.softmax((gate_query * gate_keys).sum(-1) / math.sqrt(h * d), dim=-1)
        fused = (gate.unsqueeze(-1) * context).sum(-2)
        return self.out_projection(fused)
