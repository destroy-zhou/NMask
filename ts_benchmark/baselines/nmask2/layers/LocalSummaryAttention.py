"""Target-to-covariate attention over local patches and pooled summaries."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .Embed import RotaryEmbedding


def validate_channel_attention(attention_type, window, summaries):
    if attention_type not in ("full", "local_summary"):
        raise ValueError("channel_attn_type must be full or local_summary")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("channel_window must be a positive integer")
    if isinstance(summaries, bool) or not isinstance(summaries, int) or summaries < 0:
        raise ValueError("channel_summaries must be a nonnegative integer")


def validate_channel_fusion(mode):
    if mode not in ("dot", "qk", "mlp", "cross_attn"):
        raise ValueError("channel_fusion_mode must be dot, qk, mlp or cross_attn")


class LocalSummaryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_channels, target_channels,
                 window=5, summaries=4, mode="rope", dropout=0.0, head_dim=None,
                 local_time_rope=True, channel_fusion_mode="dot", calendar_channels=0,
                 channel_group_gating=False):
        super().__init__()
        validate_channel_attention("local_summary", window, summaries)
        if mode not in ("rope", "none", "embedding"):
            raise ValueError("channel_attn_mode must be rope, none or embedding")
        if not 0 < target_channels < n_channels:
            raise ValueError("local_summary requires at least one target and one covariate")
        self.window, self.summaries = window, summaries
        self.n_channels, self.target_channels = n_channels, target_channels
        if (isinstance(calendar_channels, bool) or not isinstance(calendar_channels, int)
                or not 0 <= calendar_channels <= n_channels - target_channels):
            raise ValueError("calendar_channels must fit within the covariate channels")
        self.calendar_channels = calendar_channels
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        if mode == "rope" and self.head_dim % 2:
            self.head_dim += 1
        self.mode = mode
        if not isinstance(local_time_rope, bool):
            raise ValueError("local_time_rope must be a boolean")
        self.local_time_rope = local_time_rope
        if not isinstance(channel_group_gating, bool):
            raise ValueError("channel_group_gating must be a boolean")
        self.channel_group_gating = channel_group_gating
        width = n_heads * self.head_dim
        self.query_projection = nn.Linear(d_model, width)
        self.key_projection = nn.Linear(d_model, width)
        self.value_projection = nn.Linear(d_model, width)
        self.out_projection = nn.Linear(width, d_model)
        self.gate_query = nn.Linear(d_model, width)
        validate_channel_fusion(channel_fusion_mode)
        self.channel_fusion_mode = channel_fusion_mode
        # Only optional modes add parameters, preserving existing dot checkpoints.
        self.gate_key = nn.Linear(width, width, bias=channel_fusion_mode == "cross_attn") if channel_fusion_mode in ("qk", "cross_attn") else None
        self.gate_value = nn.Linear(width, width) if channel_fusion_mode == "cross_attn" else None
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * width, min(64, width)), nn.GELU(),
            nn.Linear(min(64, width), 1, bias=False),
        ) if channel_fusion_mode == "mlp" else None
        gate_input = d_model + 2 * width
        self.history_group_gate = nn.Linear(gate_input, 1) if channel_group_gating else None
        self.summary_group_gate = nn.Linear(gate_input, 1) if channel_group_gating else None
        for gate in (self.history_group_gate, self.summary_group_gate):
            if gate is not None:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -3.0)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if mode == "rope" else None
        self.channel_embedding = None
        if mode == "embedding":
            self.channel_embedding = nn.Parameter(torch.empty(1, n_channels, 1, d_model))
            nn.init.normal_(self.channel_embedding, std=0.02)

    def share_attention_projections_from(self, other):
        """Tie attention projections while retaining stream-specific layout state."""
        if not isinstance(other, LocalSummaryAttention):
            raise TypeError("Attention projections can only be shared with LocalSummaryAttention")
        if (self.n_heads != other.n_heads or self.head_dim != other.head_dim
                or self.mode != other.mode
                or self.channel_fusion_mode != other.channel_fusion_mode
                or self.channel_group_gating != other.channel_group_gating):
            raise ValueError("Shared channel attention requires matching attention settings")
        names = (
            "query_projection", "key_projection", "value_projection",
            "out_projection", "gate_query", "gate_key", "gate_value", "gate_mlp",
            "history_group_gate", "summary_group_gate",
        )
        for name in names:
            source = getattr(other, name)
            if (getattr(self, name) is None) != (source is None):
                raise ValueError("Shared channel attention requires matching fusion projections")
            if source is not None:
                setattr(self, name, source)

    def _rotate_channels(self, tensor):
        # [B, C, P, H, d] -> rotate along C, never along local memory slots.
        b, c, p, h, d = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1, 4).reshape(b * p * h, c, d)
        tensor = self.rope(tensor)
        return tensor.reshape(b, p, h, c, d).permute(0, 3, 1, 2, 4)

    def _rotate_time(self, tensor):
        # [B, C, H, P, d]: use original patch positions before window gathering.
        # Rotate adjacent pairs with equal frequencies; keep an odd tail intact.
        rotary_dim = tensor.shape[-1] // 2 * 2
        if rotary_dim == 0:
            return tensor
        dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
        positions = torch.arange(tensor.shape[-2], device=tensor.device, dtype=dtype)
        freq = 10000 ** (-torch.arange(0, rotary_dim, 2, device=tensor.device, dtype=dtype) / rotary_dim)
        angles = positions[:, None] * freq[None, :]
        cos, sin = angles.cos().to(tensor.dtype), angles.sin().to(tensor.dtype)
        even, odd = tensor[..., :rotary_dim:2], tensor[..., 1:rotary_dim:2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)

    def _local_indices(self, length, device):
        # Trailing window: W includes the current patch and W-1 past patches.
        offsets = torch.arange(1 - self.window, 1, device=device)
        indices = torch.arange(length, device=device)[:, None] + offsets
        valid = (indices >= 0) & (indices < length)
        return indices.clamp(0, length - 1), valid

    @staticmethod
    def _pool(tensor, count):
        # [B, E, H, P, d] -> ordered, per-variable temporal summaries.
        b, e, h, p, d = tensor.shape
        flat = tensor.permute(0, 1, 2, 4, 3).reshape(b * e * h, d, p)
        return F.adaptive_avg_pool1d(flat, count).reshape(b, e, h, d, count).transpose(-1, -2)

    def _group_softmax(self, scores):
        """Softmax a possibly fully masked group without producing NaNs."""
        finite = torch.isfinite(scores)
        has_value = finite.any(dim=-1, keepdim=True)
        safe_scores = torch.where(has_value, scores, torch.zeros_like(scores))
        return self.dropout(torch.softmax(safe_scores, dim=-1) * finite)

    @staticmethod
    def _flatten_context(context):
        # [B, S, H, E, P, d] -> [B, S, P, E, H*d]
        b, s, h, e, p, d = context.shape
        return context.permute(0, 1, 4, 3, 2, 5).reshape(b, s, p, e, h * d)

    def forward(self, x, memory=None):
        if memory is not None:
            # A separate Q stream and read-only K/V stream. Concatenation here
            # only preserves channel positions for the legacy RoPE/embeddings;
            # the result contains updates for targets alone.
            if (x.ndim != 4 or memory.ndim != 4
                    or x.shape[1] != self.target_channels
                    or memory.shape[1] != self.n_channels - self.target_channels
                    or x.shape[0] != memory.shape[0] or x.shape[2:] != memory.shape[2:]):
                raise ValueError("Targets and covariate memory must have aligned batch/patch/features")
            x = torch.cat((x, memory), dim=1)
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
        local_q = self._rotate_time(q) if self.local_time_rope else q
        local_keys = self._rotate_time(k) if self.local_time_rope else k
        local_k, local_v = local_keys[:, :, :, indices, :], v[:, :, :, indices, :]
        scores = torch.einsum("bshpd,behpwd->bshepw", local_q, local_k) / math.sqrt(d)
        scores = scores.masked_fill(~valid, float("-inf"))
        count = min(self.summaries, p)
        if count:
            summary_k, summary_v = self._pool(k, count), self._pool(v, count)
            global_scores = torch.einsum("bshpd,behrd->bshepr", q, summary_k) / math.sqrt(d)
            scores = torch.cat([scores, global_scores], dim=-1)
        if self.calendar_channels:
            # Calendar channels are last. Permit only the current (last) local slot;
            # all neighboring slots and pooled summaries are invisible to them.
            allowed = torch.ones(c - s, 1, scores.shape[-1], dtype=torch.bool, device=x.device)
            allowed[-self.calendar_channels:] = False
            allowed[-self.calendar_channels:, :, self.window - 1] = True
            scores = scores.masked_fill(~allowed, float("-inf"))
        # Either retain the original joint softmax or aggregate current,
        # historical and summary sources independently before gated fusion.
        if self.channel_group_gating:
            current = local_v[..., self.window - 1, :].unsqueeze(1)
            current = current.expand(-1, s, -1, -1, -1, -1).permute(0, 1, 3, 2, 4, 5)
            current = self._flatten_context(current)
            target = x[:, :s].unsqueeze(-2).expand(-1, -1, -1, c - s, -1)
            context = current
            if self.window > 1:
                history_weights = self._group_softmax(scores[..., :self.window - 1])
                history = torch.einsum(
                    "bshepw,behpwd->bshepd", history_weights,
                    local_v[..., :self.window - 1, :],
                )
                history = self._flatten_context(history)
                history_gate = torch.sigmoid(self.history_group_gate(
                    torch.cat((target, current, history), dim=-1)
                ))
                context = context + history_gate * history
            if count:
                summary_weights = self._group_softmax(scores[..., self.window:])
                summary = torch.einsum("bshepr,behrd->bshepd",
                                       summary_weights, summary_v)
                summary = self._flatten_context(summary)
                summary_gate = torch.sigmoid(self.summary_group_gate(
                    torch.cat((target, current, summary), dim=-1)
                ))
                context = context + summary_gate * summary
        else:
            # Normalize within each variable first, then fuse variables dynamically.
            weights = self.dropout(torch.softmax(scores, dim=-1))
            context = torch.einsum("bshepw,behpwd->bshepd", weights[..., :self.window], local_v)
            if count:
                context = context + torch.einsum("bshepr,behrd->bshepd", weights[..., self.window:], summary_v)
            context = self._flatten_context(context)
        gate_query = self.gate_query(qk[:, :s]).unsqueeze(-2)
        gate_keys = context
        if self.channel_embedding is not None:
            # Include variable identity in selection without changing values.
            # Without time RoPE, a constant K offset cancels within a variable.
            identities = F.linear(self.channel_embedding[:, s:, 0], self.key_projection.weight)
            gate_keys = context + identities[:, None, None]
        if self.channel_fusion_mode == "cross_attn":
            # Each target patch is one query; covariates are the key/value axis.
            # Q/K/V projections span all heads, followed by per-head softmax.
            fusion_q = gate_query.reshape(b, s, p, h, d)
            fusion_k = self.gate_key(gate_keys).reshape(b, s, p, c - s, h, d)
            fusion_v = self.gate_value(context).reshape(b, s, p, c - s, h, d)
            fusion_scores = torch.einsum("bsphd,bspehd->bsphe", fusion_q, fusion_k) / math.sqrt(d)
            fusion_weights = self.dropout(torch.softmax(fusion_scores, dim=-1))
            fused = torch.einsum("bsphe,bspehd->bsphd", fusion_weights, fusion_v)
            return self.out_projection(fused.reshape(b, s, p, h * d))
        if self.channel_fusion_mode == "qk":
            gate_keys = self.gate_key(gate_keys)
        if self.channel_fusion_mode == "mlp":
            # One scorer shared over covariates, target channels and patches.
            gate_input = torch.cat((gate_query.expand_as(gate_keys), gate_keys), dim=-1)
            gate_scores = self.gate_mlp(gate_input).squeeze(-1)
        else:
            gate_scores = (gate_query * gate_keys).sum(-1) / math.sqrt(h * d)
        gate = torch.softmax(gate_scores, dim=-1)
        fused = (gate.unsqueeze(-1) * context).sum(-2)
        return self.out_projection(fused)
