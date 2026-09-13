"""Encode covariates once, then decode targets against read-only memory."""

import torch
from torch import nn
import torch.nn.functional as F

from .LocalSummaryAttention import LocalSummaryAttention
from .SelfAttention_Family import AttentionLayer, FullAttention
from .Transformer_EncDec import Encoder, EncoderLayer


def time_attention(d_model, n_heads, dropout, factor, use_rope):
    return AttentionLayer(
        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
        d_model, n_heads, use_rope=use_rope,
    )


class CovariateChannelAttention(nn.Module):
    """Mix covariates within each patch after temporal attention."""

    def __init__(self, d_model, n_heads, total_channels, dropout, factor, mode):
        super().__init__()
        self.total_channels = total_channels
        self.attention = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout,
                          output_attention=False),
            d_model, n_heads, use_rope=mode == "rope",
        )
        self.channel_embedding = (
            nn.Parameter(torch.empty(1, total_channels, d_model))
            if mode == "embedding" else None
        )
        if self.channel_embedding is not None:
            nn.init.normal_(self.channel_embedding, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, channels, channel_indices=None):
        if channels <= 0 or x.shape[0] % channels:
            raise ValueError("Covariate channel layout is invalid")
        batch, patches, d_model = x.shape[0] // channels, x.shape[1], x.shape[2]
        tokens = x.reshape(batch, channels, patches, d_model)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * patches, channels, d_model)
        qk = tokens
        if self.channel_embedding is not None:
            if channel_indices is None:
                channel_indices = torch.arange(channels, device=x.device)
            else:
                channel_indices = torch.as_tensor(channel_indices, device=x.device)
            if (channel_indices.ndim != 1 or channel_indices.numel() != channels
                    or channel_indices.min().item() < 0
                    or channel_indices.max().item() >= self.total_channels):
                raise ValueError("Covariate channel indices are invalid")
            qk = tokens + self.channel_embedding[:, channel_indices].expand_as(tokens)
        update, _ = self.attention(qk, qk, tokens, attn_mask=None)
        tokens = self.norm(tokens + self.dropout(update))
        return (
            tokens.reshape(batch, patches, channels, d_model)
            .permute(0, 2, 1, 3)
            .reshape_as(x)
        )


class CovariateEncoder(Encoder):
    def __init__(self, attn_layers, norm_layer, channel_layers=None):
        super().__init__(attn_layers, norm_layer=norm_layer)
        self.channel_layers = nn.ModuleList(channel_layers or [])

    def forward_intermediates(self, x, channels, channel_indices=None):
        """Save each time-layer result before optional channel mixing."""
        outputs, attentions = [], []
        for index, layer in enumerate(self.attn_layers):
            time_state, attention = layer(x, channels, stop_after_time=True)
            outputs.append(
                self.norm(time_state) if self.norm is not None else time_state
            )
            attentions.append(attention)
            x = time_state
            if index < len(self.channel_layers):
                x = self.channel_layers[index](
                    time_state, channels, channel_indices,
                )
            x = layer.feed_forward(x)
        final_state = self.norm(x) if self.norm is not None else x
        return outputs, attentions, final_state


def build_covariate_channel_layers(d_model, n_heads, layers, total_channels,
                                     dropout, factor, channel_attn_mode):
    return [
        CovariateChannelAttention(
            d_model, n_heads, total_channels, dropout, factor,
            channel_attn_mode,
        )
        for _ in range(layers)
    ]


def build_covariate_encoder(d_model, d_ff, n_heads, layers, dropout,
                            factor, activation, use_rope,
                            self_channel_attn=False, channel_attn_mode="none",
                            total_channels=None):
    # Covariates are encoded independently of targets. The mixed post-FFN state
    # feeds the next temporal layer or, at the final depth, the auxiliary head.
    temporal_layers = [
        EncoderLayer(time_attention(d_model, n_heads, dropout, factor, use_rope),
                     None, d_model, d_ff, dropout, activation)
        for _ in range(layers)
    ]
    channel_layers = []
    if self_channel_attn:
        if total_channels is None or total_channels <= 0:
            raise ValueError("Covariate self channel attention requires covariates")
        channel_layers = build_covariate_channel_layers(
            d_model, n_heads, layers, total_channels, dropout, factor,
            channel_attn_mode,
        )
    return CovariateEncoder(
        temporal_layers, norm_layer=nn.LayerNorm(d_model),
        channel_layers=channel_layers,
    )


class TargetDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_channels, target_channels,
                 dropout, factor, activation, use_rope, channel_attn_mode,
                 channel_window, channel_summaries, local_time_rope, channel_fusion_mode="dot",
                 calendar_channels=0):
        super().__init__()
        self.time_attention = time_attention(d_model, n_heads, dropout, factor, use_rope)
        head_dim = d_model // n_heads
        if use_rope or channel_attn_mode == "rope":
            head_dim += head_dim % 2
        self.cross_attention = LocalSummaryAttention(
            d_model, n_heads, n_channels, target_channels,
            window=channel_window, summaries=channel_summaries,
            mode=channel_attn_mode, dropout=dropout, head_dim=head_dim,
            local_time_rope=local_time_rope,
            channel_fusion_mode=channel_fusion_mode,
            calendar_channels=calendar_channels,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward_time(self, targets):
        b, s, p, d = targets.shape
        flat = targets.reshape(b * s, p, d)
        update, _ = self.time_attention(flat, flat, flat, attn_mask=None)
        return self.norm1(flat + self.dropout(update)).reshape(b, s, p, d)

    def forward_cross(self, targets, memory):
        return self.norm2(
            targets + self.dropout(self.cross_attention(targets, memory))
        )

    def feed_forward(self, targets):
        update = self.linear2(self.dropout(self.activation(self.linear1(targets))))
        return self.norm3(targets + self.dropout(update))

    def forward(self, targets, memory, stop_after_time=False):
        # Only target states pass through the decoder's attention, FFN and norms.
        targets = self.forward_time(targets)
        if stop_after_time:
            return targets
        targets = self.forward_cross(targets, memory)
        return self.feed_forward(targets)


class TargetDecoder(nn.Module):
    def __init__(self, layers, d_model, channel_layers=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.channel_layers = nn.ModuleList(channel_layers or [])

    def _memories(self, memory):
        if isinstance(memory, (list, tuple)):
            if len(memory) != len(self.layers):
                raise ValueError("One memory tensor is required for each decoder layer")
            return memory
        return [memory] * len(self.layers)

    def forward(self, targets, memory):
        # H_i consumes H_(i-1) and the covariate representation M_i.
        for index, (layer, layer_memory) in enumerate(zip(self.layers, self._memories(memory))):
            if self.channel_layers:
                time_state = layer(targets, layer_memory, stop_after_time=True)
            else:
                targets = layer(targets, layer_memory)
                continue
            targets = time_state
            if index < len(self.channel_layers):
                b, c, p, d = targets.shape
                mixed = self.channel_layers[index](targets.reshape(b * c, p, d), c)
                targets = mixed.reshape(b, c, p, d)
                targets = layer.forward_cross(targets, layer_memory)
                targets = layer.feed_forward(targets)
        return self.norm(targets)

    def forward_intermediates(self, targets, memory):
        """Save each time-attention result before channel mixing and FFN."""
        outputs = []
        for index, (layer, layer_memory) in enumerate(zip(self.layers, self._memories(memory))):
            time_state = layer(targets, layer_memory, stop_after_time=True)
            outputs.append(self.norm(time_state))
            targets = time_state
            if index < len(self.channel_layers):
                b, c, p, d = targets.shape
                mixed = self.channel_layers[index](
                    targets.reshape(b * c, p, d), c,
                )
                targets = mixed.reshape(b, c, p, d)
            targets = layer.forward_cross(targets, layer_memory)
            targets = layer.feed_forward(targets)
        return outputs, self.norm(targets)
