"""Encode covariates once, then decode targets against read-only memory."""

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


def build_covariate_encoder(d_model, d_ff, n_heads, layers, dropout,
                            factor, activation, use_rope):
    # Each covariate is encoded along time, independently of the targets.
    return Encoder([
        EncoderLayer(time_attention(d_model, n_heads, dropout, factor, use_rope),
                     None, d_model, d_ff, dropout, activation)
        for _ in range(layers)
    ], norm_layer=nn.LayerNorm(d_model))


class TargetDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_channels, target_channels,
                 dropout, factor, activation, use_rope, channel_attn_mode,
                 channel_window, channel_summaries, local_time_rope, channel_fusion_mode="dot"):
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
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, targets, memory):
        # Only target states pass through the decoder's attention, FFN and norms.
        b, s, p, d = targets.shape
        flat = targets.reshape(b * s, p, d)
        update, _ = self.time_attention(flat, flat, flat, attn_mask=None)
        targets = self.norm1(flat + self.dropout(update)).reshape(b, s, p, d)
        targets = self.norm2(targets + self.dropout(self.cross_attention(targets, memory)))
        update = self.linear2(self.dropout(self.activation(self.linear1(targets))))
        return self.norm3(targets + self.dropout(update))


class TargetDecoder(nn.Module):
    def __init__(self, layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, targets, memory):
        # Reuse the same tensor without detach: target loss trains its encoder.
        for layer in self.layers:
            targets = layer(targets, memory)
        return self.norm(targets)
