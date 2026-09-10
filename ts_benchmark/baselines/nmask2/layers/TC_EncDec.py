import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from ts_benchmark.baselines.nmask2.layers.Embed import PatchEmbedding, CompressAndProject, PositionalEmbedding
from ts_benchmark.baselines.nmask2.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ts_benchmark.baselines.nmask2.layers.Transformer_EncDec import Encoder, EncoderLayer
from ts_benchmark.baselines.nmask2.layers.LocalSummaryAttention import LocalSummaryAttention, validate_channel_attention


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


def FFT_for_Period(x, k=1):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TemporalCausalityEncoder(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, series_dim,
                 patch_len, stride, d_model, d_ff, n_heads, e_layers,
                 dropout, factor, activation, pad_method, predict_method, use_future_exog, use_rope=True,
                 channel_attn_mode="rope", channel_attn_type="full",
                 channel_window=5, channel_summaries=4
                 ):
        super(TemporalCausalityEncoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series_dim = series_dim
        # self.criterion = criterion
        self.c_in = enc_in
        self.pad_method = pad_method
        self.predict_method = predict_method
        self.use_future_exog = use_future_exog
        self.use_rope = use_rope
        if channel_attn_mode not in ("rope", "none", "embedding"):
            raise ValueError("channel_attn_mode must be one of: rope, none, embedding")
        self.channel_attn_mode = channel_attn_mode
        validate_channel_attention(channel_attn_type, channel_window, channel_summaries)
        self.channel_attn_type = channel_attn_type
        self.channel_window, self.channel_summaries = channel_window, channel_summaries
        stride = patch_len
        padding = stride
        future_patch_num = int((pred_len - patch_len) / stride + 2)
        history_patch_num = int((seq_len - patch_len) / stride + 2)

        # self.patch_embedding = PatchEmbedding(
        #     d_model, patch_len, stride, padding, dropout
        # )

        # self.exog_patch_embedding = PatchEmbedding(
        #     d_model, patch_len, stride, padding, dropout
        # )
        self.x_patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        self.position_embedding = PositionalEmbedding(d_model)

        # self.encoder_exg = self._build_encoder(
        #     d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation, output_attention=True,
        #     factor=factor, e_layers=e_layers, use_rope=use_rope
        # )
        self.encoder_x = self._build_encoder(
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation, output_attention=False,
            factor=factor, e_layers=e_layers, use_rope=use_rope
        )

        # self.x_projector = CompressAndProject(self.series_dim, self.seq_len, d_model)
        # self.exog_projector = CompressAndProject(enc_in - series_dim, self.seq_len, d_model)

        # Prediction Head
        # self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.head_nf = d_model
        # self.head = FlattenHead(
        #     enc_in,
        #     self.head_nf,
        #     pred_len,
        #     head_dropout=dropout,
        # )

        # self.exog_head = FlattenHead(
        #     enc_in,
        #     self.head_nf,
        #     pred_len,
        #     head_dropout=dropout,
        # )
        # self.x_head = FlattenHead(
        #     enc_in,
        #     self.head_nf,
        #     pred_len,
        #     head_dropout=dropout,
        # )
        # patch_pred_len = int(np.ceil(pred_len / future_patch_num))
        # self.c_mix = nn.Linear(self.c_in, self.series_dim)
        patch_pred_len = patch_len
        if self.predict_method == 'future_patch':
            # patch_pred_len = patch_len
            self.x_head = nn.Sequential(
                nn.Linear(d_model, patch_pred_len),
                nn.Dropout(dropout)
            )
        elif self.predict_method == 'all_history':
            self.x_head = nn.Sequential(
                nn.Linear(history_patch_num * d_model, pred_len),
                nn.Dropout(dropout)
            )
            # self.x_head = nn.Sequential(
            #     nn.Linear(history_patch_num * d_model, d_ff),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(d_ff, pred_len),
            # )
        elif self.predict_method == 'all_future':
            self.x_head = nn.Sequential(
                nn.Linear(future_patch_num * d_model, pred_len),
                nn.Dropout(dropout)
            )
        elif self.predict_method == 'all_sequence':
            self.x_head = nn.Sequential(
                nn.Linear((history_patch_num + future_patch_num) * d_model, pred_len),
                nn.Dropout(dropout)
            )
            # self.x_head = nn.Sequential(
            #     nn.Linear((history_patch_num + future_patch_num) * d_model, d_ff),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(d_ff, pred_len),
            # )
        
        
        if self.pad_method == 'learn':
            self.x_future = nn.Parameter(
                # torch.randn(args.num_classes, bottle_dim // 1)
                torch.randn(1, self.series_dim, future_patch_num, d_model)
                # torch.randn(args.num_classes, out_dim * self.c_in)
            )
        elif self.pad_method == 'nlearn':
            self.x_future = nn.Parameter(
                torch.randn(1, self.series_dim, future_patch_num, d_model),
                requires_grad=False
            )

        if not self.use_future_exog:
            self.exog_future = nn.Parameter(
                # torch.randn(args.num_classes, bottle_dim // 1)
                torch.randn(1, self.c_in - self.series_dim, future_patch_num, d_model)
                # torch.randn(args.num_classes, out_dim * self.c_in)
            )
            self.exog_head = nn.Sequential(
                nn.Linear(d_model, patch_pred_len),
                nn.Dropout(dropout)
            )
        

    def forward(self, x, exog_future, use_exog=True):
        exog_history = x[:, :, self.series_dim:]
        x_history = x[:, :, :self.series_dim]

        _, _, EXOG_D = exog_history.shape
        B, L, X_D = x_history.shape

        # print(f"{exog_history.shape = }, {exog_future.shape = }")
        if self.use_future_exog:
            exog_history = torch.cat([exog_history, exog_future], dim=-2)
        # print(f"{exog_history.shape = }")

        # period_x, _ = FFT_for_Period(x_history, k=1)
        # period_exog, _ = FFT_for_Period(exog_history, k=1)
        # max_period = max(int(period_x), int(period_exog))

        exog_history_means = exog_history.mean(1, keepdim=True).detach()
        x_history_means = x_history.mean(1, keepdim=True).detach()

        exog_history_stdev = torch.sqrt(torch.var(exog_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_history_stdev = torch.sqrt(torch.var(x_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        exog_history = self.sample_norm(exog_history, exog_history_means, exog_history_stdev)
        x_history = self.sample_norm(x_history, x_history_means, x_history_stdev)

        exog_history = exog_history.permute(0, 2, 1)
        x_history = x_history.permute(0, 2, 1)

        # patch_exog, exog_vars = self.patch_embedding(exog_history)
        # patch_x, x_vars = self.patch_embedding(x_history)

        # patch_exog, exog_vars = self.exog_patch_embedding(exog_history)
        if self.use_future_exog:
            exog_future = exog_history[:,:,L:]
            exog_history, exog_vars = self.x_patch_embedding(exog_history[:,:,:L])
            exog_future, exog_vars = self.x_patch_embedding(exog_future)
            patch_exog = torch.cat([exog_history, exog_future], dim=-2)
        else:
            exog_history, exog_vars = self.x_patch_embedding(exog_history)
            # print(f"{exog_history.shape = }")
            exog_history = exog_history.view(B, EXOG_D, exog_history.shape[-2], exog_history.shape[-1])
            # x_history_L = patch_x.shape[-2]
            exog_future = self.exog_future.expand(B, -1, -1, -1)
            patch_exog = torch.cat([exog_history, exog_future], dim=-2)
            # print(f"{patch_exog.shape = }")

        patch_x, x_vars = self.x_patch_embedding(x_history)
        patch_x = patch_x.view(B, X_D, patch_x.shape[-2], patch_x.shape[-1])
        x_history_L = patch_x.shape[-2]
        x_future = self.x_future.expand(B, -1, -1, -1)
        patch_x = torch.cat([patch_x, x_future], dim=-2)
        patch_exog = patch_exog.view(B, EXOG_D, patch_exog.shape[-2], patch_exog.shape[-1])
        # print(f"{patch_x.shape = }, {patch_exog.shape = }")

        patch_x = torch.cat([patch_x, patch_exog], dim=1)
        # print(f"{patch_x.shape = }")
        patch_x = patch_x.view(-1, patch_x.shape[-2], patch_x.shape[-1])
        if not self.use_rope:
            patch_x = patch_x + self.position_embedding(patch_x)

        # enc_exog_out, _ = self.encoder_exg(patch_exog)
        # if use_exog:
        #     _, causality_attns = self.encoder_exg(patch_x)
        # else:
        #     causality_attns = None

        # exog_history = exog_history.permute(0, 2, 1)
        # x_history = x_history.permute(0, 2, 1)

        # exog_history_projection = self.exog_projector(exog_history)  # batch size, d_model TODO
        # x_history_projection = self.x_projector(x_history)

        # attn_alpha = F.sigmoid(torch.einsum('bd,bd->b', x_history_projection, exog_history_projection)).view(-1, 1, 1, 1)
        # print("tc attn_alpha mean:", torch.mean(attn_alpha))
        # print(f"{patch_x.shape = }")
        enc_x_out, _ = self.encoder_x(patch_x, self.c_in, exog_attns=None)
        # print(f"{enc_x_out.shape = }")

        # enc_exog_out = torch.reshape(
        #     enc_exog_out, (-1, exog_vars, enc_exog_out.shape[-2], enc_exog_out.shape[-1])
        # ).permute(0, 1, 3, 2)
        enc_x_out = torch.reshape(
            enc_x_out, (B, -1, enc_x_out.shape[-2], enc_x_out.shape[-1])
        ) #.permute(0, 1, 3, 2)
        # print(f"{enc_x_out.shape = }")
        exog_out = enc_x_out[:,X_D:,x_history_L:]
        if self.predict_method == 'future_patch':
            enc_x_out = enc_x_out[:,:X_D,x_history_L:]
            # enc_x_out = enc_x_out[:,:,x_history_L:].permute(0, 3, 2, 1)
            # enc_x_out = self.c_mix(enc_x_out)
            # enc_x_out = enc_x_out.permute(0, 3, 2, 1)
        elif self.predict_method == 'all_history':
            enc_x_out = enc_x_out[:,:X_D,:x_history_L]
            enc_x_out = enc_x_out.view(B, X_D, -1)
        elif self.predict_method == 'all_future':
            enc_x_out = enc_x_out[:,:X_D,x_history_L:]
            enc_x_out = enc_x_out.view(B, X_D, -1)
        elif self.predict_method == 'all_sequence':
            enc_x_out = enc_x_out[:,:X_D,:]
            enc_x_out = enc_x_out.view(B, X_D, -1)
        # print(f"{enc_x_out.shape = }")

        # enc_out = torch.cat([enc_x_out, enc_exog_out], dim=1)
        # out = self.head(enc_out)
        # out = out.permute(0, 2, 1)
        #
        # exog_out = out[:, :, self.series_dim:]
        # x_out = out[:, :, :self.series_dim]

        # exog_out = self.exog_head(enc_exog_out)
        x_out = self.x_head(enc_x_out)
        x_out = x_out.view(B, X_D, -1)
        x_out = x_out[:,:,:self.pred_len]

        # exog_out = exog_out.permute(0, 2, 1)
        x_out = x_out.permute(0, 2, 1)

        # exog_out = self.sample_denorm(exog_out, exog_history_means, exog_history_stdev)
        x_out = self.sample_denorm(x_out, x_history_means, x_history_stdev)

        if not self.use_future_exog:
            exog_out = self.exog_head(exog_out)
            exog_out = exog_out.view(B, EXOG_D, -1)
            exog_out = exog_out[:,:,:self.pred_len]

            exog_out = exog_out.permute(0, 2, 1)

            exog_out = self.sample_denorm(exog_out, exog_history_means, exog_history_stdev)

        # if use_exog and exog_future is not None:
        #     temporal_causality_loss = self.criterion(exog_out, exog_future)
        # else:
        #     temporal_causality_loss = torch.tensor(0.0, device=exog_out.device)

        # return x_out #, exog_out, temporal_causality_loss
        return x_out, exog_out

    def _build_encoder(self, d_model, d_ff, n_heads, dropout, activation, output_attention, factor, e_layers, use_rope=False):
        # Keep projection dimensions identical across modes, including odd head sizes.
        channel_head_dim = d_model // n_heads
        if use_rope:
            channel_head_dim += channel_head_dim % 2
        return Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        use_rope=use_rope,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        d_keys=channel_head_dim,
                        d_values=channel_head_dim,
                        use_rope=use_rope and self.channel_attn_mode == "rope",
                    ) if self.channel_attn_type == "full" else None,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    n_channels=self.c_in if self.channel_attn_type == "full" and self.channel_attn_mode == "embedding" else None,
                    local_channel_attention=LocalSummaryAttention(
                        d_model, n_heads, self.c_in, self.series_dim,
                        window=self.channel_window, summaries=self.channel_summaries,
                        mode=self.channel_attn_mode, dropout=dropout, head_dim=channel_head_dim,
                    ) if self.channel_attn_type == "local_summary" else None,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def sample_norm(self, x, means, stdev):
        x = x - means
        x /= stdev
        return x

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x
