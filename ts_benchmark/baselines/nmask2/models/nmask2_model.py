from torch import nn

from ts_benchmark.baselines.nmask2.layers.CC_EncDec import CovCausalityEncoder
from ts_benchmark.baselines.nmask2.layers.TC_EncDec import TemporalCausalityEncoder
from ts_benchmark.baselines.time_series_library.utils.timefeatures import time_features_from_frequency_str


class Nmask2Model(nn.Module):
    def __init__(self, config):
        super(Nmask2Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.use_c = config.use_c
        self.use_t = config.use_t
        self.use_c_exog = config.use_c_exog
        self.use_t_exog = config.use_t_exog
        self.alpha = config.alpha
        # self.beta = config.beta
        self.series_dim = config.series_dim
        self.use_future_exog = config.use_future_exog
        self.infer_use_future = config.infer_use_future
        self.criterion = config.criterion
        self.calendar_channels = (len(time_features_from_frequency_str(config.freq))
                                  if config.use_calendar_exog else 0)
        if config.use_calendar_exog and not self.calendar_channels:
            raise ValueError("No calendar features are available for this frequency")
        assert self.use_c or self.use_t, "At least one of use_c or use_t must be True"

        self.temporal_encoder = TemporalCausalityEncoder(
            enc_in=config.enc_in + self.calendar_channels,
            calendar_channels=self.calendar_channels,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            pad_method=config.pad_method,
            predict_method=config.predict_method,
            use_future_exog=config.use_future_exog,
            channel_attn_mode=config.channel_attn_mode,
            channel_attn_type=config.channel_attn_type,
            channel_window=config.channel_window,
            channel_summaries=config.channel_summaries,
            channel_fusion_mode=config.channel_fusion_mode,
            local_time_rope=config.local_time_rope,
            temporal_attn_scope=config.temporal_attn_scope,
            architecture=config.architecture,
            covariate_layers=config.covariate_layers,
        )

        # self.cov_encoder = CovCausalityEncoder(
        #     enc_in=config.enc_in,
        #     seq_len=self.seq_len,
        #     pred_len=self.pred_len,
        #     series_dim=config.series_dim,
        #     d_model=config.d_model,
        #     d_ff=config.d_ff,
        #     n_heads=config.n_heads,
        #     e_layers=config.e_layers,
        #     dropout=config.dropout,
        #     factor=config.factor,
        #     activation=config.activation,
        #     criterion=config.criterion

        # )

        # self.history_x_projector = CompressAndProject(self.series_dim, self.seq_len, config.d_model)
        # self.future_exog_projector = CompressAndProject(config.enc_in - self.series_dim, self.pred_len, config.d_model)

    def compute_loss(self, pred_w, pred_wo, patch_w, patch_wo, target,
                 lambda_task=1.0, lambda_distill=0.5, lambda_feat=0.3):
        # 任务损失：两条路径都对 target 负责
        target = target[:, -self.pred_len:, :self.series_dim]
        # print(f"{pred_w.shape = }, {target.shape = }")
        loss_task = self.criterion(pred_w, target) #+ self.criterion(pred_wo, target)
        
        # 预测蒸馏：让无外生路径的预测向有外生路径看齐
        # detach teacher，避免梯度污染有外生路径
        loss_distill = self.criterion(pred_wo, pred_w.detach())
        # 特征蒸馏（可选）：patch 表示层对齐
        # loss_feat = self.criterion(patch_wo, patch_w.detach())
        return lambda_task * loss_task + lambda_distill * loss_distill #+ lambda_feat * loss_feat


    def forward(self, input, exog_future, target, input_mark=None, target_mark=None):
        # input: [batch_size, seq_len, n_vars]
        temporal_causality_loss = 0
        cov_causality_loss = 0
        causality_loss = 0

        # if not self.training and not self.infer_use_future:
        #     if self.use_t:
        #         t_output, t_exog_output, temporal_causality_loss = self.temporal_encoder(input, None, self.use_t_exog)
        #     if self.use_c:
        #         c_output, cov_causality_loss = self.cov_encoder(input, t_exog_output, self.use_c_exog)

        #     output = self.alpha * t_output + (1 - self.alpha) * c_output
        #     return output, 0
        # else:
        #     if self.use_t:
        #         t_output, _, temporal_causality_loss = self.temporal_encoder(input, exog_future, self.use_t_exog)
        #     if self.use_c:
        #         c_output, cov_causality_loss = self.cov_encoder(input, exog_future, self.use_c_exog)

        #     output = self.alpha * t_output + (1 - self.alpha) * c_output
        #     causality_loss = self.beta * (temporal_causality_loss + cov_causality_loss)

        # if not self.training and not self.infer_use_future:
        #     output = self.temporal_encoder(input, None, self.use_t_exog)
        #     return output
        # else:
        # if not self.use_future_exog:
        #     if self.training:
        #         output, no_future_out, exog_out = self.temporal_encoder(input, exog_future, self.use_t_exog)
        #         lambda_distill = 0.5  #self.alpha
        #         causality_loss = self.compute_loss(no_future_out, output, None, exog_out, target, lambda_distill=lambda_distill)
        #         causality_loss += self.alpha * self.criterion(exog_out, exog_future)
        #     else:
        #         output, no_future_out, exog_out = self.temporal_encoder(input, None, self.use_t_exog)
        # else:
        #     output, exog_out = self.temporal_encoder(input, exog_future, self.use_t_exog)
        output, exog_out = self.temporal_encoder(
            input, exog_future, self.use_t_exog, input_mark, target_mark,
        )
        if (self.training and not self.use_future_exog and exog_future is not None
                and self.temporal_encoder.regular_covariates):
            # print(f"{exog_out.shape = }, {exog_future.shape = }")
            causality_loss = self.alpha * self.criterion(exog_out, exog_future)
            # causality_loss = self.compute_loss(exog_out, output, None, None, exog_future)
        return output, causality_loss
