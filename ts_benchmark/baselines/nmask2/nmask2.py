import torch.nn as nn
import numpy as np
import pandas as pd

from ts_benchmark.baselines.nmask2.models.nmask2_model import Nmask2Model
from ts_benchmark.baselines.nmask2.layers.LocalSummaryAttention import validate_channel_attention, validate_channel_fusion
from ts_benchmark.baselines.utils import (
    DBLoss,
    get_time_mark,
)
from ..deep_forecasting_model_base import DeepForecastingModelBase

MODEL_HYPER_PARAMS = {

    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 2,
    "factor": 1,
    "patch_len": 16,
    "stride": 8,
    "activation": "gelu",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MAE",
    "dbloss_alpha": 0.2,
    "dbloss_beta": 0.5,
    "patience": 10,
    "alpha": 0.2,        # the weight of auxiliary loss
    # "beta": 0.1,
    'pad_method': 'learn',
    'predict_method': 'future_patch',
    "use_future_exog": True,   # match the current server nmask configuration
    "use_c_exog": True,
    "use_t_exog": True,
    "use_c": True,
    "use_t": True,
    "warm_up_epoch": 4,

    "infer_use_future": False,
    "channel_attn_mode": "embedding",  # rope | none | embedding
    "architecture": "encoder_decoder",  # encoder_decoder | joint (legacy)
    "channel_attn_type": "local_summary",  # full is supported by joint only
    "channel_window": 1,
    "channel_summaries": 2,
    "use_calendar_exog": False,  # append time marks as known future covariates
    "calendar_temporal_attn": False,  # False keeps calendar patches position-local
    "covariate_calendar_attn": False,  # let ordinary covariates read calendar patches
    "use_patch_mask_embedding": False,  # add a projected observed/missing mask to each patch
    "covariate_self_channel_attn": False,  # mix exogenous channels after temporal attention
    "share_temporal_attn": False,  # tie target/covariate time attention at matching depths
    "channel_group_gating": True,  # gated current + history + Exo-Summary aggregation
    "channel_group_logit_bias": False,  # learned group priors in the joint local/summary softmax
    "channel_fusion_mode": "cross_attn",  # dot | qk | mlp | cross_attn; local_summary fusion
    "local_time_rope": True,  # independent of channel_attn_mode; local Q/K only
    "temporal_attn_scope": "target_only",  # all is supported by joint only

}


class Nmask2(DeepForecastingModelBase):
    """
    Nmask2 adapter with independent covariate encoding and target decoding.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DAGModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
        _init_criterion_and_optimizer: Defines the loss function and optimizer.
    """

    def __init__(self, **kwargs):
        if "covariate_layers" in kwargs:
            raise ValueError("covariate_layers was removed; use e_layers for both streams")
        if kwargs.get("architecture") == "joint":
            # Reproduce the old defaults when explicitly selecting the baseline.
            kwargs.setdefault("channel_attn_type", "full")
            kwargs.setdefault("channel_window", 5)
            kwargs.setdefault("temporal_attn_scope", "all")
        super(Nmask2, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        if not isinstance(self.config.share_temporal_attn, bool):
            raise ValueError("share_temporal_attn must be a boolean")
        if not isinstance(self.config.channel_group_gating, bool):
            raise ValueError("channel_group_gating must be a boolean")
        if self.config.channel_group_gating and self.config.channel_attn_type != "local_summary":
            raise ValueError("channel_group_gating requires channel_attn_type=local_summary")
        if not isinstance(self.config.channel_group_logit_bias, bool):
            raise ValueError("channel_group_logit_bias must be a boolean")
        if self.config.channel_group_logit_bias and self.config.channel_attn_type != "local_summary":
            raise ValueError("channel_group_logit_bias requires channel_attn_type=local_summary")
        if self.config.channel_group_logit_bias and self.config.channel_group_gating:
            raise ValueError("channel_group_logit_bias and channel_group_gating are mutually exclusive")
        if self.config.share_temporal_attn and self.config.architecture != "encoder_decoder":
            raise ValueError("share_temporal_attn requires architecture=encoder_decoder")
        if not isinstance(self.config.use_calendar_exog, bool):
            raise ValueError("use_calendar_exog must be a boolean")
        if not isinstance(self.config.calendar_temporal_attn, bool):
            raise ValueError("calendar_temporal_attn must be a boolean")
        if not isinstance(self.config.covariate_calendar_attn, bool):
            raise ValueError("covariate_calendar_attn must be a boolean")
        if not isinstance(self.config.use_patch_mask_embedding, bool):
            raise ValueError("use_patch_mask_embedding must be a boolean")
        if not isinstance(self.config.covariate_self_channel_attn, bool):
            raise ValueError("covariate_self_channel_attn must be a boolean")
        if self.config.covariate_calendar_attn and not self.config.use_calendar_exog:
            raise ValueError("covariate_calendar_attn requires use_calendar_exog=true")
        if self.config.use_calendar_exog and self.config.channel_attn_type != "local_summary":
            raise ValueError("use_calendar_exog requires channel_attn_type=local_summary")
        if self.config.covariate_calendar_attn and self.config.architecture != "encoder_decoder":
            raise ValueError("covariate_calendar_attn requires architecture=encoder_decoder")
        if self.config.covariate_self_channel_attn and self.config.architecture != "encoder_decoder":
            raise ValueError("covariate_self_channel_attn requires architecture=encoder_decoder")
        if (self.config.use_calendar_exog and not self.config.calendar_temporal_attn
                and self.config.architecture != "encoder_decoder"):
            raise ValueError("calendar_temporal_attn=false requires architecture=encoder_decoder")
        validate_channel_fusion(self.config.channel_fusion_mode)
        if self.config.channel_attn_type != "local_summary" and self.config.channel_fusion_mode != "dot":
            raise ValueError("channel_fusion_mode qk/mlp/cross_attn requires channel_attn_type=local_summary")
        if self.config.architecture not in ("encoder_decoder", "joint"):
            raise ValueError("architecture must be encoder_decoder or joint")
        if self.config.architecture == "encoder_decoder":
            if self.config.channel_attn_type != "local_summary" or self.config.temporal_attn_scope != "target_only":
                raise ValueError("encoder_decoder requires channel_attn_type=local_summary and temporal_attn_scope=target_only; use architecture=joint for the old model")
        depth = self.config.e_layers
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("e_layers must be a positive integer")
        if self.config.temporal_attn_scope not in ("all", "target_only"):
            raise ValueError("temporal_attn_scope must be all or target_only")
        validate_channel_attention(self.config.channel_attn_type,
                                   self.config.channel_window, self.config.channel_summaries)
        if self.config.channel_attn_mode not in ("rope", "none", "embedding"):
            raise ValueError(
                "channel_attn_mode must be one of: rope, none, embedding; "
                f"got {self.config.channel_attn_mode!r}"
            )

    @property
    def model_name(self):
        return "Nmask2"

    def _init_criterion(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.config.loss == "DBLoss":
            criterion = DBLoss(self.config.dbloss_alpha, self.config.dbloss_beta)
        else:
            criterion = nn.HuberLoss(delta=0.5)
        self.config.criterion = criterion
        return criterion

    def _init_model(self):
        return Nmask2Model(self.config)

    def multi_forecasting_hyper_param_tune(self, train_data):
        super().multi_forecasting_hyper_param_tune(train_data)
        if self.config.use_calendar_exog:
            self.config.freq = pd.infer_freq(train_data.index)

    def single_forecasting_hyper_param_tune(self, train_data):
        super().single_forecasting_hyper_param_tune(train_data)
        if self.config.use_calendar_exog:
            self.config.freq = pd.infer_freq(train_data.index)

    def _padding_time_stamp_mark(self, time_stamps_list, padding_len):
        if not self.config.use_calendar_exog:
            return super()._padding_time_stamp_mark(time_stamps_list, padding_len)
        # Preserve multipliers (10min, 2h, ...) and pandas offset spelling.
        # The base path uppercases/truncates frequency, losing real timestamps.
        future = np.stack([
            pd.date_range(start=stamps[-1], periods=padding_len + 1,
                          freq=self.config.freq)[1:].to_numpy()
            for stamps in time_stamps_list
        ])
        return get_time_mark(np.concatenate((time_stamps_list, future), axis=1), 1, self.config.freq)

    def _process(self, input, target, input_mark, target_mark, exog_future=None, epoch=0):
        output, causality_loss = self.model(input, exog_future, target, input_mark, target_mark)
        # if self.model.training and epoch < self.config.warm_up_epoch:
        #     # output = target
        #     causality_loss *= 2
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = causality_loss
        return out_loss
