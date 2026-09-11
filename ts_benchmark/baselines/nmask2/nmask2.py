import torch.nn as nn

from ts_benchmark.baselines.nmask2.models.nmask2_model import Nmask2Model
from ts_benchmark.baselines.nmask2.layers.LocalSummaryAttention import validate_channel_attention
from ts_benchmark.baselines.utils import (
    DBLoss,
)
from ..deep_forecasting_model_base import DeepForecastingModelBase

MODEL_HYPER_PARAMS = {

    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
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
    "channel_attn_mode": "rope",  # rope | none | embedding
    "channel_attn_type": "full",  # full | local_summary
    "channel_window": 5,
    "channel_summaries": 4,
    "local_time_rope": True,  # independent of channel_attn_mode; local Q/K only

}


class Nmask2(DeepForecastingModelBase):
    """
    Nmask2 adapter with configurable channel positional information.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DAGModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
        _init_criterion_and_optimizer: Defines the loss function and optimizer.
    """

    def __init__(self, **kwargs):
        super(Nmask2, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
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

    def _process(self, input, target, input_mark, target_mark, exog_future=None, epoch=0):
        output, causality_loss = self.model(input, exog_future, target)
        # if self.model.training and epoch < self.config.warm_up_epoch:
        #     # output = target
        #     causality_loss *= 2
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = causality_loss
        return out_loss
