"""Zero-shot Chronos-2 adapter with known-future covariates."""

from pathlib import Path

import numpy as np

from ts_benchmark.models.model_base import ModelBase


class Chronos2(ModelBase):
    def __init__(self, seq_len=720, horizon=24, batch_size=16,
                 model_batch_size=256, model_path=None, device_map="auto"):
        for name, value in (("seq_len", seq_len), ("horizon", horizon),
                            ("batch_size", batch_size), ("model_batch_size", model_batch_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        local = Path(__file__).resolve().parents[3] / "pretrained_models" / "chronos-2"
        self.model_path = str(model_path or (local if (local / "model.safetensors").is_file()
                                            else "amazon/chronos-2"))
        self.device_map = device_map
        self.pipeline = None

    @property
    def model_name(self):
        return "Chronos2"

    @staticmethod
    def required_hyper_params():
        return {}

    def _load_pipeline(self):
        if self.pipeline is None:
            try:
                import torch
                from chronos import Chronos2Pipeline
            except ImportError as exc:
                raise ImportError(
                    "Chronos2 requires Python >=3.10 and the dependencies in "
                    "ts_benchmark/baselines/chronos2/requirements.txt"
                ) from exc
            device = self.device_map
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = Chronos2Pipeline.from_pretrained(self.model_path, device_map=device)

    def forecast_fit(self, train_valid_data, *, covariates=None,
                     train_ratio_in_tv=1.0, **kwargs):
        # Pretrained zero-shot inference: no test data, fitting or external scaler.
        self._load_pipeline()
        return self

    def _predict(self, targets, past, future, horizon):
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim != 3 or not targets.shape[1] or not targets.shape[2]:
            raise ValueError("targets must have shape (batch, history, targets)")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if past is not None:
            past = np.asarray(past, dtype=np.float32)
            if past.ndim != 3 or past.shape[:2] != targets.shape[:2]:
                raise ValueError("Historical covariates must align with target history")
        if future is not None:
            future = np.asarray(future, dtype=np.float32)
            if past is None or future.shape != (len(targets), horizon, past.shape[2]):
                raise ValueError("Future covariates must have shape (batch, horizon, covariates)")
        tasks = []
        for row in range(len(targets)):
            task = {"target": targets[row].T.copy()}
            if past is not None and past.shape[2]:
                task["past_covariates"] = {
                    f"exog_{col}": past[row, :, col].copy() for col in range(past.shape[2])
                }
                if future is not None:
                    task["future_covariates"] = {
                        f"exog_{col}": future[row, :, col].copy() for col in range(past.shape[2])
                    }
            tasks.append(task)
        self._load_pipeline()
        _, points = self.pipeline.predict_quantiles(
            tasks, prediction_length=horizon, quantile_levels=[0.5],
            batch_size=self.model_batch_size, context_length=self.seq_len,
            cross_learning=False,
        )
        # Official point forecasts are the median, returned as (targets, horizon).
        predictions = np.stack([point.detach().cpu().numpy().T for point in points])
        if predictions.shape != (len(targets), horizon, targets.shape[2]):
            raise RuntimeError(f"Unexpected Chronos-2 output shape: {predictions.shape}")
        return predictions

    def batch_forecast(self, horizon, batch_maker, exog_future, i, **kwargs):
        batch = batch_maker.make_batch(self.batch_size, self.seq_len)
        targets = batch["input"]
        past = (batch.get("covariates") or {}).get("exog")
        # i is the batch number, including the final partial batch.
        start = i * self.batch_size
        future = None if exog_future is None else exog_future[start:start + len(targets)]
        return self._predict(targets, past, future, horizon)

    def forecast(self, horizon, series, *, covariates=None):
        covariates = covariates or {}
        targets = np.asarray(series, dtype=np.float32)[-self.seq_len:]
        past = covariates.get("exog")
        future = covariates.get("exog_future")
        if past is not None:
            past = np.asarray(past, dtype=np.float32)[-len(targets):][None]
        if future is not None:
            future = np.asarray(future, dtype=np.float32)[None]
        return self._predict(targets[None], past, future, horizon)[0]
