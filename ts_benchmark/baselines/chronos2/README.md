# Chronos-2

Uses the full official [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2)
120M checkpoint for zero-shot forecasting, with historical and known-future
exogenous variables. This is the official full model, not a guarantee of the
best accuracy on every dataset. No fine-tuning is performed.

## Setup (Linux/GPU server)

Use a separate Python 3.10+ environment (3.11 recommended). The original
`paper/requirements.txt` pins incompatible older pandas; do not install both.

```bash
python3.11 -m venv .venv-chronos2
source .venv-chronos2/bin/activate
pip install -r ts_benchmark/baselines/chronos2/requirements.txt
python scripts/download_chronos2.py
bash scripts/covariate_forecasting/Chronos2.sh
```

Install a CUDA-enabled PyTorch build appropriate for the server if necessary.
Put the same CSV datasets used by `nmask.sh` in `dataset/forecasting/`.
The download script pins revision `29ec3766d36d6f73f0696f85560a422f50e8498c`
and checks the weight SHA256 against Hugging Face metadata. Weights live in
`pretrained_models/chronos-2/` (ignored by Git). Run the downloader on each
machine or copy that entire directory; a Git push does not transfer weights.
The adapter automatically uses local weights if available, otherwise the
official Hugging Face model ID. `model_path` can override this.

## Evaluation

The shell script runs all 24 configurations from `nmask.sh`: NP, PJM, BE, FR,
DE, Energy, Sdwpfm1/2 and Sdwpfh1/2 use horizon/context 24/168 and 360/720;
Colbun and Rapel use 10/60 and 30/180. The last channel is the target.
Splits, stride and metrics come from the original rolling evaluation config.
Results go under `result/<dataset>/chronos2/zero_shot/`.

`forecast_fit` only loads pretrained weights. Chronos-2 handles normalization
internally; predictions remain in original units. The adapter returns the
official median point forecast. Each rolling window forms its own task:
`cross_learning=False` prevents information from later rolling windows leaking
into earlier ones, while targets and covariates within a window can interact.
The batch path receives known future covariates from the benchmark; no future
target values are supplied. The single-window `forecast` method accepts
historical `covariates['exog']` and optional `covariates['exog_future']`.

`GPU=1 BATCH_SIZE=8 MODEL_BATCH_SIZE=128 bash scripts/covariate_forecasting/Chronos2.sh`
selects a GPU and reduces memory use. `batch_size` counts rolling windows;
`model_batch_size` counts target and covariate channels in the official pipeline.
Long rolling evaluations can take substantial time; the script preserves the
original maximum 48,000 windows instead of silently reducing evaluation size.

Run adapter regression checks with `python -m unittest discover -s tests -p 'test_chronos2.py'`.
