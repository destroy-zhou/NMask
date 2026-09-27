# NMask: A Lightweight Network via Masked Alignment for Time Series Forecasting with Exogenous Variables

## Quickstart

### Installation

Given a python environment (**note**: this project is tested under python 3.11), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Train and evaluate model

- Datasets can be downloaded from public sources. Please place the downloaded data under `./dataset/forecasting`.

- We provide the experiment scripts under `./scripts/covariate_forecasting`. You can reproduce the results with:

```shell
bash ./scripts/covariate_forecasting/nmask.sh
```
