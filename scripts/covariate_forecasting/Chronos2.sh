#!/usr/bin/env bash
set -euo pipefail
cd "${BASH_SOURCE[0]%/*}/../.."

# Activate the Chronos-2 environment first; optional overrides below.
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-256}"

run_case() {
    local dataset="$1" horizon="$2" seq_len="$3"
    "$PYTHON" ./scripts/run_benchmark.py \
        --config-path rolling_forecast_config.json \
        --data-name-list "${dataset}.csv" \
        --strategy-args "{\"horizon\": ${horizon}, \"target_channel\": [-1]}" \
        --model-name chronos2.Chronos2 \
        --model-hyper-params "{\"horizon\": ${horizon}, \"seq_len\": ${seq_len}, \"batch_size\": ${BATCH_SIZE}, \"model_batch_size\": ${MODEL_BATCH_SIZE}}" \
        --gpus "$GPU" --num-workers 1 --timeout 60000 \
        --save-path "${dataset}/chronos2/zero_shot"
}

# Same datasets, horizons, context lengths and target channel as nmask.sh.
for dataset in NP PJM BE FR DE Energy Sdwpfm1 Sdwpfm2 Sdwpfh1 Sdwpfh2; do
    run_case "$dataset" 24 168
    run_case "$dataset" 360 720
done
for dataset in Colbun Rapel; do
    run_case "$dataset" 10 60
    run_case "$dataset" 30 180
done
