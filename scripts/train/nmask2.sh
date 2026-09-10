#!/usr/bin/env bash

# Run from the repository root. 24 cases x 25 (d_model, d_ff) combinations = 600 runs.

# Fixed parameters follow scripts/covariate_forecasting/nmask2.sh (server configuration).

# channel_attn_mode=embedding; original channel attention (channel_attn_type defaults to full).

set -e

# NP.csv: horizon=24, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running NP.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask2/embedding/search"
done
done

# NP.csv: horizon=360, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running NP.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask2/embedding/search"
done
done

# PJM.csv: horizon=24, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running PJM.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask2/embedding/search"
done
done

# PJM.csv: horizon=360, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running PJM.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask2/embedding/search"
done
done

# BE.csv: horizon=24, e_layers=1, lr=0.0001, patch_len=24
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running BE.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask2/embedding/search"
done
done

# BE.csv: horizon=360, e_layers=1, lr=0.0001, patch_len=24
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running BE.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 720, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask2/embedding/search"
done
done

# FR.csv: horizon=24, e_layers=1, lr=0.0001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running FR.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/nmask2/embedding/search"
done
done

# FR.csv: horizon=360, e_layers=1, lr=0.0001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running FR.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/nmask2/embedding/search"
done
done

# DE.csv: horizon=24, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running DE.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/nmask2/embedding/search"
done
done

# DE.csv: horizon=360, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running DE.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/nmask2/embedding/search"
done
done

# Energy.csv: horizon=24, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Energy.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask2/embedding/search"
done
done

# Energy.csv: horizon=360, e_layers=1, lr=0.001, patch_len=48
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Energy.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask2/embedding/search"
done
done

# Sdwpfm1.csv: horizon=24, e_layers=1, lr=0.001, patch_len=8
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfm1.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask2/embedding/search"
done
done

# Sdwpfm1.csv: horizon=360, e_layers=1, lr=0.001, patch_len=8
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfm1.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask2/embedding/search"
done
done

# Sdwpfm2.csv: horizon=24, e_layers=1, lr=0.001, patch_len=4
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfm2.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 4, "patience": 5, "seq_len": 168, "stride": 4, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask2/embedding/search"
done
done

# Sdwpfm2.csv: horizon=360, e_layers=1, lr=0.001, patch_len=4
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfm2.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 4, "patience": 5, "seq_len": 720, "stride": 4, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask2/embedding/search"
done
done

# Sdwpfh1.csv: horizon=24, e_layers=1, lr=0.001, patch_len=8
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfh1.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/nmask2/embedding/search"
done
done

# Sdwpfh1.csv: horizon=360, e_layers=1, lr=0.001, patch_len=8
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfh1.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/nmask2/embedding/search"
done
done

# Sdwpfh2.csv: horizon=24, e_layers=1, lr=0.001, patch_len=8
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfh2.csv, horizon=24, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/nmask2/embedding/search"
done
done

# Sdwpfh2.csv: horizon=360, e_layers=1, lr=0.001, patch_len=8
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Sdwpfh2.csv, horizon=360, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/nmask2/embedding/search"
done
done

# Colbun.csv: horizon=10, e_layers=1, lr=0.001, patch_len=60
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Colbun.csv, horizon=10, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 60, "patience": 5, "seq_len": 60, "stride": 60, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask2/embedding/search"
done
done

# Colbun.csv: horizon=30, e_layers=1, lr=0.001, patch_len=60
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Colbun.csv, horizon=30, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 60, "patience": 5, "seq_len": 180, "stride": 60, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask2/embedding/search"
done
done

# Rapel.csv: horizon=10, e_layers=1, lr=0.001, patch_len=10
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Rapel.csv, horizon=10, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 10, "patience": 5, "seq_len": 60, "stride": 10, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask2/embedding/search"
done
done

# Rapel.csv: horizon=30, e_layers=1, lr=0.001, patch_len=10
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running Rapel.csv, horizon=30, d_model=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask2.Nmask2" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 10, "patience": 5, "seq_len": 180, "stride": 10, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1, "channel_attn_mode": "embedding", "use_future_exog": true}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask2/embedding/search"
done
done
