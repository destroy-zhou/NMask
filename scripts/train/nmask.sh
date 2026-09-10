
for df in 32 64 128
do
    for lr in 0.0006 0.0008 0.001 0.002 0.003 0.004 0.006 0.008 0.01
    do
        echo "Running with d_ff=$df, lr=$lr"
        python ./scripts/run_benchmark.py \
            --config-path "rolling_forecast_config.json" \
            --data-name-list "Sdwpfh1.csv" \
            --strategy-args '{"horizon": 24, "target_channel": [-1]}' \
            --model-name "nmask.Nmask" \
            --model-hyper-params "{\"batch_size\": 64, \"d_ff\": $df, \"d_model\": 64, \"dropout\": 0.0, \"e_layers\": 1, \"horizon\": 24, \"loss\": \"MAE\", \"lr\": $lr, \"lradj\": \"type3\", \"n_heads\": 4, \"norm\": true, \"num_epochs\": 50, \"patch_len\": 8, \"patience\": 5, \"seq_len\": 168, \"stride\": 8, \"pad_method\": \"nlearn\", \"predict_method\": \"future_patch\", \"use_t\": 1, \"use_t_exog\": 1}" \
            --gpus 0 \
            --num-workers 1 \
            --timeout 60000 \
            --save-path "Sdwpfh1/nmask"
    done
done


for dm in 32 64 128 256
do
for df in 32 64 128 256
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 720, "stride": 24, "pad_method": "learn", "predict_method": "all_sequence", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask/all_seq"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 256, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask/a800"


for dm in 32 64 128 256
do
for df in 32 64 128 256
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask/a800"



for dm in 32 64 128 256
do
for df in 32 64 128 256
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask/a800"



for dm in 512
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask/a800"
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 64, "d_model": 256, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask/a800"


for dm in 32 64 128 256
do
for df in 32 64 128 256
do
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.7, "batch_size": 64, "d_ff": 64, "d_model": 512, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask/a800"
done
done

for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 512, "d_model": 128, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/nmask/a800"


for dm in 32 64 128 256
do
for df in 32 64 128 256
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 128, "d_model": 128, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/nmask/a800"



for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask/a800"
done
done

for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 512, "d_model": 512, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask/a800"


for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 64, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask/a800"


for dm in 32 64 128 256 512
for dm in 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 4, "patience": 5, "seq_len": 720, "stride": 4, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 64, "d_model": 256, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 4, "patience": 5, "seq_len": 168, "stride": 4, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask/a800"


for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 64, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/nmask/a800"


for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 512, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/nmask/a800"


for pl in 40 50 60
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 180, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask/a800"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 512, "d_model": 256, "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 60, "patience": 5, "seq_len": 60, "stride": 45, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask/a800"


for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 30, "patience": 5, "seq_len": 60, "stride": 30, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask/a800"
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 256, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 30, "patience": 5, "seq_len": 180, "stride": 30, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask/a800"



for pl in 10 20 30 40 50 60
do
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 180, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask/a800"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": 64, "d_model": 256, "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 10, "patience": 5, "seq_len": 60, "stride": 15, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask/a800"


for pl in 10 20 30
do
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 10, "patience": 5, "seq_len": 60, "stride": 10, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask/a800"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": 128, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 10, "patience": 5, "seq_len": 60, "stride": 10, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 6 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask"
