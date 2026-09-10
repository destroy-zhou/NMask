
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




for pl in 4 12 32
for pl in 20 30 40
for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future_fu_p_pred"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": 512, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future_fu_p_pred"

for pl in 4 12 32
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 168, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future_fu_p_pred"
done
done
done


for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future_distill"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future_distill"


for pl in 8 24 48
do
for dm in 16 32 64 128 256
do
for df in 16 32 64 128 256
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future/twin/fu_p"
done
done
done

for pl in 8 16 24 48
do
for dm in 16 32 64 128 256
do
for df in 16 32 64 128 256
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/nmask_no_future/twin/all_his"
done
done
done


for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask_no_future_distill"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.2, "batch_size": 64, "d_ff": 64, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask_no_future_distill"



for pl in 12

for pl in 24
for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.8, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask_no_future_fu_p_pred"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 64, "d_model": 128, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask_no_future_fu_p_pred"

for lr in 0.002 0.003 0.004 0.005
for lr in 0.0005 0.0006 0.0007 0.0008 0.0009
do
echo "Running with lr=$lr"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": '$lr', "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask_no_future_fu_p_pred"
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.0008, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/nmask_no_future_fu_p_pred"


for pl in 8 24 48
do
for dm in 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask_no_future_fu_p_pred"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 256, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/nmask_no_future_fu_p_pred"


for pl in 48
do
for dm in 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/nmask_no_future_fu_p_pred"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 256, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/nmask_no_future_fu_p_pred"


for pl in 8 24 48
do
for dm in 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/nmask_no_future_fu_p_pred"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 24, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/nmask_no_future_fu_p_pred"



for pl in 4 10 16
for pl in 4 8
for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.1, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 32, "dropout": 0.1, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p"

for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p_pred/wi_ch/warm_up"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 32, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p_pred/wi_ch/warm_up"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 32, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p_pred/wi_ch/warm_up"


for pl in 4 12 16
for pl in 8 24 48
for pl in 36 60 72
for pl in 8 24 36
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 168, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p_pred/wo_ch"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 12, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/nmask_no_future_fu_p_pred/wo_ch"


for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask_no_future_fu_p_pred"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 64, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask_no_future_fu_p_pred"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 32, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask_no_future_fu_p_pred"

# predict exog
for dm in 32 64 128 256 512
do
for df in 32 64 128 256 512
do
echo "Running with d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.9, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/nmask_no_future_sin/predict_fup"
done
done



for pl in 4
for pl in 8 24 48
for pl in 12 36 60 72
for pl in 4 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask_no_future_fu_p_pred/warm_up"
done
done
done

for pl in 12 36 60 72
for pl in 30 40 50 70 80
for pl in 4 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask_no_future_fu_p_pred/warm_up"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 512, "d_model": 16, "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 8, "patience": 5, "seq_len": 720, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/nmask_no_future_fu_p_pred/warm_up"


for pl in 8 24 48
for pl in 12 36 60 72
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/nmask_no_future_fu_p_pred"
done
done
done

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 512, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 36, "patience": 5, "seq_len": 168, "stride": 48, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/nmask_no_future_fu_p_pred"


for pl in 8 24 48
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 360, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 720, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/nmask_no_future_fu_p_pred"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 32, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 24, "patience": 5, "seq_len": 168, "stride": 8, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/nmask_no_future_fu_p_pred"


for pl in 10 20 30
for pl in 10 20 30 45 60
for pl in 24 48 72 90
for pl in 50 55 65 70 75
for pl in 32 64 76 96
for pl in 10 30 60
for pl in 20 45
for pl in 10 30 45 60
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 180, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask_no_future_fu_p_pred/warm_up"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 256, "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 60, "patience": 5, "seq_len": 60, "stride": 60, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask_no_future_fu_p_pred/warm_up"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.5, "batch_size": 64, "d_ff": 128, "d_model": 16, "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 60, "patience": 5, "seq_len": 180, "stride": 60, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/nmask_no_future_fu_p_pred/warm_up"


for pl in 45 60
for pl in 5 15

for pl in 8 24 48
for pl in 25 35 40
for pl in 10 20 30
for pl in 5 10 15 20 25 30
for pl in 5 15 25
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.9, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 180, "stride": '$pl', "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask_no_future_fu_p_pred"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.9, "batch_size": 64, "d_ff": 512, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.01, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 15, "patience": 5, "seq_len": 60, "stride": 10, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask_no_future_fu_p_pred"

for pl in 10 20 30
do
for dm in 16 32 64 128 256 512
do
for df in 16 32 64 128 256 512
do
echo "Running with p_l=$pl, d_m=$dm, d_ff=$df"
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": '$df', "d_model": '$dm', "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": '$pl', "patience": 5, "seq_len": 60, "stride": 30, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask_no_future_fu_p_pred"
done
done
done
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": 512, "d_model": 128, "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 10, "patience": 5, "seq_len": 180, "stride": 10, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask_no_future_fu_p_pred"


for al in 0.1 0.2 0.3 0.4 0.6 0.8 0.9
do
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": '$al', "batch_size": 64, "d_ff": 256, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 30, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 30, "patience": 5, "seq_len": 180, "stride": 30, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask_no_future_fu_p_pred"
done

for al in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
do
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "nmask.Nmask" --model-hyper-params '{"alpha": '$al', "batch_size": 64, "d_ff": 256, "d_model": 32, "dropout": 0.0, "e_layers": 1, "horizon": 10, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 30, "patience": 5, "seq_len": 60, "stride": 30, "pad_method": "learn", "predict_method": "future_patch", "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/nmask_no_future_fu_p_pred"
done


