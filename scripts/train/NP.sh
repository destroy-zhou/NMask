
for df in 64 128 256
do
    for lr in 0.0006 0.0008 0.001 0.002 0.003 0.004
    do
        echo "Running with d_ff=$df, lr=$lr"
        python ./scripts/run_benchmark.py \
            --config-path "rolling_forecast_config.json" \
            --data-name-list "NP.csv" \
            --strategy-args '{"horizon": 24, "target_channel": [-1]}' \
            --model-name "nmask.Nmask" \
            --model-hyper-params "{\"alpha\": 0.9, \"batch_size\": 64, \"d_ff\": $df, \"d_model\": 64, \"dropout\": 0.0, \"e_layers\": 1, \"horizon\": 24, \"loss\": \"MAE\", \"lr\": $lr, \"lradj\": \"type3\", \"n_heads\": 4, \"norm\": true, \"num_epochs\": 50, \"patch_len\": 48, \"patience\": 5, \"seq_len\": 168, \"stride\": 48, \"use_c\": 1, \"use_c_exog\": 1, \"use_t\": 1, \"use_t_exog\": 1}" \
            --gpus 0 \
            --num-workers 1 \
            --timeout 60000 \
            --save-path "NP/nmask"
    done
done
