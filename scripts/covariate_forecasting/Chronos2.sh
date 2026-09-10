python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NP.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NP/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PJM.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PJM/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "BE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "BE/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FR.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FR/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Energy.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Energy/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm1/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfm2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfm2/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh1.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh1/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 24, "seq_len": 168, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Sdwpfh2.csv" --strategy-args '{"horizon": 360, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 360, "seq_len": 720, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Sdwpfh2/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 10, "seq_len": 60, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Colbun.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 30, "seq_len": 180, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Colbun/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 10, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 10, "seq_len": 60, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/chronos2/zero_shot"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Rapel.csv" --strategy-args '{"horizon": 30, "target_channel": [-1]}' --model-name "chronos2.Chronos2" --model-hyper-params '{"horizon": 30, "seq_len": 180, "batch_size": 16, "model_batch_size": 256}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Rapel/chronos2/zero_shot"
