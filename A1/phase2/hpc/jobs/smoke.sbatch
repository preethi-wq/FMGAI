#!/bin/bash
# ADDED FILE: GPU job, about 10 min. One short epoch on 16 images under the modified schedule, then test-set
# inference on 4 images with the saved model. Checks the whole loop (offline caches, CSV logs, validation
# metrics, samples, final save, infer.py) before the long runs. Writes to $WORK_ROOT/smoke only.
#SBATCH --job-name=fmgai-smoke
source hpc/common.sh
activate_env; offline; job_header

"$PYTHON" env_report.py
SMOKE="$WORK_ROOT/smoke"
rm -rf "$SMOKE"
bash run_train.sh power "$SMOKE/runB" "$DATA_DIR" --max_train_samples 16 --num_train_epochs 1 --num_val_metric_images 8
"$PYTHON" infer.py --run runB --data_dir "$DATA_DIR" --out_dir "$SMOKE/eval" --model_dir "$SMOKE/runB" --limit 4
echo "== smoke outputs"
cat "$SMOKE/runB/train_log.csv" "$SMOKE/runB/val_log.csv" "$SMOKE/runB/val_metrics.csv" "$SMOKE/runB/schedule_check.json"
ls "$SMOKE/runB/samples" "$SMOKE/eval/runB/trajectories"
echo "SMOKE TEST PASSED"
