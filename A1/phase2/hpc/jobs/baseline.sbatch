#!/bin/bash
# ADDED FILE: GPU job. Records the environment, then scores the bicubic reference and Run 0 (pretrained,
# no fine-tuning) on the test split and saves Run 0's trajectories.
#SBATCH --job-name=fmgai-baseline
source hpc/common.sh
activate_env; offline; job_header

"$PYTHON" env_report.py | tee "$RESULTS_DIR/environment.txt"
"$PYTHON" infer.py --run bicubic --data_dir "$DATA_DIR" --out_dir "$RESULTS_DIR/eval"
"$PYTHON" infer.py --run run0 --data_dir "$DATA_DIR" --out_dir "$RESULTS_DIR/eval"
