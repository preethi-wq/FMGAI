#!/bin/bash
# ADDED FILE: CPU job. Builds the LR/HR pairs and the 1470/315/315 split, plus the Schedule Analysis
# figures and the original-vs-modified code listing. Submitted by hpc/submit.sh (which sets -D, -t, --mem).
#SBATCH --job-name=fmgai-prepare
source hpc/common.sh
activate_env; offline; job_header

"$PYTHON" prepare_data.py --data_dir "$DATA_DIR"
cp "$DATA_DIR/splits.csv" "$DATA_DIR/dataset_summary.json" "$RESULTS_DIR/"
"$PYTHON" plot_schedule.py --out_dir "$RESULTS_DIR/figures"
"$PYTHON" make_diff.py --out_dir "$RESULTS_DIR/figures"
