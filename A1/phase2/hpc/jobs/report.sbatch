#!/bin/bash
# ADDED FILE: CPU job. Report figures, the results table and the submission zip.
#SBATCH --job-name=fmgai-report
source hpc/common.sh
activate_env; offline; job_header

"$PYTHON" make_figures.py --runs_dir "$RESULTS_DIR/outputs" --eval_dir "$RESULTS_DIR/eval" --data_dir "$DATA_DIR" --out_dir "$RESULTS_DIR/figures"
cat "$RESULTS_DIR/figures/results_table.md"
"$PYTHON" make_submission.py --results_dir "$RESULTS_DIR"
