#!/bin/bash
# ADDED FILE: GPU job. Test-set scores and inference trajectories of one fine-tuned model.
#   sbatch ... hpc/jobs/evaluate.sbatch runA|runB
#SBATCH --job-name=fmgai-eval
source hpc/common.sh
activate_env; offline; job_header

RUN=${1:?runA or runB}
"$PYTHON" infer.py --run "$RUN" --data_dir "$DATA_DIR" --out_dir "$RESULTS_DIR/eval" --model_dir "$RESULTS_DIR/outputs/$RUN"
