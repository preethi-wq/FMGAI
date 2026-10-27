#!/bin/bash
# ADDED FILE: GPU job. One fine-tuning run with the shared hyperparameters in run_train.sh.
#   sbatch ... hpc/jobs/train.sbatch runA     # original schedule {beta_t}
#   sbatch ... hpc/jobs/train.sbatch runB     # modified schedule {beta_t ** EXPONENT_X}
# Resumes from the last saved state (every 2 epochs), so after a TIMEOUT just submit it again.
#SBATCH --job-name=fmgai-train
source hpc/common.sh
activate_env; offline; job_header

RUN=${1:?runA or runB}
case "$RUN" in runA) SCHEDULE=original ;; runB) SCHEDULE=power ;; *) echo "runA or runB" >&2; exit 2 ;; esac
bash run_train.sh "$SCHEDULE" "$RESULTS_DIR/outputs/$RUN" "$DATA_DIR"
