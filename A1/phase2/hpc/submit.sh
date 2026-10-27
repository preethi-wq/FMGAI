#!/usr/bin/env bash
# ADDED FILE: submits the Slurm jobs from the login node, from the phase2 folder.
#
#   bash hpc/submit.sh smoke            # ~10 min check of the whole loop (runs data prep first if needed)
#   bash hpc/submit.sh all              # prepare -> baseline + Run A + Run B -> evaluations -> figures + zip
#   bash hpc/submit.sh train [runA|runB]  # (re)submit training, e.g. after a TIMEOUT (it resumes)
#   bash hpc/submit.sh finish           # evaluate Run A and Run B, then figures + zip
#   bash hpc/submit.sh prepare | baseline | evaluate runA|runB | report
#
# Each job gets -D (the phase2 folder), -t and --mem explicitly, as the IITJ HPC guide requires.
# Logs: $RESULTS_DIR/slurm/<job-name>-<job-id>.out.  Watch with: squeue -u $USER
source hpc/common.sh
shopt -s inherit_errexit  # a failed sbatch inside $(...) stops the script instead of chaining on an empty id
command -v sbatch >/dev/null || { echo "sbatch not found: run this on the HPC login node" >&2; exit 1; }
mkdir -p "$RESULTS_DIR/slurm"

# submit <cpu|gpu> <afterok job ids, comma-separated or empty> <script> [args]  -> prints the job id
submit() {
  local kind=$1 deps=$2; shift 2
  local opts=(--parsable -D "$CODE_DIR" -o "$RESULTS_DIR/slurm/%x-%j.out")
  if [ "$kind" = gpu ]; then
    opts+=(-p "$GPU_PARTITION" $GPU_REQUEST -c "$GPU_CPUS" --mem "$GPU_MEM")
  else
    opts+=(-p "$CPU_PARTITION" -c "$CPU_CPUS" --mem "$CPU_MEM" -t "$CPU_TIME")
  fi
  case "$1" in
    *train.sbatch) opts+=(-t "$TRAIN_TIME") ;;
    *baseline.sbatch|*evaluate.sbatch|*smoke.sbatch) opts+=(-t "$EVAL_TIME") ;;
  esac
  if [ -n "$ACCOUNT" ]; then opts+=(-A "$ACCOUNT"); fi
  if [ -n "$deps" ]; then opts+=(--dependency="afterok:${deps//,/:}"); fi
  sbatch "${opts[@]}" "$@"
}

say() { echo "submitted $1: job $2" >&2; }

prepare_if_needed() {
  if [ -f "$DATA_DIR/splits.csv" ]; then echo ""; return; fi
  local j; j=$(submit cpu "" hpc/jobs/prepare.sbatch); say prepare "$j"; echo "$j"
}

STAGE=${1:-}
ARG=${2:-}
case "$STAGE" in
  smoke)
    p=$(prepare_if_needed)
    j=$(submit gpu "$p" hpc/jobs/smoke.sbatch); say smoke "$j"
    echo "when it finishes: tail -30 $RESULTS_DIR/slurm/fmgai-smoke-$j.out  (expect SMOKE TEST PASSED)" ;;
  all)
    p=$(prepare_if_needed)
    b=$(submit gpu "$p" hpc/jobs/baseline.sbatch); say baseline "$b"
    ta=$(submit gpu "$p" hpc/jobs/train.sbatch runA); say "train runA" "$ta"
    tb=$(submit gpu "$p" hpc/jobs/train.sbatch runB); say "train runB" "$tb"
    ea=$(submit gpu "$ta" hpc/jobs/evaluate.sbatch runA); say "evaluate runA" "$ea"
    eb=$(submit gpu "$tb" hpc/jobs/evaluate.sbatch runB); say "evaluate runB" "$eb"
    r=$(submit cpu "$b,$ea,$eb" hpc/jobs/report.sbatch); say report "$r"
    echo "If a training job ends in TIMEOUT, its evaluation and the report are not run (squeue shows"
    echo "DependencyNeverSatisfied; scancel them). Then: bash hpc/submit.sh train runX, and when done: bash hpc/submit.sh finish" ;;
  prepare)  j=$(submit cpu "" hpc/jobs/prepare.sbatch); say prepare "$j" ;;
  baseline) j=$(submit gpu "" hpc/jobs/baseline.sbatch); say baseline "$j" ;;
  train)
    for run in ${ARG:-runA runB}; do j=$(submit gpu "" hpc/jobs/train.sbatch "$run"); say "train $run" "$j"; done ;;
  evaluate)
    for run in ${ARG:-runA runB}; do j=$(submit gpu "" hpc/jobs/evaluate.sbatch "$run"); say "evaluate $run" "$j"; done ;;
  finish)
    ea=$(submit gpu "" hpc/jobs/evaluate.sbatch runA); say "evaluate runA" "$ea"
    eb=$(submit gpu "" hpc/jobs/evaluate.sbatch runB); say "evaluate runB" "$eb"
    r=$(submit cpu "$ea,$eb" hpc/jobs/report.sbatch); say report "$r" ;;
  report)   j=$(submit cpu "" hpc/jobs/report.sbatch); say report "$j" ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac
