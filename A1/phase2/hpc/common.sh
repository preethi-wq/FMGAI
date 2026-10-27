# ADDED FILE: sourced by every job script and by setup_env.sh. Loads config.sh, activates the conda env
# and points the Hugging Face / torch caches at WORK_ROOT. Jobs run with --chdir set to the phase2 folder.
set -euo pipefail

CODE_DIR="$(pwd)"
if [ ! -f "$CODE_DIR/train_instruct_pix2pix.py" ]; then
  echo "run from the phase2 folder (jobs get it through sbatch --chdir)" >&2
  exit 2
fi
source "$CODE_DIR/hpc/config.sh"
export HF_HOME TORCH_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p "$WORK_ROOT" "$RESULTS_DIR" "$HF_HOME" "$TORCH_HOME"

activate_env() {
  set +u  # module and conda activation scripts read unset variables
  if [ -n "$EXTRA_MODULES" ]; then module load $EXTRA_MODULES; fi
  if [ -n "$CONDA_MODULE" ]; then module load "$CONDA_MODULE"; fi
  if ! command -v conda >/dev/null 2>&1; then source "$CONDA_ROOT/etc/profile.d/conda.sh"; fi
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_DIR"
  set -u
  export PYTHON="$ENV_DIR/bin/python"
}

# Compute nodes read only the caches filled by setup_env.sh on the login node, so a job never
# depends on internet access (and a missing file fails immediately instead of hanging).
offline() {
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
}

job_header() {
  echo "job ${SLURM_JOB_ID:-local} ${SLURM_JOB_NAME:-} on $(hostname) at $(date)"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; fi
}
