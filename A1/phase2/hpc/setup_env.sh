#!/usr/bin/env bash
# ADDED FILE: one-time setup on the LOGIN node (it needs internet; compute nodes may not have it).
#   cd $WORK_ROOT/phase2 && bash hpc/setup_env.sh
# 1. installs Miniforge under WORK_ROOT if no conda is available
# 2. creates the Python env with the CUDA build of torch and requirements.txt
# 3. downloads everything the jobs need (model, dataset, LPIPS weights), so jobs can run offline
# Safe to rerun: finished steps are skipped.
source hpc/common.sh

if [ -n "$CONDA_MODULE" ]; then set +u; module load "$CONDA_MODULE"; set -u; fi
if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  echo "== installing Miniforge into $CONDA_ROOT"
  installer="$WORK_ROOT/Miniforge3-Linux-x86_64.sh"
  curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$installer" -b -p "$CONDA_ROOT"
  rm -f "$installer"
fi
set +u
if ! command -v conda >/dev/null 2>&1; then source "$CONDA_ROOT/etc/profile.d/conda.sh"; fi
eval "$(conda shell.bash hook)"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "== creating env $ENV_DIR (python $PYTHON_VERSION)"
  conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION" pip
fi
set -u

activate_env
echo "== installing torch from $TORCH_INDEX"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX"
echo "== installing requirements.txt"
"$PYTHON" -m pip install -r requirements.txt

echo "== downloading model, dataset and LPIPS weights"
"$PYTHON" hpc/prefetch.py --data_dir "$DATA_DIR"

echo
echo "Setup finished. Next (still on the login node):  bash hpc/submit.sh smoke"
