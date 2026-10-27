# ADDED FILE: settings for the IITJ HPC (Slurm). Edit this file once on the login node, nothing else.
# Values marked CHECK depend on the cluster; the comment says which command shows the right value.

# ---- Slurm ---------------------------------------------------------------------------------------
GPU_PARTITION="gpu"          # CHECK: `sinfo` lists partitions; pick one whose nodes have GPUs
CPU_PARTITION="small"        # CHECK: a CPU partition, used for data prep and the final figures/zip
GPU_REQUEST="--gres=gpu:1"   # CHECK: `sinfo -o "%P %G"` shows the GRES names, e.g. gpu:1 or gpu:a100:1
ACCOUNT=""                   # only if your jobs must be charged to an account (-A); leave empty otherwise

TRAIN_TIME="12:00:00"        # per training run; CHECK it is below the partition's limit (`sinfo` TIMELIMIT)
EVAL_TIME="04:00:00"         # baseline (Run 0) and the Run A / Run B test-set evaluation
CPU_TIME="01:00:00"          # data prep, figures and zip
GPU_MEM="32G"                # host RAM for GPU jobs (not GPU memory)
GPU_CPUS=8
CPU_MEM="8G"
CPU_CPUS=4

# ---- Paths ---------------------------------------------------------------------------------------
# Needs about 40 GB: model cache 4.3 GB, Python env 8 GB, dataset 0.3 GB, two saved pipelines and
# their last training state about 25 GB. If $HOME has a small quota, point this at scratch/project space.
WORK_ROOT="$HOME/FMGAI_A1"
RESULTS_DIR="$WORK_ROOT/results"   # outputs/, eval/, figures/, slurm logs and the submission zip
DATA_DIR="$WORK_ROOT/data"
HF_HOME="$WORK_ROOT/hf_cache"
TORCH_HOME="$WORK_ROOT/torch_cache"

# ---- Python environment --------------------------------------------------------------------------
# The cluster's python/3.7 module is too old (diffusers 0.41 needs torch >= 2.6, so Python >= 3.9).
# setup_env.sh installs Miniforge (conda-forge) under WORK_ROOT unless CONDA_MODULE names a module
# that provides conda (`module avail` shows e.g. anaconda3 or miniconda).
CONDA_MODULE=""
CONDA_ROOT="$WORK_ROOT/miniforge3"
ENV_DIR="$WORK_ROOT/env"
PYTHON_VERSION="3.11"
# CUDA build of torch. The wheel ships its own CUDA runtime, so no cuda module is needed; the GPU
# node's driver must support it (`nvidia-smi` on a GPU node: "CUDA Version" >= 12.4 for cu124).
# Use https://download.pytorch.org/whl/cu118 for an older driver.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
EXTRA_MODULES=""             # anything else to `module load` before running, normally empty
