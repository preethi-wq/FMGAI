#!/usr/bin/env bash
# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
# The one place the fine-tuning hyperparameters are written down. Colab and the HPC jobs both call it, so
# Run A and Run B always get identical arguments apart from --noise_schedule.
#
#   bash run_train.sh original <output_dir> <data_dir>   # Run A, {beta_t}
#   bash run_train.sh power    <output_dir> <data_dir>   # Run B, {beta_t ** EXPONENT_X}
#
# Extra arguments after <data_dir> are passed through (smoke tests only, e.g. --max_train_samples 16).
# Always resumes from the last saved state in <output_dir> if there is one.
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: bash run_train.sh {original|power} <output_dir> <data_dir> [extra args]" >&2
  exit 2
fi
SCHEDULE=$1; OUTPUT_DIR=$2; DATA_DIR=$3; shift 3
case "$SCHEDULE" in original|power) ;; *) echo "schedule must be 'original' or 'power'" >&2; exit 2 ;; esac

cd "$(dirname "$0")"
exec "${PYTHON:-python}" train_instruct_pix2pix.py \
  --noise_schedule "$SCHEDULE" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --resolution 256 --random_flip --train_batch_size 4 --gradient_accumulation_steps 2 \
  --num_train_epochs 10 --learning_rate 5e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --adam_weight_decay 1e-2 --max_grad_norm 1.0 --conditioning_dropout_prob 0.05 \
  --mixed_precision fp16 --use_8bit_adam --gradient_checkpointing \
  --checkpointing_steps 368 --checkpoints_total_limit 1 --seed 42 \
  --resume_from_checkpoint latest \
  "$@"
