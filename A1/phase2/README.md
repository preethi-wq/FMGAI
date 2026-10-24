# Noise Schedule Ablation: InstructPix2Pix, 4× aerial super-resolution (Phase 2)

Task, model, dataset and exponent are the ones in the Phase 1 declaration:

| | |
|---|---|
| Task | 4× super-resolution of UC Merced aerial tiles: input is the 256×256 tile bicubic-downsampled to 64×64 and upsampled back; target is the original 256×256 tile |
| Pretrained model | InstructPix2Pix, [`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix) ([paper](https://arxiv.org/abs/2211.09800), [code](https://github.com/timothybrooks/instruct-pix2pix)) |
| Native noise parameter | θₜ = βₜ, scaled-linear from 0.00085 to 0.012, T = 1000 |
| Exponent | `EXPONENT_X = 1.60`, the named constant at the top of `train_instruct_pix2pix.py` |
| Instruction (every sample) | `increase the resolution of this image` |
| Seed | **42** for the data split, training, validation and every sampler (image *i* of a split always starts from the noise seeded `42 + i`) |

## Files

| File | What it is |
|---|---|
| `original/train_instruct_pix2pix.py` | The unmodified diffusers v0.41.0 example script, kept for the side-by-side comparison |
| `train_instruct_pix2pix.py` | The modified fine-tuning script (every changed line ends in `# MODIFIED`, every added line in `# ADDED`) |
| `sr_common.py` | New helper module (data, schedules, sampler, metrics, figures) imported by the scripts below |
| `prepare_data.py` | Builds the LR/HR pairs and the stratified 70/15/15 split |
| `infer.py` | Test-set inference + LPIPS/PSNR/SSIM for Run 0, Run A, Run B and the bicubic reference; saves the trajectories |
| `plot_schedule.py` | Schedule Analysis plots and table (numpy only) |
| `make_figures.py` | Loss curves, checkpoint grids, comparison figure, results table |
| `make_diff.py` | `changed_lines.md` and `implementation_diff.html` for the Implementation section; fails if a changed line is unmarked |
| `FMGAI_A1_phase2_colab.ipynb` | Runs everything on Colab in order |

The new files (`sr_common.py`, `prepare_data.py`, `infer.py`, `plot_*`, `make_*`) are not part of the original codebase, so
each starts with an `# ADDED FILE` header instead of marking every line.

## What was changed in the codebase

All changes are in `train_instruct_pix2pix.py` (run `python make_diff.py` for the exact lines):

1. `EXPONENT_X = 1.60` at the top, and a `--noise_schedule {original,power}` flag. `power` rebuilds the `DDPMScheduler` with
   `trained_betas = beta_t ** EXPONENT_X`. This is the only difference between Run A and Run B.
2. Schedule verification: the script asserts that the scheduler's betas equal {βₜ} or {βₜˣ}, writes `schedule.json`, and on
   the first training step recomputes `z_t = sqrt(ᾱₜ) z₀ + sqrt(1 − ᾱₜ) ε` from an independently built schedule and
   checks it against what `add_noise` produced (`schedule_check.json`, which also records the error under the other schedule).
3. The UNet `conv_in` widening (4 → 8 channels) is skipped, because the InstructPix2Pix checkpoint already has 8 input channels.
   The original code would crash on it.
4. A `--data_dir` loader for the paired (LR, HR) split from `prepare_data.py`.
5. At the end of each epoch: `train_log.csv` (`epoch,train_loss`), validation loss on the full validation split, and at
   5 equally spaced epochs, 3 labelled samples plus LPIPS/PSNR/SSIM on 32 validation images.
6. The final saved pipeline carries the run's own schedule (a DDIM scheduler with `trained_betas`), so inference uses
   {βₜˣ} for Run B.

Validation never consumes the global random state, so Run A and Run B see the same data order, noise and timesteps.

## Environment

Run on Google Colab with a single GPU. `FMGAI_A1_phase2_colab.ipynb` writes the exact versions to `environment.txt`.
**Copy them into this table before submitting:**

| | Version |
|---|---|
| Python | _from environment.txt_ |
| torch / CUDA | _from environment.txt_ |
| diffusers | 0.41.0 (pinned) |
| transformers / accelerate / datasets | _from environment.txt_ |
| bitsandbytes / lpips / scikit-image | _from environment.txt_ |
| GPU | _from environment.txt_ |

```
pip install -r requirements.txt
```

## Reproducing the runs

Run from this folder. Paths are relative; the notebook uses Google Drive paths for `outputs/`, `eval/` and `figures/`.

```bash
# Data: 2,100 pairs; 1470 / 315 / 315 split; writes data/splits.csv and data/dataset_summary.json
python prepare_data.py --data_dir data

# Schedule Analysis figures and the code diff
python plot_schedule.py --out_dir figures
python make_diff.py --out_dir figures

# Run 0: pretrained model, no fine-tuning (plus the bicubic input as a reference row)
python infer.py --run bicubic --data_dir data --out_dir eval
python infer.py --run run0    --data_dir data --out_dir eval

# Run A: original schedule {beta_t}
python train_instruct_pix2pix.py --noise_schedule original --output_dir outputs/runA \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix --data_dir data \
  --resolution 256 --random_flip --train_batch_size 4 --gradient_accumulation_steps 2 \
  --num_train_epochs 10 --learning_rate 5e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --adam_weight_decay 1e-2 --max_grad_norm 1.0 --conditioning_dropout_prob 0.05 \
  --mixed_precision fp16 --use_8bit_adam --gradient_checkpointing \
  --checkpointing_steps 368 --checkpoints_total_limit 1 --seed 42

# Run B: modified schedule {beta_t ** 1.60}, identical arguments apart from the two flags on the first line
python train_instruct_pix2pix.py --noise_schedule power --output_dir outputs/runB \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix --data_dir data \
  --resolution 256 --random_flip --train_batch_size 4 --gradient_accumulation_steps 2 \
  --num_train_epochs 10 --learning_rate 5e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --adam_weight_decay 1e-2 --max_grad_norm 1.0 --conditioning_dropout_prob 0.05 \
  --mixed_precision fp16 --use_8bit_adam --gradient_checkpointing \
  --checkpointing_steps 368 --checkpoints_total_limit 1 --seed 42

# Test-set evaluation and trajectories of the final fine-tuned models
python infer.py --run runA --data_dir data --out_dir eval --model_dir outputs/runA
python infer.py --run runB --data_dir data --out_dir eval --model_dir outputs/runB

# Report figures and the results table
python make_figures.py --runs_dir outputs --eval_dir eval --data_dir data --out_dir figures
```

If Colab disconnects during training, rerun the same command with `--resume_from_checkpoint latest`. A training state is saved
every 2 epochs (368 optimizer steps).

## Hyperparameters (identical for Run A and Run B)

| | |
|---|---|
| Starting checkpoint | `timbrooks/instruct-pix2pix` (full UNet fine-tune; VAE and text encoder frozen) |
| Optimizer | AdamW 8-bit (bitsandbytes), β = (0.9, 0.999), weight decay 0.01, ε = 1e-8 |
| Learning rate | 5e-5, constant, no warm-up; gradient clipping at 1.0 |
| Batch size | 4 × 2 gradient-accumulation steps = 8 effective |
| Epochs | 10 (1,840 optimizer steps); samples at epochs 2, 4, 6, 8, 10 |
| Resolution | 256 × 256 (latents 32 × 32); random horizontal flip applied to input and target together |
| Conditioning dropout | 0.05 (as in InstructPix2Pix, so classifier-free guidance works) |
| Precision | fp16 mixed precision, gradient checkpointing |
| Loss | MSE on predicted noise ε, timesteps uniform on {0, …, 999} |

## Evaluation protocol

- **Sampler (all runs)**: DDIM, η = 0, 100 steps, trailing spacing (t = 999, 989, …, 9), text guidance 7.5, image guidance 1.5.
  It is built from the run's own betas: {βₜ} for Run 0 and Run A, {βₜˣ} for Run B.
- **Validation during fine-tuning**: noise-prediction MSE on all 315 validation pairs every epoch (fixed noise and
  timesteps), and LPIPS / PSNR / SSIM on the first 32 validation images at the 5 checkpoint epochs. The validation loss is
  computed under each run's own schedule, so the two runs' losses are not on the same scale. LPIPS is the comparable number.
- **Test set**: 315 pairs, used once at the end. The primary metric is **LPIPS** (AlexNet), with PSNR and SSIM as secondary metrics.
- **Trajectories**: for the first 3 test images, x_t after 0, 25, 50, 75 and 100 of the 100 steps (t = 999, 749, 499, 249, 0),
  decoded through the VAE.

## Outputs and where they go in the report

| Report item | File |
|---|---|
| Schedule plot, SNR, table | `figures/schedule_betas.png`, `figures/schedule_alphabar_snr.png`, `figures/schedule_table.md` |
| Original vs modified lines | `figures/changed_lines.md`, `figures/implementation_diff.html` |
| Run B uses {βₜˣ} | `outputs/runB/schedule.json`, `outputs/runB/schedule_check.json`, the training log line "Forward-process schedule check passed" |
| Fine-tuning logs (CSV) | `outputs/runA/train_log.csv`, `outputs/runB/train_log.csv` |
| Samples at 5 checkpoints | `outputs/run*/samples/Run*_epochNN.png`, combined in `figures/checkpoints_run*.png` |
| Inference trajectories | `eval/run*/trajectories/run*_trajectory.png` (and the individual `run*_tNNNN_sampleK.png`) |
| Run 0 samples | `figures/run0_samples.png` |
| Loss curves | `figures/loss_curves.png`, `figures/val_curves.png` |
| Test scores | `figures/results_table.md` |
