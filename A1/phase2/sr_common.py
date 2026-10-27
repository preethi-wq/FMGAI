# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Shared helpers for the noise-schedule ablation: data, schedules, sampling, metrics and figures.

train_instruct_pix2pix.py, infer.py and make_figures.py all import this module, so Run 0, Run A
and Run B use one implementation of the sampler, the schedule handling and the metrics.
"""
import contextlib
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PRETRAINED_MODEL = "timbrooks/instruct-pix2pix"
INSTRUCTION = "increase the resolution of this image"
SEED = 42
RESOLUTION = 256

# Sampler shared by every run: DDIM, 100 steps, "trailing" spacing so the first step is t = 999.
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 7.5
IMAGE_GUIDANCE_SCALE = 1.5
# Reverse-process states saved for the trajectory figures: state k is the latent after k DDIM steps,
# so states 0, 25, 50, 75, 100 sit at t = 999, 749, 499, 249 and 0 (x_0), 250 timesteps apart.
TRAJECTORY_STATES = [0, 25, 50, 75, 100]
NUM_TRAJECTORY_SAMPLES = 3
NUM_CHECKPOINT_SAMPLES = 3

RUN_LABELS = {"bicubic": "Bicubic input", "run0": "Run 0", "runA": "Run A", "runB": "Run B"}


# ----------------------------------------------------------------------------- data

def read_split(data_dir, split):
    """Rows of splits.csv for one split, with absolute LR/HR paths added."""
    with open(os.path.join(data_dir, "splits.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    for r in rows:
        r["lr_path"] = os.path.join(data_dir, "lr", r["id"] + ".png")
        r["hr_path"] = os.path.join(data_dir, "hr", r["id"] + ".png")
    return rows


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_sr_dataset(data_dir):
    """Training split as a DatasetDict with the columns the diffusers script expects."""
    from datasets import Dataset, DatasetDict
    from datasets import Image as ImageFeature

    rows = read_split(data_dir, "train")
    ds = Dataset.from_dict(
        {
            "input_image": [r["lr_path"] for r in rows],
            "edit_prompt": [INSTRUCTION] * len(rows),
            "edited_image": [r["hr_path"] for r in rows],
        }
    )
    ds = ds.cast_column("input_image", ImageFeature()).cast_column("edited_image", ImageFeature())
    return DatasetDict({"train": ds})


def to_model_tensor(images, device, dtype):
    """List of PIL images -> (B, 3, H, W) tensor in [-1, 1]."""
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in images])
    return (torch.from_numpy(arr).permute(0, 3, 1, 2) / 127.5 - 1.0).to(device, dtype)


# ----------------------------------------------------------------------------- schedules

def check_theta_in_unit_interval(betas):
    if not bool(torch.all((betas > 0) & (betas < 1))):
        raise ValueError("theta_t = beta_t must lie in (0, 1) for every t")


def power_betas(betas, x):
    """Elementwise power transform theta_t -> theta_t ** x, computed in float64."""
    return betas.double() ** x


def expected_betas(original_betas, noise_schedule, x):
    return power_betas(original_betas, x) if noise_schedule == "power" else original_betas.double()


def schedule_summary(betas):
    b = betas.double()
    alpha_bar = torch.cumprod(1.0 - b, dim=0)
    return {
        "beta_min": b.min().item(),
        "beta_max": b.max().item(),
        "alpha_bar_T": alpha_bar[-1].item(),
        "snr_T": (alpha_bar[-1] / (1.0 - alpha_bar[-1])).item(),
    }


def verify_run_schedule(run_betas, original_betas, noise_schedule, x):
    """Fail loudly unless the scheduler holds {beta_t} (Run A) or {beta_t ** x} (Run B)."""
    want = expected_betas(original_betas, noise_schedule, x)
    if not torch.allclose(run_betas.double(), want, rtol=1e-6, atol=0):
        raise RuntimeError(f"scheduler betas do not match the '{noise_schedule}' schedule")
    check_theta_in_unit_interval(run_betas)


def save_schedule(output_dir, run_betas, original_betas, noise_schedule, x):
    info = {
        "noise_schedule": noise_schedule,
        "exponent_x": x if noise_schedule == "power" else None,
        "native_parameter": "beta_t",
        "original": schedule_summary(original_betas),
        "run": schedule_summary(run_betas),
        "betas": [float(b) for b in run_betas],
    }
    with open(os.path.join(output_dir, "schedule.json"), "w") as f:
        json.dump(info, f, indent=1)


def verify_add_noise(noisy_latents, latents, noise, timesteps, original_betas, noise_schedule, x):
    """Recompute z_t = sqrt(abar_t) z_0 + sqrt(1 - abar_t) eps from an independently built schedule and
    compare it with what the training scheduler produced. Returns the evidence as a dict."""
    abar = torch.cumprod(1.0 - expected_betas(original_betas, noise_schedule, x), dim=0)
    abar_t = abar[timesteps.cpu()].to(latents.device).view(-1, 1, 1, 1)
    want = abar_t.sqrt() * latents.double() + (1.0 - abar_t).sqrt() * noise.double()
    err = (noisy_latents.double() - want).abs().max().item()
    # The other schedule would put these latents somewhere else entirely.
    other = "original" if noise_schedule == "power" else "power"
    abar_o = torch.cumprod(1.0 - expected_betas(original_betas, other, x), dim=0)
    abar_o = abar_o[timesteps.cpu()].to(latents.device).view(-1, 1, 1, 1)
    want_o = abar_o.sqrt() * latents.double() + (1.0 - abar_o).sqrt() * noise.double()
    err_o = (noisy_latents.double() - want_o).abs().max().item()
    if not (err < 2e-2 and err_o > 2 * err):  # tolerance covers fp16 latents
        raise RuntimeError(f"add_noise check failed: err={err:.3g} vs other schedule {err_o:.3g}")
    return {
        "noise_schedule": noise_schedule,
        "timesteps": timesteps.tolist(),
        "max_abs_error_vs_this_schedule": err,
        "max_abs_error_vs_other_schedule": err_o,
    }


def make_ddim(scheduler_config, betas):
    """The inference sampler for every run, holding that run's betas explicitly."""
    from diffusers import DDIMScheduler

    return DDIMScheduler.from_config(
        scheduler_config,
        trained_betas=[float(b) for b in betas],
        timestep_spacing="trailing",
        clip_sample=False,
    )


# ----------------------------------------------------------------------------- sampling

def build_pipeline(vae, text_encoder, tokenizer, unet, scheduler):
    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def initial_noise(pipe, seeds, device, dtype):
    """x_T for each image, drawn from a per-image seed so every run starts from the same noise."""
    size = RESOLUTION // pipe.vae_scale_factor
    noise = [
        torch.randn((1, pipe.vae.config.latent_channels, size, size), generator=torch.Generator("cpu").manual_seed(s))
        for s in seeds
    ]
    return torch.cat(noise).to(device, dtype)


@torch.no_grad()
def decode_latents(pipe, latents):
    images = pipe.vae.decode(latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    images = (images.float().clamp(-1, 1) + 1) / 2
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def super_resolve(pipe, lr_images, seeds, record_states=()):
    """Run the reverse process on a batch. Returns (outputs, trajectory) where outputs are uint8 HxWx3
    arrays and trajectory maps each requested state index to the decoded x_t images."""
    device = pipe._execution_device
    latents = initial_noise(pipe, seeds, device, pipe.text_encoder.dtype)
    states = {0: latents.clone()} if 0 in record_states else {}

    def on_step_end(_pipe, i, t, kwargs):
        if i + 1 in record_states:
            states[i + 1] = kwargs["latents"].clone()
        return kwargs

    out = pipe(
        [INSTRUCTION] * len(lr_images),
        image=lr_images,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        latents=latents,
        output_type="np",
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    outputs = (out.images * 255).round().astype(np.uint8)
    trajectory = {k: decode_latents(pipe, v) for k, v in states.items()}
    return outputs, trajectory


def state_timesteps(pipe, states):
    """Diffusion timestep of each saved state (state k = latent after k steps; the last state is x_0)."""
    ts = [int(t) for t in pipe.scheduler.timesteps]
    return [ts[k] if k < len(ts) else 0 for k in states]


# ----------------------------------------------------------------------------- validation and metrics

@contextlib.contextmanager
def preserve_rng(device):
    """Validation must not consume the global RNG, so Run A and Run B see identical training noise."""
    devices = [device] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield


def autocast_for(device):
    return torch.autocast("cuda") if torch.device(device).type == "cuda" else contextlib.nullcontext()


class Metrics:
    """LPIPS (AlexNet), PSNR and SSIM between generated and ground-truth HR images (uint8 HxWx3)."""

    def __init__(self, device):
        import lpips

        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred, target):
        from skimage.metrics import structural_similarity

        p = to_model_tensor([pred], self.device, torch.float32)
        t = to_model_tensor([target], self.device, torch.float32)
        mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
        return {
            "lpips": self.lpips(p, t).item(),
            "psnr": float(10 * np.log10(255.0**2 / max(mse, 1e-10))),
            "ssim": float(structural_similarity(pred, target, channel_axis=2, data_range=255)),
        }


@torch.no_grad()
def validation_loss(unet, vae, text_encoder, tokenizer, noise_scheduler, rows, device, weight_dtype, batch_size=8):
    """Denoising MSE on the validation split with fixed noise and timesteps (same draws every epoch)."""
    g = torch.Generator("cpu").manual_seed(SEED)
    ids = tokenizer(
        [INSTRUCTION], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    text = text_encoder(ids)[0]
    total, count = 0.0, 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        hr = to_model_tensor([load_rgb(r["hr_path"]) for r in chunk], device, weight_dtype)
        lr = to_model_tensor([load_rgb(r["lr_path"]) for r in chunk], device, weight_dtype)
        latents = vae.encode(hr).latent_dist.mode() * vae.config.scaling_factor
        cond = vae.encode(lr).latent_dist.mode()
        noise = torch.randn(latents.shape, generator=g).to(device, latents.dtype)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(chunk),), generator=g).to(device)
        noisy = noise_scheduler.add_noise(latents.float(), noise.float(), t).to(latents.dtype)  # fp32: see training loop
        pred = unet(torch.cat([noisy, cond], dim=1), t, text.expand(len(chunk), -1, -1), return_dict=False)[0]
        total += F.mse_loss(pred.float(), noise.float(), reduction="sum").item() / noise[0].numel()
        count += len(chunk)
    return total / count


def equally_spaced_epochs(num_epochs, k):
    return sorted({max(1, round(num_epochs * i / k)) for i in range(1, k + 1)})


def upsert_csv(path, fieldnames, row, key="epoch"):
    """Write one row keyed by `key`, replacing an existing row with the same key (safe on resume)."""
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = {r[key]: r for r in csv.DictReader(f)}
    rows[str(row[key])] = {k: row[k] for k in fieldnames}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(rows, key=int):
            writer.writerow(rows[k])


def end_of_epoch(epoch, train_loss, args, run_label, val_rows, sample_epochs, unet, vae, text_encoder, tokenizer,
                 noise_scheduler, device, weight_dtype, cache):
    """Per-epoch log (train_log.csv: epoch, train_loss), validation loss every epoch, and at the equally
    spaced checkpoint epochs: labelled samples plus LPIPS/PSNR/SSIM on a fixed validation subset."""
    out = args.output_dir
    upsert_csv(os.path.join(out, "train_log.csv"), ["epoch", "train_loss"], {"epoch": epoch, "train_loss": train_loss})
    if not val_rows:
        return
    was_training = unet.training
    unet.eval()
    with preserve_rng(device), autocast_for(device):
        val_loss = validation_loss(unet, vae, text_encoder, tokenizer, noise_scheduler, val_rows, device, weight_dtype)
        upsert_csv(os.path.join(out, "val_log.csv"), ["epoch", "val_loss"], {"epoch": epoch, "val_loss": val_loss})

        if epoch in sample_epochs:
            pipe = build_pipeline(vae, text_encoder, tokenizer, unet, make_ddim(noise_scheduler.config, noise_scheduler.betas))
            tag = run_label.replace(" ", "")
            sample_dir = os.path.join(out, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            shown = val_rows[:NUM_CHECKPOINT_SAMPLES]
            lr = [load_rgb(r["lr_path"]) for r in shown]
            outputs, _ = super_resolve(pipe, lr, [SEED + i for i in range(len(shown))])
            for k, img in enumerate(outputs, start=1):
                Image.fromarray(img).save(os.path.join(sample_dir, f"{tag}_epoch{epoch:02d}_sample{k}.png"))
            save_labeled_grid(
                [[np.asarray(l), o, np.asarray(load_rgb(r["hr_path"]))] for l, o, r in zip(lr, outputs, shown)],
                [r["id"] for r in shown],
                ["LR input", f"{run_label} output", "HR target"],
                f"{run_label} | fine-tuning epoch {epoch}",
                os.path.join(sample_dir, f"{tag}_epoch{epoch:02d}.png"),
            )

            if "metrics" not in cache:
                cache["metrics"] = Metrics(device)
            scored = val_rows[: args.num_val_metric_images]
            scores = []
            for i in range(0, len(scored), 8):
                chunk = scored[i : i + 8]
                preds, _ = super_resolve(pipe, [load_rgb(r["lr_path"]) for r in chunk], [SEED + i + j for j in range(len(chunk))])
                scores += [cache["metrics"](p, np.asarray(load_rgb(r["hr_path"]))) for p, r in zip(preds, chunk)]
            mean = {k: float(np.mean([s[k] for s in scores])) for k in ("lpips", "psnr", "ssim")}
            upsert_csv(os.path.join(out, "val_metrics.csv"), ["epoch", "lpips", "psnr", "ssim"], {"epoch": epoch, **mean})
            del pipe
    if was_training:
        unet.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- figures

def save_labeled_grid(rows, row_labels, col_labels, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.25 * nr + 0.5), squeeze=False)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            ax = axes[i][j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
