# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Test-set inference and evaluation for Run 0, Run A, Run B and the bicubic reference.

  python infer.py --run bicubic --data_dir data --out_dir eval
  python infer.py --run run0    --data_dir data --out_dir eval
  python infer.py --run runA    --data_dir data --out_dir eval --model_dir outputs/runA
  python infer.py --run runB    --data_dir data --out_dir eval --model_dir outputs/runB

Writes to <out_dir>/<run>/: test_outputs/*.png, per_image_metrics.csv, summary.json and, for the
diffusion runs, trajectories/ with x_t at 5 equally spaced reverse-process timesteps for 3 samples.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

import sr_common as C


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, choices=["bicubic", "run0", "runA", "runB"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="eval")
    parser.add_argument("--model_dir", default=None, help="Fine-tuned pipeline folder (runA / runB).")
    parser.add_argument("--pretrained", default=C.PRETRAINED_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N images (smoke tests).")
    args = parser.parse_args()
    if args.run in ("runA", "runB") and not args.model_dir:
        parser.error("--model_dir is required for runA / runB")
    return args


def load_pipeline(args, device):
    """Pipeline for the run, with a DDIM sampler holding that run's betas, checked against the declaration."""
    from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline

    from train_instruct_pix2pix import EXPONENT_X

    dtype = torch.float16 if device == "cuda" else torch.float32
    source = args.pretrained if args.run == "run0" else args.model_dir
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        source, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    original = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler").betas
    run_betas = original if args.run == "run0" else pipe.scheduler.betas
    noise_schedule = "power" if args.run == "runB" else "original"
    C.verify_run_schedule(run_betas, original, noise_schedule, EXPONENT_X)
    if args.run != "run0":
        with open(os.path.join(args.model_dir, "schedule.json")) as f:
            saved = json.load(f)
        if saved["noise_schedule"] != noise_schedule:
            raise RuntimeError(f"{args.model_dir} was trained with '{saved['noise_schedule']}', not '{noise_schedule}'")
    pipe.scheduler = C.make_ddim(pipe.scheduler.config, run_betas)
    pipe.set_progress_bar_config(disable=True)
    print(f"{C.RUN_LABELS[args.run]}: schedule '{noise_schedule}' {C.schedule_summary(run_betas)}")
    return pipe.to(device)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = C.read_split(args.data_dir, args.split)[: args.limit]
    label = C.RUN_LABELS[args.run]
    out = os.path.join(args.out_dir, args.run)
    os.makedirs(os.path.join(out, "test_outputs"), exist_ok=True)

    pipe = None if args.run == "bicubic" else load_pipeline(args, device)
    metrics = C.Metrics(device)
    per_image = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        lr = [C.load_rgb(r["lr_path"]) for r in chunk]
        if pipe is None:
            preds = [np.asarray(im) for im in lr]
        else:
            preds, _ = C.super_resolve(pipe, lr, [C.SEED + i + j for j in range(len(chunk))])
        for r, pred in zip(chunk, preds):
            Image.fromarray(pred).save(os.path.join(out, "test_outputs", r["id"] + ".png"))
            per_image.append({"id": r["id"], "class": r["class"], **metrics(pred, np.asarray(C.load_rgb(r["hr_path"])))})
        print(f"{label}: {min(i + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    with open(os.path.join(out, "per_image_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "lpips", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(per_image)
    summary = {"run": args.run, "label": label, "split": args.split, "num_images": len(per_image), "seed": C.SEED}
    for k in ("lpips", "psnr", "ssim"):
        vals = np.array([p[k] for p in per_image])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    if pipe is not None:
        summary["sampler"] = {
            "scheduler": "DDIM (eta=0, trailing spacing)",
            "num_inference_steps": C.NUM_INFERENCE_STEPS,
            "guidance_scale": C.GUIDANCE_SCALE,
            "image_guidance_scale": C.IMAGE_GUIDANCE_SCALE,
            "instruction": C.INSTRUCTION,
        }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    if pipe is not None:
        save_trajectories(pipe, rows[: C.NUM_TRAJECTORY_SAMPLES], label, args.run, out)


def save_trajectories(pipe, rows, label, run, out):
    """x_t after 0, 25, 50, 75 and 100 of the 100 DDIM steps (t = 999, 749, 499, 249, 0), decoded by the VAE.
    Seeds match the test outputs, so the last state equals the saved test output."""
    traj_dir = os.path.join(out, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    lr = [C.load_rgb(r["lr_path"]) for r in rows]
    _, traj = C.super_resolve(pipe, lr, [C.SEED + j for j in range(len(rows))], record_states=C.TRAJECTORY_STATES)
    timesteps = C.state_timesteps(pipe, C.TRAJECTORY_STATES)
    for state, t in zip(C.TRAJECTORY_STATES, timesteps):
        for k, img in enumerate(traj[state], start=1):
            Image.fromarray(img).save(os.path.join(traj_dir, f"{run}_t{t:04d}_sample{k}.png"))
    grid = [
        [np.asarray(lr[k])] + [traj[s][k] for s in C.TRAJECTORY_STATES] + [np.asarray(C.load_rgb(rows[k]["hr_path"]))]
        for k in range(len(rows))
    ]
    cols = ["LR input"] + [f"x_t, t = {t}" for t in timesteps] + ["HR target"]
    C.save_labeled_grid(grid, [r["id"] for r in rows], cols, f"{label} | reverse-process trajectory",
                        os.path.join(traj_dir, f"{run}_trajectory.png"))
    with open(os.path.join(traj_dir, "timesteps.json"), "w") as f:
        json.dump({"states": C.TRAJECTORY_STATES, "timesteps": timesteps, "ids": [r["id"] for r in rows]}, f)


if __name__ == "__main__":
    main()
