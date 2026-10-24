# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Results figures and the test-score table for the report, from the training and evaluation outputs.

  python make_figures.py --runs_dir outputs --eval_dir eval --data_dir data --out_dir figures

Expects outputs/runA and outputs/runB (training) and eval/{bicubic,run0,runA,runB} (infer.py). Writes:
  loss_curves.png             Run A and Run B train loss per epoch on one plot
  val_curves.png              validation loss per epoch and LPIPS at the checkpoint epochs
  checkpoints_runA.png / _runB  samples at the 5 equally spaced epochs, one row per epoch
  run0_samples.png            Run 0 outputs next to the LR input and HR target
  test_comparison.png         the same test images for every run side by side
  results_table.csv / .md     mean (std) LPIPS / PSNR / SSIM on the test split
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sr_common as C

TRAINED = ["runA", "runB"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def loss_curves(runs_dir, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run in TRAINED:
        rows = read_csv(os.path.join(runs_dir, run, "train_log.csv"))
        if rows:
            ax.plot([int(r["epoch"]) for r in rows], [float(r["train_loss"]) for r in rows], marker="o", label=C.RUN_LABELS[run])
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean train loss (noise-prediction MSE)")
    ax.set_title("Fine-tuning loss: Run A vs Run B")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in TRAINED:
        val = read_csv(os.path.join(runs_dir, run, "val_log.csv"))
        met = read_csv(os.path.join(runs_dir, run, "val_metrics.csv"))
        if val:
            axes[0].plot([int(r["epoch"]) for r in val], [float(r["val_loss"]) for r in val], marker="o", label=C.RUN_LABELS[run])
        if met:
            axes[1].plot([int(r["epoch"]) for r in met], [float(r["lpips"]) for r in met], marker="o", label=C.RUN_LABELS[run])
    axes[0].set_title("Validation loss (each run on its own schedule)")
    axes[0].set_ylabel("val noise-prediction MSE")
    axes[1].set_title("Validation LPIPS at checkpoint epochs (lower is better)")
    axes[1].set_ylabel("LPIPS")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_curves.png"), dpi=150)
    plt.close(fig)


def checkpoint_grids(runs_dir, data_dir, out_dir):
    val = C.read_split(data_dir, "val")[: C.NUM_CHECKPOINT_SAMPLES]
    for run in TRAINED:
        tag = C.RUN_LABELS[run].replace(" ", "")
        sample_dir = os.path.join(runs_dir, run, "samples")
        epochs = sorted({int(f.split("_epoch")[1][:2]) for f in os.listdir(sample_dir) if "_sample" in f}) if os.path.isdir(sample_dir) else []
        if not epochs:
            continue
        rows = [[np.asarray(C.load_rgb(r["lr_path"])) for r in val]]
        labels = ["LR input"]
        for e in epochs:
            rows.append([np.asarray(Image.open(os.path.join(sample_dir, f"{tag}_epoch{e:02d}_sample{k}.png"))) for k in range(1, len(val) + 1)])
            labels.append(f"{C.RUN_LABELS[run]}, epoch {e}")
        rows.append([np.asarray(C.load_rgb(r["hr_path"])) for r in val])
        labels.append("HR target")
        C.save_labeled_grid(rows, labels, [f"sample {k} ({r['id']})" for k, r in enumerate(val, 1)],
                            f"{C.RUN_LABELS[run]} | samples at equally spaced fine-tuning checkpoints",
                            os.path.join(out_dir, f"checkpoints_{run}.png"))


def test_figures(eval_dir, data_dir, out_dir, n=4):
    test = C.read_split(data_dir, "test")
    # Spread the shown images over different classes.
    shown, seen = [], set()
    for r in test:
        if r["class"] not in seen:
            shown.append(r)
            seen.add(r["class"])
        if len(shown) == n:
            break
    runs = [run for run in ["bicubic", "run0", "runA", "runB"] if os.path.isdir(os.path.join(eval_dir, run, "test_outputs"))]

    def out_img(run, r):
        return np.asarray(Image.open(os.path.join(eval_dir, run, "test_outputs", r["id"] + ".png")))

    if "run0" in runs:
        rows = [[np.asarray(C.load_rgb(r["lr_path"])), out_img("run0", r), np.asarray(C.load_rgb(r["hr_path"]))] for r in shown]
        C.save_labeled_grid(rows, [r["id"] for r in shown], ["LR input", "Run 0 output", "HR target"],
                            "Run 0 | pretrained InstructPix2Pix, no fine-tuning", os.path.join(out_dir, "run0_samples.png"))
    rows = [[np.asarray(C.load_rgb(r["lr_path"]))] + [out_img(run, r) for run in runs if run != "bicubic"]
            + [np.asarray(C.load_rgb(r["hr_path"]))] for r in shown]
    cols = ["LR input (bicubic)"] + [C.RUN_LABELS[run] for run in runs if run != "bicubic"] + ["HR target"]
    C.save_labeled_grid(rows, [r["id"] for r in shown], cols, "Test set: all runs on the same inputs",
                        os.path.join(out_dir, "test_comparison.png"))

    table = []
    for run in runs:
        path = os.path.join(eval_dir, run, "summary.json")
        if os.path.exists(path):
            s = json.load(open(path))
            table.append([C.RUN_LABELS[run], s["num_images"]] + [f"{s[k]['mean']:.4f} ({s[k]['std']:.4f})" for k in ("lpips", "psnr", "ssim")])
    header = ["Run", "N", "LPIPS (lower better)", "PSNR dB (higher better)", "SSIM (higher better)"]
    with open(os.path.join(out_dir, "results_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + table)
    with open(os.path.join(out_dir, "results_table.md"), "w", encoding="utf8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in table:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")
    print(open(os.path.join(out_dir, "results_table.md"), encoding="utf8").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="outputs")
    parser.add_argument("--eval_dir", default="eval")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    loss_curves(args.runs_dir, args.out_dir)
    checkpoint_grids(args.runs_dir, args.data_dir, args.out_dir)
    test_figures(args.eval_dir, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
