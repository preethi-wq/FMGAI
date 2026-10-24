# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Schedule Analysis figures and table for the report (numpy + matplotlib only, no GPU).

  python plot_schedule.py --out_dir figures

Rebuilds the native InstructPix2Pix schedule (scaled_linear: beta_t = linspace(sqrt 0.00085, sqrt 0.012, 1000)^2),
reads EXPONENT_X from train_instruct_pix2pix.py, checks theta_t = beta_t in (0, 1), and writes:
  schedule_betas.png        {beta_t} and {beta_t^x} against t on the same axes (linear and log scale)
  schedule_alphabar_snr.png alpha_bar_t and SNR_t = alpha_bar_t / (1 - alpha_bar_t) for both schedules
  schedule_table.csv / .md  values at selected timesteps
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BETA_START, BETA_END, T = 0.00085, 0.012, 1000


def read_exponent():
    src = open(os.path.join(HERE, "train_instruct_pix2pix.py"), encoding="utf8").read()
    return float(re.search(r"^EXPONENT_X\s*=\s*([0-9.]+)", src, flags=re.M).group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    x = read_exponent()
    t = np.arange(1, T + 1)
    beta = np.linspace(BETA_START**0.5, BETA_END**0.5, T, dtype=np.float32).astype(np.float64) ** 2
    beta_x = beta**x
    assert np.all((beta > 0) & (beta < 1)) and np.all((beta_x > 0) & (beta_x < 1)), "theta_t must lie in (0, 1)"
    abar, abar_x = np.cumprod(1 - beta), np.cumprod(1 - beta_x)
    snr, snr_x = abar / (1 - abar), abar_x / (1 - abar_x)
    print(f"x = {x:.2f}; beta in [{beta.min():.6g}, {beta.max():.6g}]; beta^x in [{beta_x.min():.6g}, {beta_x.max():.6g}]")
    print(f"alpha_bar_T: {abar[-1]:.6g} -> {abar_x[-1]:.6g};  SNR_T: {snr[-1]:.6g} -> {snr_x[-1]:.6g}")

    lab, lab_x = r"$\theta_t=\beta_t$ (Run A)", rf"$\theta_t^{{x}}=\beta_t^{{{x:.2f}}}$ (Run B)"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scale in zip(axes, ["linear", "log"]):
        ax.plot(t, beta, label=lab)
        ax.plot(t, beta_x, label=lab_x)
        ax.set_yscale(scale)
        ax.set_xlabel("timestep t")
        ax.set_ylabel(r"per-step noise variance $\theta_t$")
        ax.set_title(f"Noise schedules ({scale} scale)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "schedule_betas.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, abar, label="Run A")
    axes[0].plot(t, abar_x, label="Run B")
    axes[0].set_ylabel(r"$\bar\alpha_t=\prod_{s\leq t}(1-\theta_s)$")
    axes[0].set_title("Signal kept after t forward steps")
    axes[1].plot(t, snr, label="Run A")
    axes[1].plot(t, snr_x, label="Run B")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"SNR$_t=\bar\alpha_t/(1-\bar\alpha_t)$")
    axes[1].set_title("Signal-to-noise ratio")
    for ax in axes:
        ax.set_xlabel("timestep t")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "schedule_alphabar_snr.png"), dpi=150)
    plt.close(fig)

    cols = ["t", "beta_t", "beta_t^x", "alpha_bar_t", "alpha_bar_t (x)", "SNR_t", "SNR_t (x)"]
    rows = []
    for ti in [1, 100, 250, 500, 750, 1000]:
        i = ti - 1
        rows.append([ti, beta[i], beta_x[i], abar[i], abar_x[i], snr[i], snr_x[i]])
    with open(os.path.join(args.out_dir, "schedule_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows([cols] + rows)
    with open(os.path.join(args.out_dir, "schedule_table.md"), "w", encoding="utf8") as f:
        f.write(f"Exponent x = {x:.2f}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write(f"| {r[0]} | " + " | ".join(f"{v:.4g}" for v in r[1:]) + " |\n")
    print(open(os.path.join(args.out_dir, "schedule_table.md"), encoding="utf8").read())


if __name__ == "__main__":
    main()
