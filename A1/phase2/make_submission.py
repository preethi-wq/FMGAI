# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Pack the Phase 2 submission zip: code, fine-tuning logs, checkpoint samples, test scores, trajectories,
figures and environment.txt. Model weights are left out. Add the report PDF before submitting.

  python make_submission.py --results_dir <results_dir>

<results_dir> is the folder holding outputs/, eval/, figures/, environment.txt, splits.csv and
dataset_summary.json (MyDrive/FMGAI_A1/results on Colab, $RESULTS_DIR on the HPC).
"""
import argparse
import os
import shutil
import tempfile

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_LOGS = ["train_log.csv", "val_log.csv", "val_metrics.csv", "schedule.json", "schedule_check.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--zip_name", default="FMGAI_A1_phase2_submission")
    args = parser.parse_args()
    res = args.results_dir
    missing = []

    with tempfile.TemporaryDirectory() as tmp:
        pack = os.path.join(tmp, "submission")
        shutil.copytree(CODE_DIR, os.path.join(pack, "code"), ignore=shutil.ignore_patterns(
            "data", "figures", "results", "logs", "__pycache__", ".ipynb_checkpoints", "*.out", "*.zip"))

        for run in ["runA", "runB"]:
            d = os.path.join(res, "outputs", run)
            os.makedirs(os.path.join(pack, "logs", run), exist_ok=True)
            for f in RUN_LOGS:
                src = os.path.join(d, f)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(pack, "logs", run))
                else:
                    missing.append(src)
            if os.path.isdir(os.path.join(d, "samples")):
                shutil.copytree(os.path.join(d, "samples"), os.path.join(pack, "checkpoint_samples", run))
            else:
                missing.append(os.path.join(d, "samples"))

        for run in ["bicubic", "run0", "runA", "runB"]:
            d = os.path.join(res, "eval", run)
            os.makedirs(os.path.join(pack, "eval", run), exist_ok=True)
            for f in ["summary.json", "per_image_metrics.csv"]:
                src = os.path.join(d, f)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(pack, "eval", run))
                else:
                    missing.append(src)
            if os.path.isdir(os.path.join(d, "trajectories")):
                shutil.copytree(os.path.join(d, "trajectories"), os.path.join(pack, "trajectories", run))

        if os.path.isdir(os.path.join(res, "figures")):
            shutil.copytree(os.path.join(res, "figures"), os.path.join(pack, "figures"))
        else:
            missing.append(os.path.join(res, "figures"))
        for f in ["environment.txt", "splits.csv", "dataset_summary.json"]:
            src = os.path.join(res, f)
            if os.path.exists(src):
                shutil.copy(src, pack)
            else:
                missing.append(src)

        zip_path = shutil.make_archive(os.path.join(res, args.zip_name), "zip", pack)

    print("wrote", zip_path)
    if missing:
        print("WARNING, not found (the zip is incomplete):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
