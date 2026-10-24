# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Build the paired 4x super-resolution dataset from UC Merced Land Use.

For each of the 2,100 images:
  HR target = the original RGB tile, resized to 256x256 (bicubic) only for the 44 off-size tiles
  LR input  = HR bicubic-downsampled to 64x64, then bicubic-upsampled back to 256x256
Split: stratified by class, 70 / 15 / 15 images per class (1470 / 315 / 315), seeded.

Output layout (data_dir):
  hr/<id>.png  lr/<id>.png  splits.csv (id, class, split, orig_width, orig_height)  dataset_summary.json

Usage:
  python prepare_data.py --data_dir data                    # Hugging Face mirror (default)
  python prepare_data.py --data_dir data --source official  # zip from the UC Merced site
"""
import argparse
import collections
import csv
import io
import json
import os
import re
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from sr_common import RESOLUTION, SEED

SCALE = 4
PER_CLASS_SPLIT = {"train": 70, "val": 15, "test": 15}
HF_REPO = "blanchon/UC_Merced"
HF_REVISION = "b79f9bfb55cd7731306d583c3e072c8c010bac61"
HF_FILE = "data/train-00000-of-00001.parquet"
OFFICIAL_ZIP = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"


def iter_hf(cache_dir):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", revision=HF_REVISION, cache_dir=cache_dir)
    for item in pq.read_table(path, columns=["image"]).column("image").to_pylist():
        yield item["path"], item["bytes"]


def iter_official(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "UCMerced_LandUse.zip")
    if not os.path.exists(path):
        urllib.request.urlretrieve(OFFICIAL_ZIP, path)
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith(".tif"):
                yield os.path.basename(name), z.read(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--source", choices=["hf", "official"], default="hf")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    for sub in ("hr", "lr"):
        os.makedirs(os.path.join(args.data_dir, sub), exist_ok=True)
    cache = args.cache_dir or os.path.join(args.data_dir, "_download")
    source = iter_hf(cache) if args.source == "hf" else iter_official(cache)

    records, resized = [], []
    small = RESOLUTION // SCALE
    for filename, data in source:
        stem = os.path.splitext(filename)[0]
        cls = re.sub(r"\d+$", "", stem)
        hr = Image.open(io.BytesIO(data)).convert("RGB")
        orig_w, orig_h = hr.size
        if hr.size != (RESOLUTION, RESOLUTION):
            hr = hr.resize((RESOLUTION, RESOLUTION), Image.Resampling.BICUBIC)
            resized.append({"id": stem, "size": [orig_w, orig_h]})
        lr = hr.resize((small, small), Image.Resampling.BICUBIC).resize((RESOLUTION, RESOLUTION), Image.Resampling.BICUBIC)
        hr.save(os.path.join(args.data_dir, "hr", stem + ".png"))
        lr.save(os.path.join(args.data_dir, "lr", stem + ".png"))
        records.append({"id": stem, "class": cls, "orig_width": orig_w, "orig_height": orig_h})

    # Stratified split: shuffle each class's ids with a fixed seed, then cut 70 / 15 / 15.
    rng = np.random.default_rng(SEED)
    by_class = collections.defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda r: r["id"])
        order = rng.permutation(len(rows))
        n_train, n_val = PER_CLASS_SPLIT["train"], PER_CLASS_SPLIT["val"]
        for rank, idx in enumerate(order):
            rows[idx]["split"] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    records.sort(key=lambda r: (r["class"], r["id"]))
    with open(os.path.join(args.data_dir, "splits.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "split", "orig_width", "orig_height"])
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "source": f"{HF_REPO}@{HF_REVISION}" if args.source == "hf" else OFFICIAL_ZIP,
        "num_images": len(records),
        "num_classes": len(by_class),
        "split_sizes": dict(collections.Counter(r["split"] for r in records)),
        "hr_resolution": RESOLUTION,
        "lr_resolution": small,
        "degradation": f"bicubic {RESOLUTION}->{small}->{RESOLUTION} (PIL)",
        "num_resized_off_size": len(resized),
        "resized_off_size": resized,
        "seed": SEED,
    }
    with open(os.path.join(args.data_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "resized_off_size"}, indent=1))


if __name__ == "__main__":
    main()
