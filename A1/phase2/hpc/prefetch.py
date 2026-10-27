# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Download, on the login node, every remote file the jobs read, so they run with HF_HUB_OFFLINE=1:
the InstructPix2Pix pipeline (fp32 safetensors only, no safety checker: about 4.3 GB instead of the full 30 GB repo),
the UC Merced parquet used by prepare_data.py, and the AlexNet weights LPIPS loads through torchvision.
Run by hpc/setup_env.sh with HF_HOME and TORCH_HOME already set.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

import prepare_data  # noqa: E402
from sr_common import PRETRAINED_MODEL  # noqa: E402

MODEL_FILES = ["*.json", "*.txt", "*/model.safetensors", "*/diffusion_pytorch_model.safetensors"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    # The safety checker is never loaded (infer.py and the final save pass safety_checker=None).
    path = snapshot_download(PRETRAINED_MODEL, allow_patterns=MODEL_FILES, ignore_patterns=["safety_checker/*"])
    print("model:", path)

    path = hf_hub_download(
        prepare_data.HF_REPO, prepare_data.HF_FILE, repo_type="dataset", revision=prepare_data.HF_REVISION,
        cache_dir=os.path.join(args.data_dir, "_download"),  # where prepare_data.py looks by default
    )
    print("dataset:", path)

    import lpips

    lpips.LPIPS(net="alex", verbose=False)
    print("lpips: AlexNet weights cached in", os.environ.get("TORCH_HOME", "~/.cache/torch"))


if __name__ == "__main__":
    main()
