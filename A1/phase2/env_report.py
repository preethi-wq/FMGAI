# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Print the exact software and GPU versions for the README's Environment table.

  python env_report.py | tee <results_dir>/environment.txt
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

print("python", sys.version.split()[0])
print("platform", platform.platform())
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
for name in ("torchvision", "diffusers", "transformers", "accelerate", "datasets", "bitsandbytes", "scikit-image", "lpips"):
    try:
        print(name, version(name))
    except PackageNotFoundError:  # report and carry on: a missing package shows up in the file
        print(name, "not installed")
