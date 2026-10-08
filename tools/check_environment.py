"""Check Python, CUDA, dependencies, project paths, and copied model sources."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib
import platform
import sys
from pathlib import Path

from competition.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(f"Project: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or newer is required")

    modules = ["numpy", "PIL", "yaml", "cv2", "torch", "torchvision", "scipy"]
    for name in modules:
        module = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(module, '__version__', 'installed')}")
    try:
        importlib.import_module("onnx")
        print("[OK] onnx: installed (model export enabled)")
    except ImportError:
        print("[OPTIONAL] onnx is not installed; training/testing work, ONNX export is disabled")

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"  cuda:{index}: {torch.cuda.get_device_name(index)}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    required = [
        ROOT / "engines" / "rtdetr" / "src",
        ROOT / "engines" / "rtdetr" / "tools" / "train.py",
        ROOT / "project_config.yaml",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[OK] {path.relative_to(ROOT)}")
    from competition.platform_deps import configure_rtdetr_imports
    configure_rtdetr_imports()
    print("[OK] pycocotools: platform-compatible build")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
