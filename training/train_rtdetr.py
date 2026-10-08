"""PyCharm entry point: train or fine-tune RT-DETR-R18 on a prepared profile."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import subprocess
import sys
from pathlib import Path

from competition.paths import ROOT, load_project_config, prepared_profile, require_path
from competition.rtdetr_runtime import ENGINE, build_config


def main():
    defaults = load_project_config()["rtdetr"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--gpu", default=str(defaults["gpu"]))
    parser.add_argument(
        "--initial-weight",
        help="Explicit optional initialization for research. Omit for basic-item scratch training.",
    )
    parser.add_argument("--scratch", action="store_true", help="Do not use the bundled RT-DETR pretrain")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision when the local GPU produces non-finite training boxes.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.scratch and args.initial_weight:
        parser.error("--scratch cannot be combined with --initial-weight")

    require_path(prepared_profile(args.profile) / "annotations" / "instances_train.json", "Prepared dataset")
    config_path = build_config(args.profile, args.epochs, args.batch_size, args.workers)
    output_dir = ROOT / "outputs" / "rtdetr" / args.profile
    checkpoint = output_dir / "checkpoint.pth"
    if args.no_resume and (checkpoint.exists() or (output_dir / "best.pth").exists()):
        raise FileExistsError("Existing training result: use another profile or resume; refusing overwrite")
    command = [sys.executable, "-u", "tools/train.py", "-c", str(config_path), "--seed", "42"]
    if not args.no_amp:
        command.append("--amp")
    if checkpoint.is_file() and not args.no_resume:
        command.extend(["-r", str(checkpoint)])
        print(f"Resume: {checkpoint}")
    elif args.initial_weight:
        initial = args.initial_weight
        command.extend(["-t", str(require_path(initial, "Initial weight"))])
        print(f"Tune from: {initial}")
    else:
        print("Training from random initialization.")

    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    if os.name == "nt" and sys.version_info[:2] == (3, 8):
        vendor = ROOT / "vendor" / "windows_py38"
        environment["PYTHONPATH"] = str(vendor) + os.pathsep + environment.get("PYTHONPATH", "")
    print("Command:", " ".join(command))
    subprocess.run(command, cwd=str(ENGINE), env=environment, check=True)
    best = output_dir / "best.pth"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without best.pth: {best}")
    print(f"Best weight: {best}")


if __name__ == "__main__":
    main()
