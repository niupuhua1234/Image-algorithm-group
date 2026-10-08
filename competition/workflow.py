"""Internal command routing for the six RT-DETR project entrypoints."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOW = "rtdetr_gpt_low_contrast_combined_v1"


def commands():
    return {
        "data-catalog": ["tools/catalog_datasets.py"],
        "data-audit": ["tools/audit_dataset.py"],
        "voc-prepare": ["tools/prepare_voc_dataset.py"],
        "basic-prepare": ["tools/catalog_datasets.py", "--build-base", "--write-xml"],
        "basic-train": ["training/train_rtdetr.py", "--profile", "base", "--scratch",
                        "--gpu", "0", "--batch-size", "1", "--workers", "0", "--no-amp"],
        "basic-test": ["evaluation/test_rtdetr.py", "--profile", "base", "--gpu", "0"],
        "basic-test-historical": ["evaluation/test_rtdetr.py", "--profile", "base", "--gpu", "0",
                                  "--weights", "weights/rtdetr/vedai_real_synth_best.pth"],
        "transfer-support": ["meta_learning/03_prepare_fewshot_support.py", "--profile", LOW,
                             "--shots", "5", "10", "20", "--seed", "20260924"],
        "transfer-train": ["meta_learning/04_adapt_fewshot.py", "--method", "standard",
                           "--profile", LOW, "--shots", "10", "--gpu", "0",
                           "--seed", "20260924", "--initial-weight", "outputs/rtdetr/base/best.pth"],
        "transfer-test": ["meta_learning/05_compare_fewshot.py", "--profile", LOW,
                          "--shots", "10", "--methods", "standard", "--gpu", "0"],
        "incremental-prepare": ["incremental_after_transfer/prepare.py"],
        "incremental-train": ["incremental_after_transfer/train.py", "--gpu", "0"],
        "incremental-test": ["incremental_after_transfer/test.py", "--gpu", "0"],
        "five-frame-test": ["evaluation/evaluate_five_frames.py"],
        "export-rtdetr": ["tools/export_rtdetr_onnx.py"],
    }


def dispatch(action, extra=(), dry_run=False):
    route = commands()[action]
    script = ROOT / route[0]
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, "-B", "-u", *route, *extra]
    print(json.dumps({"cwd": str(ROOT), "command": command}, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return command
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=sorted(commands()))
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.action is None:
        print("\n".join(commands()))
        return
    dispatch(args.action, extra, args.dry_run)


def main_for(action):
    parser = argparse.ArgumentParser(description=f"RT-DETR {action}; extra options go to the implementation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args, extra = parser.parse_known_args()
    dispatch(action, extra, args.dry_run)


if __name__ == "__main__":
    main()
