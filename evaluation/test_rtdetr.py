"""PyCharm entry point: select threshold on val and test RT-DETR at IoU 0.40."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

from competition.paths import ROOT, load_project_config


def main():
    config = load_project_config()
    defaults = config["rtdetr"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--weights")
    parser.add_argument("--gpu", default=str(defaults["gpu"]))
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES must be set before importing/initializing a CUDA model.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from competition.rtdetr_runtime import evaluate_model
    checkpoint = args.weights or str(ROOT / "outputs" / "rtdetr" / args.profile / "best.pth")
    result = evaluate_model(
        profile=args.profile,
        checkpoint=checkpoint,
        epochs=defaults["epochs"],
        batch_size=defaults["batch_size"],
        workers=defaults["workers"],
        iou=config["evaluation"]["match_iou"],
        target_pd=config["evaluation"]["target_pd"],
    )
    print(f"Test summary: {result}")


if __name__ == "__main__":
    main()
