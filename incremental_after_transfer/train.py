"""Run two sequential stages from the old-low-contrast ordinary-transfer weight."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs-stage1", type=int, default=2)
    parser.add_argument("--epochs-stage2", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    from competition import incremental_trainer as trainer
    trainer.OUTPUT = ROOT / "outputs/incremental_after_transfer"
    first = trainer.train_stage("stage1", "inc_after_transfer_stage1", SOURCE, args.epochs_stage1, args)
    second = trainer.train_stage("stage2", "inc_after_transfer_stage2", first, args.epochs_stage2, args)
    print(f"Stage 1: {first}\nStage 2: {second}", flush=True)


if __name__ == "__main__":
    main()
