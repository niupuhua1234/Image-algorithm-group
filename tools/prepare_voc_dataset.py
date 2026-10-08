"""PyCharm entry point: validate and convert VOC data for RT-DETR."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from competition.paths import ROOT
from competition.prepare import prepare_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        help="VOC source directory; may be repeated. Default: datasets/source/base",
    )
    parser.add_argument("--profile", default="base", help="Name below datasets/prepared")
    parser.add_argument("--split-mode", choices=["auto", "existing", "random"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-other-size",
        action="store_true",
        help="Disable the official 1280x1024 image-size check (diagnostics only).",
    )
    args = parser.parse_args()

    sources = args.source or [str(ROOT / "datasets" / "source" / "base")]
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    summary = prepare_profile(
        sources=sources,
        output_dir=ROOT / "datasets" / "prepared" / args.profile,
        split_mode=args.split_mode,
        ratios=ratios,
        seed=args.seed,
        overwrite=args.overwrite,
        expected_size=None if args.allow_other_size else (1280, 1024),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
