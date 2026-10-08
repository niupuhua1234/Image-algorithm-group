#!/usr/bin/env python3
"""Create a portable train/val YAML from converted DroneVehicle infrared data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from convert_dronevehicle_to_yolo import CLASS_NAMES


def build_dataset_yaml(dataset_root: Path, output: Path) -> Path:
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Converted infrared dataset does not exist: {dataset_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset YAML: {output}")
    expected_names = dict(enumerate(CLASS_NAMES))
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            split_path = dataset_root / folder / split
            if not split_path.is_dir() or not any(split_path.iterdir()):
                raise FileNotFoundError(f"Missing or empty converted split: {split_path}")
        metadata_path = dataset_root / f"data_{split}.yaml"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
        if not isinstance(metadata, dict) or metadata.get("names") != expected_names:
            raise ValueError(f"Converted class order differs from DroneVehicle: {metadata_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_root = Path(os.path.relpath(dataset_root, output.parent)).as_posix()
    data = {
        "path": relative_root,
        "train": "images/train",
        "val": "images/val",
        "names": expected_names,
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_dataset_yaml(args.dataset_root, args.output))


if __name__ == "__main__":
    main()
