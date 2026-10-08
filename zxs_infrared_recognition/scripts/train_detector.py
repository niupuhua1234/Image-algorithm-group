#!/usr/bin/env python3
"""Train an Ultralytics detector with a portable YOLO dataset YAML.

Ultralytics 8.0.196 expects ``data`` to be a YAML file path, not an in-memory
dictionary.  This entry point resolves the dataset root from the source YAML,
writes a per-run runtime YAML without changing the source file, and passes the
runtime YAML path to Ultralytics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector while preserving the source data.yaml."
    )
    parser.add_argument("--model", type=Path, required=True, help="Local model weights or model YAML")
    parser.add_argument(
        "--transfer-from",
        type=Path,
        default=None,
        help=(
            "Optional local pretrained weights loaded into a model YAML before training. "
            "This supports partial parameter transfer to variants such as a P2 detector."
        ),
    )
    parser.add_argument("--data", type=Path, required=True, help="Source YOLO data.yaml")
    parser.add_argument(
        "--previous-data",
        type=Path,
        default=None,
        help=(
            "Previous-stage data.yaml. When provided, current class names must preserve "
            "all old class IDs and may only append new classes."
        ),
    )
    parser.add_argument("--project", type=Path, required=True, help="Experiment output directory")
    parser.add_argument("--name", required=True, help="Unique experiment name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument(
        "--amp-reference",
        type=Path,
        default=None,
        help=(
            "Local yolov8n.pt used by the Ultralytics 8.0.196 AMP check. "
            "Required for offline AMP when --model is not itself yolov8n.pt."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics plots")
    return parser.parse_args()


def load_source_yaml(source_yaml: Path) -> dict[str, Any]:
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {source_yaml}")

    with source_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source_yaml}")
    for required_key in ("train", "val", "names"):
        if required_key not in data:
            raise ValueError(f"Dataset YAML is missing '{required_key}': {source_yaml}")
    if not isinstance(data["names"], (dict, list)):
        raise ValueError("Dataset 'names' must be a mapping or list; class order was not changed")
    return data


def ordered_class_names(data: dict[str, Any]) -> list[str]:
    raw_names = data["names"]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    class_ids = sorted(int(class_id) for class_id in raw_names)
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"Class IDs must be continuous from zero: {class_ids}")
    return [str(raw_names.get(class_id, raw_names.get(str(class_id)))) for class_id in class_ids]


def validate_incremental_class_order(
    previous_data: dict[str, Any], current_data: dict[str, Any]
) -> tuple[list[str], list[str]]:
    previous_names = ordered_class_names(previous_data)
    current_names = ordered_class_names(current_data)
    if len(current_names) < len(previous_names):
        raise ValueError(
            "Incremental dataset removed old classes; old class IDs must be preserved. "
            f"previous={previous_names}, current={current_names}"
        )
    if current_names[: len(previous_names)] != previous_names:
        raise ValueError(
            "Incremental class order changed. New classes may only be appended after all old classes. "
            f"previous={previous_names}, current={current_names}"
        )
    return previous_names, current_names


def resolve_dataset_root(source_yaml: Path, data: dict[str, Any]) -> Path:
    configured_root = Path(str(data.get("path", ".")))
    if not configured_root.is_absolute():
        configured_root = source_yaml.parent / configured_root
    dataset_root = configured_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Resolved dataset root does not exist: {dataset_root}")
    return dataset_root


def validate_split(dataset_root: Path, split_value: Any, split_name: str) -> None:
    split_items = split_value if isinstance(split_value, list) else [split_value]
    for item in split_items:
        split_path = Path(str(item))
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.exists():
            raise FileNotFoundError(
                f"Dataset split '{split_name}' does not exist after resolution: {split_path}"
            )


def write_runtime_yaml(
    source_yaml: Path,
    source_data: dict[str, Any],
    dataset_root: Path,
    project: Path,
    run_name: str,
) -> Path:
    runtime_dir = project / "_runtime_data"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"{run_name}.yaml"
    if runtime_yaml.exists():
        raise FileExistsError(
            f"Runtime YAML already exists; use a new experiment name: {runtime_yaml}"
        )

    runtime_data = dict(source_data)
    runtime_data["path"] = str(dataset_root)
    runtime_data["source_yaml"] = str(source_yaml.resolve())
    with runtime_yaml.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(runtime_data, handle, allow_unicode=True, sort_keys=False)
    return runtime_yaml


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    transfer_from = args.transfer_from.resolve() if args.transfer_from is not None else None
    source_yaml = args.data.resolve()
    project = args.project.resolve()
    run_dir = project / args.name

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Local model file does not exist (automatic download is disabled): {model_path}"
        )
    if transfer_from is not None:
        if model_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("--transfer-from requires --model to be a model YAML")
        if not transfer_from.is_file():
            raise FileNotFoundError(f"Transfer weight does not exist: {transfer_from}")
    if run_dir.exists():
        raise FileExistsError(f"Experiment output already exists; choose a new --name: {run_dir}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0:
        raise ValueError("epochs and imgsz must be positive; batch must not be zero")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1]")

    amp_reference: Path | None = None
    if not args.no_amp:
        if args.amp_reference is not None:
            amp_reference = args.amp_reference.resolve()
        elif model_path.name.lower() == "yolov8n.pt":
            amp_reference = model_path
        else:
            raise ValueError(
                "Offline AMP requires --amp-reference pointing to a local yolov8n.pt, "
                "or use --no-amp."
            )
        if not amp_reference.is_file():
            raise FileNotFoundError(f"Local AMP reference does not exist: {amp_reference}")
        if amp_reference.name.lower() != "yolov8n.pt":
            raise ValueError(
                "Ultralytics 8.0.196 requests the literal filename yolov8n.pt during AMP checks; "
                f"got: {amp_reference}"
            )

    source_data = load_source_yaml(source_yaml)
    current_names = ordered_class_names(source_data)
    if args.previous_data is not None:
        previous_yaml = args.previous_data.resolve()
        previous_data = load_source_yaml(previous_yaml)
        previous_names, current_names = validate_incremental_class_order(
            previous_data=previous_data,
            current_data=source_data,
        )
        appended_names = current_names[len(previous_names) :]
        print("Incremental mode: cumulative replay dataset required")
        print(f"Previous YAML: {previous_yaml}")
        print(f"Preserved classes: {previous_names}")
        print(f"Appended classes: {appended_names}")
    dataset_root = resolve_dataset_root(source_yaml, source_data)
    validate_split(dataset_root, source_data["train"], "train")
    validate_split(dataset_root, source_data["val"], "val")

    runtime_yaml = write_runtime_yaml(
        source_yaml=source_yaml,
        source_data=source_data,
        dataset_root=dataset_root,
        project=project,
        run_name=args.name,
    )

    print(f"Source YAML preserved: {source_yaml}")
    print(f"Current class order: {current_names}")
    print(f"Runtime YAML: {runtime_yaml}")
    print(f"Experiment output: {run_dir}")
    if amp_reference is not None:
        print(f"Offline AMP reference: {amp_reference}")
    if transfer_from is not None:
        print(f"Transfer initialization: {transfer_from}")

    model = YOLO(str(model_path))
    if transfer_from is not None:
        model.load(str(transfer_from))
    original_working_directory = Path.cwd()
    try:
        if amp_reference is not None:
            # Ultralytics 8.0.196 opens the literal relative path "yolov8n.pt"
            # inside check_amp(). Running from the verified reference directory
            # preserves the upstream check without allowing an online fallback.
            os.chdir(amp_reference.parent)
        model.train(
            data=str(runtime_yaml),
            project=str(project),
            name=args.name,
            exist_ok=False,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            fraction=args.fraction,
            cache=args.cache,
            single_cls=args.single_cls,
            amp=not args.no_amp,
            plots=not args.no_plots,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
        )
    finally:
        os.chdir(original_working_directory)

    best_weight = run_dir / "weights" / "best.pt"
    last_weight = run_dir / "weights" / "last.pt"
    print(f"Training complete. best.pt exists: {best_weight.is_file()} ({best_weight})")
    print(f"Training complete. last.pt exists: {last_weight.is_file()} ({last_weight})")


if __name__ == "__main__":
    main()
