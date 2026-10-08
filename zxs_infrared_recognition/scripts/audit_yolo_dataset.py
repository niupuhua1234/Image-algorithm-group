#!/usr/bin/env python3
"""Audit a YOLO detection dataset and create boxed QA visualizations safely."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ImageAudit:
    split: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    dtype: str
    channels: int
    boxes: list[Box]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO labels and render QA boxes.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--visualizations-per-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def load_dataset_yaml(path: Path) -> tuple[dict[str, Any], Path, list[str]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")

    configured_root = Path(str(data.get("path", ".")))
    if not configured_root.is_absolute():
        configured_root = path.parent / configured_root
    dataset_root = configured_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
    elif isinstance(raw_names, dict):
        class_ids = sorted(int(class_id) for class_id in raw_names)
        if class_ids != list(range(len(class_ids))):
            raise ValueError(f"Class IDs must be continuous from zero: {class_ids}")
        names = [str(raw_names.get(class_id, raw_names.get(str(class_id)))) for class_id in class_ids]
    else:
        raise ValueError("Dataset YAML names must be a list or mapping")
    return data, dataset_root, names


def resolve_split_items(dataset_root: Path, split_value: Any) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    paths: list[Path] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = dataset_root / path
        paths.append(path.resolve())
    return paths


def collect_images(dataset_root: Path, split_value: Any) -> list[Path]:
    images: list[Path] = []
    for path in resolve_split_items(dataset_root, split_value):
        if path.is_dir():
            images.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    candidate = Path(line.strip())
                    if not candidate.is_absolute():
                        candidate = dataset_root / candidate
                    images.append(candidate.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(f"Unsupported or missing split path: {path}")
    return sorted(set(images), key=lambda item: str(item).lower())


def infer_label_path(image_path: Path, dataset_root: Path) -> Path:
    try:
        relative = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError as error:
        raise ValueError(f"Image is outside dataset root: {image_path}") from error
    parts = list(relative.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot infer label path because 'images' is absent: {relative}")
    image_index = parts.index("images")
    parts[image_index] = "labels"
    return (dataset_root / Path(*parts)).with_suffix(".txt")


def read_boxes(label_path: Path, class_count: int) -> list[Box]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")
    boxes: list[Box] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected 5 label fields at {label_path}:{line_number}")
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if class_id < 0 or class_id >= class_count:
                raise ValueError(f"Class ID out of range at {label_path}:{line_number}: {class_id}")
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"Non-finite coordinate at {label_path}:{line_number}")
            x_center, y_center, width, height = values
            if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
                raise ValueError(f"Invalid normalized size at {label_path}:{line_number}")
            if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
                raise ValueError(f"Invalid normalized center at {label_path}:{line_number}")
            if x_center - width / 2 < -1e-4 or x_center + width / 2 > 1.0001:
                raise ValueError(f"Horizontal box exceeds image at {label_path}:{line_number}")
            if y_center - height / 2 < -1e-4 or y_center + height / 2 > 1.0001:
                raise ValueError(f"Vertical box exceeds image at {label_path}:{line_number}")
            boxes.append(Box(class_id, x_center, y_center, width, height))
    return boxes


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def to_bgr_preview(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        preview = image
    else:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError("Image contains no finite pixel values")
        low, high = np.percentile(finite, [0.5, 99.5])
        if high <= low:
            preview = np.zeros(image.shape, dtype=np.uint8)
        else:
            preview = np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    if preview.ndim == 2:
        return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    if preview.shape[2] == 1:
        return cv2.cvtColor(preview[:, :, 0], cv2.COLOR_GRAY2BGR)
    if preview.shape[2] == 4:
        return cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)
    return preview[:, :, :3].copy()


def render_boxes(audit: ImageAudit, names: list[str]) -> np.ndarray:
    preview = to_bgr_preview(read_image(audit.image_path))
    palette = [(0, 255, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255), (0, 128, 255)]
    for box in audit.boxes:
        x1 = int(round((box.x_center - box.width / 2) * audit.width))
        y1 = int(round((box.y_center - box.height / 2) * audit.height))
        x2 = int(round((box.x_center + box.width / 2) * audit.width))
        y2 = int(round((box.y_center + box.height / 2) * audit.height))
        color = palette[box.class_id % len(palette)]
        thickness = max(1, round(min(audit.width, audit.height) / 350))
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
        label = f"{box.class_id}:{names[box.class_id]}"
        cv2.putText(
            preview,
            label,
            (max(0, x1), max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return preview


def write_image_exclusive(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Visualization already exists: {path}")
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"OpenCV could not encode visualization: {path}")
    path.write_bytes(encoded.tobytes())


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if args.visualizations_per_split < 0:
        raise ValueError("visualizations-per-split must not be negative")
    output.mkdir(parents=True, exist_ok=False)

    data, dataset_root, names = load_dataset_yaml(data_yaml)
    audits: list[ImageAudit] = []
    errors: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    dtype_counts: Counter[str] = Counter()
    channel_counts: Counter[int] = Counter()
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_areas: list[float] = []

    for split in args.splits:
        if split not in data:
            errors.append({"split": split, "path": str(data_yaml), "error": "split missing from data.yaml"})
            continue
        for image_path in collect_images(dataset_root, data[split]):
            try:
                image = read_image(image_path)
                height, width = image.shape[:2]
                channels = 1 if image.ndim == 2 else image.shape[2]
                label_path = infer_label_path(image_path, dataset_root)
                boxes = read_boxes(label_path, len(names))
                audit = ImageAudit(split, image_path, label_path, width, height, str(image.dtype), channels, boxes)
                audits.append(audit)
                split_counts[split] += 1
                dtype_counts[str(image.dtype)] += 1
                channel_counts[channels] += 1
                for box in boxes:
                    class_counts[box.class_id] += 1
                    pixel_width = box.width * width
                    pixel_height = box.height * height
                    box_widths.append(pixel_width)
                    box_heights.append(pixel_height)
                    box_areas.append(pixel_width * pixel_height)
            except Exception as error:  # noqa: BLE001 - audit must record every invalid sample
                errors.append({"split": split, "path": str(image_path), "error": str(error)})

    rng = random.Random(args.seed)
    visualization_counts: Counter[str] = Counter()
    for split in args.splits:
        candidates = [audit for audit in audits if audit.split == split]
        sample_count = min(args.visualizations_per_split, len(candidates))
        selected = rng.sample(candidates, sample_count) if sample_count else []
        split_output = output / "visualizations" / split
        split_output.mkdir(parents=True, exist_ok=True)
        for audit in selected:
            destination = split_output / f"{audit.image_path.stem}.png"
            write_image_exclusive(destination, render_boxes(audit, names))
            visualization_counts[split] += 1

    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "class_names": names,
        "images_valid": len(audits),
        "images_invalid": len(errors),
        "images_by_split": dict(split_counts),
        "objects": sum(class_counts.values()),
        "objects_by_class": {str(class_id): class_counts[class_id] for class_id in range(len(names))},
        "image_dtypes": dict(dtype_counts),
        "image_channels": {str(key): value for key, value in channel_counts.items()},
        "bbox_pixel_width": {
            "min": min(box_widths) if box_widths else None,
            "median": statistics.median(box_widths) if box_widths else None,
            "p95": percentile(box_widths, 0.95),
            "max": max(box_widths) if box_widths else None,
        },
        "bbox_pixel_height": {
            "min": min(box_heights) if box_heights else None,
            "median": statistics.median(box_heights) if box_heights else None,
            "p95": percentile(box_heights, 0.95),
            "max": max(box_heights) if box_heights else None,
        },
        "bbox_pixel_area": {
            "min": min(box_areas) if box_areas else None,
            "median": statistics.median(box_areas) if box_areas else None,
            "p95": percentile(box_areas, 0.95),
            "max": max(box_areas) if box_areas else None,
            "area_le_9": sum(area <= 9.0 for area in box_areas),
            "area_le_25": sum(area <= 25.0 for area in box_areas),
        },
        "visualizations_by_split": dict(visualization_counts),
        "errors": errors,
    }
    with (output / "audit_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "errors"}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
