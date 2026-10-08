#!/usr/bin/env python3
"""Convert official DroneVehicle RGB/infrared polygon XML labels to YOLO HBB.

The source archive is never modified.  By default the converter removes the
official 100-pixel white border (840x712 -> 640x512) and clips annotations to
the retained image area.  RGB and infrared images remain separate datasets so
they can train the two competition experts independently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


CLASS_NAMES = ["car", "freight_car", "truck", "bus", "van"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_ALIASES = {
    "car": "car",
    "freight_car": "freight_car",
    "feright_car": "freight_car",
    "feright": "freight_car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODALITIES = {
    "rgb": ("{split}img", "{split}label"),
    "infrared": ("{split}imgr", "{split}labelr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DroneVehicle XML polygons to YOLO HBB.")
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Official split root, e.g. train=datasets/raw/DroneVehicle/train",
    )
    parser.add_argument("--output", type=Path, help="New parent output directory")
    parser.add_argument(
        "--crop-border",
        type=int,
        default=100,
        help="Crop this many pixels from all four sides; use 0 to preserve 840x712 images.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, pairing, XML structure and classes without writing output.",
    )
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "Resume an interrupted conversion without overwriting files. Existing image/label pairs "
            "are validated against the source before being skipped."
        ),
    )
    return parser.parse_args()


def parse_splits(values: list[str]) -> dict[str, Path]:
    splits: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Split must use NAME=DIR syntax: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if not name or name in splits:
            raise ValueError(f"Split name is empty or duplicated: {name}")
        root = Path(raw_path.strip()).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Split root does not exist: {root}")
        splits[name] = root
    return splits


def find_unique_directory(root: Path, expected_name: str) -> Path:
    direct = root / expected_name
    if direct.is_dir():
        return direct
    candidates = [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_dir() and candidate.name.lower() == expected_name.lower()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{expected_name}' directory under {root}; found {len(candidates)}"
        )
    return candidates[0]


def collect_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    files = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    if not files:
        raise ValueError(f"No matching files found in {directory}")
    return files


def normalized_class_name(raw_name: str) -> str:
    key = "_".join(raw_name.strip().lower().replace("-", " ").split())
    if key not in CLASS_ALIASES:
        raise ValueError(f"Unknown DroneVehicle class name: {raw_name!r}")
    return CLASS_ALIASES[key]


def parse_xml(
    xml_path: Path,
    crop_border: int,
) -> tuple[int, int, list[tuple[int, float, float, float, float]], Counter[str]]:
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Malformed XML: {xml_path}: {error}") from error

    size = root.find("size")
    if size is None or size.findtext("width") is None or size.findtext("height") is None:
        raise ValueError(f"Missing image size in {xml_path}")
    source_width = int(float(size.findtext("width", "0")))
    source_height = int(float(size.findtext("height", "0")))
    width = source_width - 2 * crop_border
    height = source_height - 2 * crop_border
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Crop border {crop_border} is invalid for {source_width}x{source_height}: {xml_path}"
        )

    boxes: list[tuple[int, float, float, float, float]] = []
    stats: Counter[str] = Counter()
    for object_node in root.findall("object"):
        raw_class_name = object_node.findtext("name", "").strip()
        if raw_class_name == "*":
            # The official archive contains a tiny number of explicitly
            # unknown ('*') objects.  They cannot be assigned to the official
            # five-class taxonomy and are excluded from supervised labels.
            stats["skipped_unknown_objects"] += 1
            continue
        class_name = normalized_class_name(raw_class_name)
        polygon = object_node.find("polygon")
        bounding_box = object_node.find("bndbox")
        point = object_node.find("point")
        xs: list[float] = []
        ys: list[float] = []
        if polygon is not None:
            for point_index in range(1, 5):
                x_text = polygon.findtext(f"x{point_index}")
                y_text = polygon.findtext(f"y{point_index}")
                if x_text is None or y_text is None:
                    raise ValueError(f"Incomplete four-point polygon in {xml_path}")
                xs.append(float(x_text) - crop_border)
                ys.append(float(y_text) - crop_border)
            stats["polygon_objects"] += 1
        elif bounding_box is not None:
            xmin = bounding_box.findtext("xmin")
            ymin = bounding_box.findtext("ymin")
            xmax = bounding_box.findtext("xmax")
            ymax = bounding_box.findtext("ymax")
            if None in {xmin, ymin, xmax, ymax}:
                raise ValueError(f"Incomplete bndbox in {xml_path}")
            xs = [float(xmin) - crop_border, float(xmax) - crop_border]
            ys = [float(ymin) - crop_border, float(ymax) - crop_border]
            stats["axis_aligned_objects"] += 1
        elif point is not None:
            # A point does not contain enough information for an IoU-based
            # detection target.  Preserve its count in the report and omit it.
            stats["skipped_point_objects"] += 1
            continue
        else:
            raise ValueError(f"Object has neither polygon nor bndbox in {xml_path}")

        raw_x1, raw_y1, raw_x2, raw_y2 = min(xs), min(ys), max(xs), max(ys)
        x1 = max(0.0, min(float(width), raw_x1))
        y1 = max(0.0, min(float(height), raw_y1))
        x2 = max(0.0, min(float(width), raw_x2))
        y2 = max(0.0, min(float(height), raw_y2))
        if (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2):
            stats["clipped_boxes"] += 1
        if x2 <= x1 or y2 <= y1:
            stats["outside_boxes"] += 1
            continue

        class_id = CLASS_TO_ID[class_name]
        boxes.append((class_id, x1, y1, x2, y2))
        stats["objects"] += 1
        stats[f"class_{class_id}_{class_name}"] += 1
    return width, height, boxes, stats


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def write_image_exclusive(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite image: {path}")
    extension = path.suffix.lower()
    encode_params: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success, encoded = cv2.imencode(extension, image, encode_params)
    if not success:
        raise ValueError(f"OpenCV could not encode image: {path}")
    encoded.tofile(str(path))


def write_lines_exclusive(path: Path, lines: Iterable[str]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def validate_pairing(images: dict[str, Path], labels: dict[str, Path], context: str) -> None:
    image_stems = set(images)
    label_stems = set(labels)
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)[:10]
        missing_images = sorted(label_stems - image_stems)[:10]
        raise ValueError(
            f"Image/XML pairing mismatch for {context}; missing labels={missing_labels}, "
            f"missing images={missing_images}"
        )


def make_yolo_lines(
    boxes: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
) -> list[str]:
    lines: list[str] = []
    for class_id, x1, y1, x2, y2 in boxes:
        x_center = (x1 + x2) / 2.0 / width
        y_center = (y1 + y2) / 2.0 / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return lines


def validate_split(split_name: str, split_root: Path, crop_border: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for modality, (image_pattern, label_pattern) in MODALITIES.items():
        image_dir = find_unique_directory(split_root, image_pattern.format(split=split_name))
        label_dir = find_unique_directory(split_root, label_pattern.format(split=split_name))
        images = collect_files(image_dir, IMAGE_SUFFIXES)
        labels = collect_files(label_dir, {".xml"})
        validate_pairing(images, labels, f"{split_name}/{modality}")
        stats: Counter[str] = Counter()
        sizes: Counter[str] = Counter()
        for stem in sorted(images):
            width, height, boxes, xml_stats = parse_xml(labels[stem], crop_border)
            stats.update(xml_stats)
            stats["images"] += 1
            if not boxes:
                stats["background_images"] += 1
            sizes[f"{width}x{height}"] += 1
        result[modality] = {
            "source_images": str(image_dir),
            "source_labels": str(label_dir),
            "stats": dict(stats),
            "output_sizes": dict(sizes),
        }
    return result


def convert_split(
    split_name: str,
    split_root: Path,
    output: Path,
    crop_border: int,
    resume_output: bool,
) -> dict[str, object]:
    split_report: dict[str, object] = {}
    for modality, (image_pattern, label_pattern) in MODALITIES.items():
        image_dir = find_unique_directory(split_root, image_pattern.format(split=split_name))
        label_dir = find_unique_directory(split_root, label_pattern.format(split=split_name))
        images = collect_files(image_dir, IMAGE_SUFFIXES)
        labels = collect_files(label_dir, {".xml"})
        validate_pairing(images, labels, f"{split_name}/{modality}")

        destination_images = output / modality / "images" / split_name
        destination_labels = output / modality / "labels" / split_name
        destination_images.mkdir(parents=True, exist_ok=resume_output)
        destination_labels.mkdir(parents=True, exist_ok=resume_output)
        stats: Counter[str] = Counter()

        for image_index, stem in enumerate(sorted(images), start=1):
            image_path = images[stem]
            width, height, boxes, xml_stats = parse_xml(labels[stem], crop_border)
            stats.update(xml_stats)
            label_lines = make_yolo_lines(boxes, width, height)
            destination_image = destination_images / image_path.name
            destination_label = destination_labels / f"{stem}.txt"
            image_exists = destination_image.exists()
            label_exists = destination_label.exists()
            if image_exists != label_exists:
                raise FileExistsError(
                    f"Interrupted pair is incomplete; refusing to guess: "
                    f"{destination_image}, {destination_label}"
                )
            if image_exists:
                if not resume_output:
                    raise FileExistsError(f"Refusing to overwrite existing pair: {destination_image}")
                existing_lines = destination_label.read_text(encoding="utf-8").splitlines()
                if existing_lines != label_lines:
                    raise ValueError(f"Existing label differs from source conversion: {destination_label}")
                existing_image = read_image(destination_image)
                if (existing_image.shape[1], existing_image.shape[0]) != (width, height):
                    raise ValueError(f"Existing image has unexpected shape: {destination_image}")
                stats["resumed_pairs"] += 1
                stats["images"] += 1
                if not boxes:
                    stats["background_images"] += 1
                continue

            image = read_image(image_path)
            source_height, source_width = image.shape[:2]
            if (source_width - 2 * crop_border, source_height - 2 * crop_border) != (width, height):
                raise ValueError(
                    f"Image/XML size mismatch after crop for {image_path}: "
                    f"image={source_width}x{source_height}, XML output={width}x{height}"
                )

            if crop_border:
                cropped = image[
                    crop_border : source_height - crop_border,
                    crop_border : source_width - crop_border,
                ]
                write_image_exclusive(destination_image, cropped)
            else:
                try:
                    os.link(str(image_path), str(destination_image))
                except OSError:
                    shutil.copy2(str(image_path), str(destination_image))
            write_lines_exclusive(destination_label, label_lines)
            stats["images"] += 1
            if not boxes:
                stats["background_images"] += 1
            if image_index % 1000 == 0:
                print(f"{split_name}/{modality}: {image_index}/{len(images)} images", flush=True)

        modality_root = output / modality
        data_yaml = {
            "path": ".",
            split_name: f"images/{split_name}",
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }
        data_yaml_path = modality_root / f"data_{split_name}.yaml"
        if data_yaml_path.exists():
            if not resume_output:
                raise FileExistsError(f"Refusing to overwrite metadata: {data_yaml_path}")
            with data_yaml_path.open("r", encoding="utf-8") as handle:
                if yaml.safe_load(handle) != data_yaml:
                    raise ValueError(f"Existing dataset YAML differs from expected: {data_yaml_path}")
        else:
            with data_yaml_path.open("x", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(data_yaml, handle, allow_unicode=True, sort_keys=False)
        split_report[modality] = {
            "source_images": str(image_dir),
            "source_labels": str(label_dir),
            "stats": dict(stats),
            "output_image_size": [width, height],
        }
    return split_report


def main() -> None:
    args = parse_args()
    if args.crop_border < 0:
        raise ValueError("--crop-border must be non-negative")
    splits = parse_splits(args.split)

    validation = {
        split_name: validate_split(split_name, split_root, args.crop_border)
        for split_name, split_root in splits.items()
    }
    if args.validate_only:
        print(json.dumps({"class_names": CLASS_NAMES, "splits": validation}, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")

    output = args.output.resolve()
    if output.exists():
        if not args.resume_output or not output.is_dir():
            raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
        if (output / "reports" / "conversion_report.json").exists():
            raise FileExistsError(f"Conversion report already exists; output is complete: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)

    report: dict[str, object] = {
        "class_names": CLASS_NAMES,
        "annotation_conversion": "oriented four-point polygon to clipped horizontal bounding box",
        "crop_border_pixels": args.crop_border,
        "source_data_unchanged": True,
        "validation": validation,
        "splits": {},
    }
    for split_name, split_root in splits.items():
        report["splits"][split_name] = convert_split(
            split_name=split_name,
            split_root=split_root,
            output=output,
            crop_border=args.crop_border,
            resume_output=args.resume_output,
        )

    reports_dir = output / "reports"
    reports_dir.mkdir(exist_ok=False)
    with (reports_dir / "conversion_report.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
