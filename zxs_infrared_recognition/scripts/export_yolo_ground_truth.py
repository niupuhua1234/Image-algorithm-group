#!/usr/bin/env python3
"""Export one YOLO dataset split to the competition JSONL ground-truth schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_yolo_dataset import (
    collect_images,
    infer_label_path,
    load_dataset_yaml,
    read_boxes,
    read_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO labels as pixel-xyxy JSONL.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def image_id_for(image_path: Path, dataset_root: Path) -> str:
    return image_path.resolve().relative_to(dataset_root.resolve()).as_posix()


def main() -> None:
    args = parse_args()
    data, dataset_root, names = load_dataset_yaml(args.data)
    if args.split not in data:
        raise ValueError(f"Split '{args.split}' is absent from data.yaml")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")

    images = collect_images(dataset_root, data[args.split])
    if args.max_images is not None:
        images = images[: args.max_images]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    object_count = 0
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for image_path in images:
            image = read_image(image_path)
            height, width = image.shape[:2]
            label_path = infer_label_path(image_path, dataset_root)
            boxes = read_boxes(label_path, len(names))
            objects = []
            for box in boxes:
                x1 = (box.x_center - box.width / 2.0) * width
                y1 = (box.y_center - box.height / 2.0) * height
                x2 = (box.x_center + box.width / 2.0) * width
                y2 = (box.y_center + box.height / 2.0) * height
                objects.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": box.class_id,
                    }
                )
            object_count += len(objects)
            record = {"image_id": image_id_for(image_path, dataset_root), "objects": objects}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"images": len(images), "objects": object_count, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
