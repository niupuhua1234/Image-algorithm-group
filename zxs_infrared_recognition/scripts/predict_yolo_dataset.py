#!/usr/bin/env python3
"""Run a routed expert over a YOLO split and export prediction JSONL safely."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from audit_yolo_dataset import collect_images, load_dataset_yaml, read_image
from export_yolo_ground_truth import image_id_for
from unified_inference import UnifiedDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unified-detector predictions as JSONL.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, dataset_root, _names = load_dataset_yaml(args.data)
    if args.split not in data:
        raise ValueError(f"Split '{args.split}' is absent from data.yaml")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")

    images = collect_images(dataset_root, data[args.split])
    if args.max_images is not None:
        images = images[: args.max_images]
    output = args.output.resolve()
    partial = output.with_name(output.name + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or partial.exists():
        raise FileExistsError(
            f"Output or partial output already exists; refusing to overwrite: {output}"
        )

    detector = UnifiedDetector(args.registry, device=args.device, use_fp16=not args.fp32)
    detector.synchronize()
    started = time.perf_counter()
    detection_count = 0
    invalid_detection_count = 0
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        for image_path in images:
            detections = detector.predict(args.task, read_image(image_path))
            objects = []
            for detection in detections:
                bbox = [float(value) for value in detection["bbox"]]
                score = float(detection["score"])
                if (
                    len(bbox) != 4
                    or not all(math.isfinite(value) for value in bbox)
                    or not math.isfinite(score)
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                ):
                    # Ultralytics may clip a prediction lying completely outside
                    # the image to a zero-area boundary box. Such a box cannot
                    # match any ground truth and is invalid in the metric schema.
                    invalid_detection_count += 1
                    continue
                objects.append(
                    {
                        "bbox": bbox,
                        "class_id": detection["class_id"],
                        "score": score,
                    }
                )
            detection_count += len(objects)
            record = {"image_id": image_id_for(image_path, dataset_root), "objects": objects}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    detector.synchronize()
    elapsed = time.perf_counter() - started
    partial.replace(output)
    print(
        json.dumps(
            {
                "images": len(images),
                "detections": detection_count,
                "invalid_detections_skipped": invalid_detection_count,
                "elapsed_seconds": elapsed,
                "mean_ms": 1000.0 * elapsed / len(images) if images else 0.0,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
