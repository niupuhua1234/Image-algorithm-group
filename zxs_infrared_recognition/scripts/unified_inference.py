#!/usr/bin/env python3
"""Unified task router for three resident Ultralytics detection experts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    weight: Path
    imgsz: int
    confidence: float
    nms_iou: float
    max_det: int
    class_alternatives: dict[int, tuple[int, ...]]
    max_output: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one image through a routed expert model.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 inference on CUDA")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_relative(base_file: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_file.parent / path
    return path.resolve()


def load_registry(path: Path) -> dict[str, TaskConfig]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registry does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), dict):
        raise ValueError(f"Registry must contain a tasks mapping: {path}")

    configs: dict[str, TaskConfig] = {}
    for task_id, task_raw in raw["tasks"].items():
        if not isinstance(task_raw, dict):
            raise ValueError(f"Task config must be a mapping: {task_id}")
        weight_value = task_raw.get("weight")
        if not isinstance(weight_value, str) or not weight_value.strip():
            raise ValueError(f"Task '{task_id}' has no weight path")
        weight = resolve_relative(path, weight_value)
        if not weight.is_file():
            raise FileNotFoundError(f"Task '{task_id}' weight does not exist: {weight}")

        raw_alternatives = task_raw.get("class_alternatives", {})
        if not isinstance(raw_alternatives, dict):
            raise ValueError(f"Task '{task_id}' class_alternatives must be a mapping")
        class_alternatives: dict[int, tuple[int, ...]] = {}
        for source_class, alternatives_raw in raw_alternatives.items():
            source_id = int(source_class)
            if not isinstance(alternatives_raw, list) or not alternatives_raw:
                raise ValueError(
                    f"Task '{task_id}' alternatives for class {source_id} must be a non-empty list"
                )
            ordered: list[int] = []
            for class_value in [source_id, *alternatives_raw]:
                class_id = int(class_value)
                if class_id < 0:
                    raise ValueError(f"Task '{task_id}' has a negative alternative class ID")
                if class_id not in ordered:
                    ordered.append(class_id)
            class_alternatives[source_id] = tuple(ordered)

        max_det = int(task_raw.get("max_det", 300))
        config = TaskConfig(
            task_id=str(task_id),
            weight=weight,
            imgsz=int(task_raw.get("imgsz", 640)),
            confidence=float(task_raw.get("confidence", 0.001)),
            nms_iou=float(task_raw.get("nms_iou", 0.70)),
            max_det=max_det,
            class_alternatives=class_alternatives,
            max_output=int(task_raw.get("max_output", max_det)),
        )
        if config.imgsz <= 0 or config.max_det <= 0 or config.max_output <= 0:
            raise ValueError(f"Task '{task_id}' has invalid imgsz, max_det, or max_output")
        if not 0.0 <= config.confidence <= 1.0 or not 0.0 < config.nms_iou <= 1.0:
            raise ValueError(f"Task '{task_id}' has invalid confidence or nms_iou")
        configs[config.task_id] = config
    return configs


class UnifiedDetector:
    """Load all configured experts once and route each image by task ID."""

    def __init__(self, registry: Path, device: str = "0", use_fp16: bool = True) -> None:
        self.registry_path = registry.resolve()
        self.device = device
        self.use_fp16 = use_fp16 and device != "cpu" and torch.cuda.is_available()
        self.configs = load_registry(self.registry_path)
        self.models = {task_id: YOLO(str(config.weight)) for task_id, config in self.configs.items()}

    def synchronize(self) -> None:
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup_all(self) -> dict[str, int]:
        """Materialize every expert on the target device before timed requests."""
        detections_by_task: dict[str, int] = {}
        for task_id, config in self.configs.items():
            dummy = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
            detections_by_task[task_id] = len(self.predict(task_id, dummy))
        self.synchronize()
        return detections_by_task

    def predict(self, task_id: str, image: np.ndarray) -> list[dict[str, Any]]:
        if task_id not in self.models:
            raise KeyError(f"Unknown task_id '{task_id}'. Available: {sorted(self.models)}")
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("image must be a two- or three-dimensional NumPy array")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3:
            raise ValueError(
                "three-dimensional image must have 1, 3, or 4 channels; "
                f"got shape {image.shape}"
            )
        image = np.ascontiguousarray(image)
        config = self.configs[task_id]
        model = self.models[task_id]
        results = model.predict(
            source=image,
            imgsz=config.imgsz,
            conf=config.confidence,
            iou=config.nms_iou,
            max_det=config.max_det,
            device=self.device,
            half=self.use_fp16,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names
        detections: list[dict[str, Any]] = []
        for box, score, class_id in zip(xyxy, confidence, classes):
            class_id = int(class_id)
            alternatives = config.class_alternatives.get(class_id, (class_id,))
            for output_class_id in alternatives:
                if output_class_id not in names:
                    raise ValueError(
                        f"Task '{task_id}' alternative class {output_class_id} is absent from model names"
                    )
                detections.append(
                    {
                        "bbox": [float(value) for value in box.tolist()],
                        "score": float(score),
                        "class_id": output_class_id,
                        "class_name": str(names[output_class_id]),
                    }
                )
                if len(detections) >= config.max_output:
                    return detections
        return detections


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    # imdecode/fromfile also supports Chinese paths on Windows.
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    detector = UnifiedDetector(
        registry=args.registry,
        device=args.device,
        use_fp16=not args.fp32,
    )
    detector.warmup_all()
    detections = detector.predict(args.task, image)
    detector.synchronize()
    payload = {
        "image": str(image_path),
        "task_id": args.task,
        "detections": detections,
    }
    if args.output is not None:
        write_json_exclusive(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
