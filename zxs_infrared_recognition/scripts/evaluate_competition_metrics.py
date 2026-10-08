#!/usr/bin/env python3
"""Evaluate competition-style target detection and recognition rates.

Input files are JSON Lines. Each line has an ``image_id`` and ``objects``:

Ground truth::

    {"image_id":"0001", "objects":[{"bbox":[x1,y1,x2,y2],"class_id":0}]}

Predictions add ``score`` to each object. Matching is one-to-one, in descending
IoU order, with a configurable IoU threshold (0.40 by default).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ObjectBox:
    bbox: tuple[float, float, float, float]
    class_id: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IoU>=0.40 competition metrics.")
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def parse_object(raw: dict[str, Any], prediction: bool) -> ObjectBox:
    bbox_raw = raw.get("bbox")
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        raise ValueError(f"bbox must be a four-element list: {raw}")
    bbox = tuple(float(value) for value in bbox_raw)
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"bbox contains a non-finite value: {bbox}")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2>x1 and y2>y1: {bbox}")

    class_id = int(raw.get("class_id"))
    if class_id < 0:
        raise ValueError(f"class_id must be non-negative: {class_id}")
    score = float(raw.get("score", 1.0 if not prediction else -1.0))
    if prediction and not 0.0 <= score <= 1.0:
        raise ValueError(f"prediction score must be in [0, 1]: {score}")
    return ObjectBox(bbox=bbox, class_id=class_id, score=score)


def read_jsonl(path: Path, prediction: bool) -> dict[str, list[ObjectBox]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: dict[str, list[ObjectBox]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"Each JSONL line must be an object: {path}:{line_number}")
            image_id = str(raw.get("image_id", "")).strip()
            if not image_id:
                raise ValueError(f"Missing image_id: {path}:{line_number}")
            if image_id in records:
                raise ValueError(f"Duplicate image_id '{image_id}': {path}:{line_number}")
            objects_raw = raw.get("objects", [])
            if not isinstance(objects_raw, list):
                raise ValueError(f"objects must be a list: {path}:{line_number}")
            records[image_id] = [parse_object(item, prediction) for item in objects_raw]
    return records


def intersection_over_union(first: ObjectBox, second: ObjectBox) -> float:
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_match(
    ground_truth: list[ObjectBox],
    predictions: list[ObjectBox],
    iou_threshold: float,
    require_same_class: bool = False,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, float, int, int]] = []
    for gt_index, gt_box in enumerate(ground_truth):
        for prediction_index, prediction_box in enumerate(predictions):
            if require_same_class and gt_box.class_id != prediction_box.class_id:
                continue
            iou = intersection_over_union(gt_box, prediction_box)
            if iou >= iou_threshold:
                candidates.append((iou, prediction_box.score, gt_index, prediction_index))
    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, _score, gt_index, prediction_index in candidates:
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index, iou))
    return matches


def evaluate(
    ground_truth_records: dict[str, list[ObjectBox]],
    prediction_records: dict[str, list[ObjectBox]],
    iou_threshold: float,
    confidence_threshold: float,
    max_det: int,
) -> dict[str, Any]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    if max_det <= 0:
        raise ValueError("max_det must be positive")

    unknown_prediction_ids = sorted(set(prediction_records) - set(ground_truth_records))
    if unknown_prediction_ids:
        preview = ", ".join(unknown_prediction_ids[:5])
        raise ValueError(f"Predictions contain image IDs absent from ground truth: {preview}")

    total_gt = 0
    total_predictions = 0
    correctly_detected = 0
    correctly_recognized = 0
    sum_matched_iou = 0.0
    per_image: list[dict[str, Any]] = []

    for image_id, gt_objects in ground_truth_records.items():
        predictions = [
            item
            for item in prediction_records.get(image_id, [])
            if item.score >= confidence_threshold
        ]
        predictions.sort(key=lambda item: item.score, reverse=True)
        predictions = predictions[:max_det]
        detection_matches = greedy_match(gt_objects, predictions, iou_threshold)
        recognition_matches = greedy_match(
            gt_objects,
            predictions,
            iou_threshold,
            require_same_class=True,
        )
        recognized = len(recognition_matches)

        total_gt += len(gt_objects)
        total_predictions += len(predictions)
        correctly_detected += len(detection_matches)
        correctly_recognized += recognized
        sum_matched_iou += sum(iou for _gt, _prediction, iou in detection_matches)
        per_image.append(
            {
                "image_id": image_id,
                "ground_truth": len(gt_objects),
                "predictions": len(predictions),
                "correctly_detected": len(detection_matches),
                "correctly_recognized": recognized,
                "false_positives": len(predictions) - len(detection_matches),
                "missed": len(gt_objects) - len(detection_matches),
            }
        )

    detection_rate = correctly_detected / total_gt if total_gt else 0.0
    recognition_rate = correctly_recognized / total_gt if total_gt else 0.0
    precision = correctly_detected / total_predictions if total_predictions else 0.0
    recognition_precision = correctly_recognized / total_predictions if total_predictions else 0.0
    detection_false_alarm_rate = (
        (total_predictions - correctly_detected) / total_predictions
        if total_predictions
        else 0.0
    )
    recognition_false_alarm_rate = (
        (total_predictions - correctly_recognized) / total_predictions
        if total_predictions
        else 0.0
    )
    mean_matched_iou = sum_matched_iou / correctly_detected if correctly_detected else 0.0

    return {
        "detection_matching": "one_to_one_greedy_descending_iou_class_agnostic",
        "recognition_matching": "one_to_one_greedy_descending_iou_same_class",
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "max_det": max_det,
        "images": len(ground_truth_records),
        "ground_truth_targets": total_gt,
        "predictions": total_predictions,
        "correctly_detected": correctly_detected,
        "correctly_recognized": correctly_recognized,
        "detection_rate": detection_rate,
        "recognition_rate": recognition_rate,
        "detection_precision": precision,
        "recognition_precision": recognition_precision,
        "false_positives": total_predictions - correctly_detected,
        "wrong_recognition": total_predictions - correctly_recognized,
        "detection_false_alarm_rate": detection_false_alarm_rate,
        "recognition_false_alarm_rate": recognition_false_alarm_rate,
        "missed_targets": total_gt - correctly_detected,
        "mean_matched_iou": mean_matched_iou,
        "per_image": per_image,
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def self_test() -> None:
    gt = {
        "a": [
            ObjectBox((0.0, 0.0, 10.0, 10.0), 0, 1.0),
            ObjectBox((20.0, 20.0, 30.0, 30.0), 1, 1.0),
        ],
        "b": [ObjectBox((0.0, 0.0, 5.0, 5.0), 2, 1.0)],
    }
    predictions = {
        "a": [
            ObjectBox((0.0, 0.0, 10.0, 10.0), 0, 0.9),
            ObjectBox((20.0, 20.0, 30.0, 30.0), 9, 0.8),
            ObjectBox((40.0, 40.0, 50.0, 50.0), 1, 0.7),
        ],
        "b": [],
    }
    result = evaluate(gt, predictions, 0.40, 0.001, 300)
    assert result["ground_truth_targets"] == 3
    assert result["correctly_detected"] == 2
    assert result["correctly_recognized"] == 1
    assert math.isclose(result["detection_rate"], 2.0 / 3.0)
    assert math.isclose(result["recognition_rate"], 1.0 / 3.0)

    # A higher-IoU wrong-class box must not steal recognition credit from a
    # valid lower-IoU same-class box. Detection and recognition are matched
    # independently because the competition defines two separate rates.
    competing_predictions = {
        "a": [
            ObjectBox((0.0, 0.0, 10.0, 10.0), 9, 0.9),
            ObjectBox((0.0, 0.0, 8.0, 8.0), 0, 0.8),
        ]
    }
    competing_gt = {"a": [ObjectBox((0.0, 0.0, 10.0, 10.0), 0, 1.0)]}
    competing_result = evaluate(competing_gt, competing_predictions, 0.40, 0.001, 300)
    assert competing_result["correctly_detected"] == 1
    assert competing_result["correctly_recognized"] == 1
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.ground_truth is None or args.predictions is None or args.output is None:
        raise ValueError("--ground-truth, --predictions, and --output are required")

    ground_truth = read_jsonl(args.ground_truth.resolve(), prediction=False)
    predictions = read_jsonl(args.predictions.resolve(), prediction=True)
    result = evaluate(
        ground_truth_records=ground_truth,
        prediction_records=predictions,
        iou_threshold=args.iou,
        confidence_threshold=args.confidence,
        max_det=args.max_det,
    )
    write_json_exclusive(args.output.resolve(), result)

    summary = {key: value for key, value in result.items() if key != "per_image"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
