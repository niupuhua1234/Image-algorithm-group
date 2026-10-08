#!/usr/bin/env python3
"""Sweep confidence thresholds using the competition IoU/recognition metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_competition_metrics import evaluate, read_jsonl, write_json_exclusive


DEFAULT_THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a confidence threshold on a validation set.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-metric", choices=["detection_rate", "recognition_rate"], required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def select_threshold(rows: list[dict[str, Any]], metric: str, target: float) -> tuple[dict[str, Any], bool]:
    eligible = [row for row in rows if row[metric] >= target]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["confidence_threshold"],
                row[metric],
                -float(row.get("recognition_false_alarm_rate", row.get("detection_false_alarm_rate", 0.0))),
                row["detection_precision"],
            ),
        )
        return selected, True
    selected = max(
        rows,
        key=lambda row: (
            row[metric],
            -float(row.get("recognition_false_alarm_rate", row.get("detection_false_alarm_rate", 0.0))),
            row["detection_precision"],
            row["confidence_threshold"],
        ),
    )
    return selected, False


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be unique values in [0, 1]")

    ground_truth = read_jsonl(args.ground_truth.resolve(), prediction=False)
    predictions = read_jsonl(args.predictions.resolve(), prediction=True)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        result = evaluate(
            ground_truth_records=ground_truth,
            prediction_records=predictions,
            iou_threshold=args.iou,
            confidence_threshold=threshold,
            max_det=args.max_det,
        )
        rows.append({key: value for key, value in result.items() if key != "per_image"})

    selected, target_met = select_threshold(rows, args.task_metric, args.target)
    payload = {
        "task_metric": args.task_metric,
        "target": args.target,
        "target_met": target_met,
        "selection_rule": (
            "highest confidence threshold meeting target, then lower recognition false alarm rate; "
            "otherwise maximum metric, then lower false alarm rate, then precision"
        ),
        "selected": selected,
        "sweep": rows,
    }
    write_json_exclusive(args.output.resolve(), payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "sweep"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
