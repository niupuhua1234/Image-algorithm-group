"""Tune class-agnostic RT-DETR NMS and confidence on validation predictions only."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from competition.metrics import box_iou, evaluate_detections, save_evaluation, select_threshold
from competition.paths import ROOT, prepared_profile


def load_records(profile, split, prediction_path):
    annotation_path = prepared_profile(profile) / "annotations" / f"instances_{split}.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    targets = defaultdict(list)
    for item in payload["annotations"]:
        x, y, width, height = item["bbox"]
        targets[images[item["image_id"]]["file_name"]].append([x, y, x + width, y + height])
    predictions = defaultdict(list)
    with prediction_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            predictions[row["image"]].append(
                [float(row[key]) for key in ("x1", "y1", "x2", "y2", "confidence")]
            )
    return [
        {
            "image": image["file_name"],
            "targets": np.asarray(targets[image["file_name"]], dtype=np.float32).reshape(-1, 4),
            "predictions": np.asarray(predictions[image["file_name"]], dtype=np.float32).reshape(-1, 5),
        }
        for image in payload["images"]
    ]


def nms(predictions, iou_threshold):
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 5)
    if not len(predictions):
        return predictions
    order = np.argsort(-predictions[:, 4])
    kept = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        overlaps = box_iou(predictions[current, :4], predictions[order[1:], :4])
        order = order[1:][overlaps < iou_threshold]
    return predictions[kept]


def apply_nms(records, threshold):
    return [
        {**record, "predictions": nms(record["predictions"], threshold)} for record in records
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--target-pd", type=float, default=0.90)
    parser.add_argument("--iou", type=float, default=0.40)
    args = parser.parse_args()
    model_output = args.model_output or ROOT / "outputs" / "rtdetr" / args.profile / "competition_evaluation"
    val_records = load_records(args.profile, "val", model_output / "val_predictions.csv")
    test_records = load_records(args.profile, "test", model_output / "test_predictions.csv")

    candidates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
    rows = []
    for nms_iou in candidates:
        filtered = apply_nms(val_records, nms_iou)
        selected, _, reached = select_threshold(filtered, args.iou, args.target_pd)
        rows.append({**selected, "nms_iou": nms_iou, "target_pd_reached": reached})
    feasible = [row for row in rows if row["target_pd_reached"]]
    if feasible:
        best = min(feasible, key=lambda row: (row["false_alarm_rate"], -row["pd"], row["fp"]))
    else:
        best = max(rows, key=lambda row: (row["pd"], -row["false_alarm_rate"]))

    test_filtered = apply_nms(test_records, best["nms_iou"])
    test_summary = evaluate_detections(test_filtered, best["confidence"], args.iou)
    test_summary.update(
        {
            "threshold_mode": "confidence_and_nms_fixed_from_validation",
            "nms_iou": best["nms_iou"],
            "target_pd": args.target_pd,
            "target_pd_reached": test_summary["pd"] >= args.target_pd,
        }
    )
    output = model_output / "postprocess_tuned"
    save_evaluation(output / "test", test_summary)
    validation_summary = dict(best)
    save_evaluation(output / "validation", validation_summary)
    fields = [
        "nms_iou", "confidence", "target_pd_reached", "tp", "fp", "fn", "pd",
        "precision", "false_alarm_rate", "f1",
    ]
    with (output / "validation_grid.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    # Keep per-image matching details in the saved JSON, but avoid flooding the
    # PyCharm/VS Code console with thousands of lines during routine evaluation.
    console_validation = {key: value for key, value in best.items() if key != "details"}
    console_test = {key: value for key, value in test_summary.items() if key != "details"}
    print(
        json.dumps(
            {"validation": console_validation, "test": console_test},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
