"""Evaluate the final frame of every ordered five-frame group."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from competition.metrics import evaluate_detections, save_evaluation
from competition.paths import ROOT, prepared_profile, require_path


def load_predictions(path):
    predictions = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            predictions[Path(row["image"]).name].append(
                [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                    float(row["confidence"]),
                ]
            )
    return predictions


def load_targets(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {row["id"]: row["file_name"] for row in data["images"]}
    targets = defaultdict(list)
    for row in data["annotations"]:
        x, y, width, height = row["bbox"]
        targets[images[row["image_id"]]].append([x, y, x + width, y + height])
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rtdetr"], default="rtdetr")
    parser.add_argument("--profile", default="base")
    parser.add_argument("--predictions")
    parser.add_argument("--groups")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()

    evaluation_root = ROOT / "outputs" / args.model / args.profile / "competition_evaluation"
    prediction_path = Path(args.predictions) if args.predictions else evaluation_root / "test_predictions.csv"
    groups_path = Path(args.groups) if args.groups else prepared_profile(args.profile) / "sequence_groups.csv"
    summary_path = evaluation_root / "test" / "summary.json"
    annotations = prepared_profile(args.profile) / "annotations" / "instances_test.json"
    require_path(prediction_path, "Prediction CSV")
    require_path(groups_path, "Five-frame group CSV")
    require_path(annotations, "Test annotations")

    if args.confidence is None:
        require_path(summary_path, "Single-frame summary containing the validation threshold")
        confidence = json.loads(summary_path.read_text(encoding="utf-8"))["confidence"]
    else:
        confidence = args.confidence

    predictions = load_predictions(prediction_path)
    targets = load_targets(annotations)
    groups = defaultdict(list)
    with groups_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["group_id"]].append((int(row["frame_index"]), Path(row["image"]).name))

    records = []
    incomplete = []
    for group_id, frames in sorted(groups.items()):
        frames.sort()
        if len(frames) != 5:
            incomplete.append(group_id)
            continue
        final_image = frames[-1][1]
        records.append(
            {
                "image": final_image,
                "targets": np.asarray(targets[final_image], dtype=np.float32).reshape(-1, 4),
                "predictions": np.asarray(predictions[final_image], dtype=np.float32).reshape(-1, 5),
            }
        )
    if not records:
        raise RuntimeError("No complete five-frame groups were found")

    result = evaluate_detections(records, confidence, 0.40)
    result.update(
        {
            "groups": len(records),
            "ignored_incomplete_groups": incomplete,
            "method": "final frame of each non-overlapping five-frame group",
            "sequence_order_file": str(groups_path),
        }
    )
    output = Path(args.output) if args.output else evaluation_root / "five_frame"
    save_evaluation(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
