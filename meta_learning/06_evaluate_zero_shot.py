"""Evaluate an unadapted RT-DETR source weight on a target-domain profile.

The confidence and class-agnostic NMS thresholds are selected only on the
validation split. The fixed pair is then applied once to the test split.
"""

import argparse
import csv
import json
import os
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--target-pd", type=float, default=0.90)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def save_predictions(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["image", "confidence", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for x1, y1, x2, y2, confidence in record["predictions"]:
                writer.writerow(
                    {
                        "image": record["image"],
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )


def main():
    args = parse_args()
    if not args.weight.is_file():
        raise FileNotFoundError(f"Source weight does not exist: {args.weight}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from common import OUTPUT_ROOT, build_yaml_config
    from competition.metrics import evaluate_detections, save_evaluation, select_threshold
    from competition.rtdetr_runtime import infer_split

    apply_nms = import_module("competition.postprocess").apply_nms
    nms_candidates = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)
    config_path = build_yaml_config(args.profile, epochs=1, batch_size=1, workers=0)[1]
    output = OUTPUT_ROOT / "evaluation" / args.profile / "zero_shot"

    val_records, val_speed = infer_split(
        config_path, args.weight, args.profile, "val", warmup=args.warmup
    )
    candidates = []
    sweeps = {}
    for nms_iou in nms_candidates:
        selected, sweep, reached = select_threshold(
            apply_nms(val_records, nms_iou), args.iou, args.target_pd
        )
        candidates.append({**selected, "nms_iou": nms_iou, "target_pd_reached": reached})
        sweeps[str(nms_iou)] = sweep

    feasible = [item for item in candidates if item["target_pd_reached"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda item: (item["false_alarm_rate"], -item["pd"], item["fp"]),
        )
    else:
        selected = max(
            candidates, key=lambda item: (item["pd"], -item["false_alarm_rate"])
        )

    val_summary = dict(selected)
    val_summary.update({"target_pd": args.target_pd, "speed": val_speed})
    save_predictions(output / "val_predictions.csv", val_records)
    save_evaluation(output / "validation", val_summary, sweeps[str(selected["nms_iou"])])
    (output / "validation" / "nms_grid.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test_records, test_speed = infer_split(
        config_path, args.weight, args.profile, "test", warmup=args.warmup
    )
    test_summary = evaluate_detections(
        apply_nms(test_records, selected["nms_iou"]),
        selected["confidence"],
        args.iou,
    )
    test_summary.update(
        {
            "method": "zero_shot",
            "shots": 0,
            "threshold_mode": "confidence_and_nms_fixed_from_validation",
            "nms_iou": selected["nms_iou"],
            "target_pd": args.target_pd,
            "target_pd_reached": test_summary["pd"] >= args.target_pd,
            "speed": test_speed,
            "weight": str(args.weight.resolve()),
        }
    )
    save_predictions(output / "test_predictions.csv", test_records)
    save_evaluation(output / "test", test_summary)
    (output / "result.json").write_text(
        json.dumps(test_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console_result = {key: value for key, value in test_summary.items() if key != "details"}
    print(json.dumps(console_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
