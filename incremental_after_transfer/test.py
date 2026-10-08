"""Evaluate transfer, stage 1, and stage 2 with validation-fixed thresholds."""

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/incremental_after_transfer/evaluation"
SOURCE = ROOT / "outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth"
WEIGHTS = (("transfer", SOURCE),
           ("stage1", ROOT / "outputs/incremental_after_transfer/stage1/best.pth"),
           ("stage2", ROOT / "outputs/incremental_after_transfer/stage2/best.pth"))


def prediction_rows(path, records, confidence):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "confidence", "x1", "y1", "x2", "y2"))
        for record in records:
            for x1, y1, x2, y2, score in record["predictions"]:
                if score >= confidence:
                    writer.writerow((record["image"], score, x1, y1, x2, y2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from competition.metrics import evaluate_detections, save_evaluation, select_threshold
    from competition.rtdetr_runtime import build_config, infer_split

    apply_nms = importlib.import_module("competition.postprocess").apply_nms
    profile = "inc_after_transfer_stage2"
    config = build_config(profile, 1, 1, 0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weight in WEIGHTS:
        if not weight.is_file():
            raise FileNotFoundError(weight)
        target = OUTPUT / name
        target.mkdir(exist_ok=True)
        val, _ = infer_split(config, weight, profile, "val", warmup=args.warmup)
        seen = {"transfer": ("old",), "stage1": ("old", "v3"),
                "stage2": ("old", "v3", "new")}[name]
        selection_val = [record for record in val if record["image"].split("__", 1)[0] in seen]
        candidates = []
        for nms in (0.05, 0.1, 0.2, 0.4):
            selected, _, reached = select_threshold(apply_nms(selection_val, nms), 0.40, 0.90)
            candidates.append((reached, selected["false_alarm_rate"], -selected["pd"],
                               -selected["confidence"], nms, selected["confidence"]))
        feasible = [item for item in candidates if item[0]]
        chosen = min(feasible, key=lambda item: item[1:4]) if feasible else min(
            candidates, key=lambda item: (item[2], item[1], item[3]))
        nms, confidence = chosen[4], chosen[5]
        validation = evaluate_detections(apply_nms(selection_val, nms), confidence, 0.40)
        validation.update({"nms_iou": nms, "threshold_mode": "fixed_from_validation",
                           "target_pd_reached": chosen[0], "seen_domains": seen})
        save_evaluation(target / "validation", validation)
        test, speed = infer_split(config, weight, profile, "test", warmup=args.warmup)
        test = apply_nms(test, nms)
        prediction_rows(target / "test_predictions.csv", test, confidence)
        for domain in ("old", "v3", "new"):
            subset = [record for record in test if record["image"].startswith(domain + "__")]
            result = evaluate_detections(subset, confidence, 0.40)
            result.update({"nms_iou": nms, "threshold_mode": "fixed_from_validation",
                           "weight": str(weight), "domain": domain, "speed": speed,
                           "five_frame_status": "unavailable: independent still images, not a temporal sequence"})
            save_evaluation(target / ("test_" + domain), result)
            row = {"run": name, "domain": domain, "images": result["images"],
                   "targets": result["targets"], "tp": result["tp"], "fp": result["fp"],
                   "fn": result["fn"], "pd": result["pd"], "far": result["false_alarm_rate"],
                   "f1": result["f1"], "confidence": confidence, "nms_iou": nms,
                   "pipeline_ms_mean": speed["pipeline_ms_mean"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {OUTPUT / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
