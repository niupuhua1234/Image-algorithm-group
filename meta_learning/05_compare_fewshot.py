"""Evaluate ordinary and FOMAML few-shot weights with fixed val thresholds."""

import argparse
import csv
import json
import os
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    import yaml

    config = yaml.safe_load((ROOT / "project_config.yaml").read_text(encoding="utf-8"))
    defaults = config["meta_learning"]
    evaluation = config["evaluation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=defaults["target_profile"])
    parser.add_argument("--shots", type=int, nargs="+", default=defaults["shots"])
    parser.add_argument("--gpu", default=str(defaults["gpu"]))
    parser.add_argument("--iou", type=float, default=evaluation["match_iou"])
    parser.add_argument("--target-pd", type=float, default=evaluation["target_pd"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--run-suffix", default="", help="Optional adaptation run suffix")
    parser.add_argument("--methods", nargs="+", choices=["standard", "fomaml"],
                        default=["standard", "fomaml"])
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from common import OUTPUT_ROOT, build_yaml_config
    from competition.metrics import evaluate_detections, save_evaluation, select_threshold
    from competition.rtdetr_runtime import infer_split

    apply_nms = import_module("competition.postprocess").apply_nms
    nms_candidates = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)
    config_path = build_yaml_config(args.profile, epochs=1, batch_size=1, workers=0)[1]
    summary_rows = []
    for shots in sorted(set(args.shots)):
        for method in args.methods:
            run_name = f"{method}_{shots:03d}shot{args.run_suffix}"
            checkpoint = OUTPUT_ROOT / "adaptation" / args.profile / run_name / "best.pth"
            if not checkpoint.is_file():
                print(f"Skip missing weight: {checkpoint}")
                continue
            output = OUTPUT_ROOT / "evaluation" / args.profile / run_name
            val_records, val_speed = infer_split(
                config_path, checkpoint, args.profile, "val", warmup=args.warmup
            )
            candidates = []
            sweeps = {}
            for nms_iou in nms_candidates:
                selected, sweep, reached = select_threshold(
                    apply_nms(val_records, nms_iou), args.iou, args.target_pd
                )
                candidates.append(
                    {**selected, "nms_iou": nms_iou, "target_pd_reached": reached}
                )
                sweeps[str(nms_iou)] = sweep
            feasible = [item for item in candidates if item["target_pd_reached"]]
            if feasible:
                selected = min(
                    feasible,
                    key=lambda item: (
                        item["false_alarm_rate"], -item["pd"], item["fp"]
                    ),
                )
            else:
                selected = max(
                    candidates,
                    key=lambda item: (item["pd"], -item["false_alarm_rate"]),
                )
            val_summary = dict(selected)
            val_summary.update(
                {"target_pd": args.target_pd, "speed": val_speed}
            )
            save_predictions(output / "val_predictions.csv", val_records)
            save_evaluation(
                output / "validation",
                val_summary,
                sweeps[str(selected["nms_iou"])],
            )
            (output / "validation" / "nms_grid.json").write_text(
                json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            test_records, test_speed = infer_split(
                config_path, checkpoint, args.profile, "test", warmup=args.warmup
            )
            test_summary = evaluate_detections(
                apply_nms(test_records, selected["nms_iou"]),
                selected["confidence"],
                args.iou,
            )
            test_summary.update(
                {
                    "method": method,
                    "shots": shots,
                    "threshold_mode": "confidence_and_nms_fixed_from_validation",
                    "nms_iou": selected["nms_iou"],
                    "target_pd": args.target_pd,
                    "target_pd_reached": test_summary["pd"] >= args.target_pd,
                    "speed": test_speed,
                }
            )
            save_predictions(output / "test_predictions.csv", test_records)
            save_evaluation(output / "test", test_summary)
            summary_rows.append(
                {
                    "method": method,
                    "shots": shots,
                    "confidence": selected["confidence"],
                    "nms_iou": selected["nms_iou"],
                    "tp": test_summary["tp"],
                    "fp": test_summary["fp"],
                    "fn": test_summary["fn"],
                    "pd": test_summary["pd"],
                    "false_alarm_rate": test_summary["false_alarm_rate"],
                    "f1": test_summary["f1"],
                    "model_ms_mean": test_speed["model_ms_mean"],
                    "pipeline_ms_mean": test_speed["pipeline_ms_mean"],
                    "weight": str(checkpoint),
                }
            )
            print(json.dumps(summary_rows[-1], indent=2, ensure_ascii=False))

    if not summary_rows:
        raise FileNotFoundError("No adapted best.pth files were found. Run 04_adapt_fewshot.py.")
    output = OUTPUT_ROOT / "evaluation" / args.profile
    output.mkdir(parents=True, exist_ok=True)
    # A one-method check must not replace the historical two-method comparison.
    basename = "comparison" if set(args.methods) == {"standard", "fomaml"} else "comparison_" + "_".join(args.methods)
    with (output / f"{basename}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output / f"{basename}.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Comparison: {output / (basename + '.csv')}")


if __name__ == "__main__":
    main()
