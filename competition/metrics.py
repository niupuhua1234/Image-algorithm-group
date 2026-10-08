"""Official-style class-agnostic object metrics at a fixed IoU threshold."""

import csv
import json
from pathlib import Path

import numpy as np


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=1)
    box_area = np.prod(np.clip(box[2:] - box[:2], 0.0, None))
    boxes_area = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None), axis=1)
    union = box_area + boxes_area - intersection
    return intersection / np.clip(union, 1e-12, None)


def match_image(predictions, targets, confidence, iou_threshold):
    """Greedily match predictions by descending confidence, ignoring class names."""
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 5)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 4)
    predictions = predictions[predictions[:, 4] >= confidence]
    predictions = predictions[np.argsort(-predictions[:, 4])]
    unmatched = set(range(len(targets)))
    details = []
    for prediction in predictions:
        candidate_ids = sorted(unmatched)
        matched_id = None
        matched_iou = 0.0
        if candidate_ids:
            ious = box_iou(prediction[:4], targets[candidate_ids])
            best_local = int(np.argmax(ious))
            if float(ious[best_local]) >= iou_threshold:
                matched_id = candidate_ids[best_local]
                matched_iou = float(ious[best_local])
                unmatched.remove(matched_id)
        details.append(
            {
                "prediction": prediction.tolist(),
                "status": "TP" if matched_id is not None else "FP",
                "target_index": matched_id,
                "matched_iou": matched_iou,
            }
        )
    return details, sorted(unmatched)


def evaluate_detections(records, confidence, iou_threshold=0.40):
    tp = fp = fn = 0
    details = []
    for record in records:
        matches, unmatched = match_image(
            record.get("predictions", []), record.get("targets", []), confidence, iou_threshold
        )
        image_tp = sum(item["status"] == "TP" for item in matches)
        image_fp = len(matches) - image_tp
        image_fn = len(unmatched)
        tp += image_tp
        fp += image_fp
        fn += image_fn
        details.append(
            {
                "image": record["image"],
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "matches": matches,
                "unmatched_target_indices": unmatched,
            }
        )
    targets = tp + fn
    detections = tp + fp
    pd = tp / targets if targets else 0.0
    precision = tp / detections if detections else 0.0
    far = fp / detections if detections else 0.0
    f1 = 2.0 * pd * precision / (pd + precision) if pd + precision else 0.0
    return {
        "confidence": float(confidence),
        "iou": float(iou_threshold),
        "images": len(records),
        "targets": targets,
        "detections": detections,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pd": pd,
        "precision": precision,
        "false_alarm_rate": far,
        "f1": f1,
        "matching": "class-agnostic confidence-ordered one-to-one matching",
        "details": details,
    }


def select_threshold(records, iou_threshold=0.40, target_pd=0.90):
    """Select the lowest-FAR validation operating point while preserving Pd.

    Matching is confidence ordered, so lowering the threshold only appends new
    predictions to each image's existing match sequence.  Match every image
    once, then accumulate globally by score instead of rerunning all images for
    every unique confidence.  The selected result is evaluated once more to
    retain the original per-image details API.
    """
    confidences = {0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}
    events = []
    target_count = 0
    for record in records:
        predictions = record.get("predictions", [])
        confidences.update(float(row[4]) for row in predictions)
        matches, _ = match_image(
            predictions,
            record.get("targets", []),
            confidence=float("-inf"),
            iou_threshold=iou_threshold,
        )
        target_count += len(record.get("targets", []))
        events.extend((float(item["prediction"][4]), item["status"]) for item in matches)

    events.sort(key=lambda item: -item[0])
    rows_by_confidence = {}
    tp = fp = event_index = 0
    for confidence in sorted(confidences, reverse=True):
        while event_index < len(events) and events[event_index][0] >= confidence:
            if events[event_index][1] == "TP":
                tp += 1
            else:
                fp += 1
            event_index += 1
        fn = target_count - tp
        detections = tp + fp
        pd = tp / target_count if target_count else 0.0
        precision = tp / detections if detections else 0.0
        far = fp / detections if detections else 0.0
        f1 = 2.0 * pd * precision / (pd + precision) if pd + precision else 0.0
        rows_by_confidence[confidence] = {
            "confidence": float(confidence),
            "iou": float(iou_threshold),
            "images": len(records),
            "targets": target_count,
            "detections": detections,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pd": pd,
            "precision": precision,
            "false_alarm_rate": far,
            "f1": f1,
            "matching": "class-agnostic confidence-ordered one-to-one matching",
        }
    rows = [rows_by_confidence[value] for value in sorted(confidences)]
    feasible = [row for row in rows if row["pd"] >= target_pd]
    if feasible:
        compact_selected = min(
            feasible,
            key=lambda row: (row["false_alarm_rate"], -row["pd"], row["fp"], -row["confidence"]),
        )
        selected = evaluate_detections(records, compact_selected["confidence"], iou_threshold)
        return selected, rows, True
    compact_selected = max(rows, key=lambda row: (row["pd"], -row["fp"], row["confidence"]))
    selected = evaluate_detections(records, compact_selected["confidence"], iou_threshold)
    return selected, rows, False


def save_evaluation(output_dir, summary, threshold_rows=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = summary.pop("details", [])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in details)
    if threshold_rows:
        fields = [
            "confidence", "iou", "images", "targets", "detections", "tp", "fp", "fn",
            "pd", "precision", "false_alarm_rate", "f1",
        ]
        with (output_dir / "threshold_sweep.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in threshold_rows)
    summary["details"] = details
