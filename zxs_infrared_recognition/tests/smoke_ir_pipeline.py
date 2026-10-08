#!/usr/bin/env python3
"""Exercise IR conversion/evaluation with synthetic inputs in a new directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from build_ir_dataset_yaml import build_dataset_yaml
from convert_dronevehicle_to_yolo import CLASS_NAMES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_script(name: str, *arguments: object) -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "scripts" / name), *map(str, arguments)],
        cwd=PACKAGE_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"{name} failed:\n{result.stdout}\n{result.stderr}")


def source_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    for split in ("train", "val"):
        for image_suffix, label_suffix in (("img", "label"), ("imgr", "labelr")):
            image_dir = raw / split / f"{split}{image_suffix}"
            label_dir = raw / split / f"{split}{label_suffix}"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for class_id, class_name in enumerate(CLASS_NAMES):
                stem = f"{split}_{class_id}"
                image = np.full((712, 840, 3), 127, dtype=np.uint8)
                success, encoded = cv2.imencode(".png", image)
                require(success, "Could not encode fixture image")
                encoded.tofile(str(image_dir / f"{stem}.png"))
                xml = (
                    '<annotation><size><width>840</width><height>712</height></size>'
                    f'<object><name>{class_name}</name><polygon>'
                    '<x1>120</x1><y1>130</y1><x2>160</x2><y2>130</y2>'
                    '<x3>160</x3><y3>170</y3><x4>120</x4><y4>170</y4>'
                    '</polygon></object></annotation>'
                )
                with (label_dir / f"{stem}.xml").open("x", encoding="utf-8") as handle:
                    handle.write(xml)
    before = source_hashes(raw)
    converted = output / "converted"
    run_script(
        "convert_dronevehicle_to_yolo.py",
        "--split", f"train={raw / 'train'}", "--split", f"val={raw / 'val'}",
        "--output", converted,
    )
    require(source_hashes(raw) == before, "Converter changed the source inputs")
    dataset = converted / "infrared"
    data_yaml = build_dataset_yaml(dataset, dataset / "data.yaml")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    require(data["names"] == dict(enumerate(CLASS_NAMES)), "Class order changed")
    require(data["train"] != data["val"], "Train and val are not independent")
    try:
        build_dataset_yaml(dataset, data_yaml)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing data YAML was overwritten")
    for class_id in range(5):
        image_path = dataset / "images" / "val" / f"val_{class_id}.png"
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image.shape[:2] == (512, 640), "Wrong converted image dimensions")
        label = (dataset / "labels" / "val" / f"val_{class_id}.txt").read_text(encoding="utf-8")
        require(int(label.split()[0]) == class_id, "Wrong converted class ID")
    run_script("audit_yolo_dataset.py", "--data", data_yaml, "--output", output / "audit")
    audit = json.loads((output / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    require(not audit["errors"], "Dataset audit reported errors")
    ground_truth = output / "ground_truth.jsonl"
    run_script("export_yolo_ground_truth.py", "--data", data_yaml, "--output", ground_truth)
    records = [json.loads(line) for line in ground_truth.read_text(encoding="utf-8").splitlines()]
    require(len(records) == 5, "Expected five validation images")
    for record in records:
        bbox = record["objects"][0]["bbox"]
        require(np.allclose(bbox, [20, 30, 60, 70], atol=1e-4), "Crop coordinates changed")
    predictions = output / "perfect_predictions.jsonl"
    with predictions.open("x", encoding="utf-8") as handle:
        for record in records:
            for obj in record["objects"]:
                obj["score"] = 0.9
            handle.write(json.dumps(record) + "\n")
    metrics_path = output / "metrics.json"
    run_script(
        "evaluate_competition_metrics.py", "--ground-truth", ground_truth,
        "--predictions", predictions, "--output", metrics_path,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    require(metrics["recognition_rate"] == 1.0, "Perfect fixture recognition failed")
    run_script(
        "sweep_competition_thresholds.py", "--ground-truth", ground_truth,
        "--predictions", predictions, "--task-metric", "recognition_rate",
        "--target", 0.93, "--output", output / "thresholds.json",
    )
    run_script("evaluate_competition_metrics.py", "--self-test")
    if args.model_smoke:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "ultralytics"))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        import torch
        from ultralytics import YOLO
        import ultralytics
        from unified_inference import UnifiedDetector

        torch.set_num_threads(2)
        model = YOLO("yolov8n.yaml")
        model_config = dict(model.model.yaml)
        model_config["nc"] = 5
        model_config["scale"] = "n"
        model_yaml = output / "yolov8n_smoke.yaml"
        with model_yaml.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(model_config, handle)
        model = YOLO(str(model_yaml))
        model.model.names = dict(enumerate(CLASS_NAMES))
        weight = output / "random_five_class_smoke.pt"
        torch.save({"model": model.model.float()}, weight)
        registry = output / "registry.yaml"
        with registry.open("x", encoding="utf-8") as handle:
            yaml.safe_dump({"version": 1, "tasks": {"infrared_recognition": {
                "weight": weight.name, "imgsz": 64, "confidence": 0.99,
                "nms_iou": 0.7, "max_det": 10,
            }}}, handle)
        detector = UnifiedDetector(registry, device="cpu", use_fp16=False)
        require(list(detector.models) == ["infrared_recognition"], "Loaded unrelated experts")
        result = detector.predict("infrared_recognition", np.zeros((64, 64), dtype=np.uint8))
        require(isinstance(result, list), "Prediction did not return a list")
        run_script(
            "unified_inference.py", "--registry", registry, "--task", "infrared_recognition",
            "--image", dataset / "images" / "val" / "val_0.png", "--device", "cpu",
            "--fp32", "--output", output / "single_prediction.json",
        )
        run_script(
            "predict_yolo_dataset.py", "--registry", registry, "--task", "infrared_recognition",
            "--data", data_yaml, "--max-images", 1, "--device", "cpu", "--fp32",
            "--output", output / "model_predictions.jsonl",
        )
        print(json.dumps({"torch": torch.__version__, "ultralytics": ultralytics.__version__}))
    print(json.dumps({"status": "passed", "output": str(output), "model_smoke": args.model_smoke}))


if __name__ == "__main__":
    main()
