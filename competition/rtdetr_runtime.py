"""RT-DETR configuration generation and inference adapter."""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .metrics import evaluate_detections, save_evaluation, select_threshold
from .paths import ROOT, prepared_profile, require_path
from .platform_deps import configure_rtdetr_imports


ENGINE = ROOT / "engines" / "rtdetr"


def build_config(profile, epochs, batch_size, workers=0):
    """Write resolved single-class dataset/model YAML files for one profile."""
    prepared = prepared_profile(profile)
    require_path(prepared / "annotations" / "instances_train.json", "Prepared train annotations")
    dataset_path = ENGINE / "configs" / "dataset" / f"competition_{profile}.yml"
    model_path = ENGINE / "configs" / "rtdetr" / f"competition_r18_{profile}.yml"
    dataset_cfg = {
        "task": "detection",
        "num_classes": 1,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": (prepared / "images" / "train").as_posix(),
                "ann_file": (prepared / "annotations" / "instances_train.json").as_posix(),
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": True,
            "batch_size": batch_size,
            "num_workers": workers,
            "drop_last": True,
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": (prepared / "images" / "val").as_posix(),
                "ann_file": (prepared / "annotations" / "instances_val.json").as_posix(),
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": False,
            "batch_size": 1,
            "num_workers": workers,
            "drop_last": False,
        },
    }
    dataset_path.write_text(yaml.safe_dump(dataset_cfg, sort_keys=False), encoding="utf-8")
    model_cfg = {
        "__include__": [
            f"../dataset/{dataset_path.name}",
            "../runtime.yml",
            "./include/dataloader.yml",
            "./include/optimizer.yml",
            "./include/rtdetr_r50vd.yml",
        ],
        "output_dir": (ROOT / "outputs" / "rtdetr" / profile).as_posix(),
        "sync_bn": False,
        "find_unused_parameters": False,
        "use_amp": True,
        "use_ema": True,
        "epoches": int(epochs),
        "checkpoint_step": 5,
        "log_step": 20,
        "clip_max_norm": 0.1,
        "PResNet": {"depth": 18, "freeze_at": -1, "freeze_norm": True, "pretrained": False},
        "HybridEncoder": {
            "in_channels": [128, 256, 512],
            "hidden_dim": 256,
            "expansion": 0.5,
            "eval_spatial_size": [1024, 1280],
        },
        "RTDETR": {"multi_scale": None},
        "RTDETRTransformer": {
            "num_queries": 100,
            "num_decoder_layers": 3,
            "num_denoising": 100,
            "eval_idx": -1,
            "eval_spatial_size": [1024, 1280],
        },
        "RTDETRPostProcessor": {"num_top_queries": 100},
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": "backbone", "lr": 0.00001},
                {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
                {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
            ],
            "lr": 0.0001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
        },
        "lr_scheduler": {
            "type": "MultiStepLR",
            "milestones": [max(1, int(epochs * 0.83)), max(2, int(epochs * 0.94))],
            "gamma": 0.1,
        },
        "train_dataloader": {
            "dataset": {
                "return_masks": False,
                "transforms": {
                    "ops": [
                        {"type": "RandomHorizontalFlip", "p": 0.5},
                        {"type": "Resize", "size": [1024, 1280]},
                        {"type": "ToImageTensor"},
                        {"type": "ConvertDtype"},
                        {"type": "SanitizeBoundingBox", "min_size": 1},
                        {"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True},
                    ]
                },
            },
            "batch_size": batch_size,
            "num_workers": workers,
            "collate_fn": "default_collate_fn",
        },
        "val_dataloader": {
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [1024, 1280]},
                        {"type": "ToImageTensor"},
                        {"type": "ConvertDtype"},
                    ]
                }
            },
            "batch_size": 1,
            "num_workers": workers,
            "collate_fn": "default_collate_fn",
        },
    }
    model_path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return model_path


def _checkpoint_state(checkpoint):
    return checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]


def _load_targets(annotation_path):
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data["images"]}
    targets = defaultdict(list)
    for item in data["annotations"]:
        x, y, width, height = item["bbox"]
        targets[item["image_id"]].append([x, y, x + width, y + height])
    return images, targets


def _save_predictions(path, records):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
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


@torch.no_grad()
def infer_split(config_path, checkpoint_path, profile, split, warmup=20):
    """Run one RT-DETR split and return class-agnostic metric records."""
    configure_rtdetr_imports()
    from src.core import YAMLConfig

    prepared = prepared_profile(profile)
    annotation_path = prepared / "annotations" / f"instances_{split}.json"
    image_root = prepared / "images" / split
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = annotation_path.as_posix()
    cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = image_root.as_posix()
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    postprocessor = cfg.postprocessor.cuda().eval()
    loader = cfg.val_dataloader
    images, targets = _load_targets(annotation_path)

    input_size = cfg.yaml_cfg["RTDETRTransformer"].get("eval_spatial_size", [1024, 1280])
    dummy = torch.zeros(1, 3, input_size[0], input_size[1], device="cuda")
    for _ in range(warmup):
        model(dummy)
    torch.cuda.synchronize()

    records = []
    model_times = []
    pipeline_times = []
    iterator = iter(loader)
    while True:
        pipeline_start = time.perf_counter()
        try:
            samples, batch_targets = next(iterator)
        except StopIteration:
            break
        samples = samples.cuda(non_blocking=True)
        image_id = int(batch_targets[0]["image_id"].item())
        image_info = images[image_id]
        original_size = torch.tensor(
            [[image_info["width"], image_info["height"]]], device="cuda"
        )
        torch.cuda.synchronize()
        model_start = time.perf_counter()
        outputs = model(samples)
        torch.cuda.synchronize()
        model_end = time.perf_counter()
        result = postprocessor(outputs, original_size)[0]
        torch.cuda.synchronize()
        pipeline_end = time.perf_counter()

        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        predictions = np.column_stack((boxes, scores)).astype(np.float32)
        records.append(
            {
                "image": image_info["file_name"],
                "targets": np.asarray(targets[image_id], dtype=np.float32).reshape(-1, 4),
                "predictions": predictions,
            }
        )
        model_times.append((model_end - model_start) * 1000.0)
        pipeline_times.append((pipeline_end - pipeline_start) * 1000.0)
    return records, {
        "model_ms_mean": float(np.mean(model_times)) if model_times else 0.0,
        "model_ms_p95": float(np.percentile(model_times, 95)) if model_times else 0.0,
        "pipeline_ms_mean": float(np.mean(pipeline_times)) if pipeline_times else 0.0,
        "pipeline_ms_p95": float(np.percentile(pipeline_times, 95)) if pipeline_times else 0.0,
        "official_note": "Re-measure full interface latency on Jetson AGX Orin with all three models running.",
    }


def evaluate_model(profile, checkpoint, epochs=72, batch_size=4, workers=0, iou=0.40, target_pd=0.90):
    config_path = build_config(profile, epochs, batch_size, workers)
    checkpoint = require_path(checkpoint, "RT-DETR checkpoint")
    output = ROOT / "outputs" / "rtdetr" / profile / "competition_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    val_records, val_speed = infer_split(config_path, checkpoint, profile, "val")
    selected, threshold_rows, reached = select_threshold(val_records, iou, target_pd)
    _save_predictions(output / "val_predictions.csv", val_records)
    val_summary = dict(selected)
    val_summary.update({"target_pd": target_pd, "target_pd_reached": reached, "speed": val_speed})
    save_evaluation(output / "validation", val_summary, threshold_rows)

    test_records, test_speed = infer_split(config_path, checkpoint, profile, "test")
    _save_predictions(output / "test_predictions.csv", test_records)
    test_summary = evaluate_detections(test_records, selected["confidence"], iou)
    test_summary.update(
        {
            "threshold_mode": "fixed_from_validation",
            "target_pd": target_pd,
            "target_pd_reached": test_summary["pd"] >= target_pd,
            "speed": test_speed,
        }
    )
    save_evaluation(output / "test", test_summary)
    return output / "test" / "summary.json"
