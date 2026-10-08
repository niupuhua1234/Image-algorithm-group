"""Build a small, disjoint two-stage domain-incremental RT-DETR pilot."""

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent
V3 = Path(r"D:\server_upload\zzq\RT-DETR-VEDAI\dataset\competition_verified_v3")
NEW = Path(r"D:\导出后")
OLD = ROOT / "datasets/prepared/rtdetr_gpt_low_contrast_combined_v1"
PROFILES = ("inc_after_transfer_stage1", "inc_after_transfer_stage2")
SEED = 20260930


def add_records(source, split, limit, prefix, annotations, image_root):
    data = json.loads((source / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
    by_image = {}
    for box in data["annotations"]:
        by_image.setdefault(box["image_id"], []).append(box)
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if limit is not None:
        images = images[:limit]
    result = []
    for item in images:
        result.append((prefix + "__" + item["file_name"], image_root / split / item["file_name"],
                       item["width"], item["height"],
                       [box["bbox"] for box in by_image.get(item["id"], [])], False))
    return result


def new_records():
    names = sorted(path.name for path in (NEW / "images").glob("*_composite.png"))
    if len(names) < 60:
        raise ValueError(f"Expected at least 60 new images, found {len(names)}")
    random.Random(SEED).shuffle(names)
    result = {}
    for split, selected in (("train", names[:40]), ("val", names[40:50]), ("test", names[50:60])):
        rows = []
        for name in selected:
            xml = NEW / "xml" / (name.split("_")[0] + ".xml")
            tree = ET.parse(xml)
            boxes = []
            for obj in tree.findall(".//object"):
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                if not (0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 1024):
                    raise ValueError(f"Invalid box: {xml}")
                boxes.append([x1, y1, x2 - x1, y2 - y1])
            with Image.open(NEW / "images" / name) as image:
                if image.size != (1280, 1024):
                    raise ValueError(f"Wrong image size: {name}: {image.size}")
            rows.append(("new__" + name, NEW / "images" / name, 1280, 1024, boxes, True))
        result[split] = rows
    return result


def write_split(target, split, rows):
    image_dir = target / "images" / split
    annotation_dir = target / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(exist_ok=True)
    images, boxes = [], []
    for image_id, (name, source, width, height, original_boxes, convert) in enumerate(rows, 1):
        destination = image_dir / name
        if convert:
            with Image.open(source) as image:
                image.convert("RGB").save(destination)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        for x, y, w, h in original_boxes:
            boxes.append({"id": len(boxes) + 1, "image_id": image_id, "category_id": 0,
                          "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    data = {"info": {"description": "Independent staged incremental pilot"}, "licenses": [],
            "images": images, "annotations": boxes,
            "categories": [{"id": 0, "name": "target", "supercategory": "target"}]}
    (annotation_dir / f"instances_{split}.json").write_text(json.dumps(data), encoding="utf-8")
    return {"images": len(images), "targets": len(boxes)}


def main():
    # Parse before accessing data: --help must never prepare an experiment.
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not V3.is_dir() or not NEW.is_dir() or not OLD.is_dir():
        raise FileNotFoundError("One of the source datasets is missing")
    old = {split: add_records(OLD, split, 10, "old", None, OLD / "images") for split in ("train", "val", "test")}
    v3 = {split: add_records(V3, split, 40 if split == "train" else 10,
                             "v3", None, V3 / "images") for split in ("train", "val", "test")}
    new = new_records()
    profiles = {
        PROFILES[0]: {"train": old["train"] + v3["train"], "val": old["val"] + v3["val"],
                      "test": old["test"] + v3["test"]},
        PROFILES[1]: {"train": old["train"] + v3["train"][:10] + new["train"],
                      "val": old["val"] + v3["val"] + new["val"],
                      "test": old["test"] + v3["test"] + new["test"]},
    }
    summary = {}
    for profile, splits in profiles.items():
        target = ROOT / "datasets/prepared" / profile
        if target.exists():
            raise FileExistsError(f"Will not overwrite prepared profile: {target}")
        staging = target.with_name(target.name + ".building")
        if staging.exists():
            raise FileExistsError(staging)
        if any(set(row[0] for row in splits[a]) & set(row[0] for row in splits[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("Train/val/test overlap")
        summary[profile] = {split: write_split(staging, split, rows) for split, rows in splits.items()}
        (staging / "manifest.json").write_text(json.dumps({split: [row[0] for row in rows]
            for split, rows in splits.items()}, indent=2), encoding="utf-8")
        staging.rename(target)
    (BASE / "prepared_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
