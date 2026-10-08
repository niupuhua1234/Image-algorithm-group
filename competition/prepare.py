"""Validate, split, and convert one or more VOC datasets."""

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from .paths import ROOT
from .voc import discover_ids, find_voc_directories, load_voc_sample, read_split_ids


SPLITS = ("train", "val", "test")


def _random_split(ids, ratios, seed):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    count = len(ids)
    train_end = int(round(count * ratios[0]))
    val_end = train_end + int(round(count * ratios[1]))
    train_end = min(train_end, count)
    val_end = min(val_end, count)
    return {
        "train": sorted(ids[:train_end]),
        "val": sorted(ids[train_end:val_end]),
        "test": sorted(ids[val_end:]),
    }


def _source_splits(root, annotation_dir, mode, ratios, seed):
    existing = {name: read_split_ids(root, name) for name in SPLITS}
    has_complete_split = all(existing[name] is not None for name in SPLITS)
    if mode == "existing" and not has_complete_split:
        raise FileNotFoundError(f"Complete train/val/test files are missing below {root}/ImageSets/Main")
    if mode == "existing" or (mode == "auto" and has_complete_split):
        return existing, "existing"
    return _random_split(discover_ids(annotation_dir), ratios, seed), "random"


def _write_single_class_xml(sample, output_path, output_name):
    root = ElementTree.parse(str(sample.xml_path)).getroot()
    filename = root.find("filename")
    if filename is None:
        filename = ElementTree.SubElement(root, "filename")
    filename.text = output_name
    for node in root.findall("object"):
        name = node.find("name")
        if name is None:
            name = ElementTree.SubElement(node, "name")
        name.text = "target"
    ElementTree.ElementTree(root).write(str(output_path), encoding="utf-8", xml_declaration=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_profile(
    sources,
    output_dir,
    split_mode="auto",
    ratios=(0.70, 0.15, 0.15),
    seed=42,
    overwrite=False,
    expected_size=(1280, 1024),
):
    """Create one shared prepared dataset for RT-DETR."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Prepared profile is not empty; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(path).resolve() for path in sources]
    if not sources:
        raise ValueError("At least one VOC source is required")

    manifest = []
    category_counts = Counter()
    coco = {name: {"images": [], "annotations": []} for name in SPLITS}
    annotation_ids = {name: 1 for name in SPLITS}
    image_ids = {name: 1 for name in SPLITS}
    seen_ids = set()

    for source_index, source in enumerate(sources):
        image_dir, annotation_dir = find_voc_directories(source)
        splits, split_source = _source_splits(source, annotation_dir, split_mode, ratios, seed + source_index)
        for split in SPLITS:
            for sample_id in splits[split]:
                xml_path = annotation_dir / f"{sample_id}.xml"
                if not xml_path.is_file():
                    raise FileNotFoundError(f"Split references missing XML: {xml_path}")
                sample = load_voc_sample(xml_path, image_dir, verify_image=True)
                if expected_size and (sample.width, sample.height) != tuple(expected_size):
                    raise ValueError(
                        f"Competition image must be {expected_size[0]}x{expected_size[1]}: "
                        f"{sample.image_path} is {sample.width}x{sample.height}"
                    )
                prefix = f"{source.name}__" if len(sources) > 1 else ""
                output_stem = f"{prefix}{sample.sample_id}"
                if output_stem in seen_ids:
                    raise ValueError(f"Duplicate output sample id: {output_stem}")
                seen_ids.add(output_stem)
                output_name = output_stem + sample.image_path.suffix.lower()

                split_images = output_dir / "images" / split
                split_xml = output_dir / "voc_xml" / split
                split_images.mkdir(parents=True, exist_ok=True)
                split_xml.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.image_path, split_images / output_name)
                _write_single_class_xml(sample, split_xml / f"{output_stem}.xml", output_name)

                image_id = image_ids[split]
                coco[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": output_name,
                        "width": sample.width,
                        "height": sample.height,
                    }
                )
                for obj in sample.objects:
                    x1, y1, x2, y2 = obj.box
                    width = x2 - x1
                    height = y2 - y1
                    coco[split]["annotations"].append(
                        {
                            "id": annotation_ids[split],
                            "image_id": image_id,
                            "category_id": 0,
                            "bbox": [x1, y1, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                        }
                    )
                    annotation_ids[split] += 1
                    category_counts[obj.source_class] += 1
                manifest.append(
                    {
                        "sample_id": output_stem,
                        "split": split,
                        "source": str(source),
                        "source_split_mode": split_source,
                        "image": output_name,
                        "width": sample.width,
                        "height": sample.height,
                        "objects": len(sample.objects),
                        "sha256": _sha256(sample.image_path),
                    }
                )
                image_ids[split] += 1

    hashes = {}
    for row in manifest:
        earlier = hashes.get(row["sha256"])
        if earlier and earlier["split"] != row["split"]:
            raise ValueError(
                "Identical image content appears across splits: "
                f"{earlier['sample_id']} ({earlier['split']}) and {row['sample_id']} ({row['split']})"
            )
        hashes[row["sha256"]] = row

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    split_counts = {}
    for split in SPLITS:
        payload = {
            "info": {"description": "Infrared small targets; source classes collapsed to target"},
            "licenses": [],
            "images": coco[split]["images"],
            "annotations": coco[split]["annotations"],
            "categories": [{"id": 0, "name": "target", "supercategory": "target"}],
        }
        (annotations_dir / f"instances_{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        image_paths = [
            (output_dir / "images" / split / row["image"]).resolve().as_posix()
            for row in manifest
            if row["split"] == split
        ]
        (output_dir / f"{split}.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        split_counts[split] = {
            "images": len(payload["images"]),
            "targets": len(payload["annotations"]),
        }

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    test_rows = [row for row in manifest if row["split"] == "test"]
    with (output_dir / "sequence_groups.example.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group_id", "frame_index", "image"])
        writer.writeheader()
        for index, row in enumerate(sorted(test_rows, key=lambda item: item["image"])):
            writer.writerow({"group_id": index // 5, "frame_index": index % 5, "image": row["image"]})

    summary = {
        "sources": [str(path) for path in sources],
        "single_class": True,
        "class_name": "target",
        "split_counts": split_counts,
        "original_category_counts": dict(category_counts),
        "iou_threshold": 0.40,
        "expected_image_size": list(expected_size) if expected_size else None,
        "sequence_manifest_note": "Auto-generated sorted blocks; replace with official sequence order when provided.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
