"""Inventory fixed dataset splits; optionally import base and export VOC XML.

Never reshuffles data or deletes a profile. Generated annotations preserve the
existing box coordinates. Image hardlinks save space and must be treated as
read-only. Exact-file hashes detect duplicates, not semantic near-duplicates.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "datasets/prepared"
REPORT = ROOT / "reports/organization_20261002"
ACTIVE = {"base": "basic", "rtdetr_gpt_low_contrast_combined_v1": "transfer"}
SPLITS = ("train", "val", "test")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def import_base(source, annotations):
    target = PREPARED / "base"
    if target.exists():
        print(f"Preserving existing base: {target}")
        return
    jobs = {s: annotations / ("instances_train_real_synth.json" if s == "train" else f"instances_{s}.json") for s in SPLITS}
    payloads = {s: read_json(p) for s, p in jobs.items()}
    # Resolve every input first, before creating a new profile.
    for split, data in payloads.items():
        for item in data["images"]:
            path = source / "images" / split / item["file_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
    staging = target.with_name("base.building")
    staging.mkdir(parents=True, exist_ok=False)
    for split, data in payloads.items():
        image_dir = staging / "images" / split
        image_dir.mkdir(parents=True)
        for item in data["images"]:
            src = source / "images" / split / item["file_name"]
            dst = image_dir / item["file_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        write_json(staging / "annotations" / f"instances_{split}.json", data)
    write_json(staging / "provenance.json", {
        "source": str(source), "annotations": {s: str(p) for s, p in jobs.items()},
        "split_changed": False, "image_storage": "hardlink where possible; read-only",
        "note": "Imported existing dataset, not evidence of any checkpoint training lineage.",
    })
    staging.rename(target)


def export_labels(profile, split, data):
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    listing = []
    for item in data["images"]:
        name = Path(item["file_name"])
        width, height = item["width"], item["height"]
        xml_file = profile / "voc_xml" / split / name.with_suffix(".xml")
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = name.as_posix()
        size = ET.SubElement(root, "size")
        for key, value in (("width", width), ("height", height), ("depth", 3)):
            ET.SubElement(size, key).text = str(value)
        for annotation in by_image[item["id"]]:
            x, y, w, h = annotation["bbox"]
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = "target"
            box = ET.SubElement(obj, "bndbox")
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (x, y, x+w, y+h)):
                ET.SubElement(box, key).text = format(value, ".12g")
        if not xml_file.exists():
            xml_file.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(root).write(xml_file, encoding="utf-8", xml_declaration=True)
        listing.append((profile / "images" / split / name).as_posix())
    split_list = profile / f"{split}.txt"
    if not split_list.exists():
        split_list.write_text("\n".join(listing) + "\n", encoding="utf-8")


def inventory(hash_images=False, write_xml=False):
    rows, problems, cache = [], [], {}
    for profile in sorted(PREPARED.iterdir()):
        if not profile.is_dir():
            continue
        hashes = defaultdict(list)
        for split in SPLITS:
            annotation = profile / "annotations" / f"instances_{split}.json"
            if not annotation.is_file():
                problems.append({"profile": profile.name, "split": split, "error": "missing annotations"})
                continue
            data = read_json(annotation)
            ids = {item["id"]: item for item in data["images"]}
            if len(ids) != len(data["images"]):
                problems.append({"profile": profile.name, "split": split, "error": "duplicate image ids"})
            missing = 0
            for item in data["images"]:
                image = profile / "images" / split / item["file_name"]
                if not image.is_file():
                    missing += 1
                elif hash_images:
                    stat = image.stat()
                    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if key not in cache:
                        digest = hashlib.sha256()
                        with image.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                        cache[key] = digest.hexdigest()
                    hashes[cache[key]].append((split, item["file_name"]))
            invalid = 0
            for box in data["annotations"]:
                item = ids.get(box["image_id"])
                x, y, w, h = box["bbox"]
                if item is None or not all(math.isfinite(v) for v in (x,y,w,h)) or not (
                    x >= -1e-4 and y >= -1e-4 and w > 0 and h > 0
                    and x+w <= item["width"]+1e-3 and y+h <= item["height"]+1e-3
                ):
                    invalid += 1
            if write_xml and profile.name in ACTIVE and not missing and not invalid:
                export_labels(profile, split, data)
            rows.append({"profile": profile.name, "role": ACTIVE.get(profile.name, "research_reference"),
                         "split": split, "images": len(ids), "targets": len(data["annotations"]),
                         "missing_images": missing, "invalid_boxes": invalid,
                         "xml": len(list((profile/"voc_xml"/split).rglob("*.xml"))),
                         "path": str(profile)})
            if missing or invalid:
                problems.append({"profile": profile.name, "split": split,
                                 "missing_images": missing, "invalid_boxes": invalid})
        for digest, entries in hashes.items():
            if len({split for split, _ in entries}) > 1:
                problems.append({"profile": profile.name, "error": "exact cross-split duplicate",
                                 "sha256": digest, "files": entries})
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT/"dataset_catalog.json", {"rows": rows, "issues": problems,
               "hash_checked": hash_images, "incremental_status": "deleted at user request; not rebuilt"})
    with (REPORT/"dataset_catalog.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["profile"])
        writer.writeheader()
        writer.writerows(rows)
    table = ["# Dataset Catalog", "", "Fixed existing splits; no random repartition.", "",
             "| Profile | Role | Split | Images | Targets | Missing | Invalid boxes | XML |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    table.extend("| " + " | ".join(str(row[k]) for k in
                 ("profile","role","split","images","targets","missing_images","invalid_boxes","xml")) + " |" for row in rows)
    table.extend(["", f"Audit issues: {len(problems)}. Exact-file hashes checked: {hash_images}.",
                  "See reports/organization_20261002/dataset_catalog.json for details.",
                  "XML exports preserve COCO numeric box coordinates; they are not new manual annotations.",
                  "Incremental weights and derived splits were deleted; no incremental dataset was rebuilt."])
    (ROOT/"datasets/CATALOG.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    print(f"Catalog: {len(rows)} splits; {len(problems)} audit issues; {REPORT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-base", action="store_true")
    parser.add_argument("--write-xml", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--base-source", type=Path, default=Path(r"D:\server_upload\zzq\datasets\VEDAI_COMPETITION"))
    parser.add_argument("--base-annotations", type=Path, default=Path(r"D:\server_upload\zzq\RT-DETR-VEDAI\dataset\vedai_competition\annotations"))
    args = parser.parse_args()
    if args.build_base:
        import_base(args.base_source, args.base_annotations)
    inventory(args.hash_images, args.write_xml)


if __name__ == "__main__":
    main()
