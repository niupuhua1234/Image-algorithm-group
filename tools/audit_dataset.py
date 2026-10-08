"""Audit one prepared single-class dataset before spending GPU training time."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from competition.paths import ROOT, prepared_profile


def file_hash(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def audit_split(profile_root, split, expected_size, hash_overlap):
    annotation_path = profile_root / "annotations" / f"instances_{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco.get("images", [])}
    target_counts = Counter()
    invalid_boxes = []
    widths, heights = [], []
    for item in coco.get("annotations", []):
        target_counts[item["image_id"]] += 1
        x, y, width, height = item["bbox"]
        widths.append(float(width))
        heights.append(float(height))
        image = images.get(item["image_id"])
        if (
            image is None
            or width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > image["width"] + 1e-6
            or y + height > image["height"] + 1e-6
        ):
            invalid_boxes.append(item.get("id"))

    missing, wrong_size, corrupt, hashes = [], [], [], {}
    for item in images.values():
        path = profile_root / "images" / split / item["file_name"]
        if not path.is_file():
            missing.append(item["file_name"])
            continue
        try:
            with Image.open(path) as image:
                size = image.size
                image.verify()
            if tuple(size) != tuple(expected_size):
                wrong_size.append({"file": item["file_name"], "size": list(size)})
            if hash_overlap:
                hashes[file_hash(path)] = item["file_name"]
        except Exception as error:
            corrupt.append({"file": item["file_name"], "error": str(error)})

    sorted_widths = sorted(widths)
    sorted_heights = sorted(heights)
    median = lambda values: values[len(values) // 2] if values else 0.0
    return {
        "images": len(images),
        "targets": len(widths),
        "empty_images": sum(target_counts[item_id] == 0 for item_id in images),
        "median_target_width": median(sorted_widths),
        "median_target_height": median(sorted_heights),
        "targets_up_to_64x96": sum(w <= 64 and h <= 96 for w, h in zip(widths, heights)),
        "missing_images": missing,
        "wrong_size_images": wrong_size,
        "corrupt_images": corrupt,
        "invalid_boxes": invalid_boxes,
        "hashes": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--hash-overlap",
        action="store_true",
        help="Hash every image and fail if identical bytes occur across splits.",
    )
    args = parser.parse_args()
    profile_root = prepared_profile(args.profile)
    results = {
        split: audit_split(
            profile_root, split, (args.width, args.height), args.hash_overlap
        )
        for split in ("train", "val", "test")
    }

    overlap = []
    if args.hash_overlap:
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(results[left]["hashes"]) & set(results[right]["hashes"])
            overlap.extend(
                {
                    "splits": [left, right],
                    "left": results[left]["hashes"][digest],
                    "right": results[right]["hashes"][digest],
                }
                for digest in sorted(shared)
            )
    for result in results.values():
        result.pop("hashes")
    report = {
        "profile": args.profile,
        "expected_size": [args.width, args.height],
        "splits": results,
        "cross_split_identical_images": overlap,
        "note": (
            "Structural checks cannot prove competition-domain similarity. "
            "Visually review target contrast, road scenes, and box alignment."
        ),
    }
    output = ROOT / "reports" / f"dataset_audit_{args.profile}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    failures = []
    for split, result in results.items():
        for key in ("missing_images", "wrong_size_images", "corrupt_images", "invalid_boxes"):
            if result[key]:
                failures.append(f"{split}.{key}={len(result[key])}")
    if overlap:
        failures.append(f"cross_split_identical_images={len(overlap)}")
    if failures:
        raise SystemExit("Dataset audit failed: " + ", ".join(failures))
    print(f"Dataset audit passed: {output}")


if __name__ == "__main__":
    main()
