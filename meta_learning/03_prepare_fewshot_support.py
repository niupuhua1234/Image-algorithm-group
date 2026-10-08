"""Create deterministic nested 5/10/20-shot supports from target train only."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def subset_coco(source, selected_images):
    image_ids = {item["id"] for item in selected_images}
    return {
        "info": source.get("info", {}),
        "licenses": source.get("licenses", []),
        "images": selected_images,
        "annotations": [item for item in source["annotations"] if item["image_id"] in image_ids],
        "categories": source["categories"],
    }


def main():
    defaults = yaml.safe_load((ROOT / "project_config.yaml").read_text(encoding="utf-8"))[
        "meta_learning"
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=defaults["target_profile"])
    parser.add_argument("--shots", type=int, nargs="+", default=defaults["shots"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    args = parser.parse_args()

    source_path = ROOT / "datasets" / "prepared" / args.profile / "annotations" / "instances_train.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Prepare the {args.profile} dataset first: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if max(args.shots) > len(source["images"]):
        raise ValueError(f"Largest shot count exceeds train images: {len(source['images'])}")

    annotations = defaultdict(list)
    for annotation in source["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    groups = defaultdict(list)
    for image in sorted(source["images"], key=lambda item: item["file_name"]):
        groups[len(annotations[image["id"]])].append(image)
    rng = random.Random(args.seed)
    for images in groups.values():
        rng.shuffle(images)

    # Round-robin target counts keeps very small supports representative.
    ordered = []
    keys = sorted(groups, reverse=True)
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())

    output = ROOT / "meta_learning" / "support" / args.profile
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, image in enumerate(ordered, start=1):
        row = {
            "rank": rank,
            "image_id": image["id"],
            "file_name": image["file_name"],
            "targets": len(annotations[image["id"]]),
        }
        for shots in sorted(set(args.shots)):
            row[f"in_{shots}shot"] = rank <= shots
        rows.append(row)
    with (output / "support_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"seed": args.seed, "source": str(source_path), "nested": True, "splits": {}}
    for shots in sorted(set(args.shots)):
        data = subset_coco(source, ordered[:shots])
        annotation_path = output / f"support_{shots:03d}.json"
        annotation_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        summary["splits"][str(shots)] = {
            "images": len(data["images"]),
            "targets": len(data["annotations"]),
            "annotation": str(annotation_path),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
