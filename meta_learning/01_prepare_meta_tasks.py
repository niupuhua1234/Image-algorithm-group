"""Build source-domain task labels from the prepared base training split."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inferred_domain(row, multiple_sources):
    if multiple_sources:
        return Path(row["source"]).name
    name = row["image"].lower()
    synthetic_markers = ("_aug", "synth", "synthetic", "composite")
    return "synthetic" if any(marker in name for marker in synthetic_markers) else "real"


def read_domain_map(path):
    if not path:
        return {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image", "domain"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Domain map must contain image,domain columns")
    return {Path(row["image"]).name: row["domain"].strip() for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--domain-map", help="Optional CSV with image,domain columns")
    parser.add_argument("--minimum-images", type=int, default=4)
    args = parser.parse_args()

    prepared = ROOT / "datasets" / "prepared" / args.profile
    manifest_path = prepared / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare the {args.profile} dataset first: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    if not rows:
        raise RuntimeError("The prepared training split is empty")

    mapping = read_domain_map(args.domain_map)
    sources = {row["source"] for row in rows}
    multiple_sources = len(sources) > 1
    output_rows = []
    for row in rows:
        image = Path(row["image"]).name
        domain = mapping.get(image) or inferred_domain(row, multiple_sources)
        output_rows.append(
            {
                "image": image,
                "domain": domain,
                "targets": int(row["objects"]),
                "source": row["source"],
            }
        )

    counts = Counter(row["domain"] for row in output_rows)
    if len(counts) < 2:
        raise RuntimeError(
            "FOMAML requires at least two source domains. Provide --domain-map with "
            "image,domain columns, or prepare base from separate real/synthetic source folders."
        )
    too_small = {name: count for name, count in counts.items() if count < args.minimum_images}
    if too_small:
        raise RuntimeError(f"Meta domains need at least {args.minimum_images} images: {too_small}")

    output = ROOT / "meta_learning" / "tasks" / args.profile
    output.mkdir(parents=True, exist_ok=True)
    with (output / "domain_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "domain", "targets", "source"])
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "profile": args.profile,
        "train_only": True,
        "domain_assignment": "explicit CSV" if mapping else "source folder / filename inference",
        "domains": dict(sorted(counts.items())),
        "images": len(output_rows),
        "manifest": str(output / "domain_manifest.csv"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
