"""Run task preparation, FOMAML, all adaptations, and the final comparison."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def run(arguments):
    print("\nRUN:", " ".join(str(item) for item in arguments), flush=True)
    subprocess.run([str(item) for item in arguments], cwd=str(ROOT), check=True)


def main():
    defaults = yaml.safe_load((ROOT / "project_config.yaml").read_text(encoding="utf-8"))[
        "meta_learning"
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=str(defaults["gpu"]))
    parser.add_argument("--shots", type=int, nargs="+", default=defaults["shots"])
    parser.add_argument("--initial-weight")
    args = parser.parse_args()

    run([sys.executable, HERE / "01_prepare_meta_tasks.py"])
    meta_command = [sys.executable, HERE / "02_train_fomaml.py", "--gpu", args.gpu]
    if args.initial_weight:
        meta_command.extend(["--initial-weight", args.initial_weight])
    run(meta_command)
    run([sys.executable, HERE / "03_prepare_fewshot_support.py", "--shots", *args.shots])
    for shots in args.shots:
        for method in ("standard", "fomaml"):
            command = [
                sys.executable,
                HERE / "04_adapt_fewshot.py",
                "--method",
                method,
                "--shots",
                shots,
                "--gpu",
                args.gpu,
            ]
            if args.initial_weight and method == "standard":
                command.extend(["--initial-weight", args.initial_weight])
            run(command)
    run(
        [
            sys.executable,
            HERE / "05_compare_fewshot.py",
            "--gpu",
            args.gpu,
            "--shots",
            *args.shots,
        ]
    )


if __name__ == "__main__":
    main()
