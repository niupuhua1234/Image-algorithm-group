"""Evaluate two-stage RT-DETR domain increment; requires newly prepared runs."""
from competition.workflow import main_for

if __name__ == "__main__":
    main_for("incremental-test")
