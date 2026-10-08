"""Ordinary few-shot RT-DETR transfer; existing runs are protected from overwrite."""
from competition.workflow import main_for

if __name__ == "__main__":
    main_for("transfer-train")
