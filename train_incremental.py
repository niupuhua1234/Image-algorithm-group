"""Sequentially train stage 1 from transfer and stage 2 from stage 1 (no FOMAML)."""
from competition.workflow import main_for

if __name__ == "__main__":
    main_for("incremental-train")
