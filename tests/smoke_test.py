"""Check metrics and RT-DETR on CPU without training or touching datasets."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true", help="Run one CPU 1024x1280 forward")
    args = parser.parse_args()
    from competition.metrics import evaluate_detections
    records = [{"image": "fixture.png", "targets": [[10, 10, 30, 22]],
                "predictions": [[10, 10, 30, 22, 0.9]]}]
    result = evaluate_detections(records, 0.5, iou_threshold=0.4)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    from competition.platform_deps import configure_rtdetr_imports
    configure_rtdetr_imports()
    import torch
    from src.core import YAMLConfig
    config = ROOT / "engines/rtdetr/configs/rtdetr/competition_r18_base.yml"
    cfg = YAMLConfig(str(config))
    model = cfg.model.cpu().eval()
    if args.forward:
        # No checkpoint or data-loader is accessed: this is only an execution check.
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 1024, 1280))
        assert tuple(output["pred_boxes"].shape) == (1, 100, 4)
        assert torch.isfinite(output["pred_boxes"]).all()
    print("RT-DETR metrics/model smoke test passed; forward:", args.forward)


if __name__ == "__main__":
    main()
