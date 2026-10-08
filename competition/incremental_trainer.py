"""Shared supervised replay trainer; no FOMAML update is performed here.

The caller supplies the previous-stage checkpoint and the prepared dataset.
competition.incremental_trainer is a library, not an experiment entrypoint.
"""

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "outputs/incremental_after_transfer"


def train_stage(name, profile, source, epochs, args):
    import torch

    from competition.platform_deps import configure_rtdetr_imports
    from meta_learning.common import (
        build_yaml_config,
        load_matching_state,
        save_model_checkpoint,
        set_adaptation_parameters,
        set_seed,
        write_json,
    )

    configure_rtdetr_imports()
    from src.data import get_coco_api_from_dataset
    from src.solver.det_engine import evaluate, train_one_epoch

    output = OUTPUT / name
    complete = output / "training_complete.json"
    if complete.is_file():
        result = json.loads(complete.read_text(encoding="utf-8"))
        if int(result["epochs_completed"]) < epochs:
            raise ValueError(f"{name} was completed for fewer epochs; use a new run name for comparison")
        print(f"Already complete: {output / 'best.pth'}", flush=True)
        return output / "best.pth"
    if not source.is_file():
        raise FileNotFoundError(source)
    set_seed(args.seed)
    cfg, config_path = build_yaml_config(profile, epochs=epochs, batch_size=1, workers=0)
    cfg.yaml_cfg["train_dataloader"]["drop_last"] = False
    device = torch.device("cuda")
    model = cfg.model.to(device)
    matched, total = load_matching_state(model, source)
    if matched < total * 0.95:
        raise RuntimeError(f"Only {matched}/{total} source tensors matched")
    trainable_names = set_adaptation_parameters(model)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    base_ds = get_coco_api_from_dataset(val_loader.dataset)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pth"
    progress_path = output / "progress.json"
    best_map, best_epoch, start_epoch = -1.0, 0, 1
    if last.is_file():
        checkpoint = torch.load(str(last), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_epoch = int(checkpoint["last_epoch"]) + 1
        best_map = float(progress["best_val_map_50_95"])
        best_epoch = int(progress["best_epoch"])
        print(f"Resume {name} from epoch {start_epoch}", flush=True)
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch - 1,
            max_norm=0.1, print_freq=max(len(train_loader), 1), ema=None,
            scaler=scaler if scaler.is_enabled() else None,
        )
        val_stats, _ = evaluate(
            model, criterion, postprocessor, val_loader, base_ds, device, output
        )
        bbox = val_stats.get("coco_eval_bbox", [])
        current_map = float(bbox[0]) if bbox else -1.0
        row = {
            "epoch": epoch,
            "train_loss": float(stats["loss"]),
            "val_map_50_95": current_map,
            "val_map_50": float(bbox[1]) if len(bbox) > 1 else None,
            "seconds_elapsed_this_run": time.time() - start,
        }
        with (output / "log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        metadata = {
            "method": "sequential_domain_incremental_replay",
            "stage": name,
            "profile": profile,
            "source_weight": str(source),
            "trainable_modules": ["encoder", "decoder"],
            "trainable_tensor_names": trainable_names,
            "validation_used_for_selection": True,
            "test_used": False,
            "amp": scaler.is_enabled(),
            "seed": args.seed,
        }
        if current_map > best_map:
            best_map, best_epoch = current_map, epoch
            save_model_checkpoint(output / "best.pth", model, metadata, epoch=epoch)
        save_model_checkpoint(last, model, metadata, optimizer, epoch)
        write_json(progress_path, {
            "last_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_map_50_95": best_map,
            "target_epochs": epochs,
        })
        print(
            f"{name} [{epoch}/{epochs}] loss={row['train_loss']:.5f} "
            f"val_mAP50:95={current_map:.5f} best_epoch={best_epoch}",
            flush=True,
        )
    write_json(complete, {
        "best_weight": str(output / "best.pth"),
        "best_epoch": best_epoch,
        "best_val_map_50_95": best_map,
        "epochs_completed": epochs,
        "elapsed_seconds_this_run": time.time() - start,
        "source_weight": str(source),
        "profile": profile,
        "test_used": False,
    })
    del model, criterion, postprocessor, optimizer, cfg
    torch.cuda.empty_cache()
    return output / "best.pth"


