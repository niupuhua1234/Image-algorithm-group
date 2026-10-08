"""Adapt standard or FOMAML RT-DETR initialization on a target K-shot support."""

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    import yaml

    defaults = yaml.safe_load((ROOT / "project_config.yaml").read_text(encoding="utf-8"))[
        "meta_learning"
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("standard", "fomaml"), required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--profile", default=defaults["target_profile"])
    parser.add_argument("--source-profile", default=defaults["source_profile"])
    parser.add_argument("--gpu", default=str(defaults["gpu"]))
    parser.add_argument("--epochs", type=int, default=defaults["adaptation_epochs"])
    parser.add_argument("--lr", type=float, default=defaults["adaptation_lr"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--initial-weight", help="Override ordinary-transfer initialization")
    parser.add_argument("--fomaml-weight", help="Override FOMAML initialization")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    # Never overwrite a completed historical adaptation when reorganizing workflows.
    run_name = f"{args.method}_{args.shots:03d}shot" + ("_smoke" if args.smoke_test else "")
    existing = ROOT / "outputs" / "meta_learning" / "adaptation" / args.profile / run_name
    if existing.exists() and any(existing.iterdir()):
        raise FileExistsError(f"Run already exists: {existing}. Use a new profile for a new experiment.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from common import (
        OUTPUT_ROOT,
        SUPPORT_ROOT,
        build_yaml_config,
        load_matching_state,
        resolve_source_weight,
        save_model_checkpoint,
        set_adaptation_parameters,
        set_seed,
        write_json,
    )
    from competition.platform_deps import configure_rtdetr_imports

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select the CUDA PyCharm interpreter.")
    if args.smoke_test:
        args.epochs = 1
    set_seed(args.seed + args.shots)
    support_json = SUPPORT_ROOT / args.profile / f"support_{args.shots:03d}.json"
    if not support_json.is_file():
        raise FileNotFoundError(f"Run 03_prepare_fewshot_support.py first: {support_json}")
    if args.method == "standard":
        initial_weight = resolve_source_weight(args.initial_weight)
    else:
        initial_weight = Path(args.fomaml_weight) if args.fomaml_weight else (
            OUTPUT_ROOT / "fomaml" / args.source_profile / "best.pth"
        )
        if not initial_weight.is_file():
            raise FileNotFoundError(f"Run 02_train_fomaml.py first: {initial_weight}")

    cfg, config_path = build_yaml_config(args.profile, epochs=args.epochs, batch_size=1, workers=0)
    prepared = ROOT / "datasets" / "prepared" / args.profile
    cfg.yaml_cfg["train_dataloader"]["dataset"]["ann_file"] = support_json.as_posix()
    cfg.yaml_cfg["train_dataloader"]["dataset"]["img_folder"] = (
        prepared / "images" / "train"
    ).as_posix()
    cfg.yaml_cfg["train_dataloader"]["drop_last"] = False
    device = torch.device("cuda")
    model = cfg.model.to(device)
    load_matching_state(model, initial_weight)
    trainable_names = set_adaptation_parameters(model)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    configure_rtdetr_imports()
    from src.data import get_coco_api_from_dataset
    from src.solver.det_engine import evaluate, train_one_epoch

    base_ds = get_coco_api_from_dataset(val_loader.dataset)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )
    run_name = f"{args.method}_{args.shots:03d}shot"
    if args.smoke_test:
        run_name += "_smoke"
    output = OUTPUT_ROOT / "adaptation" / args.profile / run_name
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tensorboard"))
    best_map = -1.0
    best_epoch = -1
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch - 1,
            max_norm=0.1,
            print_freq=max(len(train_loader), 1),
            ema=None,
            scaler=None,
        )
        val_stats, _ = evaluate(
            model, criterion, postprocessor, val_loader, base_ds, device, output
        )
        bbox = val_stats.get("coco_eval_bbox", [])
        current_map = float(bbox[0]) if bbox else -1.0
        row = {
            "epoch": epoch,
            "train_loss": train_stats.get("loss"),
            "val_map_50_95": current_map,
            "val_map_50": float(bbox[1]) if len(bbox) > 1 else None,
        }
        with (output / "log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        writer.add_scalar("train/loss", row["train_loss"], epoch)
        writer.add_scalar("val/mAP_50_95", current_map, epoch)
        if row["val_map_50"] is not None:
            writer.add_scalar("val/mAP_50", row["val_map_50"], epoch)
        writer.flush()
        metadata = {
            "method": args.method,
            "shots": args.shots,
            "support_annotation": str(support_json),
            "initial_weight": str(initial_weight),
            "validation_used_for_selection": True,
            "test_used": False,
            "trainable_modules": ["encoder", "decoder"],
            "trainable_tensor_names": trainable_names,
            "config": str(config_path),
            "arguments": vars(args),
        }
        save_model_checkpoint(output / "last.pth", model, metadata, optimizer, epoch)
        if current_map > best_map:
            best_map = current_map
            best_epoch = epoch
            metadata.update({"best_epoch": best_epoch, "best_val_map_50_95": best_map})
            save_model_checkpoint(output / "best.pth", model, metadata, epoch=epoch)
        print(
            f"{args.method} {args.shots}-shot [{epoch}/{args.epochs}] "
            f"loss={row['train_loss']:.5f} val_mAP50:95={current_map:.5f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    writer.close()
    write_json(
        output / "training_complete.json",
        {
            "best_weight": str(output / "best.pth"),
            "best_epoch": best_epoch,
            "best_val_map_50_95": best_map,
            "elapsed_seconds": time.time() - start,
            "test_used": False,
        },
    )
    print(f"Adapted best weight: {output / 'best.pth'}")


if __name__ == "__main__":
    main()
