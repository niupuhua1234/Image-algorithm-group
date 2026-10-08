"""Shared helpers for the independent RT-DETR FOMAML workflow."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
META_ROOT = ROOT / "meta_learning"
OUTPUT_ROOT = ROOT / "outputs" / "meta_learning"
TASK_ROOT = META_ROOT / "tasks"
SUPPORT_ROOT = META_ROOT / "support"
TRAINABLE_PREFIXES = ("encoder.", "decoder.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_model_state(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_matching_state(model, path):
    source = checkpoint_model_state(path)
    current = model.state_dict()
    matched = {
        name: value
        for name, value in source.items()
        if name in current and current[name].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(
        f"Loaded {len(matched)}/{len(current)} tensors from {path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return len(matched), len(current)


def set_adaptation_parameters(model):
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise RuntimeError("No encoder/decoder parameters were selected for adaptation.")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} (encoder + decoder)")
    return names


def move_batch(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    return samples, targets


def detection_loss(model, criterion, samples, targets):
    outputs = model(samples, targets)
    losses = criterion(outputs, targets)
    return sum(losses.values()), losses


def save_model_checkpoint(path, model, metadata, optimizer=None, epoch=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metadata": metadata,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        state["last_epoch"] = int(epoch)
    torch.save(state, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_source_weight(explicit=None):
    candidates = [
        Path(explicit) if explicit else None,
        ROOT / "outputs" / "rtdetr" / "base" / "best.pth",
        ROOT / "weights" / "rtdetr" / "vedai_real_synth_best.pth",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No RT-DETR source weight found. Train base first or pass --initial-weight."
    )


def build_yaml_config(profile, epochs=1, batch_size=1, workers=0):
    from competition.platform_deps import configure_rtdetr_imports
    from competition.rtdetr_runtime import build_config

    config_path = build_config(profile, epochs, batch_size, workers)
    configure_rtdetr_imports()
    from src.core import YAMLConfig

    return YAMLConfig(str(config_path)), config_path
