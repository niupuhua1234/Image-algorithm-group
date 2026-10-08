"""Refine an RT-DETR initialization with first-order MAML source tasks."""

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    import yaml

    defaults = yaml.safe_load((ROOT / "project_config.yaml").read_text(encoding="utf-8"))[
        "meta_learning"
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=defaults["source_profile"])
    parser.add_argument("--gpu", default=str(defaults["gpu"]))
    parser.add_argument("--epochs", type=int, default=defaults["meta_epochs"])
    parser.add_argument("--episodes-per-epoch", type=int, default=defaults["episodes_per_epoch"])
    parser.add_argument("--support", type=int, default=defaults["support_per_task"])
    parser.add_argument("--query", type=int, default=defaults["query_per_task"])
    parser.add_argument("--inner-lr", type=float, default=defaults["inner_lr"])
    parser.add_argument("--meta-lr", type=float, default=defaults["meta_lr"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--initial-weight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sample_episode(rng, population, support, query):
    count = support + query
    if count > len(population):
        raise ValueError(f"Requested {count} images from a task containing {len(population)}")
    return rng.sample(population, count)


def run_task(task_model, criterion, dataset, indices, support_size, inner_lr, device):
    from common import detection_loss, move_batch

    task_model.train()
    inner_optimizer = __import__("torch").optim.SGD(
        (parameter for parameter in task_model.parameters() if parameter.requires_grad),
        lr=inner_lr,
    )
    inner_optimizer.zero_grad(set_to_none=True)
    support_loss = 0.0
    for index in indices[:support_size]:
        image, target = dataset[index]
        samples, targets = move_batch(image.unsqueeze(0), [target], device)
        loss, _ = detection_loss(task_model, criterion, samples, targets)
        (loss / support_size).backward()
        support_loss += float(loss.detach()) / support_size
    __import__("torch").nn.utils.clip_grad_norm_(
        (parameter for parameter in task_model.parameters() if parameter.requires_grad), 0.1
    )
    inner_optimizer.step()
    inner_optimizer.zero_grad(set_to_none=True)

    query_indices = indices[support_size:]
    query_loss = 0.0
    for index in query_indices:
        image, target = dataset[index]
        samples, targets = move_batch(image.unsqueeze(0), [target], device)
        loss, _ = detection_loss(task_model, criterion, samples, targets)
        (loss / len(query_indices)).backward()
        query_loss += float(loss.detach()) / len(query_indices)
    return support_loss, query_loss


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from common import (
        OUTPUT_ROOT,
        TASK_ROOT,
        build_yaml_config,
        load_matching_state,
        resolve_source_weight,
        save_model_checkpoint,
        set_adaptation_parameters,
        set_seed,
        write_json,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select the CUDA PyCharm interpreter.")
    if args.smoke_test:
        args.epochs = 1
        args.episodes_per_epoch = 1
        args.support = 1
        args.query = 1
    set_seed(args.seed)
    device = torch.device("cuda")
    source_weight = resolve_source_weight(args.initial_weight)
    domain_manifest = TASK_ROOT / args.profile / "domain_manifest.csv"
    if not domain_manifest.is_file():
        raise FileNotFoundError(f"Run 01_prepare_meta_tasks.py first: {domain_manifest}")

    cfg, config_path = build_yaml_config(args.profile, epochs=1, batch_size=1, workers=0)
    dataset = cfg.train_dataloader.dataset
    image_to_index = {
        Path(dataset.coco.imgs[image_id]["file_name"]).name: index
        for index, image_id in enumerate(dataset.ids)
    }
    domains = defaultdict(list)
    with domain_manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["image"] in image_to_index:
                domains[row["domain"]].append(image_to_index[row["image"]])
    if len(domains) < 2:
        raise RuntimeError(f"At least two usable meta tasks are required, found {dict(domains)}")
    for name, population in domains.items():
        if len(population) < args.support + args.query:
            raise RuntimeError(f"Task {name} has only {len(population)} usable images")

    output = OUTPUT_ROOT / "fomaml" / args.profile
    if args.smoke_test:
        output = OUTPUT_ROOT / "smoke" / "fomaml"
    output.mkdir(parents=True, exist_ok=True)
    meta_model = cfg.model.cpu()
    load_matching_state(meta_model, source_weight)
    trainable_names = set_adaptation_parameters(meta_model)
    criterion = cfg.criterion.to(device)
    task_model = copy.deepcopy(meta_model).to(device)
    set_adaptation_parameters(task_model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in meta_model.parameters() if parameter.requires_grad),
        lr=args.meta_lr,
        weight_decay=1e-4,
    )
    start_epoch = 1
    best_query_loss = float("inf")
    last_path = output / "last.pth"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(str(last_path), map_location="cpu")
        meta_model.load_state_dict(checkpoint["model"], strict=True)
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("last_epoch", 0)) + 1
        best_query_loss = float(checkpoint.get("metadata", {}).get("best_query_loss", best_query_loss))
        print(f"Resume from epoch {start_epoch}: {last_path}")

    rng = random.Random(args.seed)
    writer = SummaryWriter(str(output / "tensorboard"))
    log_path = output / "log.jsonl"
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_support = 0.0
        epoch_query = 0.0
        for episode in range(1, args.episodes_per_epoch + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated = {
                name: torch.zeros_like(parameter)
                for name, parameter in meta_model.named_parameters()
                if parameter.requires_grad
            }
            support_total = 0.0
            query_total = 0.0
            for domain_name, population in sorted(domains.items()):
                task_model.load_state_dict(meta_model.state_dict(), strict=True)
                selected = sample_episode(rng, population, args.support, args.query)
                support_loss, query_loss = run_task(
                    task_model, criterion, dataset, selected, args.support, args.inner_lr, device
                )
                support_total += support_loss
                query_total += query_loss
                for name, parameter in task_model.named_parameters():
                    if name in accumulated and parameter.grad is not None:
                        accumulated[name].add_(parameter.grad.detach().cpu())
                task_model.zero_grad(set_to_none=True)

            for name, parameter in meta_model.named_parameters():
                if name in accumulated:
                    parameter.grad = accumulated[name] / len(domains)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in meta_model.parameters() if parameter.requires_grad), 0.1
            )
            optimizer.step()
            support_total /= len(domains)
            query_total /= len(domains)
            epoch_support += support_total
            epoch_query += query_total
            print(
                f"Epoch {epoch:02d}/{args.epochs} episode "
                f"{episode:02d}/{args.episodes_per_epoch} support={support_total:.5f} "
                f"query={query_total:.5f}",
                flush=True,
            )

        epoch_support /= args.episodes_per_epoch
        epoch_query /= args.episodes_per_epoch
        row = {"epoch": epoch, "support_loss": epoch_support, "query_loss": epoch_query}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        writer.add_scalar("meta/support_loss", epoch_support, epoch)
        writer.add_scalar("meta/query_loss", epoch_query, epoch)
        writer.flush()
        metadata = {
            "method": "FOMAML",
            "profile": args.profile,
            "source_weight": str(source_weight),
            "source_domains": {name: len(values) for name, values in domains.items()},
            "target_profile_used": False,
            "trainable_modules": ["encoder", "decoder"],
            "trainable_tensor_names": trainable_names,
            "config": str(config_path),
            "arguments": vars(args),
            "best_query_loss": min(best_query_loss, epoch_query),
        }
        save_model_checkpoint(last_path, meta_model, metadata, optimizer, epoch)
        if epoch_query < best_query_loss:
            best_query_loss = epoch_query
            metadata["best_query_loss"] = best_query_loss
            save_model_checkpoint(output / "best.pth", meta_model, metadata, epoch=epoch)

    writer.close()
    write_json(
        output / "training_complete.json",
        {
            "best_weight": str(output / "best.pth"),
            "best_source_query_loss": best_query_loss,
            "elapsed_seconds": time.time() - start,
            "target_profile_used": False,
        },
    )
    print(f"FOMAML best initialization: {output / 'best.pth'}")


if __name__ == "__main__":
    main()
