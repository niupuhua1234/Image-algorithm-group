"""Export a fixed-shape RT-DETR graph for TensorRT conversion on Jetson."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import sys
from pathlib import Path

from competition.paths import ROOT, load_project_config, require_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--weights")
    parser.add_argument("--output")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from competition.platform_deps import configure_rtdetr_imports
    from competition.rtdetr_runtime import ENGINE, build_config

    defaults = load_project_config()["rtdetr"]
    config_path = build_config(
        args.profile, defaults["epochs"], defaults["batch_size"], defaults["workers"]
    )
    weights = Path(args.weights) if args.weights else ROOT / "outputs" / "rtdetr" / args.profile / "best.pth"
    output = Path(args.output) if args.output else ROOT / "outputs" / "exports" / f"rtdetr_{args.profile}_1280x1024.onnx"
    require_path(weights, "RT-DETR weight")
    output.parent.mkdir(parents=True, exist_ok=True)
    configure_rtdetr_imports()
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    checkpoint = torch.load(str(weights), map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=True)

    class ExportModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy().eval()
            self.postprocessor = cfg.postprocessor.deploy().eval()

        def forward(self, images, original_sizes):
            return self.postprocessor(self.model(images), original_sizes)

    model = ExportModel().cpu()

    # PyTorch 2.0 may fuse MultiheadAttention into aten::scaled_dot_product_attention,
    # which its legacy ONNX exporter cannot translate. This mathematically
    # equivalent implementation exports as MatMul/Softmax/MatMul operations.
    def exportable_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        scale = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal:
            causal = torch.ones(
                scores.size(-2), scores.size(-1), dtype=torch.bool, device=scores.device
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        probabilities = torch.softmax(scores, dim=-1)
        if dropout_p:
            probabilities = torch.dropout(probabilities, dropout_p, train=False)
        return torch.matmul(probabilities, value)

    functional.scaled_dot_product_attention = exportable_attention
    images = torch.zeros(1, 3, 1024, 1280, dtype=torch.float32)
    original_sizes = torch.tensor([[1280, 1024]], dtype=torch.int64)
    torch.onnx.export(
        model,
        (images, original_sizes),
        str(output),
        input_names=["images", "original_sizes"],
        output_names=["labels", "boxes", "scores"],
        opset_version=16,
        do_constant_folding=True,
    )
    import onnx
    onnx.checker.check_model(onnx.load(str(output)))
    print(f"ONNX check passed: {output}")


if __name__ == "__main__":
    main()
