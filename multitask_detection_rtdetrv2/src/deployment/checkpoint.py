# -*- coding: utf-8 -*-
"""权重加载。

放在 src/ 而不是 tools/, 因为推理运行时也要用 —— tools/ 只放 CLI 入口。
"""
from pathlib import Path
from typing import Tuple

import torch


def load_ckpt_into(model, ckpt_path, device: str = 'cuda', verbose: bool = True):
    """把权重加载进模型。

    DEIM / RT-DETR 的 checkpoint 同时含 `model` 与 `ema` 两份权重,
    推理时应优先用 **EMA**(训练侧实测更稳)。
    """
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and isinstance(ck.get('ema'), dict) and 'module' in ck['ema']:
        sd, which = ck['ema']['module'], 'ema'
    elif isinstance(ck, dict) and 'model' in ck:
        sd, which = ck['model'], 'model'
    else:
        sd, which = ck, 'raw'

    msd = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd if k not in matched]
    shape_bad = [k for k in sd if k in msd and msd[k].shape != sd[k].shape]

    model.load_state_dict(matched, strict=False)
    if verbose:
        print(f'  权重来源: {which}   匹配 {len(matched)}/{len(msd)}')
        if missing:
            print(f'  [!] 未匹配 {len(missing)} 个, 例如 {missing[:3]}')
        if shape_bad:
            print(f'  [!] 形状不符 {len(shape_bad)} 个, 例如 {shape_bad[:3]}')
    return model.to(device).eval()


def describe_ckpt(ckpt_path) -> dict:
    """看一眼 checkpoint 里有什么, 便于排查。"""
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    out = {'path': str(ckpt_path), 'size_mb': round(Path(ckpt_path).stat().st_size / 1e6, 1)}
    if isinstance(ck, dict):
        out['top_keys'] = list(ck.keys())
        for k in ('date', 'last_epoch'):
            if k in ck:
                out[k] = str(ck[k])
        for k in ('ema', 'model'):
            if k in ck and isinstance(ck[k], dict):
                inner = ck[k].get('module', ck[k])
                out[f'{k}_num_tensors'] = len(inner)
    return out
