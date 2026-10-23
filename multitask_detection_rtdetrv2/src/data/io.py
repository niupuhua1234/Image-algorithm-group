# -*- coding: utf-8 -*-
"""图像读写与预处理。

关键性能事实(实测, RTX 5080):
    PIL 读图 + PIL resize + numpy 转 float + .cuda()   ~61 ms
    cv2 解码 + uint8 上卡 + GPU resize                 ~16 ms

差异来自 PIL 的单线程 CPU 重采样。因此训练用 dataloader 走 torchvision 链路,
**推理**统一走这里的 GPU 预处理路径。
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

_IMREAD_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_image_tensor(path, imgsz: int = 1024, device: str = 'cuda'
                      ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """读图并预处理成模型输入。

    流程: cv2 解码 -> BGR2RGB -> uint8 上卡 -> permute/float/归一化 -> GPU 双线性 resize

    Args:
        path:   图片路径
        imgsz:  目标边长(训练 960, 推理默认 1024)
        device: 'cuda' 或 'cpu'

    Returns:
        (x, (W, H)) —— x 形状 [1,3,imgsz,imgsz], 值为原图尺寸用于坐标还原
    """
    import cv2

    p = Path(path)
    if p.suffix.lower() not in _IMREAD_EXT:
        raise ValueError(f'不支持的图片格式: {p.suffix}')
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(f'读图失败: {p}')
    H, W = im.shape[:2]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if device == 'cpu':
        x = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        if max(W, H) != imgsz:
            x = F.interpolate(x, size=(imgsz, imgsz), mode='bilinear', align_corners=False)
    else:
        x = torch.from_numpy(im).to(device)                      # uint8 传输, 1/4 流量
        x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255)
        if max(W, H) != imgsz:
            x = F.interpolate(x, size=(imgsz, imgsz), mode='bilinear', align_corners=False)
    return x, (W, H)


def load_image_pil(path, imgsz: int = 1024):
    """PIL 路径(仅供对照与训练链路使用, 推理不建议)。

    返回 (numpy [1,3,H,W] float32 0~1, (W,H))。
    """
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB')
        W, H = im.size
        if max(W, H) != imgsz:
            im = im.resize((imgsz, imgsz), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return arr, (W, H)


def boxes_cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, size) -> torch.Tensor:
    """归一化 cxcywh -> 原图绝对 xyxy。"""
    import torchvision
    W, H = float(size[0]), float(size[1])
    xyxy = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
    return xyxy * torch.tensor([W, H, W, H], device=xyxy.device, dtype=xyxy.dtype)
