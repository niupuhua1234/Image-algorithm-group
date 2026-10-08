# -*- coding: utf-8 -*-
"""DEIM car_union v4 部署验收: 类别一致 + IoU>=0.40 贪心匹配 recall 式计数。

路径自适应(基于本文件位置), 无需修改任何绝对路径:
    python eval_deploy.py                 # conf 0.005(验收口径), 单次
    python eval_deploy.py --scan          # 扫描多个 conf
    python eval_deploy.py --imgsz 1280    # 改推理分辨率
"""
import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent          # evaluation/
ROOT = BASE.parent                              # 仓库根目录
DEIM = ROOT / 'engines' / 'deim'                # 引擎代码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# torchvision 0.26 兼容补丁必须在 import deim 之前
sys.path.insert(0, str(DEIM))
os.chdir(str(DEIM))
try:
    import patch_tw
    if hasattr(patch_tw, 'apply'):
        patch_tw.apply()
    print('[patch_tw] 兼容补丁已应用')
except Exception as e:
    print(f'[patch_tw] 未应用: {e}')

import numpy as np
import torch
import torchvision
from PIL import Image
from engine.core import YAMLConfig

IOU = 0.40
NAMES = ['car', 'truck', 'bus', 'van', 'pickup', 'tank']


def iou_matrix(a, b):
    a = a[:, None, :].astype(np.float64)
    b = b[None, :, :].astype(np.float64)
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.maximum(ix2 - ix1, 0); ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'weights' / 'best_stg1.pth'))
    ap.add_argument('--config', default='configs/deim_rtdetrv2/deim_car_union_v4.yml')
    ap.add_argument('--json', default=str(ROOT / 'test' / 'annotations.json'))
    ap.add_argument('--imgroot', default=str(ROOT / 'test' / 'images'))
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--conf', type=float, default=0.005)
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = YAMLConfig(args.config)
    model = cfg.model
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    model.to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device)
    if isinstance(ck.get('ema'), dict) and 'module' in ck['ema']:
        sd = ck['ema']['module']
        print('使用 EMA 权重')
    else:
        sd = ck.get('model', ck)
        print('使用 model 权重')
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()

    d = json.loads(Path(args.json).read_text(encoding='utf-8'))
    ann_by = {}
    for a in d['annotations']:
        ann_by.setdefault(a['image_id'], []).append(a)
    print(f'test 块数={len(d["images"])} GT目标={len(d["annotations"])}')
    print(f'推理输入={args.imgsz}  图片目录={args.imgroot}')

    items = []
    missing = 0
    for im in d['images']:
        p = os.path.join(args.imgroot, im['file_name'])
        if os.path.exists(p):
            items.append((p, ann_by.get(im['id'], [])))
        else:
            missing += 1
    if missing:
        print(f'警告: {missing} 张图片未找到')
    print(f'参与评估 {len(items)} 块')

    IMGSZ = args.imgsz

    def decode(pred_logits, pred_boxes, conf_thr):
        bbox = torchvision.ops.box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox = bbox * IMGSZ
        scores = torch.sigmoid(pred_logits)
        scores, index = torch.topk(scores.flatten(1), 300, dim=-1)
        labels = index % 6
        index = index // 6
        boxes = bbox.gather(1, index.unsqueeze(-1).repeat(1, 1, 4))
        keep = scores[0] >= conf_thr
        return labels[0][keep].cpu().numpy(), boxes[0][keep].cpu().numpy(), scores[0][keep].cpu().numpy()

    def evaluate(conf, warmup=3):
        n_gt = n_ok = 0
        times, scores_all = [], []
        pc, po = Counter(), Counter()
        for i, (p, anns) in enumerate(items):
            with Image.open(p) as ip:
                ip = ip.copy()
            W, H = ip.size
            sc = IMGSZ / max(W, H)
            img = ip.resize((IMGSZ, IMGSZ), Image.BILINEAR) if IMGSZ != max(W, H) else ip
            gtb = [a['bbox'] for a in anns]
            gtc = [a['category_id'] for a in anns]
            n_gt += len(gtc)
            gt = np.array([(x * sc, y * sc, (x + w) * sc, (y + h) * sc) for (x, y, w, h) in gtb]).reshape(-1, 4) \
                if gtb else np.zeros((0, 4))
            x = torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            torch.cuda.synchronize() if args.device == 'cuda' else None
            t1 = time.perf_counter()
            if i >= warmup:
                times.append((t1 - t0) * 1000)
            labels, boxes, scores = decode(out['pred_logits'], out['pred_boxes'], conf)
            scores_all.extend(scores.tolist())
            for c in gtc:
                pc[c] += 1
            if boxes.shape[0] == 0 or not gtc:
                continue
            iou = iou_matrix(boxes, gt)
            matched = set()
            for pi in range(boxes.shape[0]):
                bg, bi = -1, 0.0
                for gi in range(len(gtc)):
                    if gi in matched:
                        continue
                    if iou[pi, gi] > bi:
                        bi, bg = iou[pi, gi], gi
                if bg >= 0 and bi >= IOU and int(labels[pi]) == gtc[bg]:
                    matched.add(bg)
                    n_ok += 1
                    po[gtc[bg]] += 1
        acc = n_ok / n_gt if n_gt else 0
        ms = float(np.mean(times)) if times else float('nan')
        sm = float(np.mean(scores_all)) if scores_all else float('nan')
        return acc, n_ok, n_gt, ms, sm, pc, po

    if args.scan:
        for c in (0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30):
            acc, ok, gt, ms, sm, pc, po = evaluate(c)
            print(f'  conf>={c:<5}: acc={acc:.4f} ({ok}/{gt})  ms={ms:.1f}  mean_score={sm:.3f}')
    else:
        acc, ok, gt, ms, sm, pc, po = evaluate(args.conf)
        print(f'\nconf={args.conf}: acc={acc:.4f} ({ok}/{gt})  avg_ms={ms:.1f}  mean_score={sm:.3f}')
        print('各类准确率:')
        for i, nm in enumerate(NAMES):
            if pc.get(i, 0):
                print(f'  {nm:10s}: {po.get(i,0)}/{pc[i]} = {po.get(i,0)/pc[i]:.2%}')
        print('\n参考: 训练侧实测 90.36% @conf 0.005 (TRT 单块 5.5ms / PT 单块 ~15ms)')


if __name__ == '__main__':
    main()
