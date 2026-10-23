# -*- coding: utf-8 -*-
"""按类别分别定置信度阈值, 并输出各阶段最终可用的操作点表。

为什么需要按类别定阈值:
    实测发现同一份模型对不同类别的置信度分布差异很大 ——
    3B 新类需要 conf 0.60 才达标, 而 base 老类要压到 0.20 才够召回。
    用统一阈值必然要么误检暴涨, 要么老类召回崩塌。

方法论(保持干净口径):
    ① 在 val 上, 对每个类别独立地找出「该类 recall >= 目标 的最高 conf」
    ② 把这些 conf 固定下来, 拿到 test 上跑一次, 报每类的 recall/precision
    ③ test 全程不参与阈值选择

用法:
    python per_class_threshold.py --config <cfg> --ckpt <pth> --tag base \
        --ann-val <val.json> --ann-test <test.json> --img-dir <root> --target 0.90
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
sys.path.insert(0, str(DEIM))

import numpy as np                     # noqa: E402
import torch                           # noqa: E402

CLASSES = ['car1', 'car2', 'car3', 'car4',
           'military_fuel_tanker', 'radar_vehicle',
           'military_motorcycle', 'bridge_layer']

CONF_GRID = [round(0.05 + i * 0.02, 2) for i in range(46)]   # 0.05 ~ 0.95


def iou_matrix(a, b):
    a = a[:, None, :].astype(np.float64)
    b = b[None, :, :].astype(np.float64)
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def collect(model, items, ann_by, imgsz, device, tag):
    """对每张图跑一次前向, 缓存 (boxes_px, scores, labels) 与 GT。"""
    import torchvision
    from PIL import Image
    out = []
    t0 = __import__('time').time()
    for idx, (p, im) in enumerate(items):
        W, H = im['width'], im['height']
        img = Image.open(p).convert('RGB').resize((imgsz, imgsz), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).astype(np.float32).transpose(2, 0, 1)[None]
                             / 255.0).to(device)
        with torch.no_grad():
            o = model(x)
        sc = torch.sigmoid(o['pred_logits'])
        n_cls = sc.shape[-1]
        vals, index = torch.topk(sc.flatten(1), sc.shape[1], dim=-1)
        labels = (index % n_cls)[0].cpu().numpy()
        qidx = (index // n_cls)[0]
        boxes = torchvision.ops.box_convert(o['pred_boxes'], 'cxcywh', 'xyxy') * imgsz
        boxes = boxes.gather(1, qidx.unsqueeze(-1).repeat(1, 1, 4))[0].cpu().numpy()
        scores = vals[0].cpu().numpy()

        anns = ann_by.get(im['id'], [])
        sx, sy = imgsz / W, imgsz / H
        gtb = np.array([[a['bbox'][0] * sx, a['bbox'][1] * sy,
                         (a['bbox'][0] + a['bbox'][2]) * sx,
                         (a['bbox'][1] + a['bbox'][3]) * sy]
                        for a in anns], dtype=np.float64).reshape(-1, 4)
        gtl = np.array([a['category_id'] for a in anns], dtype=np.int64)
        out.append((boxes, scores, labels, gtb, gtl))
        if (idx + 1) % 200 == 0:
            print(f'    {tag} {idx+1}/{len(items)}')
    print(f'    {tag} 完成 {len(items)} 张, {__import__("time").time()-t0:.0f}s')
    return out


def evaluate(cached, conf_by_cls, iou_thr=0.40):
    """按 per-class conf 过滤, 返回逐类 recall/precision/tp/fp/gt。"""
    stats = {}
    for boxes, scores, labels, gtb, gtl in cached:
        keep = np.array([scores[i] >= conf_by_cls.get(int(labels[i]), 1.0)
                         for i in range(len(scores))], dtype=bool)
        b, l = boxes[keep], labels[keep]
        iou = iou_matrix(b, gtb) if (len(b) and len(gtb)) else None
        taken = set()
        for pi in range(len(b)):
            if iou is None:
                break
            bi, bg = 0.0, -1
            for gi in range(len(gtl)):
                if gi in taken:
                    continue
                if iou[pi, gi] > bi:
                    bi, bg = iou[pi, gi], gi
            if bg >= 0 and bi >= iou_thr and int(l[pi]) == int(gtl[bg]):
                taken.add(bg)
                s = stats.setdefault(int(gtl[bg]), {'tp': 0, 'gt': 0, 'pred': 0})
                s['tp'] += 1
        for gi in range(len(gtl)):
            stats.setdefault(int(gtl[gi]), {'tp': 0, 'gt': 0, 'pred': 0})['gt'] += 1
        for li in l:
            stats.setdefault(int(li), {'tp': 0, 'gt': 0, 'pred': 0})['pred'] += 1
    for k, s in stats.items():
        s['recall'] = s['tp'] / s['gt'] if s['gt'] else 0.0
        s['precision'] = s['tp'] / s['pred'] if s['pred'] else 0.0
        s['fp'] = s['pred'] - s['tp']
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ann-val', required=True)
    ap.add_argument('--ann-test', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--target', type=float, default=0.90, help='每类 recall 目标')
    ap.add_argument('--total-budget', type=float, default=0.90,
                    help='整体(全部类别合计)recall 目标')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    os.chdir(str(DEIM))
    try:
        import patch_tw
        patch_tw.apply()
    except Exception:
        pass
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.config)
    model = cfg.model
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['ema']['module'] if isinstance(ck.get('ema'), dict) and 'module' in ck['ema'] else ck.get('model', ck)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model = model.cuda().eval()

    print('=' * 90)
    print(f'按类别定阈值  [{args.tag}]')
    print('=' * 90)

    def prep(ann_file):
        d = json.loads(Path(ann_file).read_text(encoding='utf-8'))
        ann_by = {}
        for a in d['annotations']:
            ann_by.setdefault(a['image_id'], []).append(a)
        items = []
        root = Path(args.img_dir)
        for im in d['images']:
            p = root / im['file_name']
            if p.exists():
                items.append((p, im))
        return items, ann_by

    val_items, val_ann = prep(args.ann_val)
    test_items, test_ann = prep(args.ann_test)
    print(f'  val  {len(val_items)} 图    test {len(test_items)} 图')

    print('\n[1/3] val 上收集预测')
    val_cached = collect(model, val_items, val_ann, args.imgsz, 'cuda', 'val')
    print('\n[2/3] test 上收集预测')
    test_cached = collect(model, test_items, test_ann, args.imgsz, 'cuda', 'test')

    # ---- 每类在 val 上找 recall 达标的最高 conf ----
    print('\n[3/3] 逐类定阈值')
    present = sorted({int(c) for _, _, _, _, gtl in val_cached for c in gtl})
    chosen = {}
    val_report = {}
    for cls in present:
        best = None
        for c in CONF_GRID:
            cfg_c = {k: 1.0 for k in range(8)}
            cfg_c[cls] = c
            st = evaluate(val_cached, cfg_c)
            s = st.get(cls, {'recall': 0, 'precision': 0})
            if s['recall'] >= args.target:
                if best is None or c > best[0]:
                    best = (c, s['recall'], s['precision'])
        if best:
            chosen[cls] = best[0]
            val_report[cls] = {'conf': best[0], 'recall': best[1], 'precision': best[2]}
        else:
            # val 上达不到目标, 用最低阈值
            chosen[cls] = CONF_GRID[0]
            cfg_c = {k: 1.0 for k in range(8)}
            cfg_c[cls] = CONF_GRID[0]
            st = evaluate(val_cached, cfg_c)
            s = st.get(cls, {'recall': 0, 'precision': 0})
            val_report[cls] = {'conf': CONF_GRID[0], 'recall': s['recall'],
                               'precision': s['precision'], 'note': 'val 未达标, 取最低阈值'}

    # ---- 用 chosen 到 test 上验证 ----
    test_stats = evaluate(test_cached, chosen)
    total_tp = sum(s['tp'] for s in test_stats.values())
    total_gt = sum(s['gt'] for s in test_stats.values())
    total_pred = sum(s['pred'] for s in test_stats.values())
    total_recall = total_tp / total_gt if total_gt else 0
    total_prec = total_tp / total_pred if total_pred else 0

    print()
    print(f'  {"类别":<24}{"conf":>7}{"recall":>9}{"precision":>11}{"TP":>7}{"GT":>7}{"FP":>7}')
    print('  ' + '-' * 72)
    for cls in sorted(present):
        s = test_stats.get(cls, {'recall': 0, 'precision': 0, 'tp': 0, 'gt': 0, 'fp': 0})
        name = CLASSES[cls] if cls < len(CLASSES) else f'cls{cls}'
        print(f'  {name:<24}{chosen[cls]:>7.2f}{s["recall"]:>9.4f}{s["precision"]:>11.4f}'
              f'{s["tp"]:>7}{s["gt"]:>7}{s["fp"]:>7}')
    print('  ' + '-' * 72)
    print(f'  {"合计":<24}{"":>7}{total_recall:>9.4f}{total_prec:>11.4f}'
          f'{total_tp:>7}{total_gt:>7}{total_pred-total_tp:>7}')

    print()
    print(f'  整体 recall {total_recall:.4f}   precision {total_prec:.4f}   '
          f'误检/图 {(total_pred-total_tp)/max(len(test_items),1):.2f}')

    def cname(k):
        return CLASSES[k] if 0 <= k < len(CLASSES) else f'cls{k}'

    out = {
        'tag': args.tag, 'config': args.config, 'ckpt': str(args.ckpt),
        'target': args.target, 'imgsz': args.imgsz,
        'chosen_conf': {cname(k): v for k, v in chosen.items()},
        'val_report': {cname(k): v for k, v in val_report.items()},
        'test_per_class': {cname(k): test_stats.get(k, {}) for k in present},
        'test_overall': {'recall': round(total_recall, 4), 'precision': round(total_prec, 4),
                         'fp': int(total_pred - total_tp),
                         'fp_per_img': round((total_pred - total_tp) / max(len(test_items), 1), 2),
                         'tp': int(total_tp), 'gt': int(total_gt)},
    }
    p = Path(args.out) if args.out else (DEIM / 'output' / f'perclass_{args.tag}.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  已保存 {p}')


if __name__ == '__main__':
    main()
