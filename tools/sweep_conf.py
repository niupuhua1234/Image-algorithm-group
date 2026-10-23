# -*- coding: utf-8 -*-
"""扫描置信度阈值, 在"recall 达标"的前提下把误警率压到最低。

背景:
    项目验收口径是 recall 式(类别一致 + IoU>=0.40, 分母为 GT 总数, 不惩罚误检),
    所以只要 recall 过线, 阈值就可以往上抬 —— 每抬一点, 误检就少一批。

    本脚本对同一个模型只跑一遍前向, 收集全部预测后再按不同 conf 过滤,
    因此扫 10 个阈值和扫 1 个阈值耗时几乎一样。

输出:
    每个操作点的 recall / precision / FP 数 / 每图误检数, 并推荐
    "recall 仍达标(默认 >=0.90)的最高 conf"。

用法:
    python sweep_conf.py \
        --config configs/deim_rtdetrv2/deim_vistarget_base.yml \
        --ckpt output/deim_vistarget_base/best_stg1.pth \
        --split test --target-recall 0.90
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
sys.path.insert(0, str(DEIM))

import numpy as np                     # noqa: E402
import torch                           # noqa: E402

CONF_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15,
             0.20, 0.25, 0.30, 0.40, 0.50]


def iou_matrix(a, b):
    a = a[:, None, :].astype(np.float64)
    b = b[None, :, :].astype(np.float64)
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--iou', type=float, default=0.40)
    ap.add_argument('--target-recall', type=float, default=0.90)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--conf-grid', default=None,
                    help='自定义阈值网格, 如 "0.30:0.60:0.02" (起:止:步长)')
    ap.add_argument('--ann-file', default=None,
                    help='直接指定标注 json(用于老域回归等非常规 split)')
    ap.add_argument('--img-dir', default=None,
                    help='直接指定图片目录(配合 --ann-file)')
    ap.add_argument('--safety-margin', type=float, default=0.0,
                    help='给 recall 目标留的余量(pp), 例如 2 表示按 target+2%% 选点')
    args = ap.parse_args()

    global CONF_GRID
    if args.conf_grid:
        lo, hi, step = (float(x) for x in args.conf_grid.split(':'))
        g = []
        v = lo
        while v <= hi + 1e-9:
            g.append(round(v, 4))
            v += step
        CONF_GRID = g

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
    model = model.to(args.device).eval()

    num_classes = cfg.yaml_cfg.get('num_classes', 4)
    ann_file = cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']
    img_folder = cfg.yaml_cfg['val_dataloader']['dataset']['img_folder']
    if args.split == 'train':
        ann_file = cfg.yaml_cfg['train_dataloader']['dataset']['ann_file']
        img_folder = cfg.yaml_cfg['train_dataloader']['dataset']['img_folder']
    elif args.split == 'test':
        # test 划分与 val 同目录结构, 文件名不同。
        # 只有 img_folder 形如 .../images/val 时, 才可以换成 .../images/test;
        # 增量阶段的 img_folder 直接指向数据集根目录(file_name 自带子路径),
        # 此时任何拼接都会指错地方, 必须原样保留。
        cand = Path(ann_file).parent / 'test.json'
        if cand.exists():
            ann_file = str(cand)
            cur = Path(img_folder)
            if cur.name in ('val', 'train'):
                img_folder = str(cur.parent / 'test')
            # else: img_folder 是根目录, 保持不变
    # 显式指定优先(用于老域回归等场景, 例如用迁移后的模型跑 base/test)
    if args.ann_file:
        ann_file = args.ann_file
    if args.img_dir:
        img_folder = args.img_dir

    d = json.loads(Path(ann_file).read_text(encoding='utf-8'))
    ann_by = {}
    for a in d['annotations']:
        ann_by.setdefault(a['image_id'], []).append(a)
    items = []
    for im in d['images']:
        p = Path(img_folder) / im['file_name']
        if p.exists():
            items.append((p, im))
    if args.limit:
        items = items[:args.limit]

    print('=' * 82)
    print(f'置信度扫描   {Path(args.config).name}   split={args.split}')
    print('=' * 82)
    print(f'  权重 {args.ckpt}')
    print(f'  图片 {len(items)}   GT {sum(len(ann_by.get(im["id"], [])) for _, im in items)}')
    print(f'  IoU 阈值 {args.iou}   目标 recall {args.target_recall:.0%}')

    # 快速预处理(src/data/io.py)是可选依赖 —— 不在同一工程目录时会回退到 PIL
    use_fast = False
    try:
        _MT = Path(r'D:\BaiduNetdiskDownload\multitask_detection')
        if str(_MT) not in sys.path:
            sys.path.insert(0, str(_MT))
        from src.data.io import load_image_tensor as _ld  # noqa: F401
        use_fast = True
    except Exception:
        use_fast = False

    # ---------- 单遍前向, 缓存全部预测 ----------
    print('\n[1/2] 前向收集预测')
    t0 = time.time()
    cached = []
    for idx, (p, im) in enumerate(items):
        from PIL import Image
        with Image.open(p) as _im:
            W, H = _im.size
        try:
            from src.data.io import load_image_tensor
            x, orig = load_image_tensor(p, args.imgsz, args.device)
        except Exception:
            img = Image.open(p).convert('RGB')
            sc = args.imgsz / max(img.size)
            inp = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(inp).astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(args.device)
            orig = (W, H)
        with torch.no_grad():
            out = model(x)
        sc = torch.sigmoid(out['pred_logits'])
        n_cls = sc.shape[-1]
        vals, index = torch.topk(sc.flatten(1), sc.shape[1], dim=-1)
        labels = (index % n_cls)[0].cpu().numpy()
        qidx = (index // n_cls)[0]
        import torchvision
        boxes = torchvision.ops.box_convert(out['pred_boxes'], 'cxcywh', 'xyxy')
        boxes = boxes * args.imgsz
        boxes = boxes.gather(1, qidx.unsqueeze(-1).repeat(1, 1, 4))[0].cpu().numpy()
        scores = vals[0].cpu().numpy()

        anns = ann_by.get(im['id'], [])
        # ⚠️ 输入是拉伸到方形 (imgsz x imgsz), x/y 缩放比例不同, 不能共用一个 scale
        sx = args.imgsz / W
        sy = args.imgsz / H
        gtb = np.array([[(a['bbox'][0]) * sx,
                         (a['bbox'][1]) * sy,
                         (a['bbox'][0] + a['bbox'][2]) * sx,
                         (a['bbox'][1] + a['bbox'][3]) * sy]
                        for a in anns], dtype=np.float64).reshape(-1, 4)
        gtl = np.array([a['category_id'] for a in anns], dtype=np.int64)
        cached.append((boxes, scores, labels, gtb, gtl))
        if (idx + 1) % 200 == 0:
            print(f'    {idx+1}/{len(items)}')
    print(f'  完成, 耗时 {time.time()-t0:.0f}s')

    # ---------- 多阈值统计 ----------
    print('\n[2/2] 按阈值统计')
    rows = []
    for conf in CONF_GRID:
        n_gt = n_ok = n_pred = n_tp = 0
        for boxes, scores, labels, gtb, gtl in cached:
            n_gt += len(gtl)
            keep = scores >= conf
            b, l = boxes[keep], labels[keep]
            n_pred += len(b)
            if len(gtl) == 0 or len(b) == 0:
                continue
            iou = iou_matrix(b, gtb)
            taken = set()
            for pi in range(len(b)):
                bg, bi = -1, 0.0
                for gi in range(len(gtl)):
                    if gi in taken:
                        continue
                    if iou[pi, gi] > bi:
                        bi, bg = iou[pi, gi], gi
                if bg >= 0 and bi >= args.iou and int(l[pi]) == int(gtl[bg]):
                    taken.add(bg)
                    n_ok += 1
            n_tp = n_ok
        rec = n_ok / n_gt if n_gt else 0.0
        prec = n_tp / n_pred if n_pred else 0.0
        fp = n_pred - n_tp
        rows.append({'conf': conf, 'recall': rec, 'precision': prec,
                     'tp': n_tp, 'pred': n_pred, 'fp': fp,
                     'fp_per_img': fp / len(items) if items else 0.0,
                     'pred_per_img': n_pred / len(items) if items else 0.0})

    print()
    print(f'  {"conf":>7}{"recall":>9}{"precision":>11}{"TP":>8}{"预测数":>9}{"FP":>9}'
          f'{"FP/图":>9}{"预测/图":>9}')
    print('  ' + '-' * 74)
    for r in rows:
        flag = ''
        if r['recall'] >= args.target_recall:
            flag = '  <- 达标'
        print(f'  {r["conf"]:>7.3f}{r["recall"]:>9.4f}{r["precision"]:>11.4f}{r["tp"]:>8,}'
              f'{r["pred"]:>9,}{r["fp"]:>9,}{r["fp_per_img"]:>9.1f}{r["pred_per_img"]:>9.1f}{flag}')

    # ---------- 推荐操作点 ----------
    ok = [r for r in rows if r['recall'] >= args.target_recall]
    print()
    print('=' * 82)
    print('推荐')
    print('=' * 82)
    if not ok:
        best = max(rows, key=lambda r: r['recall'])
        print(f'  [!] 没有阈值能让 recall 达到 {args.target_recall:.0%}')
        print(f'      最高 recall {best["recall"]:.4f} @conf {best["conf"]:.3f}')
        print(f'      => 模型本身还不够, 需要继续训练或补数据, 不是阈值能解决的')
    else:
        pick = max(ok, key=lambda r: r['conf'])
        print(f'  在 recall >= {args.target_recall:.0%} 前提下, 误警最少的操作点:')
        print(f'      conf = {pick["conf"]:.3f}')
        print(f'      recall    = {pick["recall"]:.4f}')
        print(f'      precision = {pick["precision"]:.4f}')
        print(f'      误检数    = {pick["fp"]:,}  ({pick["fp_per_img"]:.1f} 个/图)')
        base = next(r for r in rows if r['conf'] == CONF_GRID[0])
        print(f'  相比 conf=0.005:')
        print(f'      误检 {base["fp"]:,} -> {pick["fp"]:,}  '
              f'(减少 {(1 - pick["fp"]/max(base["fp"],1))*100:.1f}%)')
        print(f'      recall {base["recall"]:.4f} -> {pick["recall"]:.4f}  '
              f'(变化 {(pick["recall"]-base["recall"])*100:+.2f} pp)')

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({'rows': rows,
                                              'target_recall': args.target_recall,
                                              'iou': args.iou,
                                              'recommended': max(ok, key=lambda r: r['conf']) if ok else None},
                                             ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n  已保存 {args.out}')


if __name__ == '__main__':
    main()
