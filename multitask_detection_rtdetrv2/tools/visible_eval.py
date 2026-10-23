# -*- coding: utf-8 -*-
"""评测入口 —— 按任务标识跑指定划分。

技术路线书第 6 节的指标要求:
    框级识别: 检测 AP、逐类别精度与召回、匹配目标上的类别混淆情况、完整业务时延

用法:
    python tools/eval.py --task visible_recognition --split test \\
        --ckpt /path/to/best_stg1.pth

    # 快速验证(只跑前 200 张)
    python tools/eval.py --task visible_recognition --split test --ckpt ... --limit 200

    # 同时给出项目历史口径(recall 式), 便于与既有结果对照
    python tools/eval.py --task visible_recognition --split test --ckpt ... --legacy

    # 列出所有任务
    python tools/eval.py --list
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np            # noqa: E402
import torch                  # noqa: E402

from src.core.config import load_config, resolve_path   # noqa: E402
from src.core.registry import available_tasks, get_adapter, load_all_adapters  # noqa: E402


# --------------------------------------------------------------------------- #
def load_ckpt_into(model, ckpt_path, device='cuda', verbose=True):
    """加载权重。优先 EMA(训练侧实测更稳)。"""
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
    model.load_state_dict(matched, strict=False)
    if verbose:
        print(f'  权重来源: {which}   匹配 {len(matched)}/{len(msd)}')
        if missing:
            print(f'  [!] 未匹配 {len(missing)} 个, 例如 {missing[:3]}')
    return model.to(device).eval()


def read_coco(json_path: Path):
    """读 COCO 标注。"""
    d = json.loads(Path(json_path).read_text(encoding='utf-8'))
    ann_by_img = {}
    for a in d['annotations']:
        ann_by_img.setdefault(a['image_id'], []).append(a)
    return d, ann_by_img


def resolve_image(img_folder: Path, im: dict) -> Path:
    """定位图片。

    交接说明坑 2: JSON 里的 path 是绝对路径, 换机器即失效。
    因此优先用 img_folder + file_name, 仅在缺失时回退到 path。
    """
    p = img_folder / im['file_name']
    if p.exists():
        return p
    fallback = im.get('path')
    if fallback and Path(fallback).exists():
        return Path(fallback)
    return p


def legacy_recall_metrics(preds, gts, class_names, iou_thr=0.40):
    """项目历史口径: 类别一致 + IoU>=阈值的贪心匹配, 分母为 GT 总数(不惩罚误检)。"""
    n_gt = n_ok = 0
    pc, po = Counter(), Counter()
    for det, g in zip(preds, gts):
        gb = np.asarray(g['boxes'], dtype=np.float64).reshape(-1, 4)
        gl = np.asarray(g['labels'], dtype=np.int64).reshape(-1)
        n_gt += len(gl)
        for c in gl:
            pc[int(c)] += 1
        if len(gl) == 0 or len(det) == 0:
            continue
        # 贪心: 按预测顺序, 每个框找 IoU 最大的未匹配 GT
        iou = np.zeros((len(det), len(gl)))
        for i in range(len(det)):
            for j in range(len(gl)):
                x1 = max(det.boxes[i][0], gb[j][0]); y1 = max(det.boxes[i][1], gb[j][1])
                x2 = min(det.boxes[i][2], gb[j][2]); y2 = min(det.boxes[i][3], gb[j][3])
                iw, ih = max(x2 - x1, 0), max(y2 - y1, 0)
                inter = iw * ih
                a1 = max(det.boxes[i][2] - det.boxes[i][0], 0) * max(det.boxes[i][3] - det.boxes[i][1], 0)
                a2 = max(gb[j][2] - gb[j][0], 0) * max(gb[j][3] - gb[j][1], 0)
                iou[i, j] = inter / max(a1 + a2 - inter, 1e-9)
        taken = set()
        for i in range(len(det)):
            bg, bi = -1, 0.0
            for j in range(len(gl)):
                if j in taken:
                    continue
                if iou[i, j] > bi:
                    bi, bg = iou[i, j], j
            if bg >= 0 and bi >= iou_thr and int(det.labels[i]) == int(gl[bg]):
                taken.add(bg); n_ok += 1; po[int(gl[bg])] += 1
    acc = n_ok / n_gt if n_gt else 0.0
    return {
        'protocol': f'recall_style (类别一致 且 IoU>={iou_thr}, 分母为 GT 总数)',
        'accuracy': round(acc, 4), 'correct': n_ok, 'num_gt': n_gt,
        'per_class': {class_names[i]: {
            'correct': int(po.get(i, 0)), 'num_gt': int(pc.get(i, 0)),
            'accuracy': round(po.get(i, 0) / pc[i], 4) if pc.get(i, 0) else None,
        } for i in range(len(class_names)) if pc.get(i, 0)},
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='多任务检测 —— 评测')
    ap.add_argument('--task', default='visible_recognition')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--imgsz', type=int, default=None)
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张(0=全部)')
    ap.add_argument('--legacy', action='store_true', help='额外给出项目历史 recall 口径')
    ap.add_argument('--list', action='store_true', help='列出全部任务后退出')
    ap.add_argument('--out', default=None, help='指标保存到 json')
    args = ap.parse_args()

    load_all_adapters()
    if args.list:
        for t in available_tasks():
            print(f'  {t["task_id"]:<24}{t["status"]:<14}{t["description"]}')
        return
    if not args.ckpt:
        ap.error('--ckpt 必填(或先用 --list 查看任务)')

    cfg = load_config(args.task)
    adapter = get_adapter(args.task)()
    info = adapter.describe()

    imgsz = args.imgsz or cfg.get('deployment', {}).get('default_imgsz', 1024)
    conf = args.conf if args.conf is not None else cfg.get('deployment', {}).get('conf_threshold', 0.005)

    print('=' * 78)
    print(f'任务 {info["task_id"]}  ({info["business_name"]})   划分 {args.split}')
    print('=' * 78)
    print(f'  类别({info["num_classes"]}): {", ".join(info["class_names"])}')
    print(f'  配置: {cfg["_config_path"]}')
    print(f'  imgsz={imgsz}  conf={conf}  device={args.device}')
    if info['status'] != 'implemented':
        print(f'\n  [!] 该任务状态为 {info["status"]}, 尚未实现。跳过。')
        return

    # ---- 模型 ----
    print('\n[1/3] 构建模型并加载权重')
    model = adapter.build_model(cfg)
    model = load_ckpt_into(model, args.ckpt, args.device)
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ---- 数据 ----
    print('\n[2/3] 读取标注')
    data_root = Path(cfg['data']['root'])
    d_split = cfg['data'][args.split]
    img_folder = data_root / d_split['img_folder']
    ann_file = data_root / d_split['ann_file']
    d, ann_by = read_coco(ann_file)
    items = []
    for im in d['images']:
        p = resolve_image(img_folder, im)
        if p.exists():
            items.append((p, im))
    if args.limit:
        items = items[:args.limit]
    n_gt = sum(len(ann_by.get(im['id'], [])) for _, im in items)
    print(f'  图片 {len(items)}  GT {n_gt}')
    if not items:
        print('  [!] 没有找到图片, 检查 data.root 与 img_folder 配置')
        return

    # ---- 推理 ----
    print('\n[3/3] 推理与评测')
    from src.data.io import load_image_tensor
    preds, gts = [], []
    times = []
    t_start = time.time()
    for idx, (p, im) in enumerate(items):
        x, orig = load_image_tensor(p, imgsz, args.device)
        if idx >= 3 and args.device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            raw = model(x)
        det = adapter.postprocess(raw, orig, conf)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        if idx >= 3:
            times.append((time.perf_counter() - t0) * 1000)

        anns = ann_by.get(im['id'], [])
        gb = np.array([[a['bbox'][0], a['bbox'][1],
                        a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                       for a in anns], dtype=np.float64).reshape(-1, 4)
        gl = np.array([a['category_id'] for a in anns], dtype=np.int64)
        preds.append(det)
        gts.append({'boxes': gb, 'labels': gl})
        if (idx + 1) % 200 == 0:
            print(f'    {idx+1}/{len(items)} ...')

    ms = float(np.mean(times)) if times else float('nan')
    print(f'\n  推理完成, 单帧 forward 均值 {ms:.1f} ms (不含预处理)')

    # ---- 指标 ----
    iou_thrs = cfg.get('evaluation', {}).get('report_iou_range') or [0.5, 0.95]
    metrics = adapter.evaluate(preds, gts,
                               iou_thr=cfg.get('evaluation', {}).get('primary_iou', 0.5),
                               iou_thrs=[round(0.5 + 0.05 * i, 2) for i in range(10)]
                               if iou_thrs[0] == 0.5 else None)
    metrics['forward_ms_mean'] = round(ms, 2)
    metrics['num_images'] = len(items)
    metrics['split'] = args.split

    print()
    print('=' * 78)
    print('评测结果(路线书口径: 检测 AP / 逐类别精度与召回 / 类别混淆)')
    print('=' * 78)
    print(f'  GT 总数        {metrics["num_gt"]}')
    print(f'  匹配成功       {metrics["num_matched"]}')
    print(f'  匹配但类别错   {metrics["num_misclassified"]}')
    print(f'  mAP@{metrics["iou_thr"]}          {metrics["mAP"]}')
    if 'mAP_50_95' in metrics:
        print(f'  mAP@.50:.95    {metrics["mAP_50_95"]}')
    print()
    print(f'  {"类别":<10}{"GT":>8}{"TP":>8}{"FP":>8}{"FN":>8}{"P":>9}{"R":>9}{"F1":>9}{"AP":>9}')
    for name, m in metrics['per_class'].items():
        ap = f'{m["ap"]:.4f}' if m['ap'] is not None else '  -  '
        print(f'  {name:<10}{m["num_gt"]:>8}{m["tp"]:>8}{m["fp"]:>8}{m["fn"]:>8}'
              f'{m["precision"]:>9.4f}{m["recall"]:>9.4f}{m["f1"]:>9.4f}{ap:>9}')
    if metrics['confusion']:
        print()
        print('  类别混淆(匹配上但类别判断错误):')
        for c in metrics['confusion'][:12]:
            print(f'      预测 {c["predicted"]:<8} 实为 {c["actual"]:<8} {c["count"]} 次')

    if args.legacy:
        lm = legacy_recall_metrics(preds, gts, info['class_names'])
        print()
        print('=' * 78)
        print('项目历史口径对照(recall 式)')
        print('=' * 78)
        print(f'  {lm["protocol"]}')
        print(f'  准确率 {lm["accuracy"]:.4f}  ({lm["correct"]}/{lm["num_gt"]})')
        for name, m in lm['per_class'].items():
            if m['accuracy'] is not None:
                print(f'      {name:<10}{m["correct"]:>7}/{m["num_gt"]:<7}= {m["accuracy"]:.2%}')
        metrics['legacy'] = lm

    print(f'\n  总耗时 {time.time()-t_start:.0f}s')

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'  指标已保存: {out}')


if __name__ == '__main__':
    main()
