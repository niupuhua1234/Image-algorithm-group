# -*- coding: utf-8 -*-
"""推理入口 —— 单图推理 + 完整业务时延测量。

用法:
    # 单图
    python tools/infer.py --task visible_recognition --ckpt best_stg1.pth --image a.jpg --conf 0.3

    # 时延测量(对比三种后端)
    python tools/infer.py --task visible_recognition --ckpt best_stg1.pth --bench 50
    python tools/infer.py ... --backend cuda_graph
    python tools/infer.py ... --backend pytorch
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config import load_config                  # noqa: E402
from src.core.registry import get_adapter, load_all_adapters  # noqa: E402
from src.deployment.runtime import build_runtime         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='多任务检测 —— 推理与时延测量')
    ap.add_argument('--task', default='visible_recognition')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image', default=None, help='单图推理')
    ap.add_argument('--bench', type=int, default=0, help='>0 时做 N 次时延测量')
    ap.add_argument('--backend', default='pytorch', choices=['pytorch', 'cuda_graph'])
    ap.add_argument('--imgsz', type=int, default=None)
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None, help='时延结果保存到 json')
    args = ap.parse_args()

    load_all_adapters()
    cfg = load_config(args.task)
    adapter = get_adapter(args.task)()
    info = adapter.describe()
    if info['status'] != 'implemented':
        print(f'[!] 任务 {info["task_id"]} 状态为 {info["status"]}, 尚未实现。')
        return

    print('=' * 70)
    print(f'任务 {info["task_id"]}  ({info["business_name"]})')
    print('=' * 70)
    rt = build_runtime(cfg, adapter, args.ckpt, imgsz=args.imgsz,
                       device=args.device, backend=args.backend)

    if args.image:
        det = rt.infer(args.image, args.conf)
        print(f'\n单图推理: {Path(args.image).name}  检出 {len(det)} 个')
        names = info['class_names']
        order = det.scores.argsort()[::-1]
        for i in order[:20]:
            nm = names[int(det.labels[i])] if int(det.labels[i]) < len(names) else str(det.labels[i])
            b = det.boxes[i]
            print(f'    {nm:<10} {det.scores[i]:.4f}  [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]')
        if len(det) > 20:
            print(f'    ... 其余 {len(det)-20} 个')

    if args.bench > 0:
        img_dir = Path(cfg['data']['root']) / cfg['data']['test']['img_folder']
        paths = sorted(img_dir.glob('*.jpg'))[:max(args.bench, 20)]
        if not paths:
            print(f'\n[!] 找不到测试图片: {img_dir}')
            return
        print()
        res = rt.benchmark(paths, runs=args.bench, conf=args.conf)
        if args.out:
            import json
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({
                'task': info['task_id'], **res,
                'result': {'mAP_note': '精度请用 tools/eval.py'},
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'\n  已保存: {args.out}')


if __name__ == '__main__':
    main()
