# -*- coding: utf-8 -*-
"""端到端时延测量(优化版): 多线程并行解码 + 全 GPU 预处理 + CUDA Graph。

相比串行版本的三处关键优化:
    ① 4 个线程并行 cv2.imread + cvtColor (磁盘/解码是瓶颈, 并行后吞吐 ÷4)
    ② 只把 uint8 图上显存(6.2 MB), resize / div255 / normalize 全在 GPU 做
    ③ forward 用 CUDA Graph, 一次 replay 替代几百次 kernel 提交

计时分三段:
    forward    —— CUDA Graph replay
    preprocess —— 从队列取图 + H2D + GPU 变换
    end_to_end —— 从"开始取图"到"拿到结果"(不含排队等待, 反映稳态吞吐)

用法:
    python bench_latency_fast.py --config <cfg> --ckpt <pth> --tag base
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
WORK = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img-dir', default=None)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--n-warm', type=int, default=40)
    ap.add_argument('--n-run', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--out', default=None)
    ap.add_argument('--preload', action='store_true',
                    help='先把全部图片读入内存再计时, 用于验证"消除磁盘 IO"后的上限')
    args = ap.parse_args()

    os.chdir(str(DEIM))
    sys.path.insert(0, str(DEIM))
    try:
        import patch_tw
        patch_tw.apply()
    except Exception:
        pass

    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    from engine.core import YAMLConfig

    S = args.imgsz
    img_dir = Path(args.img_dir) if args.img_dir else (WORK / 'base' / 'images' / 'test')
    paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    if not paths:
        print(f'[X] 没有图片: {img_dir}')
        return
    print('=' * 74)
    print(f'端到端时延  [{args.tag}]')
    print('=' * 74)
    print(f'  配置 {args.config}')
    print(f'  权重 {Path(args.ckpt).name}')
    print(f'  图片 {len(paths)} 张   输入 {S}   并行度 {args.workers}   轮次 {args.n_run}')

    # ---------- 模型 ----------
    cfg = YAMLConfig(args.config)
    model = cfg.model
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['ema']['module'] if isinstance(ck.get('ema'), dict) and 'module' in ck['ema'] else ck.get('model', ck)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model = model.cuda().eval()

    # ---------- 预生成位置张量与 anchors(否则 CUDA Graph 捕获失败) ----------
    enc, dec = model.encoder, model.decoder
    for idx in enc.use_encoder_idx:
        stride = enc.feat_strides[idx]
        pe = enc.build_2d_sincos_position_embedding(S // stride, S // stride, enc.hidden_dim)
        setattr(enc, f'pos_embed{idx}', pe.cuda())
    enc.eval_spatial_size = [S, S]
    if hasattr(dec, '_generate_anchors'):
        strides = getattr(dec, 'feat_strides', [8, 16, 32])
        shapes = torch.as_tensor([(S // s, S // s) for s in strides])
        anchors, valid_mask = dec._generate_anchors(shapes, device='cuda')
        dec.register_buffer('anchors', anchors)
        dec.register_buffer('valid_mask', valid_mask)
        dec.eval_spatial_size = [S, S]

    # ---------- 参数(用于 GPU 侧归一化) ----------
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    static_u8 = torch.zeros(1, 3, 1080, 1920, dtype=torch.uint8, device='cuda')
    static_x = torch.zeros(1, 3, S, S, device='cuda')

    def to_input(u8_batch):
        """uint8(B,3,H,W) -> 归一化后的 (B,3,S,S) float32, 全部在 GPU 上。"""
        x = u8_batch.float().div_(255.0)
        x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        return x

    # ---------- CUDA Graph ----------
    with torch.no_grad():
        for _ in range(5):
            model(static_x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                model(static_x)
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g):
            static_out = model(static_x)
    torch.cuda.synchronize()
    print('  CUDA Graph 捕获完成')

    # ---------- 并行读取 ----------
    def read_one(p):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im).permute(2, 0, 1).contiguous()

    pool = ThreadPoolExecutor(max_workers=args.workers)
    q = Queue(maxsize=args.workers * 4)
    stop = threading.Event()

    cached_u8 = None
    if args.preload:
        # 预先把全部图片解码成 uint8 CHW 存内存(595 张 1920x1080x3 约 3.7 GB)
        print('  预加载图片到内存 ...')
        t0 = time.time()
        cached_u8 = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for t in ex.map(read_one, paths):
                if t is not None:
                    cached_u8.append(t)
        print(f'    已缓存 {len(cached_u8)} 张, 耗时 {time.time()-t0:.1f}s')

    def producer():
        i = 0
        # ⚠️ 必须批量提交再批量收: 如果 submit 一张就立刻 result(),
        #    主线程会被单张读取阻塞, 线程池退化成串行。
        batch_n = max(args.workers * 2, args.workers)
        while not stop.is_set():
            if cached_u8 is not None:
                q.put(cached_u8[i % len(cached_u8)])
                i += 1
                continue
            futs = []
            for _ in range(batch_n):
                p = paths[i % len(paths)]
                i += 1
                futs.append(pool.submit(read_one, p))
            for f in futs:
                try:
                    t = f.result()
                    if t is not None:
                        q.put(t)
                except Exception:
                    pass

    th = threading.Thread(target=producer, daemon=True)
    th.start()

    # 预热
    def consume_preprocess():
        t_chw = q.get()
        static_u8[0].copy_(t_chw.cuda(non_blocking=True))
        x = to_input(static_u8)
        static_x.copy_(x)
        return x

    for _ in range(args.n_warm):
        consume_preprocess()
        g.replay()
    torch.cuda.synchronize()

    # ---------- 正式计时 ----------
    t_pre, t_fwd, t_e2e = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.n_run):
        torch.cuda.synchronize()
        a = time.perf_counter()
        consume_preprocess()
        b = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        c = time.perf_counter()
        t_pre.append((b - a) * 1000)
        t_fwd.append((c - b) * 1000)
        t_e2e.append((c - a) * 1000)
    peak = torch.cuda.max_memory_allocated() / 1e6
    stop.set()
    th.join(timeout=2)

    def stat(v):
        v = sorted(v)
        return {'mean': round(float(np.mean(v)), 2),
                'median': round(float(v[len(v) // 2]), 2),
                'min': round(float(v[0]), 2),
                'p95': round(float(v[int(len(v) * 0.95)]), 2)}

    f, p_, e = stat(t_fwd), stat(t_pre), stat(t_e2e)
    res = {
        'tag': args.tag, 'imgsz': S, 'workers': args.workers, 'runs': args.n_run,
        'device': torch.cuda.get_device_name(0),
        'forward_ms': f, 'preprocess_ms': p_, 'end_to_end_ms': e,
        'peak_memory_mb': round(peak, 1),
    }
    print()
    print(f'  {"阶段":<12}{"mean":>9}{"median":>9}{"min":>9}{"p95":>9}')
    print('  ' + '-' * 48)
    for name, d in (('forward', f), ('preprocess', p_), ('end_to_end', e)):
        print(f'  {name:<12}{d["mean"]:>9.2f}{d["median"]:>9.2f}{d["min"]:>9.2f}{d["p95"]:>9.2f}')
    print(f'\n  峰值显存 {peak:.0f} MB')
    print(f'  判定: 端到端 p95 {e["p95"]:.2f} ms  ->  {"达标 (<40ms)" if e["p95"] < 40 else "超标"}')

    out = Path(args.out) if args.out else (DEIM / 'output' / f'latency_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  已保存 {out}')


if __name__ == '__main__':
    main()
