# -*- coding: utf-8 -*-
"""干净评估: 在 val 上确定 conf 阈值, 再用它到 test 上验证一次。

为什么要单独做这件事:
    之前的 sweep_conf.py 是在 test 上扫阈值的 —— 等于用测试集做超参选择,
    指标会偏乐观。正确流程应该是:
        ① 在 val 上扫阈值, 选出"recall 达标下误检最少"的 conf
        ② 固定这个 conf, 到 test 上跑一次, 只报这一组数
    本脚本就是这个流程, 输出:
        - val 上选出的 conf 及 val 指标
        - 同一 conf 在 test 上的指标  (这是干净数字)
        - 同时在 test 上暴力扫一遍的最优值 (对照, 说明泄漏有多大)

用法:
    python verify_val_to_test.py --config <cfg> --ckpt <pth> --tag base
"""
import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
DEPLOY = DEIM.parent
WORK = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')
PY = r'C:\Users\Administrator\.conda\envs\wdd\python.exe'


def run_sweep(cfg, ckpt, split, out, grid, target, ann_file=None, img_dir=None):
    cmd = [PY, '-X', 'utf8', str(DEPLOY / 'sweep_conf.py'),
           '--config', cfg, '--ckpt', str(ckpt), '--split', split,
           '--target-recall', str(target), '--conf-grid', grid, '--out', str(out)]
    if ann_file:
        cmd += ['--ann-file', str(ann_file)]
    if img_dir:
        cmd += ['--img-dir', str(img_dir)]
    if out.exists():
        out.unlink()
    r = subprocess.run(cmd, cwd=str(DEPLOY), capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
    if not out.exists():
        print(f'  [!] sweep 失败 exit={r.returncode}')
        print((r.stderr or '')[-500:])
        return None
    return json.loads(out.read_text(encoding='utf-8'))


def pick_at(rows, conf):
    """在 rows 里找最接近 conf 的那一行。"""
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r['conf'] - conf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--target-recall', type=float, default=0.90)
    ap.add_argument('--grid', default='0.05:0.95:0.01')
    ap.add_argument('--ann-val', default=None, help='自定义 val 标注')
    ap.add_argument('--ann-test', default=None, help='自定义 test 标注')
    ap.add_argument('--img-dir', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print('=' * 78)
    print(f'干净评估  [{args.tag}]   val 定阈值 -> test 验证')
    print('=' * 78)
    print(f'  配置 {args.config}')
    print(f'  权重 {Path(args.ckpt).name}')
    print(f'  阈值网格 {args.grid}   目标 recall {args.target_recall:.0%}')

    out_dir = DEIM / 'output' / f'cleaneval_{args.tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ① val 上定阈值 ----
    print('\n[1/3] val 上扫阈值')
    jv = run_sweep(args.config, args.ckpt, 'val', out_dir / 'val_sweep.json',
                   args.grid, args.target_recall, args.ann_val, args.img_dir)
    if not jv or not jv.get('recommended'):
        print('  [!] val 上没有任何阈值能让 recall 达标, 无法定阈值')
        if jv:
            rows = jv.get('rows') or []
            if rows:
                best = max(rows, key=lambda r: r['recall'])
                print(f'      val 最高 recall = {best["recall"]:.4f} @conf {best["conf"]:.3f}')
        return
    rec_v = jv['recommended']
    print(f'  val 选定: conf = {rec_v["conf"]:.3f}   '
          f'recall {rec_v["recall"]:.4f}  precision {rec_v["precision"]:.4f}  '
          f'FP/图 {rec_v["fp_per_img"]:.2f}')

    # ---- ② 用这个阈值到 test 验证 ----
    print('\n[2/3] 用该阈值在 test 上验证(干净数字)')
    conf_str = f'{rec_v["conf"]:.3f}'
    jt = run_sweep(args.config, args.ckpt, 'test', out_dir / 'test_at_valconf.json',
                   f'{conf_str}:{conf_str}:0.01', args.target_recall,
                   args.ann_test, args.img_dir)
    clean = pick_at((jt or {}).get('rows') or [], rec_v['conf'])
    if clean:
        print(f'  test @conf {clean["conf"]:.3f}:  recall {clean["recall"]:.4f}  '
              f'precision {clean["precision"]:.4f}  FP/图 {clean["fp_per_img"]:.2f}')
    else:
        print('  [!] test 验证失败')

    # ---- ③ 对照: test 上暴力扫(说明泄漏幅度) ----
    print('\n[3/3] 对照: test 上直接扫(有泄漏, 仅作对比)')
    jt2 = run_sweep(args.config, args.ckpt, 'test', out_dir / 'test_full_sweep.json',
                    args.grid, args.target_recall, args.ann_test, args.img_dir)
    leaky = (jt2 or {}).get('recommended')
    if leaky:
        print(f'  test 上扫出的最优: conf {leaky["conf"]:.3f}  recall {leaky["recall"]:.4f}  '
              f'precision {leaky["precision"]:.4f}  FP/图 {leaky["fp_per_img"]:.2f}')
        if clean:
            d_rec = (clean['recall'] - leaky['recall']) * 100
            d_fp = clean['fp_per_img'] - leaky['fp_per_img']
            print(f'\n  泄漏幅度: 干净 recall 比"test 上扫"低 {abs(d_rec):.2f} pp, '
                  f'误检差 {d_fp:+.2f} 个/图')

    # ---- 汇总 ----
    rep = {
        'tag': args.tag,
        'config': args.config,
        'ckpt': str(args.ckpt),
        'val_recommended': rec_v,
        'test_at_val_conf': clean,
        'test_leaky_optimum': leaky,
        'leak_delta_recall_pp': round((clean['recall'] - leaky['recall']) * 100, 2) if (clean and leaky) else None,
    }
    out = Path(args.out) if args.out else (out_dir / 'report.json')
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    print()
    print('=' * 78)
    print('结论')
    print('=' * 78)
    if clean:
        print(f'  干净指标(val 定阈值): recall {clean["recall"]:.4f}   '
              f'precision {clean["precision"]:.4f}   误检 {clean["fp_per_img"]:.2f} 个/图')
        print(f'  -> {"达标" if clean["recall"] >= args.target_recall else "未达标"}(门槛 {args.target_recall:.0%})')
    if leaky:
        print(f'  泄漏版本(test 定阈值): recall {leaky["recall"]:.4f}   '
              f'precision {leaky["precision"]:.4f}   误检 {leaky["fp_per_img"]:.2f} 个/图')
    print(f'  报告 {out}')


if __name__ == '__main__':
    main()
