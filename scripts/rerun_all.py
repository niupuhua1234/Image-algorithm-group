# -*- coding: utf-8 -*-
"""数据修正后的完整重训流水线: base -> transfer -> 3A -> 3B。

数据修正内容:
    base/train 从 3,998 降回 2,800(剔除误并入的 val 603 + test 595)
    replay 改为从干净的 base/train(2,800 张)抽取

各阶段 epoch 说明:
    base      60 轮(原配置; 2,800 图 / batch 4 = 700 iter/epoch)
    transfer 120 轮(原 200, 但上次实测 ep55 后完全不再提升, 120 足够)
    3A/3B     20 轮

用法:
    python rerun_all.py
"""
import ast
import io
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
WORK = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')
SCRIPTS = Path(r'D:\BaiduNetdiskDownload\wdd_work\scripts')
PY = r'C:\Users\Administrator\.conda\envs\wdd\python.exe'

STAGES = [
    ('deim_vistarget_base', 'configs/deim_rtdetrv2/deim_vistarget_base.yml', 60, None),
    ('deim_vistarget_transfer', 'configs/deim_rtdetrv2/deim_vistarget_transfer.yml', 120, 'base'),
    ('deim_vistarget_inc_p1', 'configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', 20, 'transfer'),
    ('deim_vistarget_inc_p2', 'configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', 20, 'inc_p1'),
]


def log(msg):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(DEIM / 'output' / 'rerun_all.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def read_log(p):
    p = Path(p)
    if not p.exists():
        return []
    return [ast.literal_eval(l) for l in p.read_text(encoding='utf-8', errors='ignore').splitlines()
            if l.strip().startswith('{')]


def procs(patterns):
    import psutil
    out = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' not in (p.info['name'] or '').lower():
                continue
            c = ' '.join(p.info['cmdline'] or [])
            if all(s in c for s in patterns):
                out.append(p.info['pid'])
        except Exception:
            pass
    return out


def run_stage(dirname, cfg, epochs, init_from, tag):
    out_dir = DEIM / 'output' / dirname
    # 归档旧结果
    if out_dir.exists():
        old = out_dir.with_name(dirname + '_old')
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        try:
            out_dir.rename(old)
            log(f'  旧结果归档 -> {old.name}')
        except Exception as e:
            log(f'  归档失败: {e}')

    cmd = [PY, '-X', 'utf8', 'train.py', '-c', cfg, '-d', 'cuda:0', '--use-amp']
    if init_from:
        p = DEIM / 'output' / init_from
        if p.exists():
            cmd += ['-t', str(p)]
            log(f'  起点权重: {p.name}')
        else:
            log(f'  [!] 起点权重不存在: {p}')
            return None
    log(f'  启动训练: {cfg}')
    outlog = open(DEIM / 'output' / f'rerun_{tag}.stdout.log', 'w', encoding='utf-8')
    subprocess.Popen(cmd, cwd=str(DEIM), stdout=outlog, stderr=subprocess.STDOUT)

    t0 = time.time()
    last = -1
    while time.time() - t0 < 10 * 3600:
        recs = read_log(out_dir / 'log.txt')
        n = len(recs)
        if n != last:
            if recs:
                b = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
                bb = b['test_coco_eval_bbox']
                log(f'    {n}/{epochs}  最优 ep{b["epoch"]} mAP50={bb[1]:.4f}')
            last = n
        if n >= epochs:
            log(f'  完成 {n}/{epochs} 轮')
            break
        if not procs(['train.py', Path(cfg).stem]):
            n2 = len(read_log(out_dir / 'log.txt'))
            log(f'  进程已退出, 完成 {n2}/{epochs} 轮')
            break
        time.sleep(60)
    return out_dir / 'best_stg1.pth'


def expand(prev_cfg, prev_ckpt, nc, tag):
    out_ckpt = DEIM / 'output' / f'{tag}_init.pth'
    out_cfg = DEIM / 'configs' / 'deim_rtdetrv2' / f'deim_vistarget_{tag}_init.yml'
    if out_ckpt.exists():
        out_ckpt.unlink()
    cmd = [PY, '-X', 'utf8', str(SCRIPTS / 'expand_head.py'),
           '--config', prev_cfg, '--ckpt', str(prev_ckpt),
           '--new-num-classes', str(nc),
           '--out-ckpt', str(out_ckpt), '--out-config', str(out_cfg)]
    log(f'  扩头 -> {nc} 类')
    r = subprocess.run(cmd, cwd=str(DEIM), capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
    ok = out_ckpt.exists()
    log(f'    {"成功" if ok else "失败"}')
    return out_ckpt if ok else None


def main():
    log('=' * 70)
    log('数据修正后完整重训')
    log('=' * 70)

    best = {}
    for dirname, cfg, epochs, init_from in STAGES:
        tag = dirname.replace('deim_vistarget_', '')
        log('')
        log(f'--- {tag} ({epochs} 轮) ---')

        init_path = None
        if init_from == 'base':
            init_path = best.get('base')
        elif init_from == 'transfer':
            # 3A 需要先扩头到 6 类
            tr = best.get('transfer')
            if not tr:
                log('[X] transfer 权重缺失, 终止')
                return
            init_path = expand('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr, 6, 'inc_p1')
        elif init_from == 'inc_p1':
            p1 = best.get('inc_p1')
            if not p1:
                log('[X] 3A 权重缺失, 终止')
                return
            init_path = expand('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', p1, 8, 'inc_p2')

        w = run_stage(dirname, cfg, epochs, init_path, tag)
        if not w or not w.exists():
            log(f'[X] {tag} 未产出权重, 终止')
            return
        best[tag] = w
        recs = read_log(DEIM / 'output' / dirname / 'log.txt')
        if recs:
            b = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
            log(f'  {tag} 最优 ep{b["epoch"]} mAP50={b["test_coco_eval_bbox"][1]:.4f}')

    log('')
    log('=' * 70)
    log('全部训练完成')
    log('=' * 70)
    for k, v in best.items():
        log(f'  {k}: {v}')


if __name__ == '__main__':
    main()
