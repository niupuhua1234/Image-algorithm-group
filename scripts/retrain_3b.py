# -*- coding: utf-8 -*-
"""只重训 3B(用新的回放配额), 然后做三次干净评估。

背景:
    上一版 3B 对 3A 学过的 2 类出现严重遗忘 —— 必须把 conf 从 0.34 压到 0.20
    才能凑出 87% recall, precision 从 71.8% 崩到 48.2%, 误检涨 2.7 倍。
    根因是 inc_p2 的回放按类别数比例(4:2)分配, 3A 的 2 类只拿到 16% 的训练框。

本次修正:
    inc_p2 回放改为 base:inc_p1 = 1:1(各 288/289 图),
    使 3A 的 2 类在训练框里的占比从 16% 提到 24.7%(每类 549 框, 是 base 每类的 2 倍)。

用法:
    python retrain_3b.py
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
LOG = DEIM / 'output' / 'retrain_3b.log'


def log(msg):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
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


def clean_eval(cfg, ckpt, tag, ann_val=None, ann_test=None, img_dir=None):
    cmd = [PY, '-X', 'utf8', str(SCRIPTS / 'verify_val_to_test.py'),
           '--config', cfg, '--ckpt', str(ckpt), '--tag', tag, '--grid', '0.10:0.90:0.02']
    if ann_val:
        cmd += ['--ann-val', str(ann_val)]
    if ann_test:
        cmd += ['--ann-test', str(ann_test)]
    if img_dir:
        cmd += ['--img-dir', str(img_dir)]
    log(f'  评估 {tag}')
    subprocess.run(cmd, cwd=str(DEIM), capture_output=True, text=True,
                   encoding='utf-8', errors='ignore')
    rep = DEIM / 'output' / f'cleaneval_{tag}' / 'report.json'
    if rep.exists():
        j = json.loads(rep.read_text(encoding='utf-8'))
        c = j.get('test_at_val_conf')
        if c:
            log(f'    {tag}: conf {c["conf"]:.3f}  recall {c["recall"]:.4f}  '
                f'precision {c["precision"]:.4f}  误检 {c["fp_per_img"]:.2f}/图')
            return c
    log(f'    [!] 无结果')
    return None


def main():
    log('=' * 70)
    log('重训 3B(新回放配额: base:inc_p1 = 1:1)')
    log('=' * 70)

    # 起点: 3A 的最优权重 -> 扩头到 8 类
    p1_ckpt = DEIM / 'output' / 'deim_vistarget_inc_p1' / 'best_stg1.pth'
    if not p1_ckpt.exists():
        log(f'[X] 3A 权重不存在: {p1_ckpt}')
        return
    log(f'3A 权重: {p1_ckpt.name}  {datetime.fromtimestamp(p1_ckpt.stat().st_mtime).strftime("%H:%M")}')

    out_ckpt = DEIM / 'output' / 'inc_p2_init_v2.pth'
    out_cfg = DEIM / 'configs' / 'deim_rtdetrv2' / 'deim_vistarget_inc_p2_init_v2.yml'
    if out_ckpt.exists():
        out_ckpt.unlink()
    cmd = [PY, '-X', 'utf8', str(SCRIPTS / 'expand_head.py'),
           '--config', 'configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml',
           '--ckpt', str(p1_ckpt), '--new-num-classes', '8',
           '--out-ckpt', str(out_ckpt), '--out-config', str(out_cfg)]
    log('扩头 -> 8 类')
    subprocess.run(cmd, cwd=str(DEIM), capture_output=True, text=True,
                   encoding='utf-8', errors='ignore')
    if not out_ckpt.exists():
        log('[X] 扩头失败')
        return
    log(f'  成功: {out_ckpt.name}')

    # 清空 3B 目录并重训
    p2 = DEIM / 'output' / 'deim_vistarget_inc_p2'
    if p2.exists():
        shutil.rmtree(p2, ignore_errors=True)
        log('旧 3B 目录已清空')
    log('启动 3B 训练(20 轮)')
    f = open(DEIM / 'output' / 'inc_p2_v4.stdout.log', 'w', encoding='utf-8')
    subprocess.Popen([PY, '-X', 'utf8', 'train.py',
                      '-c', 'configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml',
                      '-d', 'cuda:0', '--use-amp', '-t', str(out_ckpt)],
                     cwd=str(DEIM), stdout=f, stderr=subprocess.STDOUT)

    t0 = time.time()
    last = -1
    while time.time() - t0 < 3 * 3600:
        recs = read_log(p2 / 'log.txt')
        n = len(recs)
        if n != last:
            if recs:
                b = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
                log(f'    {n}/20  最优 ep{b["epoch"]} mAP50={b["test_coco_eval_bbox"][1]:.4f}')
            last = n
        if n >= 20:
            log(f'完成 {n}/20')
            break
        if not procs(['train.py', 'deim_vistarget_inc_p2']):
            log(f'进程退出, 完成 {len(read_log(p2/"log.txt"))}/20')
            break
        time.sleep(60)

    ck2 = p2 / 'best_stg1.pth'
    if not ck2.exists():
        log('[X] 3B 无权重')
        return
    recs = read_log(p2 / 'log.txt')
    if recs:
        bb = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
        log(f'3B 最优 ep{bb["epoch"]} mAP50={bb["test_coco_eval_bbox"][1]:.4f}')

    log('')
    log('三次干净评估:')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', ck2, 'inc_p2_v4',
               img_dir=WORK,
               ann_val=WORK / 'inc_p2' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'inc_p2' / 'annotations' / 'rel_test.json')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', ck2, 'inc_p2_v4_base_reg',
               img_dir=WORK,
               ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', ck2, 'inc_p2_v4_p1_reg',
               img_dir=WORK,
               ann_val=WORK / 'inc_p1' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'inc_p1' / 'annotations' / 'rel_test.json')

    log('')
    log('=' * 70)
    log('3B 重训完成')
    log('=' * 70)


if __name__ == '__main__':
    main()
