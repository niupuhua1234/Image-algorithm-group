# -*- coding: utf-8 -*-
"""从 transfer 40 轮收尾, 接力 3A / 3B(替代 auto_pipeline 的后续阶段)。

原因: transfer 在 ep17 后就饱和(ep36 才 0.4480, 19 轮没突破), 跑满 120 轮纯属空转。
      用户决定 120 -> 40 轮提前结束。

流程:
    ① 等 transfer 跑到 40 轮
    ② 停掉 auto_pipeline 与 transfer 训练
    ③ transfer 干净评估(新域 + base 老域回归)
    ④ 扩头 6 类 -> 3A(20 轮) -> 干净评估(新类 + base 回归)
    ⑤ 扩头 8 类 -> 3B(20 轮) -> 干净评估(新类 + base 回归 + 3A 类回归)

用法:
    python finish_inc.py --transfer-epochs 40
"""
import argparse
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
LOG = DEIM / 'output' / 'finish_inc.log'


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


def kill(patterns, tag=''):
    import psutil
    pids = procs(patterns)
    for pid in pids:
        try:
            psutil.Process(pid).terminate()
            log(f'  终止 {tag} PID {pid}')
        except Exception:
            pass
    return pids


def wait_gpu(max_wait=120):
    for _ in range(max_wait // 3):
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True)
        s = (out.stdout or '0').strip().split()
        if s and int(s[0]) < 2500:
            return True
        time.sleep(3)
    return False


def launch(cfg, init_ckpt, tag):
    cmd = [PY, '-X', 'utf8', 'train.py', '-c', cfg, '-d', 'cuda:0', '--use-amp']
    if init_ckpt:
        cmd += ['-t', str(init_ckpt)]
    log(f'  启动训练: {cfg}' + (f'  init={Path(init_ckpt).name}' if init_ckpt else ''))
    f = open(DEIM / 'output' / f'{tag}.stdout.log', 'w', encoding='utf-8')
    subprocess.Popen(cmd, cwd=str(DEIM), stdout=f, stderr=subprocess.STDOUT)


def wait_epochs(out_dir, target, cfg_name, timeout=6 * 3600):
    logp = Path(out_dir) / 'log.txt'
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout:
        recs = read_log(logp)
        n = len(recs)
        if n != last:
            if recs:
                b = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
                log(f'    {n}/{target}  最优 ep{b["epoch"]} mAP50={b["test_coco_eval_bbox"][1]:.4f}')
            last = n
        if n >= target:
            log(f'  完成 {n}/{target}')
            return True
        if not procs(['train.py', cfg_name]):
            log(f'  进程退出, 完成 {len(read_log(logp))}/{target}')
            return True
        time.sleep(60)
    return False


def clean_eval(cfg, ckpt, tag, ann_val=None, ann_test=None, img_dir=None):
    cmd = [PY, '-X', 'utf8', str(SCRIPTS / 'verify_val_to_test.py'),
           '--config', cfg, '--ckpt', str(ckpt), '--tag', tag, '--grid', '0.10:0.90:0.02']
    if ann_val:
        cmd += ['--ann-val', str(ann_val)]
    if ann_test:
        cmd += ['--ann-test', str(ann_test)]
    if img_dir:
        cmd += ['--img-dir', str(img_dir)]
    log(f'  干净评估 {tag}')
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
    log(f'    [!] 未拿到结果')
    return None


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
    subprocess.run(cmd, cwd=str(DEIM), capture_output=True, text=True,
                   encoding='utf-8', errors='ignore')
    return out_ckpt if out_ckpt.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transfer-epochs', type=int, default=40)
    args = ap.parse_args()

    log('=' * 70)
    log(f'transfer 收尾(目标 {args.transfer_epochs} 轮) + 接力 3A/3B')
    log('=' * 70)

    tr_dir = DEIM / 'output' / 'deim_vistarget_transfer'
    log(f'等 transfer 到 {args.transfer_epochs} 轮')
    t0 = time.time()
    while time.time() - t0 < 2 * 3600:
        n = len(read_log(tr_dir / 'log.txt'))
        if n >= args.transfer_epochs:
            break
        if not procs(['train.py', 'deim_vistarget_transfer.yml']) and n > 0:
            log('  transfer 进程已退出')
            break
        time.sleep(30)

    n = len(read_log(tr_dir / 'log.txt'))
    log(f'transfer 当前 {n} 轮, 停掉流水线与训练')
    kill(['auto_pipeline.py'], 'auto_pipeline')
    kill(['train.py', 'deim_vistarget_transfer.yml'], 'transfer 训练')
    wait_gpu()
    log('GPU 已释放')

    recs = read_log(tr_dir / 'log.txt')
    b = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
    log(f'transfer 最优: ep{b["epoch"]} mAP50={b["test_coco_eval_bbox"][1]:.4f}')
    tr_ckpt = tr_dir / 'best_stg1.pth'
    if not tr_ckpt.exists():
        log('[X] transfer 无权重, 停止')
        return

    # transfer 评估(新域 + base 老域)
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr_ckpt, 'transfer_final')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr_ckpt,
               'transfer_base_reg', img_dir=WORK,
               ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')

    # 3A
    log('')
    log('--- 类增量 3A ---')
    init1 = expand('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr_ckpt, 6, 'inc_p1')
    if not init1:
        log('[X] 扩头失败')
        return
    p1 = DEIM / 'output' / 'deim_vistarget_inc_p1'
    if p1.exists():
        shutil.rmtree(p1, ignore_errors=True)
    launch('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', init1, 'inc_p1_v3')
    wait_epochs(p1, 20, 'deim_vistarget_inc_p1.yml')
    ck1 = p1 / 'best_stg1.pth'
    if not ck1.exists():
        log('[X] 3A 无权重')
        return
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', ck1, 'inc_p1_v3')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', ck1,
               'inc_p1_v3_base_reg', img_dir=WORK,
               ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')

    # 3B
    log('')
    log('--- 类增量 3B ---')
    init2 = expand('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', ck1, 8, 'inc_p2')
    if not init2:
        log('[X] 扩头失败')
        return
    p2 = DEIM / 'output' / 'deim_vistarget_inc_p2'
    if p2.exists():
        shutil.rmtree(p2, ignore_errors=True)
    launch('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', init2, 'inc_p2_v3')
    wait_epochs(p2, 20, 'deim_vistarget_inc_p2.yml')
    ck2 = p2 / 'best_stg1.pth'
    if ck2.exists():
        clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', ck2, 'inc_p2_v3')
        clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', ck2,
                   'inc_p2_v3_base_reg', img_dir=WORK,
                   ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
                   ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')
        clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', ck2,
                   'inc_p2_v3_p1_reg', img_dir=WORK,
                   ann_val=WORK / 'inc_p1' / 'annotations' / 'rel_val.json',
                   ann_test=WORK / 'inc_p1' / 'annotations' / 'rel_test.json')

    log('')
    log('=' * 70)
    log('全部完成')
    log('=' * 70)


if __name__ == '__main__':
    main()
