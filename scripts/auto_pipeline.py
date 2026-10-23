# -*- coding: utf-8 -*-
"""全自动流水线: base 收尾 -> 续训(带饱和检测) -> 干净评估 -> 达标则接力 transfer/3A/3B。

执行顺序:
    ① 等 base 60 轮跑完
    ② 停掉 rerun_all(否则它会自动跳去 transfer)
    ③ 启动续训(base_ext.yml, 33 轮), 每 10 轮检查饱和
        饱和判据: 最近 5 轮 mAP50 增量 < 0.005
    ④ 干净评估(val 定阈值 -> test 验证)
    ⑤ recall >= 0.90 时自动接力:
         transfer(120轮) -> 扩头6类 -> 3A(20轮) -> 扩头8类 -> 3B(20轮)
         每个阶段都做一次干净评估
    ⑥ 任一步失败或未达标, 记录并停止

用法:
    python auto_pipeline.py
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
LOG = DEIM / 'output' / 'auto_pipeline.log'

SAT_WINDOW = 5          # 用最近几轮判断饱和
SAT_THRESHOLD = 0.005   # 窗口内增量低于此值视为饱和
TARGET_RECALL = 0.90


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


def m50_series(p):
    return [((r.get('test_coco_eval_bbox') or [0])[1], r.get('epoch')) for r in read_log(p)]


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
    pids = procs(patterns)
    import psutil
    for pid in pids:
        try:
            psutil.Process(pid).terminate()
            log(f'  终止 {tag} PID {pid}')
        except Exception:
            pass
    return pids


def wait_gpu_free(max_wait=120):
    for _ in range(max_wait // 3):
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True)
        s = (out.stdout or '0').strip().split()
        used = int(s[0]) if s else 0
        if used < 2500:
            return True
        time.sleep(3)
    return False


def launch_train(cfg, init_ckpt, tag, extra=None):
    cmd = [PY, '-X', 'utf8', 'train.py', '-c', cfg, '-d', 'cuda:0', '--use-amp']
    if init_ckpt:
        cmd += ['-t', str(init_ckpt)]
    if extra:
        cmd += extra
    log(f'  启动训练: {cfg}' + (f'  init={Path(init_ckpt).name}' if init_ckpt else ''))
    f = open(DEIM / 'output' / f'{tag}.stdout.log', 'w', encoding='utf-8')
    subprocess.Popen(cmd, cwd=str(DEIM), stdout=f, stderr=subprocess.STDOUT)


def wait_epochs(out_dir, target, cfg_name, poll=60, timeout=8 * 3600, sat_check=True):
    """等训练达到 target 轮; sat_check=True 时中途做饱和检测。"""
    logp = Path(out_dir) / 'log.txt'
    t0 = time.time()
    last = -1
    sat_rounds = 0
    while time.time() - t0 < timeout:
        recs = read_log(logp)
        n = len(recs)
        if n != last:
            if recs:
                b = max(recs, key=lambda r: r['test_coco_eval_bbox'][0])
                bb = b['test_coco_eval_bbox']
                log(f'    {n}/{target}  最优 ep{b["epoch"]} mAP50={bb[1]:.4f}')
            last = n
            # 饱和检测(每满 10 轮查一次)
            if sat_check and n >= 10 and n % 10 == 0:
                series = [x[0] for x in m50_series(logp)]
                if len(series) >= SAT_WINDOW:
                    delta = series[-1] - series[-SAT_WINDOW]
                    log(f'      [饱和检查] 最近 {SAT_WINDOW} 轮增量 {delta:+.4f}')
                    if delta < SAT_THRESHOLD:
                        sat_rounds += 1
                        log(f'      => 判定饱和(第 {sat_rounds} 次确认)')
                        if sat_rounds >= 1:
                            log('  提前结束训练(已饱和)')
                            return 'saturated', n
                    else:
                        sat_rounds = 0
        if n >= target:
            log(f'  完成 {n}/{target} 轮')
            return 'done', n
        if not procs(['train.py', cfg_name]):
            n2 = len(read_log(logp))
            log(f'  进程退出, 完成 {n2}/{target} 轮')
            return 'exited', n2
        time.sleep(poll)
    return 'timeout', len(read_log(logp))


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
    log(f'    [!] 评估未拿到结果')
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
    log('=' * 70)
    log('自动流水线启动')
    log('=' * 70)

    # ---------- ① 等 base 60 轮 ----------
    base_dir = DEIM / 'output' / 'deim_vistarget_base'
    log('等 base 跑满 60 轮')
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        n = len(read_log(base_dir / 'log.txt'))
        if n >= 60:
            break
        time.sleep(30)
    n = len(read_log(base_dir / 'log.txt'))
    log(f'base 当前 {n} 轮')
    time.sleep(20)

    # ---------- ② 停 rerun_all ----------
    kill(['rerun_all.py'], 'rerun_all')
    kill(['train.py', 'deim_vistarget_base.yml'], 'base 训练')
    wait_gpu_free()
    log('GPU 已释放')

    base_recs = read_log(base_dir / 'log.txt')
    b = max(base_recs, key=lambda r: r['test_coco_eval_bbox'][0])
    log(f'base 60 轮最优: ep{b["epoch"]}  mAP50={b["test_coco_eval_bbox"][1]:.4f}')
    base_ckpt = base_dir / 'best_stg1.pth'

    # ---------- ③ 续训 ----------
    ext_dir = DEIM / 'output' / 'deim_vistarget_base_ext'
    if ext_dir.exists():
        shutil.rmtree(ext_dir, ignore_errors=True)
    log('启动续训(base_ext, 33 轮)')
    launch_train('configs/deim_rtdetrv2/deim_vistarget_base_ext.yml', base_ckpt, 'base_ext')
    status, got = wait_epochs(ext_dir, 33, 'deim_vistarget_base_ext.yml', sat_check=True)
    log(f'续训结束: {status}, {got} 轮')

    # 用全局最优(60 轮 vs 续训 取较高者)
    final_ckpt = base_ckpt
    final_desc = f'base 60轮 ep{b["epoch"]} mAP50={b["test_coco_eval_bbox"][1]:.4f}'
    ext_best = max(read_log(ext_dir / 'log.txt'), key=lambda r: r['test_coco_eval_bbox'][0]) \
        if read_log(ext_dir / 'log.txt') else None
    if ext_best and ext_best['test_coco_eval_bbox'][1] > b['test_coco_eval_bbox'][1]:
        final_ckpt = ext_dir / 'best_stg1.pth'
        final_desc = (f'续训 ep{ext_best["epoch"]} '
                      f'mAP50={ext_best["test_coco_eval_bbox"][1]:.4f}')
    log(f'采用权重: {final_desc}')

    # ---------- ④ 干净评估 ----------
    kill(['train.py'], '残留训练')
    wait_gpu_free()
    c = clean_eval('configs/deim_rtdetrv2/deim_vistarget_base.yml', final_ckpt, 'base_final')
    if not c:
        log('[X] base 评估失败, 停止')
        return
    if c['recall'] < TARGET_RECALL:
        log(f'[X] base recall {c["recall"]:.4f} < {TARGET_RECALL}, 不启动后续阶段')
        return
    log(f'[OK] base 达标 recall {c["recall"]:.4f}, 接力后续阶段')

    # ---------- ⑤ transfer ----------
    log('')
    log('--- transfer (120 轮) ---')
    tr_dir = DEIM / 'output' / 'deim_vistarget_transfer'
    if tr_dir.exists():
        shutil.rmtree(tr_dir, ignore_errors=True)
    launch_train('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', final_ckpt, 'transfer')
    wait_epochs(tr_dir, 120, 'deim_vistarget_transfer.yml', sat_check=False)
    tr_ckpt = tr_dir / 'best_stg1.pth'
    if not tr_ckpt.exists():
        log('[X] transfer 无权重, 停止')
        return
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr_ckpt, 'transfer_final')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr_ckpt,
               'transfer_base_reg', img_dir=WORK,
               ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')

    # ---------- ⑥ 3A ----------
    log('')
    log('--- 类增量 3A (20 轮) ---')
    init1 = expand('configs/deim_rtdetrv2/deim_vistarget_transfer.yml', tr_ckpt, 6, 'inc_p1')
    if not init1:
        log('[X] 扩头失败, 停止')
        return
    p1_dir = DEIM / 'output' / 'deim_vistarget_inc_p1'
    if p1_dir.exists():
        shutil.rmtree(p1_dir, ignore_errors=True)
    launch_train('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', init1, 'inc_p1')
    wait_epochs(p1_dir, 20, 'deim_vistarget_inc_p1.yml', sat_check=False)
    p1_ckpt = p1_dir / 'best_stg1.pth'
    if not p1_ckpt.exists():
        log('[X] 3A 无权重, 停止')
        return
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', p1_ckpt, 'inc_p1_final')
    clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', p1_ckpt,
               'inc_p1_base_reg', img_dir=WORK,
               ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
               ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')

    # ---------- ⑦ 3B ----------
    log('')
    log('--- 类增量 3B (20 轮) ---')
    init2 = expand('configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', p1_ckpt, 8, 'inc_p2')
    if not init2:
        log('[X] 扩头失败, 停止')
        return
    p2_dir = DEIM / 'output' / 'deim_vistarget_inc_p2'
    if p2_dir.exists():
        shutil.rmtree(p2_dir, ignore_errors=True)
    launch_train('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', init2, 'inc_p2')
    wait_epochs(p2_dir, 20, 'deim_vistarget_inc_p2.yml', sat_check=False)
    p2_ckpt = p2_dir / 'best_stg1.pth'
    if p2_ckpt.exists():
        clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', p2_ckpt, 'inc_p2_final')
        clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', p2_ckpt,
                   'inc_p2_base_reg', img_dir=WORK,
                   ann_val=WORK / 'base' / 'annotations' / 'rel_val.json',
                   ann_test=WORK / 'base' / 'annotations' / 'rel_test.json')
        clean_eval('configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', p2_ckpt,
                   'inc_p2_p1_reg', img_dir=WORK,
                   ann_val=WORK / 'inc_p1' / 'annotations' / 'rel_val.json',
                   ann_test=WORK / 'inc_p1' / 'annotations' / 'rel_test.json')

    log('')
    log('=' * 70)
    log('全流程结束')
    log('=' * 70)


if __name__ == '__main__':
    main()
