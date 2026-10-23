# -*- coding: utf-8 -*-
"""生成汇报文档所需的全部图表(基于最终数据, 不含作废的中间版本)。

数据来源:
    - 数据集统计   : vistarget_work/*/annotations/*.json
    - 训练曲线     : deim/output/*/log.txt
    - 干净评估     : deim/output/cleaneval_*/report.json, perclass_*.json
    - 时延         : deim/output/latency_*.json

输出: D:\\BaiduNetdiskDownload\\wdd_work\\report_figs_v2\\
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DengXian']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

WORK = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')
DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
OUT = Path(r'D:\BaiduNetdiskDownload\wdd_work\report_figs_v2')
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ['car1', 'car2', 'car3', 'car4',
           'military_fuel_tanker', 'radar_vehicle',
           'military_motorcycle', 'bridge_layer']
SHORT = ['car1', 'car2', 'car3', 'car4', 'fuel_tanker', 'radar_veh', 'motorcycle', 'bridge']

STAGES = [('base', '基础(4类)'), ('transfer', '迁移(4类)'),
          ('inc_p1', '增量3A(6类)'), ('inc_p2', '增量3B(8类)')]
COLORS = ['#3B6DB3', '#E07B39', '#3FA45C', '#C44E52', '#8172B3', '#937860']


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def read_log(p):
    import ast
    p = Path(p)
    if not p.exists():
        return []
    return [ast.literal_eval(l) for l in p.read_text(encoding='utf-8', errors='ignore').splitlines()
            if l.strip().startswith('{')]


# ============ 图1: 数据规模 ============
def fig_size():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels, tr, va, te = [], [], [], []
    for stage, label in STAGES:
        for cand in ('train_replay.json', 'rel_train.json', 'train.json'):
            p = WORK / stage / 'annotations' / cand
            if p.exists():
                d = load_json(p)
                labels.append(label)
                tr.append(len(d['images']))
                break
        v = load_json(WORK / stage / 'annotations' / ('rel_val.json' if stage != 'transfer' else 'val.json'))
        t = load_json(WORK / stage / 'annotations' / ('rel_test.json' if stage != 'transfer' else 'test.json'))
        va.append(len(v['images']) if v else 0)
        te.append(len(t['images']) if t else 0)

    x = np.arange(len(labels))
    w = 0.26
    for i, (vals, nm, c) in enumerate(zip([tr, va, te], ['train', 'val', 'test'],
                                          ['#3B6DB3', '#E07B39', '#3FA45C'])):
        b = ax.bar(x + (i - 1) * w, vals, w, label=nm, color=c)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + max(tr) * 0.015, f'{v:,}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('图片数'); ax.set_title('图1  各阶段数据划分规模', fontsize=12, weight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, max(tr) * 1.18)
    fig.savefig(OUT / '01_size.png'); plt.close(fig)
    print('  01_size.png')


# ============ 图2: 类别分布 ============
def fig_class():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    labels, mat = [], []
    for stage, label in STAGES:
        for cand in ('train_replay.json', 'rel_train.json', 'train.json'):
            p = WORK / stage / 'annotations' / cand
            if p.exists():
                d = load_json(p)
                cnt = {}
                for a in d['annotations']:
                    cnt[a['category_id']] = cnt.get(a['category_id'], 0) + 1
                labels.append(label)
                mat.append([cnt.get(i, 0) for i in range(8)])
                break
    mat = np.array(mat, float)
    pct = mat / np.maximum(mat.sum(1, keepdims=True), 1) * 100
    x = np.arange(8); w = 0.2
    for i, (lab, row) in enumerate(zip(labels, pct)):
        ax.bar(x + (i - 1.5) * w, row, w, label=lab, color=COLORS[i])
    ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=18, ha='right')
    ax.set_ylabel('占该阶段训练框比例 (%)')
    ax.set_title('图2  各阶段训练集类别占比', fontsize=12, weight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.savefig(OUT / '02_class.png'); plt.close(fig)
    print('  02_class.png')


# ============ 图3: 目标尺寸 ============
def fig_obj():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = list(range(8, 56, 4))
    for (stage, label), c in zip(STAGES, COLORS):
        for cand in ('train_replay.json', 'rel_train.json', 'train.json'):
            p = WORK / stage / 'annotations' / cand
            if not p.exists():
                continue
            d = load_json(p)
            sizes = [(a['bbox'][2] * a['bbox'][3]) ** 0.5 for a in d['annotations']
                     if a['bbox'][2] > 0 and a['bbox'][3] > 0]
            if sizes:
                ax.hist(sizes, bins=bins, alpha=0.45, label=label, color=c, density=True)
            break
    ax.axvline(32, color='red', ls='--', lw=1.3, label='32px 参考线')
    ax.set_xlabel('目标等效边长 sqrt(w×h) (px, 原图 1920×1080)')
    ax.set_ylabel('概率密度')
    ax.set_title('图3  各阶段训练集目标尺寸分布', fontsize=12, weight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(OUT / '03_obj.png'); plt.close(fig)
    print('  03_obj.png')


# ============ 图4: 训练曲线 ============
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for (stage, label), c in zip(STAGES, COLORS):
        recs = read_log(DEIM / 'output' / f'deim_vistarget_{stage}' / 'log.txt')
        if not recs:
            continue
        ep = [r['epoch'] for r in recs]
        m = [(r.get('test_coco_eval_bbox') or [0])[1] for r in recs]
        lo = [r.get('train_loss', 0) for r in recs]
        axes[0].plot(ep, m, label=label, color=c, lw=1.7)
        axes[1].plot(ep, lo, label=label, color=c, lw=1.7)
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('mAP@0.50')
    axes[0].set_title('验证集 mAP@0.50 曲线', fontsize=11, weight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss(log 尺度)')
    axes[1].set_yscale('log')
    axes[1].set_title('训练损失曲线', fontsize=11, weight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle('图4  四阶段训练过程', fontsize=12, weight='bold')
    fig.savefig(OUT / '04_curves.png'); plt.close(fig)
    print('  04_curves.png')


# ============ 图5: 各阶段干净评估 ============
def fig_eval():
    items = [('base', '基础'), ('transfer', '迁移'), ('inc_p1', '增量3A'), ('inc_p2', '增量3B')]
    rows = []
    for tag, label in items:
        TAGMAP = {'base': 'base_final', 'transfer': 'transfer_final',
                  'inc_p1': 'inc_p1_v3', 'inc_p2': 'inc_p2_v4'}
        for f in (f'cleaneval_{TAGMAP.get(tag, tag)}',):
            j = load_json(DEIM / 'output' / f / 'report.json')
            if j and j.get('test_at_val_conf'):
                rows.append((label, j['test_at_val_conf']))
                break
    if not rows:
        print('  05_eval.png  (无数据)'); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = [r[0] for r in rows]
    rec = [r[1]['recall'] * 100 for r in rows]
    pre = [r[1]['precision'] * 100 for r in rows]
    fp = [r[1]['fp_per_img'] for r in rows]
    x = np.arange(len(labels))

    b = axes[0].bar(x, rec, color='#3B6DB3')
    axes[0].axhline(90, color='red', ls='--', lw=1.3, label='90% 门槛')
    axes[0].set_ylim(80, 100)
    for r, v in zip(b, rec):
        axes[0].text(r.get_x() + r.get_width() / 2, v + 0.3, f'{v:.2f}', ha='center', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('recall (%)'); axes[0].set_title('Recall', fontsize=11, weight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

    b = axes[1].bar(x, pre, color='#3FA45C')
    for r, v in zip(b, pre):
        axes[1].text(r.get_x() + r.get_width() / 2, v + 1, f'{v:.2f}', ha='center', fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('precision (%)'); axes[1].set_title('Precision', fontsize=11, weight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    b = axes[2].bar(x, fp, color='#C44E52')
    top = max(fp) * 1.3 if max(fp) > 0 else 1
    axes[2].set_ylim(0, top)
    for r, v in zip(b, fp):
        axes[2].text(r.get_x() + r.get_width() / 2, v + top * 0.04, f'{v:.2f}',
                     ha='center', fontsize=9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('误检数 / 图'); axes[2].set_title('每图误检(越低越好)', fontsize=11, weight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle('图5  各阶段干净评估(val 定阈值 → test 验证)', fontsize=12, weight='bold')
    fig.savefig(OUT / '05_eval.png'); plt.close(fig)
    print('  05_eval.png')


# ============ 图6: conf 曲线 ============
def fig_conf():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    shown = 0
    for (tag, label), c in zip([('base', '基础'), ('transfer', '迁移'),
                                ('inc_p1', '增量3A'), ('inc_p2', '增量3B')], COLORS):
        j = None
        TAGMAP = {'base': 'base_final', 'transfer': 'transfer_final',
                  'inc_p1': 'inc_p1_v3', 'inc_p2': 'inc_p2_v4'}
        for f in (f'cleaneval_{TAGMAP.get(tag, tag)}',):
            cand = load_json(DEIM / 'output' / f / 'test_full_sweep.json')
            if cand:
                j = cand
                break
        if not j or not j.get('rows'):
            continue
        rows = sorted(j['rows'], key=lambda r: r['conf'])
        conf = [r['conf'] for r in rows]
        axes[0].plot(conf, [r['recall'] * 100 for r in rows], label=label, color=c, lw=1.6)
        axes[1].plot(conf, [r['precision'] * 100 for r in rows], label=label, color=c, lw=1.6)
        shown += 1
    if not shown:
        print('  06_conf.png  (无数据)'); return
    axes[0].axhline(90, color='red', ls='--', lw=1.2, label='90% 门槛')
    axes[0].set_xlabel('confidence 阈值'); axes[0].set_ylabel('recall (%)')
    axes[0].set_title('阈值 — Recall', fontsize=11, weight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('confidence 阈值'); axes[1].set_ylabel('precision (%)')
    axes[1].set_title('阈值 — Precision', fontsize=11, weight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle('图6  置信度阈值扫描(test 集, 用于分析)', fontsize=12, weight='bold')
    fig.savefig(OUT / '06_conf.png'); plt.close(fig)
    print('  06_conf.png')


# ============ 图7: 时延 ============
def fig_latency():
    files = [('latency_base_clean.json', 'HDD+串行\n(修复前)'),
             ('latency_base_hdd_final.json', 'HDD+并行\n(修复后)'),
             ('latency_base_preload.json', '内存预加载')]
    rows = []
    for f, label in files:
        j = load_json(DEIM / 'output' / f)
        if j:
            rows.append((label, j))
    if not rows:
        print('  07_latency.png  (无数据)'); return
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(rows)); w = 0.26
    for i, (k, nm, c) in enumerate(zip(['forward_ms', 'preprocess_ms', 'end_to_end_ms'],
                                        ['forward', '预处理', '端到端'],
                                        ['#3B6DB3', '#E07B39', '#C44E52'])):
        vals = [r[1][k]['mean'] for r in rows]
        b = ax.bar(x + (i - 1) * w, vals, w, label=nm, color=c)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, f'{v:.1f}', ha='center',
                    va='bottom', fontsize=8)
    ax.axhline(40, color='red', ls='--', lw=1.3, label='40ms 门槛')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('时延 (ms)')
    ax.set_title('图7  端到端时延对比(mean, RTX 5080, 1280 输入)', fontsize=12, weight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.savefig(OUT / '07_latency.png'); plt.close(fig)
    print('  07_latency.png')


# ============ 图8: 遗忘检查 ============
def fig_forget():
    items = [('基础阶段', 'cleaneval_base_final'), ('迁移后', 'cleaneval_transfer_base_reg'),
             ('增量3A后', 'cleaneval_inc_p1_v3_base_reg'), ('增量3B后', 'cleaneval_inc_p2_v4_base_reg')]
    rows = []
    for label, tag in items:
        j = load_json(DEIM / 'output' / tag / 'report.json')
        c = (j or {}).get('test_at_val_conf')
        if c:
            rows.append((label, c['recall'] * 100))
    if not rows:
        print('  08_forget.png  (无数据)'); return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    b = ax.bar(np.arange(len(labels)), vals, color=['#3B6DB3', '#E07B39', '#3FA45C', '#8172B3'])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.15, f'{v:.2f}%', ha='center', fontsize=9)
    ax.set_ylim(min(vals) - 3, max(vals) + 2)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel('老类(base/test) recall (%)')
    ax.set_title('图8  灾难性遗忘检查 — 老类在各阶段的召回', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(OUT / '08_forget.png'); plt.close(fig)
    print('  08_forget.png')


# ============ 图9: 按类别定阈值对比 ============
def fig_perclass():
    stages = [('inc_p1', '增量3A'), ('inc_p2', '增量3B')]
    data = []
    for tag, label in stages:
        j = load_json(DEIM / 'output' / f'perclass_{tag}.json')
        if not j:
            continue
        for cls, s in (j.get('test_per_class') or {}).items():
            if isinstance(s, dict) and s.get('gt', 0) > 0:
                data.append((label, cls, j['chosen_conf'].get(cls, 0),
                             s.get('recall', 0) * 100, s.get('precision', 0) * 100))
    if not data:
        print('  09_perclass.png  (无数据)'); return
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(data))
    confs = [d[2] for d in data]
    b = ax.bar(x, confs, color=['#3B6DB3' if d[0] == '增量3A' else '#3FA45C' for d in data])
    for r, d in zip(b, data):
        ax.text(r.get_x() + r.get_width() / 2, d[2] + 0.01,
                f'{d[2]:.2f}\nR{d[3]:.0f}%P{d[4]:.0f}%', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d[1][:16]}' for d in data], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('按类别定出的 conf')
    ax.set_title('图9  按类别定阈值 — 各类最优 conf(标注重该类的 R/P)', fontsize=11, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(OUT / '09_perclass.png'); plt.close(fig)
    print('  09_perclass.png')


if __name__ == '__main__':
    print('=' * 56)
    print('生成汇报图表')
    print('=' * 56)
    for fn in (fig_size, fig_class, fig_obj, fig_curves, fig_eval,
               fig_conf, fig_latency, fig_forget, fig_perclass):
        try:
            fn()
        except Exception as e:
            print(f'  [!] {fn.__name__} 失败: {e}')
    print()
    for p in sorted(OUT.glob('*.png')):
        print(f'  {p.name:<22} {p.stat().st_size/1024:.0f} KB')
