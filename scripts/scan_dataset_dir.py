# -*- coding: utf-8 -*-
"""扫描 D:\\BaiduNetdiskDownload\\datasets\\dataset, 并与 vistarget 对比。

该版本没有 data.yaml, 只有 classes.txt, 结构为 <split>/<stage>/{images,labels}/。
"""
import io
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

D = Path(r'D:\BaiduNetdiskDownload\datasets\dataset')
V = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget')
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

# ---- 类别 ----
cls_file = D / 'classes.txt'
names = {}
if cls_file.exists():
    for i, ln in enumerate(cls_file.read_text(encoding='utf-8').splitlines()):
        ln = ln.strip()
        if ln:
            names[i] = ln
print('=' * 84)
print('dataset 扫描')
print('=' * 84)
print(f'  classes.txt: {len(names)} 类')
for k, v in names.items():
    print(f'    {k} = {v}')

STAGES = ('base', 'transfer', 'inc_p1', 'inc_p2')
SPLITS = ('train', 'val', 'test', 'replay')

rows = []
for stage in STAGES:
    for sp in SPLITS:
        img_dir = D / stage / sp / 'images'
        lbl_dir = D / stage / sp / 'labels'
        if not img_dir.exists():
            continue
        imgs = {p.stem for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}
        lbls = {p.stem: p for p in lbl_dir.iterdir() if p.is_file() and p.suffix == '.txt'} if lbl_dir.exists() else {}

        cls_cnt = Counter()
        eq_sizes = []
        per_img = Counter()
        empty = 0
        for stem, lp in lbls.items():
            lines = [l.strip() for l in lp.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
            if not lines:
                empty += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    w, h = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                cls_cnt[cid] += 1
                per_img[stem] += 1
                eq_sizes.append((w * 1920 * h * 1080) ** 0.5)

        n_ann = sum(cls_cnt.values())
        rows.append({
            'stage': stage, 'split': sp, 'imgs': len(imgs), 'lbls': len(lbls),
            'anns': n_ann, 'empty': empty,
            'classes': dict(cls_cnt), 'median_px': (sorted(eq_sizes)[len(eq_sizes) // 2] if eq_sizes else 0),
            'pct_lt32': (sum(1 for s in eq_sizes if s < 32) / len(eq_sizes) * 100 if eq_sizes else 0),
            'per_img_median': (sorted(per_img.values())[len(per_img) // 2] if per_img else 0),
            'missing_lbl': len(imgs - set(lbls)),
        })

print()
print('=' * 84)
print(f'{"stage":<10}{"split":<8}{"图片":>7}{"标注":>8}{"无标注":>7}{"中位px":>8}{"<32px":>7}{"图均":>6}  覆盖类别')
print('=' * 84)
for r in rows:
    covered = [names.get(c, str(c)) for c in sorted(r['classes'])]
    s = ','.join(covered) if len(covered) <= 4 else f'{len(covered)}类'
    print(f'{r["stage"]:<10}{r["split"]:<8}{r["imgs"]:>7,}{r["anns"]:>8,}{r["empty"]:>7,}'
          f'{r["median_px"]:>8.0f}{r["pct_lt32"]:>6.1f}%{r["per_img_median"]:>6}  {s}')

# 每阶段类别分布明细
print()
print('=' * 84)
print('各类别框数(train 划分)')
print('=' * 84)
print(f'  {"stage":<10}' + ''.join(f'{names.get(c,"c"+str(c))[:14]:>16}' for c in sorted(names)))
for stage in STAGES:
    r = next((x for x in rows if x['stage'] == stage and x['split'] == 'train'), None)
    if not r:
        continue
    line = f'  {stage:<10}'
    for c in sorted(names):
        n = r['classes'].get(c, 0)
        line += f'{n:>16,}'
    print(line)

# ---- 与 vistarget 对比 ----
print()
print('=' * 84)
print('与 vistarget 对比(图片数)')
print('=' * 84)
print(f'  {"stage/split":<20}{"dataset":>10}{"vistarget":>12}{"差异":>10}')
for stage in STAGES:
    for sp in SPLITS:
        cur = next((x for x in rows if x['stage'] == stage and x['split'] == sp), None)
        vd = V / stage / sp / 'images'
        v_n = len([p for p in vd.iterdir() if p.is_file()]) if vd.exists() else None
        if cur is None and v_n is None:
            continue
        d_n = cur['imgs'] if cur else 0
        if v_n is None:
            print(f'  {stage+"/"+sp:<20}{d_n:>10,}{"不存在":>12}{"":>10}')
            continue
        diff = d_n - v_n
        flag = '' if diff == 0 else ('  <<<' if abs(diff) > 0 else '')
        print(f'  {stage+"/"+sp:<20}{d_n:>10,}{v_n:>12,}{diff:>+10,}{flag}')

# vistarget 独有的 all
va = V / 'all' / 'train' / 'images'
if va.exists():
    print(f'  {"all/train(仅 vistarget)":<20}{"不存在":>10}{len([p for p in va.iterdir() if p.is_file()]):>12,}')
