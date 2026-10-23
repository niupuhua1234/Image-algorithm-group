# -*- coding: utf-8 -*-
"""base 数据集详细画像 —— 场景特点、难度、数据来源线索。"""
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

BASE = Path(r'D:\BaiduNetdiskDownload\datasets\dataset\base')
NAMES = {0: 'car1', 1: 'car2', 2: 'car3', 3: 'car4'}
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
W, H = 1920, 1080

print('=' * 80)
print('base 数据集画像')
print('=' * 80)

per_split = {}
for sp in ('train', 'val', 'test'):
    img_dir = BASE / sp / 'images'
    lbl_dir = BASE / sp / 'labels'
    imgs = {p.stem: p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}
    lbls = {p.stem: p for p in lbl_dir.iterdir() if p.is_file() and p.suffix == '.txt'}

    cls_cnt = Counter()
    cls_sizes = defaultdict(list)
    per_img = Counter()
    empty = 0
    for stem, lp in lbls.items():
        lines = [l.strip() for l in lp.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
        if not lines:
            empty += 1
            continue
        for ln in lines:
            p = ln.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            w, h = float(p[3]), float(p[4])
            cls_cnt[cid] += 1
            cls_sizes[cid].append((w * W * h * H) ** 0.5)
            per_img[stem] += 1

    per_split[sp] = dict(imgs=imgs, lbls=lbls, cls=cls_cnt, cls_sizes=cls_sizes,
                         per_img=per_img, empty=empty)

# ---- 规模 ----
print('\n【规模】')
print(f'  {"split":<8}{"图片":>8}{"标注框":>10}{"无标注图":>9}{"图均目标":>9}')
for sp, d in per_split.items():
    n_ann = sum(d['cls'].values())
    avg = n_ann / (len(d['imgs']) - d['empty']) if len(d['imgs']) > d['empty'] else 0
    print(f'  {sp:<8}{len(d["imgs"]):>8,}{n_ann:>10,}{d["empty"]:>9,}{avg:>9.1f}')

# ---- 类别 ----
print('\n【类别分布】(4 类)')
print(f'  {"类别":<10}{"train":>9}{"val":>8}{"test":>8}{"合计":>9}{"占比":>8}')
tot_all = sum(sum(d['cls'].values()) for d in per_split.values())
for cid, nm in NAMES.items():
    tr = per_split['train']['cls'].get(cid, 0)
    va = per_split['val']['cls'].get(cid, 0)
    te = per_split['test']['cls'].get(cid, 0)
    tot = tr + va + te
    print(f'  {nm:<10}{tr:>9,}{va:>8,}{te:>8,}{tot:>9,}{tot/tot_all*100:>7.1f}%')

# ---- 目标尺寸 ----
print('\n【目标尺寸】(等效边长 sqrt(w*h), 像素 @1920x1080)')
sizes = []
for d in per_split.values():
    for v in d['cls_sizes'].values():
        sizes.extend(v)
sizes.sort()
buckets = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28),
           (28, 32), (32, 40), (40, 48), (48, 64), (64, 10**9)]
print(f'  {"区间(px)":<14}{"数量":>9}{"占比":>8}   分布')
for lo, hi in buckets:
    n = sum(1 for s in sizes if lo <= s < hi)
    if not n and lo > 8:
        continue
    bar = '#' * max(0, int(n / max(1, len(sizes)) * 120))
    lbl = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
    print(f'  {lbl:<14}{n:>9,}{n/len(sizes)*100:>7.1f}%   {bar}')
def q(p):
    return sizes[min(len(sizes) - 1, int(len(sizes) * p))]
print(f'  p5 {q(.05):.0f}  p25 {q(.25):.0f}  中位 {q(.5):.0f}  p75 {q(.75):.0f}  '
      f'p95 {q(.95):.0f}  max {sizes[-1]:.0f} px')

# 类别间尺寸差异
print('\n【各类别的目标尺寸】')
print(f'  {"类别":<10}{"中位":>7}{"p25":>7}{"p75":>7}{"<32px":>8}')
for cid, nm in NAMES.items():
    ss = sorted(per_split['train']['cls_sizes'].get(cid, []))
    if not ss:
        continue
    m = ss[len(ss) // 2]
    p25 = ss[int(len(ss) * .25)]
    p75 = ss[int(len(ss) * .75)]
    lt32 = sum(1 for s in ss if s < 32) / len(ss) * 100
    print(f'  {nm:<10}{m:>7.0f}{p25:>7.0f}{p75:>7.0f}{lt32:>7.1f}%')

# ---- 每图目标数 ----
print('\n【每图目标数】(train)')
cnts = sorted(per_split['train']['per_img'].values())
c = Counter(cnts)
for k in range(0, 10):
    n = c.get(k, 0)
    if n:
        print(f'  {k} 个目标: {n:>5,} 张  ({n/len(cnts)*100:>5.1f}%)   ' + '#' * int(n / max(c.values()) * 50))
print(f'  中位 {cnts[len(cnts)//2]}   p90 {cnts[int(len(cnts)*.9)]}   max {cnts[-1]}')

# ---- 文件名来源线索 ----
print('\n【文件名前缀】(推测数据来源)')
prefix = Counter()
for d in per_split.values():
    for stem in d['imgs']:
        if '__' in stem:
            prefix[stem.split('__')[0]] += 1
        else:
            import re
            m = re.match(r'^([A-Za-z]+)', stem)
            prefix[m.group(1) if m else stem[:4]] += 1
for k, v in prefix.most_common(12):
    print(f'  {k:<28}{v:>7,} 张')

# ---- 图片属性 ----
print('\n【图片】')
from PIL import Image
sample = list(per_split['train']['imgs'].values())
dims = Counter()
tot_bytes = 0
for p in sample:
    tot_bytes += p.stat().st_size
    try:
        with Image.open(p) as im:
            dims[im.size] += 1
    except Exception:
        pass
n_chk = min(len(sample), 200)
print(f'  格式 PNG    抽样尺寸: {dict(dims.most_common(3))}')
print(f'  平均体积 {tot_bytes/len(sample)/1e6:.2f} MB/张(全量 train 约 {tot_bytes/len(sample)*len(sample)/1e9:.1f} GB)')

# ---- 跨划分一致性 ----
print('\n【跨划分一致性】')
for key in ('cls',):
    tr = set(per_split['train'][key])
    va = set(per_split['val'][key])
    te = set(per_split['test'][key])
    print(f'  类别集合  train={sorted(tr)}  val={sorted(va)}  test={sorted(te)}  '
          f'{"一致 ✓" if tr == va == te else "不一致 ✗"}')

print('\n【难度提示】')
lt32 = sum(1 for s in sizes if s < 32) / len(sizes) * 100
lt20 = sum(1 for s in sizes if s < 20) / len(sizes) * 100
print(f'  <20px 占 {lt20:.1f}%   <32px 占 {lt32:.1f}%')
print(f'  每图仅 {len(list(per_split["train"]["per_img"].values()))/len(per_split["train"]["imgs"]):.1f} 个目标 —— 稀疏场景')
