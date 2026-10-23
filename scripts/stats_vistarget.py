# -*- coding: utf-8 -*-
"""统计 vistarget 各 split 的规模、类别分布与目标尺寸(YOLO txt 标注)。

YOLO 标注格式: 每行 `class_id cx cy w h`(全部归一化到 0~1)
等效边长 = sqrt((w*W) * (h*H)), 需要图片尺寸; 这里假设用户数据统一 1920x1080,
实际会从 json/yaml 或抽样图片读取。
"""
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

D = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget')
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_names(yaml_path: Path):
    """简易 yaml 解析: 只取 names: 下的 id->name。"""
    import yaml
    d = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    names = d.get('names') or {}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def image_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


print('=' * 84)
print('vistarget 数据集统计 (YOLO txt 标注)')
print('=' * 84)

summary = {}
for split_name in ('base', 'transfer', 'inc_p1', 'inc_p2', 'all'):
    root = D / split_name
    yml = root / 'data.yaml'
    if not yml.exists():
        continue
    names = read_names(yml)
    print(f'\n{"#"*84}')
    print(f'# {split_name}   类别 {len(names)}: ' + ', '.join(f'{k}={v}' for k, v in sorted(names.items())))
    print(f'{"#"*84}')

    for sp in ('train', 'val', 'test', 'replay'):
        img_dir = root / sp / 'images'
        lbl_dir = root / sp / 'labels'
        if not img_dir.exists():
            continue
        imgs = {p.stem: p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}
        lbls = {p.stem: p for p in lbl_dir.iterdir() if p.is_file() and p.suffix == '.txt'} if lbl_dir.exists() else {}
        missing_lbl = set(imgs) - set(lbls)
        orphan_lbl = set(lbls) - set(imgs)

        # 抽样图片尺寸
        sizes = set()
        for p in list(imgs.values())[:5]:
            try:
                sizes.add(image_size(p))
            except Exception:
                pass
        WH = sizes.pop() if len(sizes) == 1 else (1920, 1080)
        W, H = WH

        cls_cnt = Counter()
        eq_sizes = []
        per_img = Counter()
        empty = 0
        bad_lines = 0
        for stem, lp in lbls.items():
            lines = [l.strip() for l in lp.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
            if not lines:
                empty += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    bad_lines += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    w, h = float(parts[3]), float(parts[4])
                except ValueError:
                    bad_lines += 1
                    continue
                cls_cnt[cid] += 1
                per_img[stem] += 1
                eq_sizes.append((w * W * h * H) ** 0.5)

        n_ann = sum(cls_cnt.values())
        print(f'\n  [{sp}]  图片 {len(imgs):,}   标注 {n_ann:,}   尺寸 {W}x{H}'
              + (f'   无标注图 {empty:,}' if empty else ''))
        if missing_lbl:
            print(f'        [!] {len(missing_lbl)} 张图没有标签文件')
        if orphan_lbl:
            print(f'        [!] {len(orphan_lbl)} 个标签文件没有对应图片')
        if bad_lines:
            print(f'        [X] {bad_lines} 行标注格式错误')

        if n_ann:
            print(f'        类别分布:')
            for cid, nm in sorted(names.items()):
                n = cls_cnt.get(cid, 0)
                if n:
                    print(f'            {str(nm):<24} id={cid:<3} {n:>7,} 框  {n/n_ann*100:>5.1f}%')
                else:
                    print(f'            {str(nm):<24} id={cid:<3} {0:>7} 框   0.0%   [!] 该划分无此类别')

            eq_sizes.sort()
            def q(p):
                return eq_sizes[min(len(eq_sizes) - 1, int(len(eq_sizes) * p))]
            s16 = sum(1 for s in eq_sizes if s < 16) / len(eq_sizes) * 100
            s32 = sum(1 for s in eq_sizes if s < 32) / len(eq_sizes) * 100
            cnts = sorted(per_img.values())
            print(f'        目标等效边长: p25 {q(.25):.0f}  中位 {q(.5):.0f}  p75 {q(.75):.0f}  '
                  f'p90 {q(.90):.0f} px    <16px {s16:.1f}%  <32px {s32:.1f}%')
            if cnts:
                print(f'        每图目标数:   p50 {cnts[len(cnts)//2]}  p90 {cnts[int(len(cnts)*.9)]}  '
                      f'max {cnts[-1]}')
            summary[(split_name, sp)] = {
                'images': len(imgs), 'anns': n_ann, 'classes': dict(cls_cnt),
                'median_px': round(q(.5), 1), 'pct_lt32': round(s32, 1),
            }

print()
print('=' * 84)
print('规模汇总')
print('=' * 84)
print(f'  {"划分":<10}{"split":<9}{"图片":>8}{"标注":>10}{"中位边长":>10}{"<32px":>8}')
for (s, sp), v in sorted(summary.items()):
    print(f'  {s:<10}{sp:<9}{v["images"]:>8,}{v["anns"]:>10,}'
          f'{v["median_px"]:>10.0f}{v["pct_lt32"]:>7.1f}%')
