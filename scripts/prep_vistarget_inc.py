# -*- coding: utf-8 -*-
"""inc_p1 / inc_p2 数据预处理: PNG->JPEG + COCO(全局类别 id)。

增量阶段的图片与 base 不同目录, 为了让引擎能用一个 img_folder 加载,
这里把 file_name 写成相对 vistarget_work 的路径, 例如:
    inc_p1/images/train/VLO0002.jpg
然后 img_folder 设为 vistarget_work 根目录即可。

用法:
    python prep_vistarget_inc.py            # 处理 inc_p1 + inc_p2
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

SRC_ROOT = Path(r'D:\BaiduNetdiskDownload\datasets\dataset')
DST_ROOT = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')
CLASSES = ['car1', 'car2', 'car3', 'car4',
           'military_fuel_tanker', 'radar_vehicle',
           'military_motorcycle', 'bridge_layer']
W, H = 1920, 1080
QUALITY = 95
NPROC = 4
STAGES = ('inc_p1', 'inc_p2')
SPLITS = ('train', 'val', 'test')


def convert_one(args):
    src, dst = args
    try:
        from PIL import Image
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            return (str(dst), True, 'skip')
        with Image.open(src) as im:
            im.convert('RGB').save(dst, 'JPEG', quality=QUALITY, optimize=True)
        return (str(dst), True, None)
    except Exception as e:
        return (str(dst), False, str(e)[:80])


def main():
    t_all = time.time()
    print('=' * 76)
    print('inc_p1 / inc_p2 预处理')
    print('=' * 76)

    # ---- 转码 ----
    for stage in STAGES:
        src = SRC_ROOT / stage
        tasks = []
        for sp in SPLITS:
            idir = src / sp / 'images'
            if not idir.exists():
                continue
            for p in idir.iterdir():
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    tasks.append((p, DST_ROOT / stage / 'images' / sp / (p.stem + '.jpg')))
        if not tasks:
            print(f'  [{stage}] 没有图片, 跳过')
            continue
        print(f'\n  [{stage}] 转码 {len(tasks):,} 张 ({NPROC} 进程)')
        t0 = time.time()
        ok_n = 0
        errs = []
        with ProcessPoolExecutor(max_workers=NPROC) as ex:
            futs = [ex.submit(convert_one, t) for t in tasks]
            for f in as_completed(futs):
                _, ok, err = f.result()
                if ok:
                    ok_n += 1
                elif err and err != 'skip':
                    errs.append(err)
        print(f'    完成 {ok_n:,}/{len(tasks):,}  耗时 {(time.time()-t0)/60:.1f} 分')
        if errs:
            print(f'    [!] {len(errs)} 张失败: {errs[:2]}')

    # ---- COCO ----
    print()
    cat_list = [{'id': i, 'name': n, 'supercategory': 'vehicle'} for i, n in enumerate(CLASSES)]
    for stage in STAGES:
        src = SRC_ROOT / stage
        for sp in SPLITS:
            idir = src / sp / 'images'
            ldir = src / sp / 'labels'
            jdir = DST_ROOT / stage / 'images' / sp
            if not idir.exists():
                continue
            images, anns = [], []
            ann_id = 1
            img_id = 1
            n_empty = 0
            n_bad = 0
            cls_cnt = {}
            for p in sorted(idir.iterdir()):
                if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                if not (jdir / (p.stem + '.jpg')).exists():
                    continue
                # file_name 相对 vistarget_work, 便于统一 img_folder
                rel = f'{stage}/images/{sp}/{p.stem}.jpg'
                images.append({'id': img_id, 'file_name': rel, 'width': W, 'height': H})
                lp = ldir / (p.stem + '.txt')
                if lp.exists():
                    lines = [l.strip() for l in lp.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
                    if not lines:
                        n_empty += 1
                    for ln in lines:
                        parts = ln.split()
                        if len(parts) < 5:
                            n_bad += 1
                            continue
                        try:
                            cid = int(float(parts[0]))
                            cx, cy, w, h = (float(v) for v in parts[1:5])
                        except ValueError:
                            n_bad += 1
                            continue
                        if cid < 0 or cid >= len(CLASSES):
                            n_bad += 1
                            continue
                        bw, bh = w * W, h * H
                        bx = max(0.0, min(cx * W - bw / 2, W - 1))
                        by = max(0.0, min(cy * H - bh / 2, H - 1))
                        bw = max(1.0, min(bw, W - bx))
                        bh = max(1.0, min(bh, H - by))
                        anns.append({'id': ann_id, 'image_id': img_id, 'category_id': cid,
                                     'bbox': [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                                     'area': round(bw * bh, 2), 'iscrowd': 0})
                        ann_id += 1
                        cls_cnt[cid] = cls_cnt.get(cid, 0) + 1
                img_id += 1
            coco = {'images': images, 'annotations': anns, 'categories': cat_list,
                    'info': {'description': f'vistarget {stage} {sp}',
                             'note': 'file_name 相对 vistarget_work 根目录'}}
            out = DST_ROOT / stage / 'annotations' / f'{sp}.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(coco, ensure_ascii=False), encoding='utf-8')
            cls_str = ', '.join(f'{CLASSES[c]}={n}' for c, n in sorted(cls_cnt.items()))
            print(f'  {stage}/{sp:<5} 图片 {len(images):>5,}  标注 {len(anns):>6,}  '
                  f'无标注 {n_empty:>3}  非法 {n_bad:>2}   类别: {cls_str or "无"}')

    print(f'\n  总耗时 {(time.time()-t_all)/60:.1f} 分')
    print(f'  输出 {DST_ROOT}')


if __name__ == '__main__':
    main()
