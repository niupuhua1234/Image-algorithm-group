# -*- coding: utf-8 -*-
"""transfer 数据预处理(PNG->JPEG + COCO), 与 base 同一套流程。

只处理 transfer 一个 stage; 进程数降到 4, 避免与正在跑的 base 训练抢 CPU。
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

SRC = Path(r'D:\BaiduNetdiskDownload\datasets\dataset\transfer')
DST = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work\transfer')
CLASSES = ['car1', 'car2', 'car3', 'car4']
W, H = 1920, 1080
QUALITY = 95
NPROC = 4
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
    print('transfer 预处理')
    print('=' * 76)

    # 转码
    tasks = []
    for sp in SPLITS:
        idir = SRC / sp / 'images'
        if not idir.exists():
            continue
        for p in idir.iterdir():
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                tasks.append((p, DST / 'images' / sp / (p.stem + '.jpg')))
    print(f'  待转码 {len(tasks):,} 张 ({NPROC} 进程)')
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
    print(f'  完成 {ok_n:,}/{len(tasks):,}  耗时 {(time.time()-t0)/60:.1f} 分')
    if errs:
        print(f'  [!] {len(errs)} 张失败: {errs[:3]}')

    # COCO
    print()
    cat_list = [{'id': i, 'name': n, 'supercategory': 'vehicle'} for i, n in enumerate(CLASSES)]
    for sp in SPLITS:
        idir = SRC / sp / 'images'
        ldir = SRC / sp / 'labels'
        jdir = DST / 'images' / sp
        if not idir.exists():
            continue
        images, anns = [], []
        ann_id = 1
        img_id = 1
        n_empty = 0
        n_bad = 0
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            if not (jdir / (p.stem + '.jpg')).exists():
                continue
            images.append({'id': img_id, 'file_name': p.stem + '.jpg', 'width': W, 'height': H})
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
            img_id += 1
        coco = {'images': images, 'annotations': anns, 'categories': cat_list,
                'info': {'description': f'vistarget transfer {sp}', 'converted_from': 'YOLO txt'}}
        out = DST / 'annotations' / f'{sp}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coco, ensure_ascii=False), encoding='utf-8')
        print(f'  {sp:<6} 图片 {len(images):>4,}  标注 {len(anns):>5,}  '
              f'无标注图 {n_empty:>3}  非法行 {n_bad:>2}')

    tot = sum(p.stat().st_size for p in DST.rglob('*.jpg'))
    print(f'\n  JPEG 合计 {tot/1e9:.2f} GB   总耗时 {(time.time()-t_all)/60:.1f} 分')
    print(f'  输出 {DST}')


if __name__ == '__main__':
    main()
