# -*- coding: utf-8 -*-
"""base 数据集预处理:
    ① PNG -> JPEG(q95) 转码, 多进程
    ② YOLO txt -> COCO json

输出到新目录, 不动原始数据:
    D:\\BaiduNetdiskDownload\\datasets\\vistarget_work\\base\\
        images/{train,val,test}/*.jpg
        annotations/{train,val,test}.json
"""
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

SRC = Path(r'D:\BaiduNetdiskDownload\datasets\dataset\base')
DST = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work\base')
CLASSES = ['car1', 'car2', 'car3', 'car4']
W, H = 1920, 1080
QUALITY = 95
NPROC = 8
SPLITS = ('train', 'val', 'test')


def convert_one(args):
    """单张 PNG -> JPEG。返回 (rel_path, ok, err)。"""
    src, dst = args
    try:
        from PIL import Image
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            return (str(dst), True, 'skip')
        with Image.open(src) as im:
            im = im.convert('RGB')
            im.save(dst, 'JPEG', quality=QUALITY, optimize=True)
        return (str(dst), True, None)
    except Exception as e:
        return (str(dst), False, str(e)[:80])


def main():
    t_all = time.time()
    print('=' * 76)
    print('base 数据集预处理')
    print('=' * 76)
    print(f'  源  {SRC}')
    print(f'  目标 {DST}')

    # ---------- ① 转码 ----------
    print('\n[1/2] PNG -> JPEG 转码')
    tasks = []
    for sp in SPLITS:
        idir = SRC / sp / 'images'
        if not idir.exists():
            print(f'  [!] 缺少 {idir}')
            continue
        odir = DST / 'images' / sp
        for p in idir.iterdir():
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                tasks.append((p, odir / (p.stem + '.jpg')))
    print(f'  待转码 {len(tasks):,} 张, {NPROC} 进程')

    t0 = time.time()
    done = ok_n = 0
    errs = []
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        futs = [ex.submit(convert_one, t) for t in tasks]
        for f in as_completed(futs):
            _, ok, err = f.result()
            done += 1
            if ok:
                ok_n += 1
            elif err and err != 'skip':
                errs.append(err)
            if done % 500 == 0:
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f'    {done:,}/{len(tasks):,}  已用 {el/60:.1f} 分  '
                      f'预计还需 {eta/60:.1f} 分')
    el = time.time() - t0
    print(f'  完成: {ok_n:,}/{len(tasks):,}  耗时 {el/60:.1f} 分 '
          f'({len(tasks)/max(el,1e-9):.1f} 张/秒)')
    if errs:
        print(f'  [!] {len(errs)} 张失败, 前 3 个: {errs[:3]}')

    # ---------- ② 生成 COCO ----------
    print('\n[2/2] YOLO txt -> COCO json')
    cat_list = [{'id': i, 'name': n, 'supercategory': 'vehicle'} for i, n in enumerate(CLASSES)]
    for sp in SPLITS:
        idir = SRC / sp / 'images'
        ldir = SRC / sp / 'labels'
        jdir = DST / 'images' / sp
        if not idir.exists():
            continue

        images, anns = [], []
        ann_id = 1
        n_empty = 0
        n_bad_line = 0
        n_missing_lbl = 0
        img_id = 1
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            jpg = jdir / (p.stem + '.jpg')
            if not jpg.exists():
                continue
            lp = ldir / (p.stem + '.txt')
            images.append({'id': img_id, 'file_name': p.stem + '.jpg',
                           'width': W, 'height': H})
            if not lp.exists():
                n_missing_lbl += 1
                img_id += 1
                continue
            lines = [l.strip() for l in lp.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
            if not lines:
                n_empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    n_bad_line += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    n_bad_line += 1
                    continue
                if cid < 0 or cid >= len(CLASSES):
                    n_bad_line += 1
                    continue
                # 转成像素 xywh, 并裁剪到图内
                bw, bh = w * W, h * H
                bx = cx * W - bw / 2
                by = cy * H - bh / 2
                bx = max(0.0, min(bx, W - 1))
                by = max(0.0, min(by, H - 1))
                bw = max(1.0, min(bw, W - bx))
                bh = max(1.0, min(bh, H - by))
                anns.append({'id': ann_id, 'image_id': img_id, 'category_id': cid,
                             'bbox': [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                             'area': round(bw * bh, 2), 'iscrowd': 0})
                ann_id += 1
            img_id += 1

        coco = {'images': images, 'annotations': anns, 'categories': cat_list,
                'info': {'description': f'vistarget base {sp}',
                         'source': str(SRC), 'converted_from': 'YOLO txt'}}
        out = DST / 'annotations' / f'{sp}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coco, ensure_ascii=False), encoding='utf-8')
        print(f'  {sp:<6} 图片 {len(images):>6,}  标注 {len(anns):>7,}  '
              f'无标注图 {n_empty:>4,}  缺标签 {n_missing_lbl:>3}  非法行 {n_bad_line:>3}  '
              f'-> {out.name} ({out.stat().st_size/1e6:.1f} MB)')

    # ---------- 汇总 ----------
    print('\n' + '=' * 76)
    print('完成')
    print('=' * 76)
    tot_src = sum(p.stat().st_size for sp in SPLITS
                  for p in (SRC / sp / 'images').glob('*.png') if p.exists())
    tot_dst = sum(p.stat().st_size for sp in SPLITS
                  for p in (DST / 'images' / sp).glob('*.jpg') if p.exists())
    print(f'  原始 PNG {tot_src/1e9:.2f} GB  ->  JPEG {tot_dst/1e9:.2f} GB '
          f'(省 {(1-tot_dst/max(tot_src,1))*100:.0f}%)')
    print(f'  总耗时 {(time.time()-t_all)/60:.1f} 分')
    print(f'  输出 {DST}')


if __name__ == '__main__':
    main()
