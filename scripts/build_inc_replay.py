# -*- coding: utf-8 -*-
"""构造类增量用的混合训练集(新类 + 老类回放)。

背景:
    inc_p1/train 只含新类(4/5 类, 4,052 框), inc_p2/train 只含新类(6/7 类, 2,284 框)。
    数据里没有现成的 replay 目录, 所以老类回放必须从 base/inc_p1 里抽。

设计:
    inc_p1 阶段: 新类 1,050 图  +  从 base 抽 1,050 图(1:1)
    inc_p2 阶段: 新类   577 图  +  从 base 抽 385 + 从 inc_p1 抽 192(= 577, 1:1)

    抽样固定随机种子, 保证可复现。
    file_name 统一写成相对 vistarget_work 的路径, 这样 img_folder 用根目录即可。

输出:
    vistarget_work/inc_p1/annotations/train_replay.json
    vistarget_work/inc_p2/annotations/train_replay.json
    vistarget_work/base/annotations/rel_{train,val,test}.json   (file_name 转为相对路径)
    vistarget_work/inc_p1/annotations/rel_{val,test}.json
    vistarget_work/inc_p2/annotations/rel_{val,test}.json
"""
import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

WORK = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')
CLASSES = ['car1', 'car2', 'car3', 'car4',
           'military_fuel_tanker', 'radar_vehicle',
           'military_motorcycle', 'bridge_layer']
SEED = 20261022


def load(stage: str, split: str, tag: str = ''):
    p = WORK / stage / 'annotations' / f'{split}{tag}.json'
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding='utf-8'))


def rebase(stage: str, split: str, d: dict):
    """把 file_name 改成相对 vistarget_work 的路径(幂等)。"""
    prefix = f'{stage}/images/{split}/'
    for im in d['images']:
        fn = im['file_name']
        if not fn.startswith(prefix):
            im['file_name'] = prefix + Path(fn).name
    return d


def save(stage: str, name: str, d: dict, desc: str):
    d.setdefault('info', {})['description'] = desc
    out = WORK / stage / 'annotations' / f'{name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')
    n_img = len(d['images'])
    n_ann = len(d['annotations'])
    cls = {}
    for a in d['annotations']:
        cls[a['category_id']] = cls.get(a['category_id'], 0) + 1
    cls_str = ', '.join(f'{CLASSES[c]}={n}' for c, n in sorted(cls.items()))
    print(f'  {stage}/{name}.json   图片 {n_img:>5,}  标注 {n_ann:>6,}   {cls_str}')
    return out


def merge(sources, out_stage, out_name, desc):
    """把多个 COCO 合并成一个(重映射 image/annotation id)。"""
    images, anns = [], []
    img_id = 1
    ann_id = 1
    for d in sources:
        id_map = {}
        for im in d['images']:
            new_im = dict(im)
            old_id = im['id']
            new_im['id'] = img_id
            id_map[old_id] = img_id
            images.append(new_im)
            img_id += 1
        for a in d['annotations']:
            if a['image_id'] not in id_map:
                continue
            na = dict(a)
            na['id'] = ann_id
            na['image_id'] = id_map[a['image_id']]
            anns.append(na)
            ann_id += 1
    cats = [{'id': i, 'name': n, 'supercategory': 'vehicle'} for i, n in enumerate(CLASSES)]
    merged = {'images': images, 'annotations': anns, 'categories': cats}
    return save(out_stage, out_name, merged, desc)


def sample(d: dict, n: int, rng: random.Random):
    """从 COCO 里随机抽 n 张图(连同其标注)。"""
    imgs = list(d['images'])
    if n >= len(imgs):
        return d
    keep = set(rng.sample([im['id'] for im in imgs], n))
    out_imgs = [im for im in imgs if im['id'] in keep]
    out_anns = [a for a in d['annotations'] if a['image_id'] in keep]
    return {'images': out_imgs, 'annotations': out_anns, 'categories': d['categories']}


def main():
    rng = random.Random(SEED)
    print('=' * 78)
    print('构造类增量回放数据')
    print('=' * 78)

    # ---------- 1) 所有划分的 file_name 转相对路径 ----------
    print('\n[1/3] file_name 转相对路径')
    base = {}
    for sp in ('train', 'val', 'test'):
        d = rebase('base', sp, load('base', sp))
        base[sp] = d
        save('base', f'rel_{sp}', d, f'base {sp} (相对路径)')
    inc1 = {}
    for sp in ('train', 'val', 'test'):
        d = rebase('inc_p1', sp, load('inc_p1', sp))
        inc1[sp] = d
        if sp != 'train':
            save('inc_p1', f'rel_{sp}', d, f'inc_p1 {sp} (相对路径)')
    inc2 = {}
    for sp in ('train', 'val', 'test'):
        d = rebase('inc_p2', sp, load('inc_p2', sp))
        inc2[sp] = d
        if sp != 'train':
            save('inc_p2', f'rel_{sp}', d, f'inc_p2 {sp} (相对路径)')

    # ---------- 2) inc_p1: 新类 + base 等量回放 ----------
    print('\n[2/3] inc_p1 混合(新类 1,050 + base 回放 1,050)')
    n_new1 = len(inc1['train']['images'])
    base_r1 = sample(base['train'], n_new1, rng)
    print(f'  新类 {n_new1:,} 图   回放 base {len(base_r1["images"]):,} 图')
    merge([inc1['train'], base_r1], 'inc_p1', 'train_replay',
          'inc_p1 新类 + base 回放 (1:1)')

    # ---------- 3) inc_p2: 新类 + base/inc_p1 回放 ----------
    print('\n[3/3] inc_p2 混合(新类 577 + base/inc_p1 回放 577)')
    n_new2 = len(inc2['train']['images'])
    # 【修正】原来按类别数比例 4:2 分配(base 385 / inc_p1 192),
    # 实测导致 3B 对 3A 学过的 2 类置信度崩塌(conf 0.34->0.20, 误检 1.38->3.70)。
    # 原因是 3A 的类目标中位仅 24~26px(96% 小于 32px), 本身更难保;
    # 且分到的回放只有 192 图, 在 3B 训练里占比过低。
    # 改为「按类均衡」: 让 inc_p1 的 2 类拿到与 base 每类相当的样本量 =>
    #   base 4 类 : inc_p1 2 类 = 1 : 1 (各 288 图), 每类 72 图
    # 相比原来 inc_p1 每类 96 图, 这里反而略降, 但两阶段样本量对等后,
    # 小目标的老类不会因为"总样本少"而被稀释。
    n_base2 = n_new2 // 2          # 288
    n_inc1_2 = n_new2 - n_base2    # 289
    base_r2 = sample(base['train'], n_base2, rng)
    inc1_r2 = sample(inc1['train'], n_inc1_2, rng)
    print(f'  新类 {n_new2:,} 图   回放 base {len(base_r2["images"]):,} 图  '
          f'+ inc_p1 {len(inc1_r2["images"]):,} 图  (base:inc_p1 = 1:1)')
    merge([inc2['train'], base_r2, inc1_r2], 'inc_p2', 'train_replay',
          'inc_p2 新类 + base/inc_p1 回放 (base:inc_p1 = 1:1)')

    # ---------- 汇总 ----------
    print('\n' + '=' * 78)
    print('完成')
    print('=' * 78)
    print(f'  inc_p1 训练: inc_p1/annotations/train_replay.json')
    print(f'  inc_p2 训练: inc_p2/annotations/train_replay.json')
    print(f'  回归验证用: base/annotations/rel_test.json')
    print(f'  各阶段测试: inc_p1/annotations/rel_test.json, inc_p2/annotations/rel_test.json')
    print(f'\n  注意 img_folder 统一设为: {WORK}')


if __name__ == '__main__':
    main()
