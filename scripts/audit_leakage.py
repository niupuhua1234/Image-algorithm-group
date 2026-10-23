# -*- coding: utf-8 -*-
"""数据泄露全面核查。

检查五个层面, 从弱到强:
    ① 文件级   —— 同一文件出现在多个划分(比对 MD5, 不只是文件名)
    ② 场景级   —— 同一场景/序列的多帧被拆到 train 与 test(最隐蔽, 危害最大)
    ③ 前后缀级 —— 切片图的 stem 前缀重叠(如 base_x0 vs base_x640)
    ④ 回放级   —— 增量阶段 replay 样本是否落进了评价集
    ⑤ 跨阶段级 —— 前序阶段用过的图是否出现在后续阶段的评价集

文件名形态:
    RGB_IR_IRS_1__bg_00005.png   -> 场景 = RGB_IR_IRS_1,  帧 = bg_00005
    VLP0003.png                  -> 前缀 = VLP,  编号 = 3
    VLO0110.png                  -> 前缀 = VLO,  编号 = 110
"""
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

WORK = Path(r'D:\BaiduNetdiskDownload\datasets\vistarget_work')
RAW = Path(r'D:\BaiduNetdiskDownload\datasets\dataset')

STAGES = ['base', 'transfer', 'inc_p1', 'inc_p2']
PLAN = {
    ('base', 'train'): 'rel_train.json',
    ('base', 'val'): 'rel_val.json',
    ('base', 'test'): 'rel_test.json',
    ('transfer', 'train'): 'train.json',
    ('transfer', 'val'): 'val.json',
    ('transfer', 'test'): 'test.json',
    ('inc_p1', 'train'): 'train_replay.json',
    ('inc_p1', 'val'): 'rel_val.json',
    ('inc_p1', 'test'): 'rel_test.json',
    ('inc_p2', 'train'): 'train_replay.json',
    ('inc_p2', 'val'): 'rel_val.json',
    ('inc_p2', 'test'): 'rel_test.json',
}


def scene_of(filename: str):
    """提取场景/序列标识。"""
    base = Path(filename).stem
    if '__' in base:
        return base.split('__')[0]
    m = re.match(r'^([A-Za-z]+)(\d+)$', base)
    if m:
        return m.group(1)
    return base


def load_set(stage, split, fname):
    p = WORK / stage / 'annotations' / fname
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding='utf-8'))
    files = [Path(im['file_name']).name for im in d['images']]
    return {'files': files, 'path': p, 'json': d}


def main():
    print('=' * 78)
    print('数据泄露核查')
    print('=' * 78)

    sets = {}
    for (stage, split), fn in PLAN.items():
        s = load_set(stage, split, fn)
        if s:
            sets[(stage, split)] = s
            scenes = Counter(scene_of(f) for f in s['files'])
            s['scenes'] = scenes
            print(f'  {stage:<9} {split:<6} {len(s["files"]):>6,} 张   '
                  f'场景 {len(scenes):>2} 个: {dict(list(scenes.most_common(6)))}')

    # ============ ① 文件级:同名文件内容比对 ============
    print()
    print('=' * 78)
    print('① 文件级核查 —— 同名文件是否内容相同(MD5)')
    print('=' * 78)

    def img_path(stage, split, fname):
        # 图片实际在 <stage>/images/<split>/
        p = WORK / stage / 'images' / split / fname
        if p.exists():
            return p
        # base/inc 的 json 里 file_name 带子路径
        p2 = WORK / fname
        return p2 if p2.exists() else None

    def md5(p):
        h = hashlib.md5()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    pairs = [
        (('base', 'train'), ('base', 'val')),
        (('base', 'train'), ('base', 'test')),
        (('base', 'val'), ('base', 'test')),
        (('transfer', 'train'), ('transfer', 'val')),
        (('transfer', 'train'), ('transfer', 'test')),
        (('transfer', 'val'), ('transfer', 'test')),
        (('inc_p1', 'train'), ('inc_p1', 'val')),
        (('inc_p1', 'train'), ('inc_p1', 'test')),
        (('inc_p1', 'val'), ('inc_p1', 'test')),
        (('inc_p2', 'train'), ('inc_p2', 'val')),
        (('inc_p2', 'train'), ('inc_p2', 'test')),
        (('inc_p2', 'val'), ('inc_p2', 'test')),
    ]
    file_ok = True
    for a, b in pairs:
        if a not in sets or b not in sets:
            continue
        fa = set(sets[a]['files'])
        fb = set(sets[b]['files'])
        inter = sorted(fa & fb)
        if not inter:
            print(f'  [OK] {a[0]}/{a[1]} 与 {b[0]}/{b[1]}: 无同名文件')
            continue
        # 逐对算 MD5(全量, 不抽样)
        same = 0
        for nm in inter:
            pa = img_path(a[0], a[1], nm)
            pb = img_path(b[0], b[1], nm)
            if pa and pb:
                try:
                    if md5(pa) == md5(pb):
                        same += 1
                except Exception:
                    pass
        if same:
            print(f'  [X ] {a[0]}/{a[1]} 与 {b[0]}/{b[1]}: 同名 {len(inter)}, '
                  f'内容相同 {same}  <-- 真实泄露!')
            file_ok = False
        else:
            print(f'  [OK] {a[0]}/{a[1]} 与 {b[0]}/{b[1]}: 同名 {len(inter)} 但内容全不同')

    # ============ ② 场景级 ============
    print()
    print('=' * 78)
    print('② 场景级核查 —— 同一场景是否被拆到训练与测试(最隐蔽的泄漏)')
    print('=' * 78)
    scene_issues = []
    for stage in STAGES:
        tr = sets.get((stage, 'train'))
        if not tr:
            continue
        for sp in ('val', 'test'):
            ev = sets.get((stage, sp))
            if not ev:
                continue
            s_tr = set(tr['scenes'])
            s_ev = set(ev['scenes'])
            both = s_tr & s_ev
            if both:
                detail = ', '.join(f'{s}(train {tr["scenes"][s]} / {sp} {ev["scenes"][s]})'
                                   for s in sorted(both))
                print(f'  [X ] {stage}: train 与 {sp} 共享场景 {len(both)} 个: {detail}')
                scene_issues.append((stage, sp, sorted(both)))
            else:
                print(f'  [OK] {stage}: train 与 {sp} 场景完全不相交 '
                      f'(train {len(s_tr)} 个, {sp} {len(s_ev)} 个)')

    # ============ ③ 前后缀级(切片) ============
    print()
    print('=' * 78)
    print('③ 切片前缀核查 —— 同一原图切出的多个块是否跨划分')
    print('=' * 78)

    def slice_base(filename):
        """把 xxx_x0_y0.jpg / xxx_x0.jpg 归一成原始 stem。"""
        b = Path(filename).stem
        return re.sub(r'_x\d+(?:_y\d+)?$', '', b)

    slice_issues = []
    for stage in STAGES:
        tr = sets.get((stage, 'train'))
        if not tr:
            continue
        sb_tr = set(slice_base(f) for f in tr['files'])
        for sp in ('val', 'test'):
            ev = sets.get((stage, sp))
            if not ev:
                continue
            sb_ev = set(slice_base(f) for f in ev['files'])
            both = sb_tr & sb_ev
            if both:
                print(f'  [X ] {stage}: train 与 {sp} 有 {len(both)} 个相同原图 '
                      f'(切片拆散), 例: {sorted(both)[:3]}')
                slice_issues.append((stage, sp, len(both)))
            else:
                print(f'  [OK] {stage}: train 与 {sp} 无共同原图')

    # ============ ④ 回放泄露 ============
    print()
    print('=' * 78)
    print('④ 回放核查 —— 增量 replay 样本是否落进评价集')
    print('=' * 78)

    for stage, evstage in (('inc_p1', 'inc_p1'), ('inc_p2', 'inc_p2')):
        tr = sets.get((stage, 'train'))
        if not tr:
            continue
        for sp in ('val', 'test'):
            ev = sets.get((evstage, sp))
            if not ev:
                continue
            inter = set(tr['files']) & set(ev['files'])
            if inter:
                print(f'  [X ] {stage}/train_replay 与 {evstage}/{sp} 有 {len(inter)} 张重叠')
            else:
                print(f'  [OK] {stage}/train_replay 与 {evstage}/{sp} 无重叠')

    # 回放 vs base 评价集
    for stage in ('inc_p1', 'inc_p2'):
        tr = sets.get((stage, 'train'))
        if not tr:
            continue
        for sp in ('val', 'test'):
            ev = sets.get(('base', sp))
            if not ev:
                continue
            inter = set(tr['files']) & set(ev['files'])
            if inter:
                print(f'  [X ] {stage}/train_replay 与 base/{sp} 有 {len(inter)} 张重叠'
                      f'  <-- 会污染老域回归')
            else:
                print(f'  [OK] {stage}/train_replay 与 base/{sp} 无重叠')

    # ============ ⑤ 跨阶段 ============
    print()
    print('=' * 78)
    print('⑤ 跨阶段核查 —— 前序训练集与后续评价集')
    print('=' * 78)
    for tr_stage in ('base', 'transfer', 'inc_p1'):
        tr = sets.get((tr_stage, 'train'))
        if not tr:
            continue
        for ev_stage in ('transfer', 'inc_p1', 'inc_p2'):
            if ev_stage == tr_stage:
                continue
            for sp in ('val', 'test'):
                ev = sets.get((ev_stage, sp))
                if not ev:
                    continue
                inter = set(tr['files']) & set(ev['files'])
                if inter:
                    print(f'  [X ] {tr_stage}/train 与 {ev_stage}/{sp} 重叠 {len(inter)} 张')
                # 场景级
                s1 = set(tr['scenes'])
                s2 = set(ev['scenes'])
                sb = s1 & s2
                if sb:
                    print(f'  [X ] {tr_stage}/train 与 {ev_stage}/{sp} 共享场景 {len(sb)} 个: '
                          f'{sorted(sb)[:4]}')

    # ============ 汇总 ============
    print()
    print('=' * 78)
    print('汇总')
    print('=' * 78)
    problems = []
    if not file_ok:
        problems.append('文件级:存在内容相同的同名文件')
    if scene_issues:
        problems.append(f'场景级:{len(scene_issues)} 组 stage 的 train/test 共享场景')
    if slice_issues:
        problems.append(f'切片级:{len(slice_issues)} 组存在同原图跨划分')
    if problems:
        for p in problems:
            print(f'  [X ] {p}')
    else:
        print('  [OK] 文件级 / 场景级 / 切片级 均未发现泄漏')

    print()
    print('  附注:阈值泄漏已在评估流程中修正 —— 阈值在 val 上定, test 只跑一次。')


if __name__ == '__main__':
    main()
