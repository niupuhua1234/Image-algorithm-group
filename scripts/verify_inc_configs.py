# -*- coding: utf-8 -*-
"""验证类增量配置能被引擎解析, 且数据路径(file_name + img_folder)能对上。"""
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
sys.path.insert(0, str(DEIM))
os.chdir(str(DEIM))
try:
    import patch_tw
    patch_tw.apply()
except Exception:
    pass
from engine.core import YAMLConfig

CASES = [
    ('inc_p1', 'configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml', 6),
    ('inc_p2', 'configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml', 8),
]

for tag, cfg_path, expect_nc in CASES:
    print('=' * 76)
    print(f'{tag}   {cfg_path}')
    print('=' * 76)
    cfg = YAMLConfig(cfg_path)
    y = cfg.yaml_cfg
    nc = y.get('num_classes')
    print(f'  num_classes      {nc}   {"OK" if nc == expect_nc else "X 期望 " + str(expect_nc)}')
    print(f'  epoches          {y.get("epoches")}')
    print(f'  lr               {y.get("optimizer", {}).get("lr")}')
    print(f'  freeze_at        {y.get("PResNet", {}).get("freeze_at")}')
    print(f'  freeze_norm      {y.get("PResNet", {}).get("freeze_norm")}')
    td = y.get('train_dataloader', {})
    ds = td.get('dataset', {})
    print(f'  batch / workers  {td.get("total_batch_size")} / {td.get("num_workers")}')
    print(f'  img_folder       {ds.get("img_folder")}')
    print(f'  ann_file         {ds.get("ann_file")}')
    pol = ds.get('transforms', {}).get('policy', {})
    print(f'  policy.epoch     {pol.get("epoch")}  (999 表示策略不触发, 等于关闭 Mosaic)')

    # 校验数据
    ip = Path(ds.get('img_folder') or '')
    ap = Path(ds.get('ann_file') or '')
    if not ap.exists():
        print(f'  [X] 标注不存在: {ap}')
        continue
    d = json.loads(ap.read_text(encoding='utf-8'))
    imgs = d['images']
    miss = 0
    for im in imgs[:200]:
        if not (ip / im['file_name']).exists():
            miss += 1
            if miss <= 2:
                print(f'  [X] 图片缺失示例: {ip / im["file_name"]}')
    cls = {}
    for a in d['annotations']:
        cls[a['category_id']] = cls.get(a['category_id'], 0) + 1
    names = {c['id']: c['name'] for c in d.get('categories', [])}
    print(f'  数据: 图片 {len(imgs):,}  标注 {len(d["annotations"]):,}  '
          f'抽样 200 张缺图 {miss}')
    print(f'  类别分布: ' + ', '.join(f'{names.get(c, c)}={n}' for c, n in sorted(cls.items())))

    # 校验 val
    vds = y.get('val_dataloader', {}).get('dataset', {})
    vap = Path(vds.get('ann_file') or '')
    print(f'  val ann_file     {vap.name}  {"存在" if vap.exists() else "X 缺失"}')
    if vap.exists():
        dv = json.loads(vap.read_text(encoding='utf-8'))
        print(f'  val 图片 {len(dv["images"]):,}  标注 {len(dv["annotations"]):,}')
    print()

print('=' * 76)
print('说明: img_folder 指向 vistarget_work 根目录, file_name 形如')
print('      base/images/train/xxx.jpg 或 inc_p1/images/train/xxx.jpg,')
print('      拼接后正好落在真实文件上。')
print('=' * 76)
