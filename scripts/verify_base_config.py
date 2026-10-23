# -*- coding: utf-8 -*-
"""验证新配置能被引擎解析, 且关键值符合预期。"""
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

cfg = YAMLConfig('configs/deim_rtdetrv2/deim_vistarget_base.yml')
y = cfg.yaml_cfg

print('=' * 76)
print('配置解析验证: deim_vistarget_base.yml')
print('=' * 76)

checks = []
def chk(name, actual, expect=None, note=''):
    ok = (expect is None) or (actual == expect)
    checks.append(ok)
    mark = 'OK ' if ok else 'X  '
    tail = f'   期望 {expect}' if (expect is not None and not ok) else ''
    print(f'  [{mark}] {name:<28} = {actual}{tail} {note}')

chk('num_classes', y.get('num_classes'), 4)
chk('epoches', y.get('epoches'), 60)
chk('flat_epoch', y.get('flat_epoch'), 30)
chk('no_aug_epoch', y.get('no_aug_epoch'), 3)
chk('eval_spatial_size', y.get('eval_spatial_size'), None)
chk('remap_mscoco_category', y.get('remap_mscoco_category'), False)

td = y.get('train_dataloader', {})
chk('train total_batch_size', td.get('total_batch_size'), 4)
chk('train num_workers', td.get('num_workers'), 8)
cf = td.get('collate_fn', {})
chk('collate.base_size', cf.get('base_size'), 1280)
chk('collate.mixup_epochs', cf.get('mixup_epochs'), [4, 30])
chk('collate.stop_epoch', cf.get('stop_epoch'), 60)

tr = td.get('dataset', {})
print(f'  [ OK ] {"train img_folder":<28} = {tr.get("img_folder")}')
print(f'  [ OK ] {"train ann_file":<28} = {tr.get("ann_file")}')
pol = tr.get('transforms', {}).get('policy', {})
chk('policy.epoch', pol.get('epoch'), [4, 30, 58])

ops = tr.get('transforms', {}).get('ops', [])
mos = next((o for o in ops if o.get('type') == 'Mosaic'), None)
rs = next((o for o in ops if o.get('type') == 'Resize'), None)
iouc = next((o for o in ops if o.get('type') == 'RandomIoUCrop'), None)
chk('Mosaic.output_size', mos.get('output_size') if mos else None, 640)
chk('Mosaic.probability', mos.get('probability') if mos else None, 0.7)
chk('Resize.size', rs.get('size') if rs else None, [1280, 1280])
chk('IoUCrop.p', iouc.get('p') if iouc else None, 0.6)

# 模型与损失(继承链)
print()
print('  --- 继承链 ---')
for k in ('PResNet', 'HybridEncoder', 'RTDETRTransformerv2', 'DEIMCriterion'):
    v = y.get(k)
    if isinstance(v, dict):
        keep = {kk: v[kk] for kk in ('depth', 'num_layers', 'hidden_dim', 'expansion',
                                     'losses', 'gamma', 'weight_dict') if kk in v}
        print(f'  {k}: {keep}')
    else:
        print(f'  {k}: {v}')

opt = y.get('optimizer', {})
print(f'  optimizer: lr={opt.get("lr")} betas={opt.get("betas")} wd={opt.get("weight_decay")}')
print(f'  lrsheduler={y.get("lrsheduler")} lr_gamma={y.get("lr_gamma")} warmup_iter={y.get("warmup_iter")}')
print(f'  ema={y.get("ema")} use_amp={y.get("use_amp")} clip_max_norm={y.get("clip_max_norm")}')

# 数据文件是否就绪
print()
print('  --- 数据就绪检查 ---')
for sp in ('train', 'val'):
    ip = Path(tr.get('img_folder', '')) if sp == 'train' else Path(
        y.get('val_dataloader', {}).get('dataset', {}).get('img_folder', ''))
    ap = Path(tr.get('ann_file', '')) if sp == 'train' else Path(
        y.get('val_dataloader', {}).get('dataset', {}).get('ann_file', ''))
    n_img = len(list(ip.glob('*.jpg'))) if ip.exists() else -1
    n_ann = 0
    if ap.exists():
        d = json.loads(ap.read_text(encoding='utf-8'))
        n_ann = len(d.get('annotations', []))
        n_img_json = len(d.get('images', []))
    else:
        n_img_json = -1
    print(f'  {sp}: 磁盘 jpg {n_img:,}  json 图片 {n_img_json:,}  json 标注 {n_ann:,}')

print()
print('=' * 76)
print(f'结论: {"全部通过" if all(checks) else f"{checks.count(False)} 项不符"}')
print('=' * 76)
