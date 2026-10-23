# -*- coding: utf-8 -*-
"""数据增强构建 —— 按配置实例化引擎的 transform 算子。

对应技术路线书 2.3 节"公共网络模块层"的思路: 增强算子属公共能力,
各任务用**配置**选择启用哪些、参数是多少, 而不是各自实现一套。

配置形态(configs/training/*.yaml):
    augmentation:
      train:
        policy: {epoch: [4, 24, 43], ops: [Mosaic, RandomPhotometricDistort, ...]}
        ops:
          - {type: Mosaic, output_size: 320, probability: 1.0, ...}
          - {type: RandomHorizontalFlip}
          - {type: Resize, size: [960, 960]}

实例化规则: 配置里的 `type` 对应 deim_engine.data.transforms 下的类名,
其余键作为关键字参数。
"""
from typing import Dict, List

# 允许的算子白名单 —— 避免配置写错类名时静默失败
_KNOWN_OPS = {
    'Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomResize', 'Resize',
    'SanitizeBoundingBoxes', 'ConvertPILImage', 'ConvertBoxes', 'RandomCrop',
    'PadToSize', 'RandomShift', 'Normalize',
}


def _import_transforms():
    try:
        from deim_engine.data import transforms as T
        return T
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f'无法导入引擎 transforms: {e}\n'
            '请确认 third_party/deim_engine 完整, 且 patch_tw 已应用。'
        )


def build_transforms(cfg: Dict, split: str) -> List:
    """按配置构建某个划分的 transform 列表。"""
    T = _import_transforms()
    spec = (cfg.get('augmentation') or {}).get(split) or {}
    ops_cfg = spec.get('ops') or []

    out = []
    for item in ops_cfg:
        if not isinstance(item, dict) or 'type' not in item:
            raise ValueError(f'增强算子配置格式错误: {item!r}')
        kwargs = {k: v for k, v in item.items() if k != 'type'}
        name = item['type']
        if name not in _KNOWN_OPS:
            raise KeyError(f'未知增强算子 {name!r}; 允许: {sorted(_KNOWN_OPS)}')
        cls = getattr(T, name, None)
        if cls is None:
            raise AttributeError(f'引擎中不存在算子 {name!r} (transforms 模块)')
        out.append(cls(**kwargs))
    return out


def build_policy(cfg: Dict, split: str) -> Dict:
    """取出该划分的增强策略(按 epoch 切换)。"""
    spec = (cfg.get('augmentation') or {}).get(split) or {}
    return spec.get('policy') or {}


def augmentation_summary(cfg: Dict, split: str = 'train') -> List[str]:
    """生成人类可读的增强说明, 写进实验记录。"""
    spec = (cfg.get('augmentation') or {}).get(split) or {}
    lines = []
    pol = spec.get('policy') or {}
    if pol:
        lines.append(f"policy.epoch = {pol.get('epoch')}   受策略控制的算子 = {pol.get('ops')}")
    for item in spec.get('ops') or []:
        name = item.get('type')
        params = {k: v for k, v in item.items() if k != 'type'}
        lines.append(f'  - {name}: {params}')
    return lines
