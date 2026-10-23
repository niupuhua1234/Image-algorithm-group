# -*- coding: utf-8 -*-
# 可见光任务保留交接版模型、损失和增强策略；可共享的 DETR 解码由父类提供。
# handover_available 表示交接实现已保留，统一入口尚未启用该任务。
"""可见光目标识别任务适配器 —— 保留交接版 DEIM 模型与训练适配。

路线书 5.2 节对该任务的定位:

    任务标识        visible_recognition
    业务内容        可见光目标识别
    输出要求        目标框、任务类别、分数
    原生对照方法    DEIM
    主线建设方式    v2 基线与 DEIM 增强训练对照; 后续开展 v4 的基础模型接入

算法族绑定(DETR 系, 不可与 YOLO/AA 组件混搭):
    模型    RT-DETRv2  (PResNet + HybridEncoder + RTDETRTransformerv2)
    匹配    匈牙利匹配 (一对一集合预测)
    损失    MAL + L1 + GIoU   (DEIM 增强)
    后处理  集合预测, **无 NMS**
"""
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.core.registry import register_task
from src.models.adapters.base import Detection, TaskAdapter, TaskMeta
from src.models.adapters.detr_task import DetrTaskAdapter

# ---------------------------------------------------------------------------
# vendored 引擎接入(技术路线书: 以官方 RT-DETR v2 PyTorch 实现为基准,
# 记录源码版本; 这里用本地 DEIM 引擎, 版本见 third_party/deim_engine/VERSION.md)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_THIRD_PARTY = _PROJECT_ROOT / 'third_party'
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

_patched = False


def _ensure_patch():
    """torchvision 0.26 兼容补丁 —— 必须在 import 引擎之前应用。"""
    global _patched
    if _patched:
        return
    try:
        import patch_tw
        if hasattr(patch_tw, 'apply'):
            patch_tw.apply()
    except Exception as e:  # pragma: no cover
        print(f'[visible_recognition] patch_tw 未应用: {e}')
    _patched = True


def _clean(d):
    """去掉配置里供人工阅读的元键(_class / _note 等), 只保留构造参数。"""
    return {k: v for k, v in (d or {}).items() if not k.startswith('_')}


# 可见光车辆识别 —— 6 类, 类别编号即此列表下标
CLASS_NAMES = ['car', 'truck', 'bus', 'van', 'pickup', 'tank']


# 该子类重写模型、损失及数据方法，保留 DEIM 配套关系，公共后处理继续继承。
@register_task(
    task_id='visible_recognition',
    description='可见光目标识别(目标框 + 任务类别 + 分数)',
    status='handover_available',
)
class VisibleRecognitionAdapter(DetrTaskAdapter):
    """可见光目标识别 —— RT-DETRv2 主干 + DEIM 增强训练。"""

    META = TaskMeta(
        task_id='visible_recognition',
        business_name='可见光目标识别',
        output_spec='目标框、任务类别、分数',
        native_baseline='DEIM',
        mainline_plan='v2 基线与 DEIM 增强训练对照; 后续开展 v4 的基础模型接入',
        status='handover_available',
        notes=[
            '类别体系独立, 不与其他任务共用',
            '集合预测后处理, 不依赖 NMS',
            'Dense O2O 与 MAL 可分别开关, 便于单因素消融',
        ],
    )
    class_names = CLASS_NAMES

    # ------------------------------------------------------------------ #
    # 模型
    # ------------------------------------------------------------------ #
    def build_model(self, cfg: Dict) -> torch.nn.Module:
        """构建 RT-DETRv2(DETR 系, 一对一集合预测)。

        配置来源: configs/models/rtdetrv2_r18_visible.yaml
        """
        self.bind(cfg)
        _ensure_patch()
        from deim_engine.backbone.presnet import PResNet
        from deim_engine.deim.deim import DEIM
        from deim_engine.deim.hybrid_encoder import HybridEncoder
        from deim_engine.deim.rtdetrv2_decoder import RTDETRTransformerv2

        m = cfg['model']
        backbone = PResNet(**_clean(m['backbone']))
        encoder = HybridEncoder(**_clean(m['encoder']))
        decoder = RTDETRTransformerv2(num_classes=self.num_classes, **_clean(m['decoder']))
        model = DEIM(backbone, encoder, decoder)

        # 推理时的固定输入尺寸假设: None 表示动态(eval 阶段再按需预生成)
        ess = m.get('eval_spatial_size')
        model.encoder.eval_spatial_size = ess
        model.decoder.eval_spatial_size = ess
        return model

    # ------------------------------------------------------------------ #
    # 损失与匹配(必须与模型同族)
    # ------------------------------------------------------------------ #
    def build_criterion(self, cfg: Dict) -> torch.nn.Module:
        """DEIM 损失: MAL + L1 + GIoU, 匈牙利匹配。

        配置里的 losses 列表决定实际启用哪些项 ——
        weight_dict 中残留的 loss_vfl 不生效(见交接说明坑 5)。
        """
        _ensure_patch()
        from deim_engine.deim.deim_criterion import DEIMCriterion
        from deim_engine.deim.matcher import HungarianMatcher

        lc = cfg['loss']
        matcher_cfg = dict(lc.get('matcher', {}))
        matcher = HungarianMatcher(**{k: v for k, v in matcher_cfg.items() if k != 'type'})
        extra = {k: v for k, v in lc.items()
                 if k not in ('weight_dict', 'losses', 'matcher') and not k.startswith('_')}
        # 类别数必须与模型一致 —— 引擎里 criterion 默认按 COCO 的 80 类算,
        # 这里由适配器统一注入, 避免配置漏填导致分类损失维度不匹配。
        extra.setdefault('num_classes', self.num_classes)
        return DEIMCriterion(matcher=matcher,
                             weight_dict=lc['weight_dict'],
                             losses=lc['losses'],
                             **extra)

    # ------------------------------------------------------------------ #
    # 数据
    # ------------------------------------------------------------------ #
    def build_dataloader(self, cfg: Dict, split: str) -> Any:
        """构建 COCO 格式数据加载器。

        注意(交接说明坑 1): train 的 JSON 只引用 37,847 张,
        磁盘目录里有 50,024 张(12,177 张旧格式残留), **必须以 JSON 引用为准**。
        """
        _ensure_patch()
        from deim_engine.data import CocoDetection, DataLoader, BatchImageCollateFunction
        from deim_engine.data.transforms import Compose
        from src.data.transforms import build_transforms

        d = cfg['data'][split]
        img_folder = str(Path(cfg['_data_root']) / d['img_folder'])
        ann_file = str(Path(cfg['_data_root']) / d['ann_file'])

        dataset = CocoDetection(img_folder=img_folder, ann_file=ann_file,
                                return_masks=False,
                                transforms=Compose(build_transforms(cfg, split)))
        collate = BatchImageCollateFunction(
            base_size=cfg['train'].get('input_size', 960),
            base_size_repeat=d.get('base_size_repeat', 1),
            stop_epoch=d.get('stop_epoch', cfg['train']['epoches']),
            mixup_prob=cfg['train'].get('mixup_prob', 0.0),
            mixup_epochs=tuple(cfg['train'].get('mixup_epochs', [0, 0])),
        ) if split == 'train' else BatchImageCollateFunction(
            base_size=cfg['train'].get('input_size', 960))

        return DataLoader(dataset, batch_size=d.get('batch_size', 8),
                          shuffle=(split == 'train'),
                          num_workers=d.get('num_workers', 4),
                          drop_last=(split == 'train'),
                          collate_fn=collate)

    # ------------------------------------------------------------------ #
    # 后处理 —— 集合预测, 无 NMS
    # ------------------------------------------------------------------ #
    # super() 调用父类共享解码；本任务的类别和不启用 NMS 的策略仍由自己的配置决定。
    def postprocess(self, raw_output, orig_size, conf=None):
        """Reuse DETR decode; this task keeps its own class map and no-NMS policy."""
        return super().postprocess(raw_output, orig_size, conf)

    # ------------------------------------------------------------------ #
    def predict_from_path(self, model, image_path, cfg: Dict, conf: float = 0.005,
                          device: str = 'cuda') -> Detection:
        """便捷方法: 从图片路径直接得到 Detection(含预处理与原图坐标还原)。"""
        from src.data.io import load_image_tensor
        x, orig_size = load_image_tensor(image_path, cfg['train'].get('input_size', 1024),
                                         device)
        raw = model(x)
        return self.postprocess(raw, orig_size, conf)
