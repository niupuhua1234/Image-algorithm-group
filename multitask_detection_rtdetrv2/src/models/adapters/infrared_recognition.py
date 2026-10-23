# 红外目标识别：导入 RGBT 工程的 RT-DETRv2-R18VD 四类任务。
"""Infrared recognition adapter for the imported R18VD source recipe."""
from typing import Dict, List

import numpy as np

from src.core.registry import register_task
from src.models.adapters.base import Detection, TaskMeta
from src.models.adapters.detr_task import DetrTaskAdapter


CLASS_NAMES = ['car1', 'car2', 'car3', 'car4']


@register_task('infrared_recognition', '红外目标识别（外部 RGBT RT-DETRv2）', 'implemented')
class InfraredRecognitionAdapter(DetrTaskAdapter):
    META = TaskMeta(
        'infrared_recognition', '红外目标识别', '原图 xyxy 框、任务类别、置信度',
        '外部 RGBT RT-DETRv2 R18VD', 'RT-DETRv2-R18VD；独立四类、输入尺寸和权重',
        'implemented',
        ['类别 car1/car2/car3/car4', '沿用来源工程 IoU=0.40 AP/Recall 选点规则'],
    )
    class_names = CLASS_NAMES

    def bind(self, cfg):
        names = list(cfg['classes']['names'])
        if names != CLASS_NAMES:
            raise ValueError(f'Infrared recognition class order mismatch: {names!r}')
        if tuple(cfg['model']['input_size']) != (1024, 1280):
            raise ValueError('Infrared recognition requires input_size [1024, 1280]')
        mapping = {int(k): int(v) for k, v in cfg['classes']['category_id_to_label'].items()}
        if mapping != {0: 0, 1: 1, 2: 2, 3: 3}:
            raise ValueError(f'Unexpected infrared recognition category mapping: {mapping!r}')
        return super().bind(cfg)

    def build_dataloader(self, cfg, split):
        from src.data.infrared_rtdetr import build_infrared_rtdetr_loader
        return build_infrared_rtdetr_loader(cfg, split)

    def evaluate(self, preds: List[Detection], gts: List[Dict], **kw) -> Dict:
        from src.metrics.detection import evaluate_detection
        from src.metrics.source_rtdetr import source_coco_metrics

        cfg = getattr(self, 'cfg', {})
        evaluation = cfg.get('evaluation', {})
        result = evaluate_detection(
            preds, gts, self.class_names,
            iou_thr=float(evaluation.get('primary_iou', 0.40)),
        )
        source = source_coco_metrics(
            preds, gts, self.class_names,
            max_detections=int(evaluation.get('source_max_detections', 200)),
        )
        per_class = result.get('per_class', {}).values()
        tp = int(sum(int(row['tp']) for row in per_class))
        fp = int(sum(int(row['fp']) for row in result.get('per_class', {}).values()))
        fn = int(sum(int(row['fn']) for row in result.get('per_class', {}).values()))
        result.update(source)
        result.update({
            'tp': tp, 'fp': fp, 'fn': fn, 'targets': tp + fn, 'detections': tp + fp,
            'pd': tp / (tp + fn) if tp + fn else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'fa': fp / (tp + fp) if tp + fp else None,
            'far_definition': 'FP / (TP + FP)',
            'matching': 'source COCO category-aware matching for AP/Recall selection',
        })
        return result

    def select_operating_point(self, records, cfg):
        # Source training evaluates the 300 DETR candidates; no business score
        # threshold is applied before AP40/R40/R50/R60 checkpoint selection.
        preds = [row['predictions'] for row in records]
        gts = [row['ground_truth'] for row in records]
        metrics = self.evaluate(preds, gts)
        return {
            'confidence': 0.0,
            'metrics': metrics,
            'target_met': True,
            'selected_on': 'source_rtdetr_coco_val',
            'selection_key': [
                metrics['Recall_IoU40'], metrics['Recall_IoU50'],
                metrics['Recall_IoU60'], metrics['AP40'],
            ],
            'ap_score': metrics['AP40'],
        }
