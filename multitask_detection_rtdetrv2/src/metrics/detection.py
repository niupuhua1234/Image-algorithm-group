# -*- coding: utf-8 -*-
"""公共检测指标 —— 框级识别任务(可见光 / 红外)共用。

技术路线书第 6 节对框级识别任务的要求:

    红外/可见光框级识别: 检测 AP、逐类别精度与召回、匹配目标上的类别混淆情况、
                         完整业务时延。

本模块提供前三项(时延由 src/deployment/runtime.py 负责)。
不依赖 pycocotools, 便于跨环境部署。

三个关键设计:
    1. 匹配按分数降序贪心, IoU >= 阈值即算命中(**不看类别**)
    2. 命中后**类别一致**记 TP, **类别不一致**记混淆(既不算 TP 也不算纯 FP)
       —— 这样才能统计出"匹配目标上的类别混淆情况"
    3. 每个 GT 只能被匹配一次, 每个预测框也只能匹配一次
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[Na,4] x [Nb,4] xyxy -> [Na,Nb] IoU。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    a = a[:, None, :].astype(np.float64)
    b = b[None, :, :].astype(np.float64)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-9)


class DetectionAccumulator:
    """按类别累积 TP/FP/FN 与混淆对, 支持计算 AP。"""

    def __init__(self, class_names: Sequence[str], iou_thr: float = 0.5):
        self.class_names = list(class_names)
        self.n_cls = len(class_names)
        self.iou_thr = iou_thr
        # 每类的 (score, is_tp) 列表, 用于画 PR 曲线
        self.records: Dict[int, List[tuple]] = defaultdict(list)
        self.n_gt: Dict[int, int] = defaultdict(int)
        # 混淆: (预测类, 真值类) -> 次数
        self.confusion: Dict[tuple, int] = defaultdict(int)
        # 匹配上但类别错误的框数
        self.n_misclassified = 0
        self.n_matched = 0

    def add_image(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """累积单张图的统计。所有输入均为 numpy。"""
        pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        pred_labels = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)

        for g in gt_labels:
            self.n_gt[int(g)] += 1

        if pred_boxes.shape[0] == 0:
            return
        if gt_boxes.shape[0] == 0:
            for s, l in zip(pred_scores, pred_labels):
                self.records[int(l)].append((float(s), 0))
            return

        order = np.argsort(-pred_scores)          # 分数降序
        iou = box_iou_matrix(pred_boxes[order], gt_boxes)
        gt_taken = np.zeros(gt_boxes.shape[0], dtype=bool)

        for row, pi in enumerate(order):
            best_gi, best_iou = -1, 0.0
            for gi in range(gt_boxes.shape[0]):
                if gt_taken[gi]:
                    continue
                if iou[row, gi] > best_iou:
                    best_iou, best_gi = iou[row, gi], gi
            pl = int(pred_labels[pi])
            s = float(pred_scores[pi])
            if best_gi >= 0 and best_iou >= self.iou_thr:
                gt_taken[best_gi] = True
                gl = int(gt_labels[best_gi])
                self.n_matched += 1
                if pl == gl:
                    self.records[gl].append((s, 1))       # TP 记在真值类下
                else:
                    # 匹配上了但类别错 —— 既不算该类 TP, 也不算 FP
                    self.records[pl].append((s, 0))
                    self.confusion[(pl, gl)] += 1
                    self.n_misclassified += 1
            else:
                self.records[pl].append((s, 0))           # FP

    # ------------------------------------------------------------------ #
    def average_precision(self, cls_id: int) -> Optional[float]:
        """全点插值 AP(与 COCO 的面积法略有差异, 但趋势一致)。"""
        rec = sorted(self.records.get(cls_id, []), key=lambda x: -x[0])
        n_gt = self.n_gt.get(cls_id, 0)
        if n_gt == 0 or not rec:
            return None
        tp = np.cumsum([r[1] for r in rec])
        fp = np.cumsum([1 - r[1] for r in rec])
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        # 单调化
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        idx = np.where(np.diff(np.concatenate([[0.0], recall])) > 0)[0]
        if len(idx) == 0:
            return 0.0
        return float(np.sum(precision[idx] * np.diff(np.concatenate([[0.0], recall]))[idx]))

    def summary(self) -> Dict:
        per_class = {}
        aps = []
        for i, name in enumerate(self.class_names):
            n_gt = self.n_gt.get(i, 0)
            rec = self.records.get(i, [])
            tp = sum(r[1] for r in rec)
            fp = len(rec) - tp
            fn = n_gt - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / n_gt if n_gt > 0 else 0.0
            ap = self.average_precision(i)
            if ap is not None:
                aps.append(ap)
            per_class[name] = {
                'num_gt': int(n_gt), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
                'precision': round(p, 4), 'recall': round(r, 4),
                'f1': round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0,
                'ap': round(ap, 4) if ap is not None else None,
            }

        conf_rows = []
        for (pl, gl), cnt in sorted(self.confusion.items(), key=lambda x: -x[1]):
            conf_rows.append({
                'predicted': self.class_names[pl] if pl < self.n_cls else str(pl),
                'actual': self.class_names[gl] if gl < self.n_cls else str(gl),
                'count': int(cnt),
            })

        return {
            'iou_thr': self.iou_thr,
            'num_gt': int(sum(self.n_gt.values())),
            'num_matched': int(self.n_matched),
            'num_misclassified': int(self.n_misclassified),
            'mAP': round(float(np.mean(aps)), 4) if aps else 0.0,
            'per_class': per_class,
            'confusion': conf_rows,
        }


def evaluate_detection(preds: List, gts: List[Dict], class_names: Sequence[str],
                       iou_thr: float = 0.5, iou_thrs: Sequence[float] = None) -> Dict:
    """公共入口: 计算 AP / 逐类别 P-R / 类别混淆矩阵。

    Args:
        preds:       List[Detection]  (src.models.adapters.base.Detection)
        gts:         List[Dict], 每项 {'boxes': [N,4] xyxy, 'labels': [N]}
        class_names: 本任务的类别名(各任务独立)
        iou_thr:     主阈值(默认 0.5, 用于 P/R 与混淆矩阵)
        iou_thrs:    若给出, 额外汇总多阈值 mAP

    Returns:
        {'mAP':..., 'per_class':..., 'confusion':..., 'mAP_50_95':...}
    """
    acc = DetectionAccumulator(class_names, iou_thr)
    assert len(preds) == len(gts), f'preds({len(preds)}) 与 gts({len(gts)}) 数量不一致'
    for p, g in zip(preds, gts):
        acc.add_image(p.boxes, p.scores, p.labels,
                      g.get('boxes', np.zeros((0, 4))), g.get('labels', np.zeros((0,))))
    out = acc.summary()

    if iou_thrs:
        aps = []
        for t in iou_thrs:
            a = DetectionAccumulator(class_names, float(t))
            for p, g in zip(preds, gts):
                a.add_image(p.boxes, p.scores, p.labels,
                            g.get('boxes', np.zeros((0, 4))), g.get('labels', np.zeros((0,))))
            for i in range(len(class_names)):
                v = a.average_precision(i)
                if v is not None:
                    aps.append(v)
        out['mAP_50_95'] = round(float(np.mean(aps)), 4) if aps else 0.0
        out['iou_thrs'] = list(iou_thrs)
    return out
