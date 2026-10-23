# -*- coding: utf-8 -*-
"""COCO AP/recall selection metrics matching the imported R18VD recipe."""
import contextlib
import io

import numpy as np


IOU_THRESHOLDS = np.asarray([0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


def _mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[values > -1.0]
    return float(valid.mean()) if valid.size else 0.0


def _index_for_iou(iou):
    found = np.where(np.isclose(IOU_THRESHOLDS, float(iou), atol=1e-6))[0]
    if not found.size:
        raise ValueError(f'IoU {iou} is absent from the source metric schedule')
    return int(found[0])


def source_coco_metrics(predictions, ground_truth, class_names, max_detections=200):
    """Return the AP40/R40/R50/R60 values used by the source trainer.

    The evaluator intentionally uses category-aware COCO matching, the source
    IoU sequence and its final maxDets entry.  It builds an in-memory COCO view
    from the unified Detection contract, so no source-project code is imported.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ImportError('pycocotools is required for infrared source metrics') from exc

    images, annotations, detections = [], [], []
    annotation_id = 1
    for image_id, (prediction, target) in enumerate(zip(predictions, ground_truth), 1):
        images.append({'id': image_id})
        boxes = np.asarray(target.get('boxes', np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(target.get('labels', np.zeros((0,))), dtype=np.int64).reshape(-1)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = (float(v) for v in box)
            width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            annotations.append({
                'id': annotation_id, 'image_id': image_id, 'category_id': int(label),
                'bbox': [x1, y1, width, height], 'area': width * height, 'iscrowd': 0,
            })
            annotation_id += 1
        for box, score, label in zip(prediction.boxes, prediction.scores, prediction.labels):
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append({
                'image_id': image_id, 'category_id': int(label),
                'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                'score': float(score),
            })

    categories = [{'id': index, 'name': name} for index, name in enumerate(class_names)]
    coco_gt = COCO()
    coco_gt.dataset = {'images': images, 'annotations': annotations, 'categories': categories}
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {'images': images, 'annotations': [], 'categories': categories}
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.params.imgIds = [row['id'] for row in images]
    evaluator.params.catIds = list(range(len(class_names)))
    evaluator.params.iouThrs = IOU_THRESHOLDS.copy()
    evaluator.params.maxDets = [1, 10, 100, int(max_detections)]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()

    precision, recall = evaluator.eval['precision'], evaluator.eval['recall']
    def at_iou(iou):
        index = _index_for_iou(iou)
        return _mean_valid(precision[index, :, :, 0, -1]), _mean_valid(recall[index, :, 0, -1])

    ap40, recall40 = at_iou(0.40)
    _, recall50 = at_iou(0.50)
    _, recall60 = at_iou(0.60)
    return {
        'AP40': ap40,
        'Recall_IoU40': recall40,
        'Recall_IoU50': recall50,
        'Recall_IoU60': recall60,
        'source_iou_thresholds': IOU_THRESHOLDS.tolist(),
        'source_max_detections': int(max_detections),
    }
