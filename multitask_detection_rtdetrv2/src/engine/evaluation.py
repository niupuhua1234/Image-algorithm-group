# 公共评估调度：采集预测与原图真值，具体业务指标由 adapter.evaluate 多态分派。
"""Shared orchestration; business metrics dispatch through adapter.evaluate."""
import time
import numpy as np
from src.core.common import save_json,digest
from src.data.coco import CocoBoxDataset

# 按标注清单逐图推理，将统一预测、原图真值和可追溯记录一起收集。
def collect(runtime,cfg,split,limit=None,conf=None):
    data=CocoBoxDataset(cfg,split); n=min(limit,len(data)) if limit else len(data)
    preds=[]; gts=[]; records=[]
    for i in range(n):
        row=data.rows[i]; pred=runtime.infer_rgb(data.pixels(i),conf)
        gt=dict(boxes=row['boxes'],labels=row['labels']); preds.append(pred); gts.append(gt)
        records.append(dict(image=row['image'],image_id=row['image_id'],predictions=pred.to_dict(),
            ground_truth=dict(boxes=row['boxes'].tolist(),labels=row['labels'].tolist())))
    return preds,gts,records,data

# 通过任务子类计算指标，并记录固定工作点、权重与标注指纹，便于复核。
def evaluate(cfg,adapter,args):
    rt=adapter.create_runtime(cfg,args.ckpt,args.device,args.backend)
    pred,gt,records,data=collect(rt,cfg,args.split,args.limit,args.conf)
    # 调用同一个 evaluate 接口时，由具体任务子类决定采用 Pd/Fa 或识别类指标。
    result=dict(task_id=cfg['task']['id'],algorithm=cfg['model']['arch'],split=args.split,
        metrics=adapter.evaluate(pred,gt),checkpoint=str(rt.checkpoint),checkpoint_sha256=digest(rt.checkpoint),
        annotation_sha256=digest(data.annotation_path),class_names=cfg['classes']['names'],
        operating_point=dict(confidence=args.conf if args.conf is not None else cfg['deployment']['conf_threshold'],
                             nms_iou=cfg['deployment'].get('nms_iou')),partial_dataset=bool(args.limit),
        test_role='existing-set regression; thresholds fixed before test')
    if args.bench:
        # 先解码测速图像，再调用运行时计时，磁盘读取和图片解码不计入接口耗时。
        images=[data.pixels(i) for i in range(min(20,len(data)))]
        result['timing']=rt.benchmark(images,args.bench,conf=args.conf)
    if args.out:
        save_json(args.out,result)
        if args.save_predictions: save_json(str(args.out)+'.predictions.json',records)
    display=dict(result)
    if 'timing' in display: display['timing']={k:v for k,v in result['timing'].items() if k!='raw_ms'}
    print(__import__('json').dumps(display,ensure_ascii=False,indent=2))
    return result
