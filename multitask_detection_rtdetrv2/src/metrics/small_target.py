# 小目标评估口径：只匹配位置，按配置的 IoU 阈值统计 TP、FP、FN。
# Pd = TP/(TP+FN)，Fa = FP/(TP+FP)；返回值为 0～1 比例，展示百分比时乘 100。
"""Competition counts with one global match, then ground-truth size recall."""
import numpy as np
from src.metrics.detection import box_iou_matrix

# 每张图先完成一次全局一对一匹配，再累计数据集计数，避免对逐图百分比直接求平均。
def evaluate_small_target(preds,gts,config,**kw):
    if len(preds)!=len(gts): raise ValueError('Prediction/ground-truth length mismatch')
    if config.get('far',{}).get('unit','per_detection')!='per_detection':
        raise ValueError('Declare a separate metric strategy for another FAR denominator')
    iou_thr=float(config.get('primary_iou',.4)); tp=fp=fn=0
    buckets={b['name']:dict(targets=0,tp=0,fn=0) for b in config.get('size_buckets',[])}
    negative_images=negative_fp_images=negative_fp=0
    for pred,gt in zip(preds,gts):
        boxes=np.asarray(gt['boxes'],np.float64).reshape(-1,4)
        overlap=box_iou_matrix(pred.boxes,boxes); taken=np.zeros(len(boxes),bool)
        if not len(boxes):
            negative_images+=1; negative_fp+=len(pred); negative_fp_images+=int(len(pred)>0)
        # 高分预测先匹配尚未占用的真值；一个真值最多计一次 TP，重复预测计 FP。
        for pi in np.argsort(-pred.scores,kind='stable'):
            candidates=np.where(~taken)[0]
            if len(candidates):
                gi=candidates[np.argmax(overlap[pi,candidates])]
                if overlap[pi,gi]>=iou_thr:
                    taken[gi]=True; tp+=1; continue
            fp+=1
        # 所有未被匹配的真值计为漏检 FN；无真值的负样本中，每个预测都计为 FP。
        fn+=int((~taken).sum())
        # 尺寸分组按原图框面积的平方根划分，使用全局匹配结果统计各组召回，避免重复匹配。
        wh=np.maximum(boxes[:,2:]-boxes[:,:2],0); side=np.sqrt(wh[:,0]*wh[:,1])
        for b in config.get('size_buckets',[]):
            mask=(side>=b['min_px'])&(side<b['max_px']); item=buckets[b['name']]
            item['targets']+=int(mask.sum()); item['tp']+=int(taken[mask].sum()); item['fn']+=int((~taken[mask]).sum())
    for b in buckets.values(): b['pd']=b['tp']/b['targets'] if b['targets'] else None
    # 分母为 0 返回 None，写成 JSON 时是 null，表示该比例没有定义。
    return dict(tp=tp,fp=fp,fn=fn,targets=tp+fn,detections=tp+fp,images=len(gts),
        pd=tp/(tp+fn) if tp+fn else None,recall=tp/(tp+fn) if tp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None,fa=fp/(tp+fp) if tp+fp else None,
        far_definition='FP / (TP + FP)',zero_denominator='null (undefined)',match_iou=iou_thr,
        matching='class-agnostic confidence-ordered one-to-one; IoU >= threshold',
        fp_per_image=fp/len(gts) if gts else None,size_definition='sqrt(original bbox width * height), [min,max)',
        size_buckets=buckets,negative_images=negative_images,negative_fp=negative_fp,
        negative_fp_image_rate=negative_fp_images/negative_images if negative_images else None)

# 仅在验证集搜索：优先满足 Pd 约束并降低 Fa；无可行点时记录最高可达 Pd。
def select_operating_point(records,adapter,cfg):
    """Validation only: lowest Fa subject to a declared Pd constraint."""
    events=[]; total_gt=0
    for record in records:
        p=record['predictions']; boxes=record['ground_truth']['boxes']; total_gt+=len(boxes)
        overlap=box_iou_matrix(p.boxes,boxes); taken=np.zeros(len(boxes),bool)
        for i in np.argsort(-p.scores,kind='stable'):
            hit=0; available=np.where(~taken)[0]
            if len(available):
                j=available[np.argmax(overlap[i,available])]
                if overlap[i,j]>=cfg['evaluation']['primary_iou']: taken[j]=True; hit=1
            events.append((float(p.scores[i]),hit))
    # 先汇总每个候选的命中状态，再按全验证集分数扫描阈值，累计 TP/FP。
    events.sort(reverse=True,key=lambda x:x[0]); tp=fp=0; points=[]
    for index,(score,hit) in enumerate(events):
        tp+=hit; fp+=1-hit
        # 相同置信度必须一起纳入，才能与实际 score >= threshold 的筛选行为一致。
        if index+1==len(events) or events[index+1][0]!=score:
            points.append((score,tp,fp,tp/total_gt if total_gt else 0.,fp/(tp+fp)))
    target=cfg['evaluation']['validation_pd_target']; feasible=[x for x in points if x[3]>=target]
    # 可行点中先选最低 Fa，再以较高 Pd、较高阈值打破平局；未达标时显式返回 target_met=False。
    chosen=min(feasible,key=lambda x:(x[4],-x[3],-x[0])) if feasible else max(points,key=lambda x:(x[3],-x[4],x[0])) if points else (1.,0,0,0.,1.)
    value=chosen[0]; metrics=adapter.evaluate([r['predictions'].filter_by_score(value) for r in records],[r['ground_truth'] for r in records])
    return dict(confidence=value,metrics=metrics,target_met=bool(feasible),selected_on='val')
