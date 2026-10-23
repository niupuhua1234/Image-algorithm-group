# 公共 DETR 后处理：解码目标框、还原原图坐标、按置信度筛选并执行任务配置的 NMS。
"""Shared DETR decode and optional task-configured class-aware NMS."""
import numpy as np
from src.models.adapters.base import Detection
from src.metrics.detection import box_iou_matrix

# 按分数由高到低保留框，移除与已保留框 IoU 超过阈值的重复候选。
def nms(rows,threshold):
    rows=np.asarray(rows,dtype=np.float32).reshape(-1,5)
    order=np.argsort(-rows[:,4],kind='stable'); keep=[]
    while len(order):
        i=order[0]; keep.append(i); rest=order[1:]
        if len(rest)==0: break
        overlap=box_iou_matrix(rows[i:i+1,:4],rows[rest,:4])[0]
        order=rest[overlap<=threshold]
    return rows[keep].copy()

# PyTorch 字典与 TensorRT 数组分别解码，最后统一成原图坐标的 Detection。
def decode_detr(raw,orig_size,cfg,conf=None):
    import torch
    dep=cfg['deployment']; conf=dep.get('conf_threshold',.005) if conf is None else conf
    if conf is None: raise ValueError('Freeze a validation operating point before inference')
    if isinstance(raw,dict):
        # 网络框格式为归一化 [中心x, 中心y, 宽, 高]，分类 logits 先经 sigmoid 转成分数。
        boxes=raw['pred_boxes'].float(); scores=raw['pred_logits'].float().sigmoid()
        classes=scores.shape[-1]
        # Source RT-DETR focal postprocessor keeps its configured query count
        # after flattening query x class scores (300 for infrared recognition).
        configured_topk=int(dep.get('num_top_queries',cfg['model'].get('decoder',{}).get('num_queries',boxes.shape[1])))
        k=min(configured_topk,scores.shape[1]*classes)
        # 展平的是查询×类别；通过取模恢复类别编号，通过整除恢复对应的查询编号。
        scores,idx=torch.topk(scores.flatten(1),k,dim=-1)
        labels=(idx%classes)[0].cpu().numpy()
        xyxy=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),-1)
        xyxy=xyxy.gather(1,(idx//classes).unsqueeze(-1).expand(-1,-1,4))[0]
        # 使用原图 (宽, 高) 将比例框还原为像素框，支持矩形图像和不同任务尺寸。
        xyxy=xyxy*xyxy.new_tensor([orig_size[0],orig_size[1]]*2)
        rows=torch.cat((xyxy,scores[0,:,None]),1).detach().cpu().numpy()
    else:
        # TensorRT exporter returns N x (4 + C): input-pixel xyxy plus one
        # sigmoid focal score per class.  Decode it like the PyTorch path.
        packed=np.asarray(raw,dtype=np.float32).reshape(raw.shape[0],-1)
        classes=len(cfg['classes']['names'])
        if packed.shape[1]!=4+classes:
            raise ValueError(f'Unexpected TensorRT output shape {packed.shape}; expected N x {4+classes}')
        score_matrix=packed[:,4:]
        configured_topk=int(dep.get('num_top_queries',cfg['model'].get('decoder',{}).get('num_queries',packed.shape[0])))
        k=min(configured_topk,score_matrix.size)
        order=np.argsort(-score_matrix.reshape(-1),kind='stable')[:k]
        labels=(order%classes).astype(np.int64)
        boxes=packed[order//classes,:4]
        scores=score_matrix.reshape(-1)[order]
        h,w=cfg['model']['input_size']; boxes*=np.array([orig_size[0]/w,orig_size[1]/h]*2,np.float32)
        rows=np.concatenate((boxes,scores[:,None]),axis=1)
    mask=rows[:,4]>=conf; rows,labels=rows[mask],labels[mask]
    # NMS 是否启用及阈值均取自当前任务；多类时分开抑制，避免不同类别互相删除。
    threshold=dep.get('nms_iou')
    if threshold is not None and len(rows):
        selected=[]
        for label in np.unique(labels):
            filtered=nms(rows[labels==label],threshold)
            selected.extend((r,int(label)) for r in filtered)
        rows=np.array([r for r,l in selected],np.float32).reshape(-1,5)
        labels=np.array([l for r,l in selected],np.int64)
    return Detection(rows[:,:4],rows[:,4],labels)
