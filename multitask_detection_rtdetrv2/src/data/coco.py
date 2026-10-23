# 公共 COCO 数据类：训练返回归一化框，评估保留原图像素框和变换记录。
# 图片清单以标注 JSON 为准；不同任务的尺寸和类别映射从各自配置读取。
"""COCO records preserve original coordinates and an explicit transform record."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from src.core.common import image_size
from src.core.config import resolve_path

# 同一数据接口支持 train/val/test，只有训练样本启用配置中的随机翻转。
class CocoBoxDataset(Dataset):
    def __init__(self,cfg,split):
        self.cfg=cfg; self.split=split; self.size=image_size(cfg)
        self.root=resolve_path(cfg,cfg['data']['root']); spec=cfg['data'][split]
        self.folder=self.root/spec['img_folder']; self.annotation_path=self.root/spec['ann_file']
        self.payload=json.loads(self.annotation_path.read_text(encoding='utf-8-sig'))
        # COCO category_id 可不连续，先映射成本任务从 0 开始的内部标签。
        mapping=cfg['classes'].get('category_id_to_label')
        if mapping is None:
            names=cfg['classes']['names']; mapping={x['id']:names.index(x['name']) for x in self.payload['categories']}
        self.mapping={int(k):int(v) for k,v in mapping.items()}
        if set(self.mapping.values())-set(range(len(cfg['classes']['names']))): raise ValueError('Invalid class mapping')
        by_image={x['id']:[] for x in self.payload['images']}
        for a in self.payload['annotations']:
            if a.get('iscrowd',0): continue
            if a['category_id'] not in self.mapping: raise ValueError('Unmapped category '+str(a['category_id']))
            by_image[a['image_id']].append(a)
        self.rows=[]
        for item in self.payload['images']:
            boxes=[]; labels=[]
            for a in by_image[item['id']]:
                x,y,w,h=a['bbox']
                if w<=0 or h<=0: raise ValueError('Invalid ground truth box')
                # COCO 的左上角+宽高格式转换为原图 xyxy，后续评估保持这一坐标系。
                boxes.append([x,y,x+w,y+h]); labels.append(self.mapping[a['category_id']])
            self.rows.append(dict(image_id=item['id'],image=item['file_name'],path=self.folder/item['file_name'],
                width=item['width'],height=item['height'],boxes=np.asarray(boxes,np.float32).reshape(-1,4),
                labels=np.asarray(labels,np.int64)))

    def __len__(self): return len(self.rows)

    # 统一读成 RGB uint8，并检查真实图片尺寸与 COCO 标注一致。
    def pixels(self,index):
        row=self.rows[index]
        with Image.open(row['path']) as image:
            image=image.convert('RGB')
            if image.size!=(row['width'],row['height']): raise ValueError('Image size differs from annotation')
            return np.asarray(image).copy()

    # 返回网络输入、训练目标和原图信息；变换后的训练目标与原图评估真值分开保存。
    def __getitem__(self,index):
        row=self.rows[index]; pixels=self.pixels(index); boxes=row['boxes'].copy()
        flip=self.split=='train' and random.random()<self.cfg['train'].get('flip_probability',0.)
        if flip:
            # 水平翻转图像时同步更新框的左右边界，保证训练输入与标注对齐。
            pixels=pixels[:,::-1].copy(); boxes[:,[0,2]]=row['width']-boxes[:,[2,0]]
        h,w=self.size
        if pixels.shape[:2]!=(h,w): pixels=np.asarray(Image.fromarray(pixels).resize((w,h),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(pixels).permute(2,0,1).contiguous().float().mul_(1./255)
        b=torch.from_numpy(boxes)
        center=(b[:,:2]+b[:,2:])/2; wh=b[:,2:]-b[:,:2]
        # 直接缩放图像时，按原图宽高归一化的中心点和宽高比例仍适用；此处无需重复缩放框。
        target=dict(boxes=torch.cat((center,wh),1)/torch.tensor([row['width'],row['height']]*2),
                    labels=torch.from_numpy(row['labels'].copy()))
        # meta 留存未增强的原图真值；original_size 为 [宽, 高]，resize 记录为 [高, 宽]。
        meta=dict(image_id=row['image_id'],file_name=row['image'],original_size=[row['width'],row['height']],
                  boxes=row['boxes'],labels=row['labels'],
                  transform=dict(resize_from=[row['height'],row['width']],resize_to=[h,w],horizontal_flip=flip))
        return x,target,meta

# 图片可堆叠成批，目标数因图片而异，因此目标和元信息保持列表。
def collate(samples):
    images,targets,meta=zip(*samples)
    return torch.stack(images),list(targets),list(meta)

def build_loader(cfg,split):
    dataset=CocoBoxDataset(cfg,split); d=cfg['data'][split]
    return DataLoader(dataset,batch_size=d.get('batch_size',1),num_workers=d.get('num_workers',0),
                      shuffle=split=='train',collate_fn=collate,drop_last=False)
