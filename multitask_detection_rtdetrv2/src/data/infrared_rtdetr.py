# -*- coding: utf-8 -*-
"""Data pipeline used only by the imported DroneVehicle infrared RT-DETR run."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datapoints
from torchvision.transforms import v2 as T

from src.core.common import image_size
from src.core.config import resolve_path
from src.data.coco import CocoBoxDataset, collate


class InfraredRTDETRDataset(CocoBoxDataset):
    """COCO dataset with the source R18VD augmentation recipe.

    This intentionally does not change ``CocoBoxDataset``: existing branches
    keep their established loader while infrared recognition opts into this
    source-compatible pipeline through its own adapter.
    """

    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _is_disabled_by_policy(self, name):
        policy = ((self.cfg.get('augmentation') or {}).get(self.split) or {}).get('policy') or {}
        if policy.get('name') != 'stop_epoch' or name not in policy.get('ops', []):
            return False
        return self.epoch >= int(policy.get('epoch', 10 ** 9))

    @staticmethod
    def _sanitize(image, target, **kwargs):
        return T.SanitizeBoundingBox(min_size=float(kwargs.get('min_size', 1)))(image, target)

    def _apply_ops(self, image, target):
        spec = ((self.cfg.get('augmentation') or {}).get(self.split) or {})
        for declared in spec.get('ops') or []:
            item = dict(declared)
            name = item.pop('type')
            if self._is_disabled_by_policy(name):
                continue
            if name == 'RandomIoUCrop':
                probability = float(item.pop('p', 1.0))
                if torch.rand(()) < probability:
                    image, target = T.RandomIoUCrop(**item)(image, target)
            elif name == 'SanitizeBoundingBoxes':
                image, target = self._sanitize(image, target, **item)
            elif name == 'RandomHorizontalFlip':
                image, target = T.RandomHorizontalFlip(**item)(image, target)
            elif name == 'RandomVerticalFlip':
                image, target = T.RandomVerticalFlip(**item)(image, target)
            elif name == 'RandomPhotometricDistort':
                image, target = T.RandomPhotometricDistort(**item)(image, target)
            elif name == 'Resize':
                image, target = T.Resize(**item)(image, target)
            elif name == 'ConvertPILImage':
                # torchvision 0.15 converts uint8 to [0, 1] for float dtype,
                # matching the source project's ConvertPILImage(scale=True).
                dtype = getattr(torch, str(item.get('dtype', 'float32')))
                image, target = T.ToImageTensor()(image, target)
                image, target = T.ConvertDtype(dtype)(image, target)
            elif name == 'ConvertBoxes':
                # Conversion to normalized cxcywh is performed below after all
                # geometry transforms, so the returned target matches criterion.
                continue
            else:
                raise KeyError(f'Unsupported infrared RT-DETR transform: {name}')
        if not torch.is_tensor(image):
            image, target = T.ToImageTensor()(image, target)
            image, target = T.ConvertDtype(torch.float32)(image, target)
        return torch.as_tensor(image), target

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as source:
            image = source.convert('RGB')
        if image.size != (row['width'], row['height']):
            raise ValueError('Image size differs from annotation')

        boxes = torch.from_numpy(row['boxes'].copy()).to(torch.float32)
        target = {
            'boxes': datapoints.BoundingBox(
                boxes, format='XYXY', spatial_size=(row['height'], row['width'])
            ),
            'labels': torch.from_numpy(row['labels'].copy()).to(torch.int64),
        }
        image, target = self._apply_ops(image, target)
        h, w = image.shape[-2:]
        transformed = torch.as_tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
        center = (transformed[:, :2] + transformed[:, 2:]) / 2
        wh = transformed[:, 2:] - transformed[:, :2]
        normalized = torch.cat((center, wh), 1) / torch.tensor(
            [w, h, w, h], dtype=torch.float32
        )
        train_target = {'boxes': normalized, 'labels': torch.as_tensor(target['labels']).to(torch.int64)}
        meta = {
            'image_id': row['image_id'],
            'file_name': row['image'],
            'original_size': [row['width'], row['height']],
            'boxes': row['boxes'],
            'labels': row['labels'],
            'transform': {
                'recipe': 'rtdetrv2_r18vd_120e_first_date_200',
                'resize_from': [row['height'], row['width']],
                'resize_to': [int(h), int(w)],
                'epoch': self.epoch,
            },
        }
        return image.contiguous(), train_target, meta


def build_infrared_rtdetr_loader(cfg, split):
    """Build the source-compatible loader without affecting other tasks."""
    dataset = InfraredRTDETRDataset(cfg, split)
    spec = cfg['data'][split]
    return DataLoader(
        dataset,
        batch_size=spec.get('batch_size', cfg['train'].get('batch_size', 1)),
        num_workers=spec.get('num_workers', cfg['train'].get('num_workers', 0)),
        shuffle=split == 'train',
        collate_fn=collate,
        drop_last=False,
    )
