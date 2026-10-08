"""Compatibility helpers for torchvision's datapoints to tv_tensors rename."""

import torch
import torchvision.transforms.v2 as T


try:
    from torchvision import datapoints

    ToImageTensor = T.ToImageTensor
    ConvertDtype = T.ConvertDtype
    SanitizeBoundingBox = T.SanitizeBoundingBox

    def make_bounding_box(data, box_format, spatial_size):
        return datapoints.BoundingBox(data, format=box_format, spatial_size=spatial_size)

    def bounding_box_spatial_size(boxes):
        return boxes.spatial_size

except ImportError:
    from torchvision import tv_tensors

    class _DatapointsNamespace:
        Image = tv_tensors.Image
        Video = tv_tensors.Video
        Mask = tv_tensors.Mask
        BoundingBox = tv_tensors.BoundingBoxes
        BoundingBoxFormat = tv_tensors.BoundingBoxFormat

    datapoints = _DatapointsNamespace()

    class ToImageTensor(T.ToImage):
        pass

    class ConvertDtype(T.ToDtype):
        def __init__(self, dtype=torch.float32):
            super().__init__(dtype=dtype, scale=True)

    class SanitizeBoundingBox(T.SanitizeBoundingBoxes):
        pass

    def make_bounding_box(data, box_format, spatial_size):
        return tv_tensors.BoundingBoxes(data, format=box_format, canvas_size=spatial_size)

    def bounding_box_spatial_size(boxes):
        return boxes.canvas_size
