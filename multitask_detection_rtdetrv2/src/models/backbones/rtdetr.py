# 骨干组件的公共导入入口；具体实现来自固定版本的 third_party/rtdetrv2。
# FrozenBatchNorm2d 使用固定统计量，供当前小批量训练配方使用。
"""Public component import; upstream implementation remains pinned."""
from third_party.rtdetrv2.nn.backbone.presnet import PResNet
from third_party.rtdetrv2.nn.backbone.common import FrozenBatchNorm2d
