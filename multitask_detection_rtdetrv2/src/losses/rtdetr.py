# 损失构造入口：匈牙利匹配确定预测与真值的一对一关系，再计算 v2 配套损失。
"""Matched loss + Hungarian matcher pair for the v2 family."""
from third_party.rtdetrv2.zoo.rtdetr.matcher import HungarianMatcher
from third_party.rtdetrv2.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2

def build_loss(cfg):
    loss=cfg['loss']; options={k:v for k,v in loss.items() if k not in ('family','matcher')}
    return RTDETRCriterionv2(matcher=HungarianMatcher(**loss['matcher']),
        num_classes=len(cfg['classes']['names']),**options)
