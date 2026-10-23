# RT-DETR v2 是完整检测架构；其中 PResNet 是骨干，后接编码器和解码器。
# 当前红外小目标采用 v2-S 配置，其他任务可在同一主线下配置自己的规模。
"""Pinned official v2 components, retaining the benchmark architecture."""
from src.models.factory import register_algorithm

# 按三段配置组装骨干、编码器、解码器；类别数取自当前任务，固定推理尺寸按高、宽传入。
def model_builder(cfg):
    from src.models.backbones.rtdetr import PResNet
    from src.models.encoders.rtdetr import HybridEncoder
    from src.models.decoders.rtdetr import RTDETRTransformerv2
    from third_party.rtdetrv2.zoo.rtdetr.rtdetr import RTDETR
    import torch
    m = cfg['model']; size = list(m['input_size'])
    model = RTDETR(PResNet(**m['backbone']),
        HybridEncoder(**dict(m['encoder'], eval_spatial_size=size)),
        RTDETRTransformerv2(**dict(m['decoder'], num_classes=len(cfg['classes']['names']), eval_spatial_size=size)))
    # 把预生成的位置编码登记为非持久缓冲区，使其随 model.to(device) 移动，同时不增加权重文件键。
    for module in model.modules():
        for key, value in list(vars(module).items()):
            if key.startswith('pos_embed') and torch.is_tensor(value) and key not in module._buffers:
                delattr(module,key)
                module.register_buffer(key,value,persistent=False)
    return model

def criterion_builder(cfg):
    from src.losses.rtdetr import build_loss
    return build_loss(cfg)

# 两个名称复用同一构造器，实际网络规模由配置中的骨干、编码器和解码器参数决定。
register_algorithm('rtdetrv2_s',model_builder,criterion_builder,'detr')
register_algorithm('rtdetrv2',model_builder,criterion_builder,'detr')
