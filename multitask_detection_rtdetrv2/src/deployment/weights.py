# 公共权重加载：核对任务和类别身份；预训练初始化与正式推理采用不同的加载严格度。
"""Checkpoint identity and strict inference loading shared by all tasks."""
from pathlib import Path
import torch
from src.core.common import digest

# 完整训练断点优先使用 EMA；普通推理权重读取 model，纯参数字典直接返回。
def state_dict(checkpoint):
    if isinstance(checkpoint.get('ema'),dict) and 'module' in checkpoint['ema']:
        return checkpoint['ema']['module']
    return checkpoint.get('model',checkpoint)

# 初始化允许跳过类别头等形状不符的张量；正式推理要求参数完整匹配。
def load_weights(model,path,cfg,initialize=False):
    path=Path(path)
    ck=torch.load(str(path),map_location='cpu')
    if 'task_id' in ck and ck['task_id']!=cfg['task']['id']: raise ValueError('Checkpoint belongs to another task')
    if 'class_names' in ck and ck['class_names']!=cfg['classes']['names']: raise ValueError('Checkpoint class map mismatch')
    expected=cfg['deployment'].get('checkpoint_sha256')
    if not initialize and expected and digest(path)!=expected: raise ValueError('Checkpoint SHA256 differs from deployment configuration')
    sd=state_dict(ck)
    if initialize:
        target=model.state_dict(); matched={k:v for k,v in sd.items() if k in target and target[k].shape==v.shape}
        # 覆盖率按参数字典中的张量条目数检查，防止把不相干模型误当作预训练初始化。
        if len(matched)<.85*len(target): raise ValueError('Insufficient pretrained tensor coverage')
        model.load_state_dict(matched,strict=False)
    else: model.load_state_dict(sd,strict=True)
    return ck
