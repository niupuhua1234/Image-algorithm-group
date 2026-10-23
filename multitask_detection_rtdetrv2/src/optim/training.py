# 公共优化组件：按配置构建参数分组和 EMA，状态随各任务实验独立保存。
"""Reusable optimizer and EMA; algorithm losses remain adapter-owned."""
import math
from copy import deepcopy
import torch

# 骨干使用可配置的较小学习率；骨干外的偏置和归一化权重按当前配方免权重衰减。
def build_optimizer(model,cfg):
    options=cfg['optimizer']
    if options['type']!='AdamW': raise NotImplementedError(options['type'])
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        lr=options.get('backbone_lr',options['lr']) if 'backbone' in name else options['lr']
        no_decay=('backbone' not in name and ('bias' in name or ('norm' in name and name.endswith('weight'))))
        wd=0. if no_decay else options['weight_decay']
        groups.setdefault((lr,wd),[]).append(p)
    return torch.optim.AdamW([dict(params=params,lr=lr,weight_decay=wd) for (lr,wd),params in groups.items()],betas=(.9,.999))

# 维护模型参数的指数滑动平均副本，用于验证和选取推理权重，不参与梯度更新。
class ModelEMA:
    def __init__(self,model,decay=.9999,warmups=2000):
        self.module=deepcopy(model).eval(); self.decay=decay; self.warmups=warmups; self.updates=0
        for p in self.module.parameters(): p.requires_grad_(False)

    # 更新初期使用逐渐升高的衰减系数；这里只对浮点参数和缓冲区做滑动平均。
    @torch.no_grad()
    def update(self,model):
        self.updates+=1; decay=self.decay*(1-math.exp(-self.updates/self.warmups))
        state=model.state_dict()
        for k,v in self.module.state_dict().items():
            if v.is_floating_point(): v.mul_(decay).add_(state[k].detach(),alpha=1-decay)

    # 保存平均模型及更新步数，断点恢复后可延续相同的 EMA 调度。
    def state_dict(self):
        return dict(module=self.module.state_dict(),updates=self.updates,decay=self.decay,warmups=self.warmups)

    def load_state_dict(self,state):
        self.module.load_state_dict(state['module'],strict=True); self.updates=state['updates']
