# 红外小目标业务子类：继承公共 DETR 流程，专门定义单类约束和 Pd/Fa 评估。
# target 标签表示目标存在，不承担车辆、人员等语义类别识别。
"""Only the small-target business rules specialize the shared detection class."""
from src.core.registry import register_task
from src.models.adapters.base import TaskMeta
from src.models.adapters.detr_task import DetrTaskAdapter

@register_task('infrared_small_target','RT-DETRv2-S 红外小目标检测','implemented')
class InfraredSmallTargetAdapter(DetrTaskAdapter):
    META=TaskMeta('infrared_small_target','红外小目标检测','原图xyxy框、分数、固定标签0',
        'AA-YOLO、原RT-DETR','RT-DETRv2-S轻量基线','implemented')
    class_names=['target']

    # 只接受单一 target 前景类，后续仍需要分类分支学习目标置信度。
    def bind(self,cfg):
        if cfg['classes']['names'] != ['target']:
            raise ValueError('Small-target task requires exactly one target label')
        return super().bind(cfg)

    # 重写父类的默认识别评估，改用当前比赛口径的类别无关 Pd/Fa。
    def evaluate(self,preds,gts,**kw):
        from src.metrics.small_target import evaluate_small_target
        return evaluate_small_target(preds,gts,self.cfg['evaluation'],**kw)

    # 调用小目标专用的验证集阈值策略，供公共训练循环复用。
    def select_operating_point(self,records,cfg):
        from src.metrics.small_target import select_operating_point
        return select_operating_point(records,self,cfg)
