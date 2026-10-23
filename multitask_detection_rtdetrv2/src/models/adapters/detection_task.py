# 检测任务公共父类：共享数据加载、推理运行时及训练/评估入口。
# 子类可覆盖同名方法表达差异；公共调用方无需按任务名称反复写判断分支。
"""Template methods shared by all box-detection business tasks."""
from src.models.adapters.base import TaskAdapter

class DetectionTaskAdapter(TaskAdapter):
    # 把配置绑定到当前实例；复制类别列表，避免多个任务实例共享可变列表。
    def bind(self,cfg):
        self.cfg=cfg
        self.class_names=list(cfg['classes']['names'])
        return self

    # 默认复用公共 COCO 加载器；有专用增强的任务可覆盖本方法。
    def build_dataloader(self,cfg,split):
        from src.data.coco import build_loader
        return build_loader(cfg,split)

    # 默认复用 DETR 解码，阈值、类别和 NMS 策略仍读取本任务配置。
    def postprocess(self,raw_output,orig_size,conf=None):
        from src.deployment.postprocess import decode_detr
        return decode_detr(raw_output,orig_size,self.cfg,conf)

    # 把当前业务子类交给统一运行时，使后续后处理继续调用该子类的接口。
    def create_runtime(self,cfg,checkpoint=None,device='cuda',backend='pytorch'):
        from src.deployment.unified_runtime import DetectionRuntime
        self.bind(cfg)
        return DetectionRuntime(cfg,self,checkpoint,device,backend)

    # 共享训练入口；模型、损失、数据和选点策略通过 self 调用，保留子类重写的行为。
    def run_training(self,cfg,args):
        from src.engine.trainer import train
        return train(cfg,self,args)

    def run_evaluation(self,cfg,args):
        from src.engine.evaluation import evaluate
        return evaluate(cfg,self,args)

    # 不同业务的模型选择标准可能不同，由具体任务声明自己的验证策略。
    def select_operating_point(self,records,cfg):
        raise NotImplementedError('Declare the validation selection policy for this business task')
