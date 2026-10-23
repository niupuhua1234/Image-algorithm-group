# DETR 任务中间父类：复用检测流程，并通过算法工厂构建配套网络和损失。
# 业务子类继承这里的默认行为，也可以覆盖具体方法接入自己的实现。
"""RT-DETR family template shared by tasks with compatible algorithms."""
from src.models.adapters.detection_task import DetectionTaskAdapter
from src.models.factory import build_model,build_criterion

class DetrTaskAdapter(DetectionTaskAdapter):
    # 先绑定任务配置，再按 model.arch 构建网络，不在父类中写死 v2-S 的规模。
    def build_model(self,cfg):
        self.bind(cfg)
        return build_model(cfg)

    # 通过同一算法工厂获得配套损失，避免网络和匹配器来自不同算法族。
    def build_criterion(self,cfg):
        return build_criterion(cfg)
