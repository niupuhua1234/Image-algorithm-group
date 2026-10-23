# -*- coding: utf-8 -*-
# 继承的起点：抽象父类规定统一接口，具体子类实现业务规则，调用方按接口使用。
"""任务适配器抽象 —— 三类业务任务的统一接口。

技术路线书 5.3 节的硬性要求:

    三类任务统一输出目标框、分数和类别标识。红外小目标检测任务可以将所有目标归为
    单一类别; 红外与可见光识别任务保留各自的业务类别。类别数量和类别编号独立配置,
    避免将不同任务的类别体系混用。

    预测适配器负责原图坐标还原、分数处理,以及该算法所需的 NMS 或集合预测后处理。
    对外结果一致不意味着内部原始输出一致。

技术路线书 2.3 节的约束:

    不能将 YOLO 检测头、AA objectness 或 DETR 匈牙利匹配任意组合。

因此适配器**同时绑定**「模型 + 损失/匹配 + 后处理」三件套,三者必须来自同一算法族。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Detection:
    """统一检测结果 —— 三类任务对外都用这个结构。

    Attributes:
        boxes:  [N, 4] float, **原图像素坐标**, xyxy 格式
        scores: [N]    float
        labels: [N]    int, 取自本任务的 class_names 下标
    """
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    # 筛选时同步索引框、分数和标签，保证三个数组仍逐项对应。
    def filter_by_score(self, thr: float) -> 'Detection':
        keep = self.scores >= thr
        return Detection(self.boxes[keep], self.scores[keep], self.labels[keep])

    def to_dict(self, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        names = class_names or []
        return {
            'boxes': self.boxes.tolist(),
            'scores': self.scores.tolist(),
            'labels': self.labels.tolist(),
            'class_names': [names[int(i)] if int(i) < len(names) else str(int(i))
                            for i in self.labels.tolist()],
        }

    # 无检出时保留规范的空数组形状，后处理和指标无需另换数据结构。
    @staticmethod
    def empty() -> 'Detection':
        return Detection(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                         np.zeros((0,), np.int64))


@dataclass
class TaskMeta:
    """任务的元信息 —— 与路线书 5.2 的表格一一对应。"""
    task_id: str
    business_name: str
    output_spec: str
    native_baseline: str
    mainline_plan: str
    status: str = 'implemented'          # implemented | placeholder
    notes: List[str] = field(default_factory=list)


# 抽象方法规定子类必须提供的能力；evaluate 的默认实现可被业务子类覆盖。
class TaskAdapter(ABC):
    """任务适配器基类。

    子类必须声明:
        META            TaskMeta, 描述这个任务在路线书里的定位
        class_names     业务类别名(顺序即 label 编号, 各任务独立)
    """

    META: TaskMeta = None                     # type: ignore
    class_names: List[str] = []

    # ------------------------------------------------------------------ #
    # 类别体系 —— 各任务独立, 禁止跨任务混用
    # ------------------------------------------------------------------ #
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def task_id(self) -> str:
        return self.META.task_id if self.META else self.__class__.__name__

    # ------------------------------------------------------------------ #
    # 构建三件套(模型 / 损失 / 数据)—— 必须来自同一算法族
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_model(self, cfg: Dict) -> Any:
        """按模型配置构建网络。"""

    @abstractmethod
    def build_criterion(self, cfg: Dict) -> Any:
        """构建与该模型配套的损失与匹配器。"""

    @abstractmethod
    def build_dataloader(self, cfg: Dict, split: str) -> Any:
        """构建某个划分(train/val/test)的数据加载器。"""

    # ------------------------------------------------------------------ #
    # 后处理 —— 把算法原始输出转成统一的 Detection
    # ------------------------------------------------------------------ #
    @abstractmethod
    def postprocess(self, raw_output: Any, orig_size, conf: float = 0.005) -> Detection:
        """原始输出 -> Detection(原图坐标)。

        Args:
            raw_output: 模型前向的原始返回(各算法不同, 对外不暴露)
            orig_size:  (w, h) 原图尺寸, 用于把预测框还原回去
            conf:       分数阈值
        """

    # ------------------------------------------------------------------ #
    # 评测 —— 默认走公共指标, 特殊口径可覆盖
    # ------------------------------------------------------------------ #
    def evaluate(self, preds: List[Detection], gts: List[Dict], **kw) -> Dict:
        """默认实现: COCO 风格 AP + 逐类别 P/R + 类别混淆矩阵。

        路线书 6 节对框级识别任务要求的指标:
            检测 AP、逐类别精度与召回、匹配目标上的类别混淆情况、完整业务时延
        """
        from src.metrics.detection import evaluate_detection
        return evaluate_detection(preds, gts, self.class_names, **kw)

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        m = self.META
        return {
            'task_id': m.task_id if m else self.task_id,
            'business_name': m.business_name if m else '',
            'num_classes': self.num_classes,
            'class_names': list(self.class_names),
            'output_spec': m.output_spec if m else '',
            'native_baseline': m.native_baseline if m else '',
            'mainline_plan': m.mainline_plan if m else '',
            'status': m.status if m else 'unknown',
            'notes': list(m.notes) if m else [],
        }


class PlaceholderAdapter(TaskAdapter):
    """占位适配器 —— 给尚未实现的任务留位置。

    路线书第一阶段的要求是"保留四套原生对照", 所以占位任务不是空的:
    它声明了任务标识、类别体系、对照方法, 只缺具体实现。
    """

    PLACEHOLDER_HINT = '该任务尚未实现, 参见 docs/ARCHITECTURE.md 的接入步骤'

    def _raise(self, what: str):
        raise NotImplementedError(f'[{self.task_id}] {what} 未实现。{self.PLACEHOLDER_HINT}')

    def build_model(self, cfg: Dict) -> Any:
        self._raise('build_model')

    def build_criterion(self, cfg: Dict) -> Any:
        self._raise('build_criterion')

    def build_dataloader(self, cfg: Dict, split: str) -> Any:
        self._raise('build_dataloader')

    def postprocess(self, raw_output: Any, orig_size, conf: float = 0.005) -> Detection:
        self._raise('postprocess')

    def bind(self, cfg):
        self.cfg = cfg
        return self

    def create_runtime(self, *args, **kwargs):
        self._raise('create_runtime')

    def run_training(self, *args, **kwargs):
        self._raise('run_training')

    def run_evaluation(self, *args, **kwargs):
        self._raise('run_evaluation')
