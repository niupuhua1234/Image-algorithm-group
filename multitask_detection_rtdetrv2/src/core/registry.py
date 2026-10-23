# -*- coding: utf-8 -*-
# 任务注册表回答“做哪一种业务”；模型算法注册表位于 src/models/factory.py。
"""组件注册表 —— 任务 / 适配器 / 模型按名查找。

对应技术路线书 2.3 节"三层复用机制"的**算法组件层**入口:
    算法组件层   YOLO 检测头、AADH、DETR 解码器、目标匹配、损失、解码与后处理
                 复用方式: 按算法配套配置, 通过适配层接入

三个任务标识(路线书 5.2):
    visible_recognition    可见光目标识别    对照: DEIM
    infrared_recognition   红外目标识别      对照: YOLOv8 / YOLO11
    infrared_small_target  红外小目标检测    对照: AA-YOLO / 现有 RT-DETR
"""
from typing import Dict, List, Optional, Type

# task_id -> {'cls': TaskAdapter子类, 'description': str, 'status': str}
_TASKS: Dict[str, dict] = {}


def register_task(task_id: str, description: str = '', status: str = 'implemented'):
    """装饰器: 把适配器类注册到某个任务标识。

    Args:
        task_id:     路线书定义的任务标识
        description: 业务内容说明
        status:      'implemented' | 'placeholder' —— placeholder 表示只留了骨架
    """
    def deco(cls):
        if task_id in _TASKS:
            raise KeyError(f'任务 {task_id} 已被 {_TASKS[task_id]["cls"].__name__} 注册')
        _TASKS[task_id] = {'cls': cls, 'description': description, 'status': status}
        # 注册表存储的是类；get_adapter 返回后还需实例化，每个任务实例再绑定自己的配置。
        cls.task_id = task_id
        return cls
    return deco


def get_adapter(task_id: str) -> Type:
    """按任务标识取适配器类。"""
    if task_id not in _TASKS:
        raise KeyError(
            f'未知任务 {task_id!r}。已注册: {sorted(_TASKS)}\n'
            f'请检查 configs/tasks/ 下的任务配置, 或确认适配器已 import。'
        )
    return _TASKS[task_id]['cls']


def available_tasks() -> List[dict]:
    """列出全部任务及其状态, 供 CLI 展示。"""
    return [
        {'task_id': k, 'description': v['description'],
         'status': v['status'], 'adapter': v['cls'].__module__ + '.' + v['cls'].__name__}
        for k, v in sorted(_TASKS.items())
    ]


def load_all_adapters() -> None:
    """import 所有适配器模块, 触发注册。

    新增任务时只需在 src/models/adapters/ 下加文件并在下面追加一行。
    """
    from src.models.adapters import (  # noqa: F401
        visible_recognition,
        infrared_recognition,
        infrared_small_target,
    )
