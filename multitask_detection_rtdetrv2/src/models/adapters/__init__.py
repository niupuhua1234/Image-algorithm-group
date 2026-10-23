# -*- coding: utf-8 -*-
"""任务适配器集合。

新增一个任务时的步骤(见 docs/ARCHITECTURE.md):
    1) 在本目录新建 <task_id>.py, 继承 TaskAdapter 或 PlaceholderAdapter
    2) 用 @register_task(task_id=...) 装饰
    3) 在 src/core/registry.py 的 load_all_adapters() 里追加 import
    4) 在 configs/tasks/ 下加同名 yaml
"""
from src.models.adapters.base import Detection, PlaceholderAdapter, TaskAdapter, TaskMeta

__all__ = ['Detection', 'TaskAdapter', 'PlaceholderAdapter', 'TaskMeta']
