# -*- coding: utf-8 -*-
# 配置负责表达各任务差异：输入尺寸、类别、训练策略和部署阈值分别设置。
"""配置加载 —— YAML + include 链 + 路径解析 + 环境变量替换。

设计原则(对应技术路线书 3.2 节):
    任务差异通过**任务配置**、**模型配置**和**训练适配器**表达,
    避免为每个任务复制一整套项目。

配置分层:
    configs/tasks/<task>.yaml        任务层: 任务标识、类别映射、数据路径、评测口径
    configs/models/<model>.yaml      模型层: 主干、编解码器、损失
    configs/training/<recipe>.yaml   训练层: 优化器、调度、增强策略
    configs/deployment/<dep>.yaml    部署层: 导出、推理后端

任务配置用 `defaults:` 引用另外三层, 后者可以覆盖前者的任意字段。
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}')


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并: override 覆盖 base, dict 逐层合并, 其他类型直接替换。"""
    # 字典递归合并，列表等其他类型整体替换，避免把不同任务的类别列表拼接。
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_env(obj: Any) -> Any:
    """把 ${VAR} / ${VAR:-默认值} 展开成环境变量, 支持嵌套 dict/list。"""
    if isinstance(obj, str):
        def sub(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else '')
        return _ENV_RE.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f'配置不存在: {path}')
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _resolve_include(cfg: Dict, base_dir: Path, seen: set) -> Dict:
    """处理 defaults 列表(可含多个), 按顺序深度合并。"""
    merged: Dict = {}
    for inc in cfg.get('defaults', []) or []:
        # allow 语法: defaults: [ {include: ../models/x.yaml} ]
        inc_path = inc.get('include') if isinstance(inc, dict) else inc
        if not inc_path:
            continue
        p = (base_dir / inc_path).resolve()
        if p in seen:
            raise RuntimeError(f'配置 include 出现环: {p}')
        seen.add(p)
        sub = _resolve_include(_load_yaml(p), p.parent, seen)
        merged = _deep_merge(merged, sub)
        seen.discard(p)
    own = {k: v for k, v in cfg.items() if k != 'defaults'}
    return _deep_merge(merged, own)


def load_config(task_or_path: str, overrides: Dict = None) -> Dict:
    """加载任务配置(含 include 链)。

    Args:
        task_or_path: 任务标识(如 'visible_recognition')或 yaml 路径
        overrides:    最上层覆盖, 例如 {'train': {'epoches': 30}}

    Returns:
        合并后的完整配置 dict, 附带 '_config_path' 与 '_project_root'
    """
    p = Path(task_or_path)
    if not p.suffix:
        p = PROJECT_ROOT / 'configs' / 'tasks' / f'{task_or_path}.yaml'
    elif not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()

    # 先合并 defaults 引用，再展开环境变量，最后应用调用方显式覆盖项。
    cfg = _resolve_include(_load_yaml(p), p.parent, {p.resolve()})
    cfg = _expand_env(cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg['_config_path'] = str(p)
    cfg['_project_root'] = str(PROJECT_ROOT)
    root = cfg.get('data', {}).get('root', '')
    cfg['_data_root'] = str(resolve_path(cfg, root)) if root else ''
    return cfg


def resolve_path(cfg: Dict, value: str) -> Path:
    """把配置里的相对路径按项目根解析成绝对路径。

    支持 ${DATA_ROOT} 之类的环境变量占位(已在 load_config 阶段展开)。
    """
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(cfg.get('_project_root', PROJECT_ROOT)) / p).resolve()


def dump_resolved(cfg: Dict) -> str:
    """打印最终生效配置(便于实验记录)。"""
    c = {k: v for k, v in cfg.items() if not k.startswith('_')}
    return yaml.safe_dump(c, allow_unicode=True, sort_keys=False, default_flow_style=False)
