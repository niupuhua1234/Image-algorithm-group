# 实验存储类：按任务、算法、阶段和实验名称隔离配置、权重、日志与报告。
"""Independent task/algorithm/run storage with atomic checkpoints."""
import json
import os
from pathlib import Path
import torch
from src.core.common import save_json
from src.core.config import resolve_path,dump_resolved

class ExperimentStore:
    def __init__(self,cfg,name,resume=False):
        if not name or Path(name).name!=name or name in ('.','..'): raise ValueError('Use a simple run name')
        # 同一运行目录只允许显式 resume 继续，避免新实验覆盖已有权重和日志。
        self.path=resolve_path(cfg,cfg['experiment']['root'])/cfg['task']['id']/cfg['model']['arch']/cfg['experiment']['stage']/name
        if self.path.exists() and not resume: raise FileExistsError('Experiment exists: '+str(self.path))
        self.path.mkdir(parents=True,exist_ok=True)
        for part in ('configs','weights','logs','reports'): (self.path/part).mkdir(exist_ok=True)
        if not resume:
            (self.path/'configs/resolved.yaml').write_text(dump_resolved(cfg),encoding='utf-8')
            save_json(self.path/'configs/classes.json',cfg['classes'])

    # 先完整写入并刷盘，再原子替换目标；通过 Python 文件句柄兼容 Windows 中文路径。
    def checkpoint(self,name,payload):
        target=self.path/'weights'/name; temp=target.with_suffix('.tmp')
        # PyTorch 2.0's native path writer can mis-handle Windows non-ASCII paths.
        # A Python file handle preserves Unicode and retains atomic replacement.
        with temp.open('wb') as stream:
            torch.save(payload,stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temp),str(target))

    # 逐轮追加 JSONL 日志，便于查看训练进度及对应的验证工作点。
    def event(self,payload):
        with (self.path/'logs/history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
