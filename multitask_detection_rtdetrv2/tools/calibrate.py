# 验证集工作点选择入口：在预设 Pd 约束下寻找较低 Fa 的置信度，并保存冻结配置。
# 工作点与所用权重、验证标注、推理后端一并记录，后续测试直接使用该配置。
"""Freeze an operating point using validation only, never test."""
import sys
import os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.engine.application import parser,setup
from src.engine.evaluation import collect
from src.core.common import save_json,digest
import yaml

if __name__=='__main__':
    p=parser('Select and freeze a validation operating point')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--pd-target',type=float,default=None)
    a=p.parse_args(); cfg,adapter=setup(a)
    if a.pd_target is not None:
        if not 0<a.pd_target<=1: p.error('--pd-target must be in (0,1]')
        cfg['evaluation']['validation_pd_target']=a.pd_target
    rt=adapter.create_runtime(cfg,a.ckpt,a.device,a.backend)
    # 固定使用 val 收集候选，不读取 test 参与阈值选择。
    preds,gts,_,dataset=collect(rt,cfg,'val',conf=0.)
    records=[dict(predictions=p,ground_truth=g) for p,g in zip(preds,gts)]
    selected=adapter.select_operating_point(records,cfg)
    selected.update(annotation_sha256=digest(dataset.annotation_path),checkpoint_sha256=digest(rt.checkpoint),backend=a.backend)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'operating_point.json').exists(): raise FileExistsError('Calibration output already exists')
    save_json(out/'operating_point.json',selected)
    reference=os.path.relpath(cfg['_config_path'],str(out.resolve()))
    checkpoint=os.path.relpath(str(rt.checkpoint),cfg['_project_root'])
    # 只覆盖权重身份和所选置信度，其余模型、尺寸及后处理配置继续继承原任务。
    frozen=dict(defaults=[reference],deployment=dict(checkpoint=checkpoint,
        checkpoint_sha256=digest(rt.checkpoint),conf_threshold=selected['confidence']))
    (out/'frozen_task.yaml').write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(str(out/'frozen_task.yaml'))
