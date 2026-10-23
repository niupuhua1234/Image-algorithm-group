# 统一命令行装配：加载配置、选择任务子类并绑定，再交给公共训练或推理流程。
"""CLI wiring: select a task once and dispatch through the adapter contract."""
import argparse
from src.core.config import load_config,resolve_path,dump_resolved
from src.core.registry import load_all_adapters,get_adapter
from src.core.common import image_size

# 训练、评估、推理共用任务、配置、设备、权重、阈值和后端参数。
def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--task',default=None)
    p.add_argument('--config',default=None)
    p.add_argument('--data-root',default=None)
    p.add_argument('--device',default='cuda')
    p.add_argument('--ckpt','--checkpoint',dest='ckpt',default=None)
    p.add_argument('--conf',type=float,default=None)
    p.add_argument('--backend',choices=['pytorch','tensorrt'],default='pytorch')
    return p

# 用注册表创建具体子类；尚未完成接入的任务在统一入口明确拒绝运行。
def setup(args):
    load_all_adapters()
    cfg=load_config(args.config or args.task or 'infrared_small_target')
    task=cfg['task']['id']
    if args.task and args.task!=task: raise ValueError('CLI task differs from configuration task')
    adapter=get_adapter(task)()
    if adapter.describe()['status']!='implemented':
        raise NotImplementedError('Reserved or handover-only task: '+task+'; see docs/EXTENDING_TASKS.md')
    if args.data_root: cfg['data']['root']=args.data_root; cfg['_data_root']=str(resolve_path(cfg,args.data_root))
    # 显式换权重时清除旧部署文件的 SHA 约束，加载器仍会检查权重内部的任务和类别元信息。
    if args.ckpt:
        cfg['deployment']['checkpoint_sha256']=''  # explicit new checkpoint; internal task/class metadata is still checked
    if args.conf is not None: cfg['deployment']['conf_threshold']=args.conf
    # 模型和部署尺寸必须显式一致；不同任务之间可以使用不同尺寸。
    declared=cfg.get('deployment',{}).get('input_size')
    if declared is not None and tuple(declared)!=image_size(cfg):
        raise ValueError('Deployment input size differs from model task size; configure both explicitly')
    adapter.bind(cfg)
    return cfg,adapter
