# 统一训练入口：选择 --task/--config 后，通过任务适配器启动相应训练流程。
# --dry-run 执行一次真实前向和反向；--max-steps、--val-limit 仅用于冒烟检查。
"""Common training entry; business behavior comes from the selected adapter."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.engine.application import parser,setup
from src.core.config import dump_resolved,load_config


def _is_infrared_recognition(argv):
    """Route the infrared task to its existing training lifecycle."""
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--task')
    probe.add_argument('--config')
    args,_=probe.parse_known_args(argv)
    if args.task=='infrared_recognition':
        return True
    return bool(args.config and load_config(args.config)['task']['id']=='infrared_recognition')

# 命令行只负责装配参数，训练行为通过适配器分派到公共调度。
def main():
    if _is_infrared_recognition(sys.argv[1:]):
        from experiments.infrared_recognition.train import main as infrared_main
        return infrared_main()
    p=parser('Unified detection training')
    p.add_argument('--name',default='run')
    p.add_argument('--resume',default=None)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--max-steps',type=int,default=None,help='SMOKE ONLY: limit steps per epoch')
    p.add_argument('--val-limit',type=int,default=None,help='SMOKE ONLY: limit validation')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--show-config',action='store_true')
    p.add_argument('--initialize',default=None)
    a=p.parse_args(); cfg,adapter=setup(a)
    if a.show_config: print(dump_resolved(cfg)); return
    # --initialize 只提供训练起始参数，--resume 则从完整断点恢复训练状态。
    if a.initialize: cfg['train']['initialize']=a.initialize
    return adapter.run_training(cfg,a)

if __name__=='__main__': main()
