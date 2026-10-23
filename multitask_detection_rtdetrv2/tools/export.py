# 统一导出入口：先用 --action onnx 导出模型，再用 --action build 构建引擎。
# 两步使用同一输出目录，输入尺寸和权重来自所选任务配置。
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.engine.application import parser,setup
from src.core.config import resolve_path
from src.deployment.export import export_onnx,build_engine

if __name__=='__main__':
    p=parser('Export configured detection model')
    p.add_argument('--action',choices=['onnx','build'],default='onnx')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp16')
    a=p.parse_args(); cfg,adapter=setup(a)
    checkpoint=resolve_path(cfg,a.ckpt or cfg['deployment']['checkpoint'])
    if a.action=='onnx': print(export_onnx(cfg,adapter,checkpoint,a.output_dir,a.device))
    else: print(build_engine(cfg,checkpoint,a.output_dir,a.precision))
