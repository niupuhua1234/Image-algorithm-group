# 统一评估入口：读取已经冻结的权重和阈值，输出当前任务定义的指标。
# --limit 仅评估部分图片；正式汇总应使用完整划分。
"""Common evaluation entry; each adapter owns its metric definition."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.engine.application import parser,setup

# 评估入口不搜索置信度；需要选点时先运行 calibrate.py 并冻结配置。
def main():
    p=parser('Unified task evaluation')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--bench',type=int,default=0)
    p.add_argument('--out',default=None)
    p.add_argument('--save-predictions',action='store_true')
    a=p.parse_args(); cfg,adapter=setup(a)
    adapter.run_evaluation(cfg,a)

if __name__=='__main__': main()
