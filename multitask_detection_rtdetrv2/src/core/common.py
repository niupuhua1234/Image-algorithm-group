# 公共工具：统一尺寸读取、文件指纹和 JSON 写入，供训练、评估、部署共同调用。
# 任务配置中的尺寸使用 [高, 宽]；PIL 和原图坐标还原接口使用 (宽, 高)。
"""Task-neutral configuration, IO, provenance and shape helpers."""
import json
import hashlib
from pathlib import Path

# 每个任务必须声明自己的输入尺寸，整数形式表示正方形，二元形式按高、宽读取。
def image_size(cfg):
    size=cfg.get('model',{}).get('input_size')
    if size is None: size=cfg.get('train',{}).get('input_size')
    if size is None: raise ValueError('Each task must declare its own input size')
    if isinstance(size,int): size=[size,size]
    if len(size)!=2 or min(size)<=0: raise ValueError('Input size must be [height,width]')
    return tuple(int(x) for x in size)

# 分块计算 SHA256，用于核对权重、引擎和标注文件的来源是否一致。
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

# 先写临时文件再替换目标文件；分母无定义等情况使用 None，避免写入非标准 NaN。
def save_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)
