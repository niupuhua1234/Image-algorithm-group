# 统一推理入口：--image 检测单图，--bench 测量预热后的平均单帧接口耗时。
"""Common inference entry; task size is resolved from its own configuration."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.engine.application import parser,setup
from src.core.common import save_json
from src.data.coco import CocoBoxDataset

# 单图结果和测速结果可写入同一个 JSON；测速图片在计时前完成解码。
def main():
    p=parser('Unified task inference')
    p.add_argument('--image',default=None)
    p.add_argument('--bench',type=int,default=0)
    p.add_argument('--out',default=None)
    a=p.parse_args(); cfg,adapter=setup(a)
    if not a.image and not a.bench: p.error('Provide --image and/or --bench')
    runtime=adapter.create_runtime(cfg,a.ckpt,a.device,a.backend); result={'task_id':cfg['task']['id']}
    if a.image: result['detections']=runtime.infer(a.image,a.conf).to_dict(adapter.class_names)
    if a.bench:
        data=CocoBoxDataset(cfg,'test'); images=[data.pixels(i) for i in range(min(20,len(data)))]
        result['timing']=runtime.benchmark(images,a.bench,conf=a.conf)
    if a.out: save_json(a.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
