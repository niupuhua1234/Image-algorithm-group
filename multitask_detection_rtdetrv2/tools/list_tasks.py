# 任务清单入口：展示已实现、交接待接入和预留任务，便于检查统一工程的扩展位置。
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core.config import load_config
from src.core.registry import load_all_adapters,available_tasks,get_adapter
from src.core.common import image_size

if __name__=='__main__':
    load_all_adapters()
    for task in available_tasks():
        cfg=load_config(task['task_id']); adapter=get_adapter(task['task_id'])()
        data=adapter.describe()
        # 预留任务可暂不声明尺寸，此时显示 null，接入后由自己的配置补齐。
        try: data['input_size_hw']=image_size(cfg)
        except ValueError: data['input_size_hw']=None
        print(json.dumps(data,ensure_ascii=False))
