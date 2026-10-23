import tempfile
from pathlib import Path
import unittest
import torch
from src.core.config import load_config
from src.engine.experiment import ExperimentStore
from src.deployment.weights import load_weights

class Checkpoints(unittest.TestCase):
    def test_unicode_atomic_save_and_task_guard(self):
        cfg=load_config('infrared_small_target')
        with tempfile.TemporaryDirectory() as tmp:
            cfg['experiment']['root']=str(Path(tmp)/'中文实验')
            store=ExperimentStore(cfg,'check')
            model=torch.nn.Linear(2,1)
            state=dict(task_id='infrared_small_target',class_names=['target'],model=model.state_dict())
            store.checkpoint('best.pth',state)
            path=store.path/'weights/best.pth'
            self.assertTrue(path.is_file())
            cfg['deployment']['checkpoint_sha256']=''
            load_weights(model,path,cfg)
            cfg['task']['id']='infrared_recognition'
            with self.assertRaises(ValueError): load_weights(model,path,cfg)

if __name__=='__main__': unittest.main()
