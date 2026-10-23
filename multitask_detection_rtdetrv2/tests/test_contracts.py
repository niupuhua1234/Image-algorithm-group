import copy
import unittest
import numpy as np
from src.core.config import load_config
from src.core.common import image_size
from src.core.registry import load_all_adapters,get_adapter
from src.models.adapters.base import Detection
from src.metrics.small_target import evaluate_small_target,select_operating_point
from src.deployment.postprocess import nms

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_all_adapters(); cls.cfg=load_config('infrared_small_target')

    def test_task_specific_dimensions_and_infrared_recognition_registration(self):
        self.assertEqual(image_size(self.cfg),(1024,1280))
        self.assertEqual(image_size(load_config('visible_recognition')),(960,960))
        infrared=get_adapter('infrared_recognition')()
        cfg=load_config('infrared_recognition')
        self.assertEqual(infrared.describe()['status'],'implemented')
        self.assertEqual(infrared.class_names,['car1','car2','car3','car4'])
        self.assertEqual(image_size(cfg),(1024,1280))

    def test_all_three_tasks_share_detr_parent(self):
        from src.models.adapters.detr_task import DetrTaskAdapter
        for task in ('infrared_small_target', 'infrared_recognition', 'visible_recognition'):
            with self.subTest(task=task):
                self.assertTrue(issubclass(get_adapter(task), DetrTaskAdapter))

    def test_iou_boundary_duplicate_and_negative(self):
        p=Detection(np.array([[0,0,4,10],[0,0,4,10]],np.float32),np.array([.9,.8]),np.zeros(2))
        g=dict(boxes=np.array([[0,0,10,10]],np.float32),labels=np.array([0]))
        m=evaluate_small_target([p,Detection.empty()],[g,dict(boxes=np.zeros((0,4)))],self.cfg['evaluation'])
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,0)); self.assertEqual(m['fa'],.5)

    def test_empty_denominators_and_size_boundaries(self):
        m=evaluate_small_target([Detection.empty()],[dict(boxes=np.zeros((0,4)))],self.cfg['evaluation'])
        self.assertIsNone(m['fa']); self.assertIsNone(m['pd'])
        m=evaluate_small_target([Detection.empty()],[dict(boxes=np.array([[0,0,8,8]]))],self.cfg['evaluation'])
        self.assertEqual(m['size_buckets']['small']['fn'],1)
        self.assertEqual(m['size_buckets']['tiny']['targets'],0)

    def test_nms_keeps_higher_confidence(self):
        rows=np.array([[0,0,10,10,.4],[0,0,10,10,.8],[20,20,30,30,.7]],np.float32)
        result=nms(rows,.05); self.assertEqual(len(result),2); self.assertAlmostEqual(float(result[0,4]),.8,6)

    def test_validation_operating_point(self):
        adapter=get_adapter('infrared_small_target')().bind(self.cfg)
        p=Detection(np.array([[0,0,10,10],[20,20,30,30]],np.float32),np.array([.8,.2],np.float32),np.zeros(2))
        selected=select_operating_point([dict(predictions=p,ground_truth=dict(boxes=np.array([[0,0,10,10]])))],adapter,self.cfg)
        self.assertAlmostEqual(selected['confidence'],.8,6); self.assertEqual(selected['metrics']['fp'],0)

    def test_algorithm_family_guard(self):
        from src.models.factory import components
        cfg=copy.deepcopy(self.cfg); cfg['loss']['family']='yolo'
        with self.assertRaises(ValueError): components(cfg)

    def test_shared_runtime_preserves_task_shape_and_original_size(self):
        from src.deployment.backends import InferenceBackend
        from src.deployment.unified_runtime import DetectionRuntime
        class FakeBackend(InferenceBackend):
            def predict(self,pixels):
                self.shape=pixels.shape
                return 'raw'
        class FakeAdapter:
            def postprocess(self,raw,size,conf):
                self.size=size
                return Detection.empty()
        runtime=object.__new__(DetectionRuntime)
        runtime.size=(320,640); runtime.backend=FakeBackend(); runtime.adapter=FakeAdapter()
        runtime.infer_rgb(np.zeros((1024,1280,3),np.uint8))
        self.assertEqual(runtime.backend.shape,(320,640,3))
        self.assertEqual(runtime.adapter.size,(1280,1024))

    def test_shared_detr_decode_restores_rectangular_coordinates(self):
        import torch
        from src.deployment.postprocess import decode_detr
        raw=dict(pred_boxes=torch.tensor([[[.5,.5,.1,.2]]]),pred_logits=torch.tensor([[[10.]]]))
        pred=decode_detr(raw,(1280,1024),self.cfg,.5)
        np.testing.assert_allclose(pred.boxes[0],[576,409.6,704,614.4],atol=1e-4)

    def test_multiclass_tensorrt_decode_matches_focal_topk_contract(self):
        from src.deployment.postprocess import decode_detr
        cfg=load_config('infrared_recognition')
        raw=np.array([[100,200,300,400,.1,.9,.4,.2]],np.float32)
        pred=decode_detr(raw,(1280,1024),cfg,0.)
        self.assertEqual(pred.labels.tolist(),[1,2,3,0])
        np.testing.assert_allclose(pred.scores,[.9,.4,.2,.1],atol=1e-6)
        np.testing.assert_allclose(pred.boxes[0],[100,200,300,400],atol=1e-6)

if __name__=='__main__': unittest.main()
