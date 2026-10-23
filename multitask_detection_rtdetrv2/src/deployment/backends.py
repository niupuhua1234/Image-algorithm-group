# 推理后端多态：运行时统一调用 predict，PyTorch 和 TensorRT 子类负责各自执行细节。
# 只有选择 TensorRT 后端时才导入其可选依赖。
"""Backend polymorphism; task code never imports TensorRT unless selected."""
from abc import ABC,abstractmethod
import importlib
import os
import sys
from pathlib import Path
import numpy as np
import torch
from src.core.common import image_size,digest
from src.core.config import resolve_path
from src.deployment.weights import load_weights

# 抽象父类只约定 predict 接口，统一运行时无需了解具体后端内部实现。
class InferenceBackend(ABC):
    @abstractmethod
    def predict(self,pixels): pass

    def resource_snapshot(self):
        return {}

# PyTorch 子类负责建模、严格加载权重和执行前向，返回模型原始字典。
class PyTorchBackend(InferenceBackend):
    def __init__(self,cfg,adapter,checkpoint,device):
        self.device=device
        self.model=adapter.build_model(cfg)
        load_weights(self.model,checkpoint,cfg)
        self.model.to(device).eval()

    # 把 HWC、0～255 的 RGB 转成 CHW、0～1 的浮点张量，与训练预处理保持一致。
    @torch.inference_mode()
    def predict(self,pixels):
        # Same multiplication and RGB values as the original R18 recipe.
        x=torch.from_numpy(pixels).permute(2,0,1).contiguous().float().mul_(1./255).unsqueeze(0)
        return self.model(x.to(self.device))

    def resource_snapshot(self):
        return dict(parameters=sum(p.numel() for p in self.model.parameters()))

# TensorRT 子类复用固定缓冲区与 CUDA 流，返回单类像素框和分数数组。
class TensorRTBackend(InferenceBackend):
    def __init__(self,cfg,adapter,checkpoint,device):
        if not str(device).startswith('cuda'): raise ValueError('TensorRT requires CUDA')
        root=cfg['deployment'].get('trt_library_root',''); self.handles=[]
        if root:
            root=Path(root)
            if str(root) not in sys.path: sys.path.insert(0,str(root))
            for rel in ('tensorrt_libs','nvidia/cuda_runtime/bin'):
                d=root/rel
                if d.exists():
                    os.environ['PATH']=str(d)+os.pathsep+os.environ['PATH']
                    if hasattr(os,'add_dll_directory'): self.handles.append(os.add_dll_directory(str(d)))
        try: trt=importlib.import_module('tensorrt')
        except ImportError: trt=importlib.import_module('tensorrt_bindings')
        import json
        dep=cfg['deployment']; path=resolve_path(cfg,dep['engine']); meta_path=resolve_path(cfg,dep['engine_metadata'])
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        # 引擎必须对应所选权重及当前 GPU；模型更新或 GPU 改变后需要重新构建。
        if meta['engine_sha256']!=digest(path) or meta['weight_sha256']!=digest(checkpoint):
            raise ValueError('Engine/checkpoint provenance mismatch')
        if meta.get('gpu')!=torch.cuda.get_device_name(device):
            raise ValueError('Rebuild this engine for the selected GPU')
        self.logger=trt.Logger(trt.Logger.WARNING); self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None: raise RuntimeError('Could not deserialize TensorRT engine')
        # 初始化时一次性建立上下文与输入输出缓冲区，逐帧推理复用这些对象。
        self.context=self.engine.create_execution_context(); self.buffers={}
        types={trt.float32:torch.float32,trt.float16:torch.float16}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i); shape=tuple(self.engine.get_tensor_shape(name))
            self.buffers[name]=torch.empty(shape,dtype=types[self.engine.get_tensor_dtype(name)],device=device)
            self.context.set_tensor_address(name,self.buffers[name].data_ptr())
        h,w=image_size(cfg)
        if tuple(self.buffers['images'].shape)!=(1,3,h,w): raise ValueError('Engine input size differs from this task')
        # 锁页内存配合 non_blocking 传输；先传 uint8 图像，再在 GPU 上转换布局和归一化。
        self.host_pixels=torch.empty((h,w,3),dtype=torch.uint8,pin_memory=True)
        self.gpu_pixels=torch.empty_like(self.host_pixels,device=device)
        shape=tuple(self.buffers['predictions'][0].shape)
        self.host_predictions=torch.empty(shape,dtype=torch.float32,pin_memory=True)
        self.stream=torch.cuda.Stream(device=device); self.stream.wait_stream(torch.cuda.current_stream(device))

    # 在同一 CUDA 流中依次传输、预处理、执行引擎、回传结果，返回前等待全部完成。
    @torch.inference_mode()
    def predict(self,pixels):
        self.host_pixels.copy_(torch.from_numpy(pixels))
        with torch.cuda.stream(self.stream):
            self.gpu_pixels.copy_(self.host_pixels,non_blocking=True)
            self.buffers['images'][0].copy_(self.gpu_pixels.permute(2,0,1)); self.buffers['images'].mul_(1./255)
            if not self.context.execute_async_v3(self.stream.cuda_stream): raise RuntimeError('TensorRT execution failed')
            self.host_predictions.copy_(self.buffers['predictions'][0],non_blocking=True)
        # 同步完成后才读取 CPU 结果，保证接口返回内容完整，也保证计时包含实际 GPU 执行。
        self.stream.synchronize(); result=self.host_predictions.numpy().copy()
        if not np.isfinite(result).all(): raise FloatingPointError('Non-finite engine output')
        return result

    # 这里报告引擎上下文显存需求；总进程显存还包括其他 CUDA 与运行时分配。
    def resource_snapshot(self):
        return dict(engine_context_device_memory_mib=self.engine.device_memory_size/1024**2,
                    note='Engine context requirement, not total process GPU memory')

# 运行时按名字实例化子类，后续只调用相同的 predict 和 resource_snapshot 接口。
BACKENDS={'pytorch':PyTorchBackend,'tensorrt':TensorRTBackend}
