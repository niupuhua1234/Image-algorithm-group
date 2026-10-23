# 统一运行时通过组合连接“业务适配器”和“推理后端”，分别处理业务差异与执行方式。
# 对外接收 RGB 图像并返回原图像素坐标的 Detection，尺寸取自当前任务配置。
"""One runtime for all detection tasks; shape and postprocess stay task-configured."""
import time
import numpy as np
import torch
from PIL import Image
from src.core.common import image_size
from src.core.config import resolve_path
from src.deployment.backends import BACKENDS

# 同一类服务不同任务，实际输入尺寸与后处理分别由配置和适配器决定。
class DetectionRuntime:
    def __init__(self,cfg,adapter,checkpoint=None,device='cuda',backend='pytorch'):
        if backend not in BACKENDS: raise ValueError('Unsupported backend '+backend)
        self.cfg=cfg; self.adapter=adapter.bind(cfg); self.device=device
        self.size=image_size(cfg); self.checkpoint=resolve_path(cfg,checkpoint or cfg['deployment']['checkpoint'])
        # 后端通过共同接口替换，业务适配器继续控制后处理和结果定义。
        self.backend=BACKENDS[backend](cfg,adapter,self.checkpoint,device)
        self.backend_name=backend

    def synchronize(self):
        if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)

    # 内存图像接口：记录原图宽高，按任务尺寸缩放，再调用后端和业务后处理。
    @torch.inference_mode()
    def infer_rgb(self,pixels,conf=None):
        if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3: raise ValueError('Expected RGB uint8 HWC input')
        height,width=pixels.shape[:2]
        if (height,width)!=self.size:
            pixels=np.asarray(Image.fromarray(pixels).resize((self.size[1],self.size[0]),Image.Resampling.BILINEAR)).copy()
        raw=self.backend.predict(pixels)
        # 通过适配器多态还原原图坐标；这里传原图 (宽, 高)，不是缩放后的网络输入尺寸。
        return self.adapter.postprocess(raw,(width,height),conf)

    # 文件便捷入口包含图片解码；性能计时使用 infer_rgb 的内存输入边界。
    def infer(self,path,conf=None):
        with Image.open(path) as image: pixels=np.asarray(image.convert('RGB')).copy()
        return self.infer_rgb(pixels,conf)

    # 先预热，再逐帧同步计时；包含预处理、传输、网络、后处理和 CPU 结果返回。
    def benchmark(self,images,runs=200,warmup=10,conf=None):
        if not images or runs<=0: raise ValueError('Benchmark requires images and positive runs')
        # 预热不进入统计，正式计时前后同步 CUDA，避免异步提交造成耗时偏小。
        for i in range(warmup): self.infer_rgb(images[i%len(images)],conf)
        self.synchronize(); values=[]
        if str(self.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(self.device)
        for i in range(runs):
            self.synchronize(); start=time.perf_counter(); self.infer_rgb(images[i%len(images)],conf)
            self.synchronize(); values.append((time.perf_counter()-start)*1000)
        # 此处为本机单模型测速；PyTorch 分配器峰值不包含全部 TensorRT/CUDA 外部分配。
        return dict(mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),
            max_ms=float(max(values)),samples=runs,over_45_ms=sum(x>45 for x in values),raw_ms=values,
            peak_allocated_mib=torch.cuda.max_memory_allocated(self.device)/1024**2 if str(self.device).startswith('cuda') else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved(self.device)/1024**2 if str(self.device).startswith('cuda') else None,
            backend_resources=self.backend.resource_snapshot(),
            resource_scope='PyTorch allocator; external TensorRT/CUDA context allocations are not fully included',
            input_size_hw=list(self.size),backend=self.backend_name,
            boundary='decoded CPU RGB uint8 -> preprocessing, H2D, model, decode, D2H, threshold/NMS, CPU Detection',
            excluded=['disk IO/image decoding','initialization','warmup','metric matching'],
            gpu=torch.cuda.get_device_name(self.device) if str(self.device).startswith('cuda') else 'CPU',
            concurrent_models=False,official_orin_three_model_verified=False)
