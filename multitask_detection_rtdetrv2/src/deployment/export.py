# 导出服务：构建并加载任务模型，生成固定尺寸 ONNX，再构建本机 TensorRT 引擎。
"""One export service using the configured adapter and input shape."""
import importlib
import os
import sys
from pathlib import Path
import torch
from src.core.common import image_size,digest,save_json
from src.deployment.weights import load_weights
from src.deployment.transforms import optimize,attention

# 导出包装器将归一化中心点框转为输入图像的像素 xyxy 框，并附加 sigmoid 分数。
class ExportWrapper(torch.nn.Module):
    def __init__(self,model,size):
        super().__init__(); self.model=model
        self.height,self.width=size

    def forward(self,images):
        raw=self.model(images); boxes=raw['pred_boxes'].float()
        xyxy=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),dim=-1)
        xyxy=xyxy*torch.tensor([self.width,self.height,self.width,self.height],device=boxes.device)
        return torch.cat((xyxy,raw['pred_logits'].float().sigmoid()),dim=-1)

# 在指定设备上完成权重折叠后转 CPU 导出，以保持已验证的浮点计算路径。
def export_onnx(cfg,adapter,checkpoint,dest,device='cuda'):
    import onnx  # fail before tracing when optional export dependency is absent
    if cfg['model']['arch'] not in ('rtdetrv2_s','rtdetrv2'):
        raise NotImplementedError('This exporter supports the configured RT-DETRv2 family only')
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True); path=dest/'model.onnx'
    if path.exists(): raise FileExistsError(path)
    model=adapter.build_model(cfg); load_weights(model,checkpoint,cfg)
    # Match the benchmark's floating-point folding device. CPU and CUDA rsqrt
    # can round folded Conv/BN weights differently before FP16 conversion.
    # 保持已验证的 CUDA 折叠顺序：CPU 与 CUDA 的 rsqrt 舍入差异可能经 FP16 转换影响临界分数。
    model.to(device); optimize(model); model.cpu()
    wrapper=ExportWrapper(model,image_size(cfg)).eval()
    # 临时替换为可导出的注意力表达式，finally 中恢复全局函数，避免影响后续代码。
    original=torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention=attention
    try:
        torch.onnx.export(wrapper,(torch.zeros(1,3,*image_size(cfg)),),str(path),
            input_names=['images'],output_names=['predictions'],opset_version=16,do_constant_folding=True)
    finally: torch.nn.functional.scaled_dot_product_attention=original
    onnx.checker.check_model(path.read_bytes())
    save_json(dest/'export.json',dict(task_id=cfg['task']['id'],algorithm=cfg['model']['arch'],
        input_shape=[1,3,*image_size(cfg)],
        output_contract='pixel xyxy + sigmoid focal score for each class',
        weight_sha256=digest(checkpoint),onnx_sha256=digest(path),
        folding_device=str(device),
        transforms=['deploy branch fusion','Conv BN/FrozenBN fusion','top100 before encoder box head']))
    return path


def _force_layernorm_fp32(trt,network):
    """Match the source R18VD accuracy mode for decomposed LayerNorm graphs."""
    markers=('/norm1/','/norm2/','/norm3/','/enc_output/norm/')
    constrained=0
    for index in range(network.num_layers):
        layer=network.get_layer(index); name=str(getattr(layer,'name','')).lower()
        if not any(marker in name for marker in markers):
            continue
        layer.precision=trt.float32
        for output_index in range(layer.num_outputs): layer.set_output_type(output_index,trt.float32)
        constrained+=1
    if not constrained:
        raise RuntimeError('Could not locate Transformer LayerNorm layers for requested FP32 accuracy mode')
    return constrained

# 从 ONNX 构建指定精度引擎，并记录权重、ONNX、引擎指纹及构建 GPU。
def build_engine(cfg,checkpoint,dest,precision='fp16'):
    root=cfg['deployment'].get('trt_library_root',''); handles=[]
    if root:
        if root not in sys.path: sys.path.insert(0,root)
        for rel in ('tensorrt_libs','nvidia/cuda_runtime/bin'):
            folder=Path(root)/rel
            if folder.exists():
                os.environ['PATH']=str(folder)+os.pathsep+os.environ['PATH']
                if hasattr(os,'add_dll_directory'): handles.append(os.add_dll_directory(str(folder)))
    try: trt=importlib.import_module('tensorrt')
    except ImportError: trt=importlib.import_module('tensorrt_bindings')
    dest=Path(dest); onnx_path=dest/'model.onnx'; path=dest/('model_'+precision+'.engine')
    if path.exists(): raise FileExistsError(path)
    logger=trt.Logger(trt.Logger.WARNING); builder=trt.Builder(logger)
    # TensorRT 10 only accepts ONNX models through an explicit-batch network.
    # The exported shape remains fixed at batch=1 for every existing task.
    flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network=builder.create_network(flags)
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx_path.read_bytes()): raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    # 关闭 TF32；选择 fp16 时允许 TensorRT 使用半精度算子，具体实现由构建器选择。
    options=builder.create_builder_config(); options.clear_flag(trt.BuilderFlag.TF32)
    layernorm_fp32=False; constrained=0
    if precision=='fp16':
        options.set_flag(trt.BuilderFlag.FP16)
        # This option is opt-in. Existing branches retain their previous build
        # settings; the imported R18VD configuration explicitly enables it.
        layernorm_fp32=bool(cfg.get('deployment',{}).get('trt_layernorm_fp32',False))
        if layernorm_fp32:
            obey=getattr(trt.BuilderFlag,'OBEY_PRECISION_CONSTRAINTS',None)
            if obey is None: raise RuntimeError('TensorRT cannot enforce requested FP32 LayerNorm constraints')
            constrained=_force_layernorm_fp32(trt,network); options.set_flag(obey)
    workspace_mib=int(cfg.get('deployment',{}).get('trt_workspace_mib',256))
    options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,workspace_mib*1024*1024); options.builder_optimization_level=3
    # timing cache 复用算子策略的测量结果，加速再次构建；模型参数仍来自当前 ONNX。
    cache_path=dest/'timing.cache'
    cache=options.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
    options.set_timing_cache(cache,False)
    plan=builder.build_serialized_network(network,options)
    if plan is None: raise RuntimeError('TensorRT engine build failed')
    path.write_bytes(bytes(plan))
    cache_path.write_bytes(bytes(options.get_timing_cache().serialize()))
    # 将 GPU 和文件指纹随引擎保存，推理加载时据此校验来源。
    save_json(path.with_suffix('.build.json'),dict(task_id=cfg['task']['id'],trt=trt.__version__,
        gpu=torch.cuda.get_device_name(0),weight_sha256=digest(checkpoint),engine_sha256=digest(path),
        onnx_sha256=digest(onnx_path),precision=precision,
        layernorm_fp32=layernorm_fp32,layernorm_fp32_layers=constrained,workspace_mib=workspace_mib))
    return path
