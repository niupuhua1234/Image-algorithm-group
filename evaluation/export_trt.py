# -*- coding: utf-8 -*-
"""在新机器上导出 TensorRT engine(必须在目标机器执行, engine 与 GPU 架构绑定)。

用法: python export_trt.py --imgsz 1024
"""
import argparse
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent          # evaluation/
ROOT = BASE.parent                              # 仓库根目录
DEIM = ROOT / 'engines' / 'deim'
sys.path.insert(0, str(DEIM))
os.chdir(str(DEIM))

import torch
from engine.core import YAMLConfig

ap = argparse.ArgumentParser()
ap.add_argument('--imgsz', type=int, default=1024)
ap.add_argument('--batch', type=int, default=1)
args = ap.parse_args()

import patch_tw
patch_tw.apply()

cfg = YAMLConfig('configs/deim_rtdetrv2/deim_car_union_v4.yml')
model = cfg.model
model.encoder.eval_spatial_size = None
model.decoder.eval_spatial_size = None
ck = torch.load(str(ROOT / 'weights' / 'best_stg1.pth'), map_location='cpu')
sd = ck['ema']['module'] if isinstance(ck.get('ema'), dict) and 'module' in ck['ema'] else ck.get('model', ck)
msd = model.state_dict()
model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
model.eval()

B, S = args.batch, args.imgsz
onnx_p = str(BASE / f'deim_v4_{S}_b{B}.onnx')
x = torch.zeros((B, 3, S, S))
# 关键: 完全定形(batch 与 H/W 固定, 无 dynamic_axes) → Gather 常量折叠 → TRT 可编译
torch.onnx.export(model, (x,), onnx_p, input_names=['images'],
                  output_names=['pred_logits', 'pred_boxes'],
                  opset_version=18, dynamo=False, do_constant_folding=True)
print(f'ONNX: {onnx_p} ({os.path.getsize(onnx_p)/1e6:.1f} MB)')

import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(onnx_p, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print('parse err:', parser.get_error(i))
        sys.exit(1)
cfg_b = builder.create_builder_config()
t0 = time.time()
serialized = builder.build_serialized_network(network, cfg_b)
if serialized is None:
    print('TRT 构建失败')
    sys.exit(1)
eng_p = str(BASE / f'deim_v4_{S}_b{B}.engine')
with open(eng_p, 'wb') as f:
    f.write(serialized)
print(f'engine: {eng_p} ({os.path.getsize(eng_p)/1e6:.1f} MB, 构建 {time.time()-t0:.0f}s)')

# 时延实测
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
ctx = engine.create_execution_context()
d_img = torch.zeros((B, 3, S, S), device='cuda').contiguous()
d_logits = torch.empty((B, 300, 6), device='cuda')
d_boxes = torch.empty((B, 300, 4), device='cuda')
stream = torch.cuda.current_stream().cuda_stream
for nm, t in [('images', d_img), ('pred_logits', d_logits), ('pred_boxes', d_boxes)]:
    try:
        ctx.set_tensor_address(nm, t.data_ptr())
    except Exception as e:
        print(f'set_tensor_address({nm}): {str(e)[:60]}')

for _ in range(15):
    ctx.execute_async_v3(stream)
torch.cuda.synchronize()
ts = []
for _ in range(60):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)
ts.sort()
print(f'TRT @{S} batch={B}: avg={sum(ts)/len(ts):.2f}ms  med={ts[len(ts)//2]:.2f}ms  p95={ts[int(len(ts)*0.95)]:.2f}ms')
print(f'参考: 原机器 TRT @1024 batch=1 = 5.5ms')
