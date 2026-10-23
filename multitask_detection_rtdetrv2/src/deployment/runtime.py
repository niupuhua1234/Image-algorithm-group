# -*- coding: utf-8 -*-
"""统一推理运行时 —— 三种后端 + 完整业务时延测量。

技术路线书 3.4 节:
    "模型时延需在目标硬件上测量, 不能由网络类别、参数量或论文 FPS 单独推断。
     本项目统一记录训练吞吐、模型推理时延、**完整业务时延**和显存占用。"

因此本模块区分三个口径:
    forward      纯模型前向(GPU 上, 不含读图与预处理)
    preprocess   读图 + 解码 + resize + 归一化 + 传输
    end_to_end   完整业务时延 = preprocess + forward   <- 路线书要求的那个

后端:
    pytorch      eager 前向。每次前向提交约 600+ 个 CUDA kernel, 在 Windows WDDM 下
                 CPU 提交开销常成为瓶颈(GPU 利用率可能只有 20%)。
    cuda_graph   把上述提交压成一次重放。实测 forward 40ms -> 7.7ms, 且数值完全一致。
    tensorrt     需先在目标机器导出 engine(与显卡架构绑定)。

⚠️ 关键前置(否则 CUDA Graph 捕获会失败):
    `eval_spatial_size=None` 时, 引擎每帧都会重建 anchors 并产生一次 CPU->GPU 拷贝。
    `prepare_static()` 把位置相关张量一次性预生成到 GPU, 既省时间也让捕获成立。
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.io import load_image_tensor


class InferenceRuntime:
    """任务无关的推理运行时。"""

    def __init__(self, model, adapter, cfg: Dict, imgsz: int = 1024,
                 device: str = 'cuda', backend: str = 'pytorch'):
        self.model = model
        self.adapter = adapter
        self.cfg = cfg
        self.imgsz = int(imgsz)
        self.device = device
        self.backend = backend
        self._graph = None
        self._static_x = None
        self._static_out = None
        self._static_ready = False

    # ------------------------------------------------------------------ #
    # 静态化: 消除每帧的 CPU->GPU 拷贝, 让 CUDA Graph 可捕获
    # ------------------------------------------------------------------ #
    def prepare_static(self, verbose: bool = True) -> int:
        """把位置相关张量预生成到 GPU。返回处理的张量个数。"""
        n = 0
        enc = getattr(self.model, 'encoder', None)
        dec = getattr(self.model, 'decoder', None)
        S = self.imgsz

        # encoder: pos_embed 用 setattr 存为普通属性, .cuda() 不会搬
        if enc is not None and hasattr(enc, 'use_encoder_idx'):
            for idx in enc.use_encoder_idx:
                stride = enc.feat_strides[idx]
                if hasattr(enc, 'build_2d_sincos_position_embedding'):
                    pe = enc.build_2d_sincos_position_embedding(
                        S // stride, S // stride, enc.hidden_dim)
                    setattr(enc, f'pos_embed{idx}', pe.to(self.device))
                    n += 1
            enc.eval_spatial_size = [S, S]

        # decoder: anchors / valid_mask 用 device 参数直接生成到 GPU(不走 CPU 中转)
        if dec is not None and hasattr(dec, '_generate_anchors'):
            strides = getattr(dec, 'feat_strides', [8, 16, 32])
            shapes = torch.as_tensor([(S // s, S // s) for s in strides])
            anchors, valid_mask = dec._generate_anchors(shapes, device=self.device)
            dec.register_buffer('anchors', anchors)
            dec.register_buffer('valid_mask', valid_mask)
            dec.eval_spatial_size = [S, S]
            n += 2

        if verbose:
            print(f'[runtime] prepare_static: 预生成 {n} 个张量到 {self.device}')
        return n

    # ------------------------------------------------------------------ #
    # CUDA Graph
    # ------------------------------------------------------------------ #
    def capture(self, verbose: bool = True):
        """捕获 CUDA Graph。要求输入尺寸固定、且已 prepare_static。"""
        if self.device != 'cuda':
            raise RuntimeError('CUDA Graph 需要 cuda 设备')
        if not self._static_ready:
            self.prepare_static(verbose=verbose)
        S = self.imgsz
        self._static_x = torch.zeros(1, 3, S, S, device=self.device)
        with torch.no_grad():
            for _ in range(5):
                self.model(self._static_x)                  # 预热
            torch.cuda.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    self.model(self._static_x)
            torch.cuda.current_stream().wait_stream(st)
            with torch.cuda.graph(self._graph):
                self._static_out = self.model(self._static_x)
        torch.cuda.synchronize()
        self._static_ready = True
        self.backend = 'cuda_graph'
        if verbose:
            print(f'[runtime] CUDA Graph 捕获完成 (imgsz={S})')
        return self

    def _forward(self, x):
        if self.backend == 'cuda_graph':
            if self._graph is None:
                self.capture(verbose=False)
            self._static_x.copy_(x)
            self._graph.replay()
            return self._static_out
        with torch.no_grad():
            return self.model(x)

    # ------------------------------------------------------------------ #
    # 三口径计时
    # ------------------------------------------------------------------ #
    def infer(self, image_path, conf: Optional[float] = None):
        """完整业务路径: 读图 -> 预处理 -> 前向 -> 后处理 -> Detection。"""
        conf = (self.cfg.get('deployment', {}).get('conf_threshold', 0.005)
                if conf is None else conf)
        x, orig = load_image_tensor(image_path, self.imgsz, self.device)
        raw = self._forward(x)
        return self.adapter.postprocess(raw, orig, conf)

    def benchmark(self, image_paths: List[Path], warmup: int = 10,
                  runs: int = 50, conf: Optional[float] = None,
                  verbose: bool = True) -> Dict:
        """分别测量 forward / preprocess / end_to_end 三条时延。"""
        paths = list(image_paths)
        if not paths:
            raise ValueError('image_paths 为空')
        conf = (self.cfg.get('deployment', {}).get('conf_threshold', 0.005)
                if conf is None else conf)

        # 预热(含 CUDA 上下文与 cuDNN 选核)
        for i in range(warmup):
            x, orig = load_image_tensor(paths[i % len(paths)], self.imgsz, self.device)
            self._forward(x)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        t_fwd, t_pre, t_e2e = [], [], []
        for i in range(runs):
            p = paths[i % len(paths)]
            if self.device == 'cuda':
                torch.cuda.synchronize()
            a = time.perf_counter()
            x, orig = load_image_tensor(p, self.imgsz, self.device)
            b = time.perf_counter()
            raw = self._forward(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            c = time.perf_counter()
            self.adapter.postprocess(raw, orig, conf)
            d = time.perf_counter()
            t_pre.append((b - a) * 1000)
            t_fwd.append((c - b) * 1000)
            t_e2e.append((d - a) * 1000)

        def stat(v):
            v = sorted(v)
            return {'mean': round(float(np.mean(v)), 2),
                    'median': round(float(v[len(v) // 2]), 2),
                    'min': round(float(v[0]), 2),
                    'p95': round(float(v[int(len(v) * 0.95)]), 2)}

        peak = (torch.cuda.max_memory_allocated() / 1e6) if self.device == 'cuda' else 0.0
        out = {
            'backend': self.backend,
            'imgsz': self.imgsz,
            'device': self.device,
            'device_name': (torch.cuda.get_device_name(0) if self.device == 'cuda' else 'cpu'),
            'forward_ms': stat(t_fwd),
            'preprocess_ms': stat(t_pre),
            'end_to_end_ms': stat(t_e2e),
            'peak_memory_mb': round(peak, 1),
            'runs': runs,
        }
        if verbose:
            print('=' * 70)
            print(f'时延测量  后端={self.backend}  imgsz={self.imgsz}  设备={out["device_name"]}')
            print('=' * 70)
            for k, label in (('forward_ms', '纯模型 forward'),
                             ('preprocess_ms', '预处理(读图+resize+传输)'),
                             ('end_to_end_ms', '**完整业务时延**')):
                s = out[k]
                print(f'  {label:<26} mean {s["mean"]:>7.2f}  median {s["median"]:>7.2f}'
                      f'  min {s["min"]:>7.2f}  p95 {s["p95"]:>7.2f} ms')
            print(f'  峰值显存 {out["peak_memory_mb"]:.0f} MB')
        return out


def build_runtime(cfg: Dict, adapter, ckpt_path, imgsz: int = None,
                  device: str = 'cuda', backend: str = 'pytorch',
                  verbose: bool = True) -> InferenceRuntime:
    """便捷构造: 建模型 -> 载权重 -> (可选)预生成 -> (可选)捕获 CUDA Graph。"""
    from src.deployment.checkpoint import load_ckpt_into

    imgsz = imgsz or cfg.get('deployment', {}).get('default_imgsz', 1024)
    model = adapter.build_model(cfg)
    model = load_ckpt_into(model, ckpt_path, device, verbose=verbose)
    rt = InferenceRuntime(model, adapter, cfg, imgsz, device, backend)
    if backend == 'cuda_graph':
        rt.prepare_static(verbose=verbose)
        rt.capture(verbose=verbose)
    return rt
