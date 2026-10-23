# -*- coding: utf-8 -*-
"""训练入口。

本工程的职责边界(技术路线书 3.2 / 2.3):
    框架提供**统一的训练入口与调度接口**, 但每种算法仍保留相容的前向输出、
    目标编码与损失计算。

因此这里做两件事:
    ① --dry-run  用适配器构建 模型/损失, 跑一次前向+反向, 验证算法族接口连通
    ② 无 --dry-run 时打印完整训练指引 —— 完整训练(AMP / EMA / flatcosine /
       增强策略按 epoch 调度)复用 third_party 里 vendored 的引擎, 不在框架层重复实现

用法:
    python tools/train.py --task visible_recognition --dry-run
    python tools/train.py --task visible_recognition            # 看训练指引
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch                                              # noqa: E402

from src.core.config import load_config, dump_resolved    # noqa: E402
from src.core.registry import get_adapter, load_all_adapters  # noqa: E402


def build_fake_batch(cfg, num_classes: int, device: str = 'cuda'):
    """构造一个假 batch 用于接口自检。

    形状与真实训练一致: 图像 [B,3,S,S], 标签为 COCO 目标字典列表。
    """
    S = cfg['train'].get('input_size', 960)
    B = 2
    images = torch.rand(B, 3, S, S, device=device)
    labels = []
    for _ in range(B):
        n = 5
        boxes = torch.rand(n, 4, device=device) * 0.5 + 0.25      # cxcywh 归一化
        boxes[:, 2:] *= 0.2
        labels.append({
            'boxes': boxes,
            'labels': torch.randint(0, num_classes, (n,), device=device),
        })
    return images, labels


def dry_run(cfg, adapter, device: str = 'cuda'):
    print('=' * 74)
    print('接口自检 (--dry-run)')
    print('=' * 74)

    print('\n[1/4] 构建模型')
    model = adapter.build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'  参数量 {n_par:,}')

    print('\n[2/4] 构建损失与匹配')
    criterion = adapter.build_criterion(cfg).to(device)
    lc = cfg['loss']
    print(f'  losses = {lc["losses"]}   weight_dict = {lc["weight_dict"]}')

    print('\n[3/4] 前向')
    images, targets = build_fake_batch(cfg, adapter.num_classes, device)
    print(f'  输入 {tuple(images.shape)}   目标 {[t["boxes"].shape[0] for t in targets]} 个/图')
    model.eval()
    with torch.no_grad():
        out = model(images)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f'  输出 {k:<18} {tuple(v.shape)}')

    print('\n[4/4] 损失 + 反向')
    model.train()
    out = model(images, targets)          # 训练态需要 targets(去噪分组要用)
    loss_dict = criterion(out, targets)
    weight_dict = criterion.weight_dict
    total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    for k in sorted(loss_dict):
        print(f'    {k:<26} {float(loss_dict[k]):.4f}  (w={weight_dict.get(k, "-")})')
    print(f'    {"total":<26} {float(total):.4f}')
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f'  反向传播: {n_grad} 个参数拿到非零梯度')
    print('\n  => 算法族接口连通(模型 + 匹配 + 损失同族, 可训练)')


def print_guidance(cfg, adapter, task_id: str):
    info = adapter.describe()
    print('=' * 74)
    print(f'训练指引 —— 任务 {task_id} ({info["business_name"]})')
    print('=' * 74)
    print(f'\n  配置: {cfg["_config_path"]}')
    print(f'  类别({info["num_classes"]}): {", ".join(info["class_names"])}')
    print(f'  轮数 {cfg["train"]["epoches"]}   输入 {cfg["train"]["input_size"]}'
          f'   batch {cfg["train"]["batch_size"]}')
    print(f'  优化器 {cfg["optimizer"]["type"]} lr={cfg["optimizer"]["lr"]}'
          f'   调度 {cfg["scheduler"]["type"]}')

    ab = cfg.get('ablation') or {}
    on = [k for k, v in ab.items() if v is True]
    off = [k for k, v in ab.items() if v is False]
    print(f'\n  消融开关(开): {on or "无"}')
    print(f'  消融开关(关): {off or "无"}')

    print('\n  完整训练请使用 vendored 引擎(含 AMP / EMA / flatcosine / 增强调度):')
    eng = _ROOT / 'third_party' / 'deim_engine'
    print(f'    cd {eng}')
    print(f'    python train.py -c <引擎格式配置> -d cuda:0 --use-amp -u epoches={cfg["train"]["epoches"]}')
    print('\n  可见光任务对应的引擎配置(本地 deim 仓库):')
    print('    configs/deim_rtdetrv2/deim_car_union_v4.yml')
    print('  其最终生效超参见 DEIM_v4_超参数与交接说明.md。')
    print('\n  提示: 引擎配置与本工程的四层配置语义不同, 迁移时注意')
    print('        "YAML 的 list 是整体替换"这一行为(交接说明坑 4)。')


def main():
    ap = argparse.ArgumentParser(description='多任务检测 —— 训练入口')
    ap.add_argument('--task', default='visible_recognition')
    ap.add_argument('--dry-run', action='store_true', help='只做接口自检, 不训练')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--show-config', action='store_true', help='打印最终生效配置')
    args = ap.parse_args()

    load_all_adapters()
    cfg = load_config(args.task)
    adapter = get_adapter(args.task)()
    info = adapter.describe()

    if args.show_config:
        print(dump_resolved(cfg))
        return

    if info['status'] != 'implemented':
        print(f'[!] 任务 {info["task_id"]} 状态为 {info["status"]}, 尚未实现。')
        print(f'    接入步骤见 docs/ARCHITECTURE.md 第三节, 或该适配器文件内的注释。')
        return

    if args.dry_run:
        dry_run(cfg, adapter, args.device)
    else:
        print_guidance(cfg, adapter, args.task)


if __name__ == '__main__':
    main()
