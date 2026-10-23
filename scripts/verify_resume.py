# -*- coding: utf-8 -*-
"""验证 last.pth 能否"完整恢复"续训到更多轮次。

检查三件事:
    ① checkpoint 里到底存了哪些状态
    ② 引擎的 resume 逻辑实际会读取哪些状态
    ③ 把 epoches 从 60 改成 90 之后, lr 调度会怎么变

只有 ①② 对得上、③ 的调度是连续的, 续训才算"完整恢复";
否则会出现 lr 跳变 / EMA 失效 / epoch 计数错乱, 破坏已训好的权重。

用法:
    python verify_resume.py --ckpt output/deim_vistarget_base/last.pth \
        --config configs/deim_rtdetrv2/deim_vistarget_base.yml --new-epoches 90
"""
import argparse
import ast
import io
import os
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
sys.path.insert(0, str(DEIM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--new-epoches', type=int, default=90)
    args = ap.parse_args()

    os.chdir(str(DEIM))
    import torch

    print('=' * 76)
    print('续训完整性验证')
    print('=' * 76)
    print(f'  checkpoint {args.ckpt}')
    print(f'  config     {args.config}')
    print(f'  目标轮数   {args.new_epoches}')

    ck = torch.load(DEIM / args.ckpt, map_location='cpu', weights_only=False)
    print()
    print('① checkpoint 内容')
    print('-' * 76)
    for k in ck.keys():
        v = ck[k]
        if isinstance(v, dict):
            print(f'  {k:<24} dict, {len(v)} 个键')
        elif torch.is_tensor(v):
            print(f'  {k:<24} tensor {tuple(v.shape)}')
        else:
            print(f'  {k:<24} {type(v).__name__} = {v}')

    # ---- 关键状态 ----
    print()
    print('  关键状态检查:')
    required = {
        'model': '模型权重',
        'optimizer': '优化器动量/二阶矩',
        'ema': 'EMA 权重',
        'scaler': 'AMP GradScaler',
        'last_epoch': 'epoch 计数',
        'lr_warmup_scheduler': '学习率调度器进度',
    }
    have = {}
    for k, desc in required.items():
        ok = k in ck
        have[k] = ok
        print(f'    {"OK " if ok else "缺失"}  {k:<22} ({desc})')
        if ok and k in ('ema', 'lr_warmup_scheduler', 'scaler'):
            v = ck[k]
            if isinstance(v, dict):
                print(f'           子键: {list(v.keys())[:6]}')
            elif v is not None:
                print(f'           值: {v}')

    # ---- 引擎的 resume 实现 ----
    print()
    print('② 引擎 resume 实现(实际会读什么)')
    print('-' * 76)
    solver = DEIM / 'engine' / 'solver' / '_solver.py'
    src = solver.read_text(encoding='utf-8', errors='ignore') if solver.exists() else ''
    # 找 load_checkpoint 附近
    lines = src.splitlines()
    hit = [i for i, l in enumerate(lines) if 'def load_checkpoint' in l]
    if hit:
        i0 = hit[0]
        seg = lines[i0:i0 + 60]
        for l in seg:
            if l.strip():
                print('  ' + l.rstrip()[:110])
    else:
        print('  未找到 load_checkpoint, 尝试搜索 resume 相关')
        for i, l in enumerate(lines):
            if 'resume' in l.lower() and ('load' in l.lower() or 'ckpt' in l.lower()):
                print(f'  L{i}: {l.strip()[:110]}')

    # ---- 调度分析 ----
    print()
    print('③ 调度影响分析')
    print('-' * 76)
    cfg_txt = (DEIM / args.config).read_text(encoding='utf-8')
    def get(key, default=None):
        m = re.search(rf'^{key}:\s*(\S+)', cfg_txt, flags=re.M)
        return m.group(1) if m else default
    cur_ep = get('epoches')
    cur_flat = get('flat_epoch')
    cur_noaug = get('no_aug_epoch')
    print(f'  当前配置: epoches={cur_ep}  flat_epoch={cur_flat}  no_aug_epoch={cur_noaug}')
    last_ep = ck.get('last_epoch')
    print(f'  已完成: last_epoch={last_ep}')
    if cur_ep and last_ep is not None:
        print(f'  -> 当前配置下已跑 {last_ep + 1 if isinstance(last_ep, int) else "?"} / {cur_ep} 轮')

    print()
    print('  若把 epoches 改为 %d:' % args.new_epoches)
    cur_ep_i = int(cur_ep) if cur_ep and cur_ep.isdigit() else 60
    cur_flat_i = int(cur_flat) if cur_flat and cur_flat.isdigit() else cur_ep_i // 2
    print(f'    flat_epoch 需从 {cur_flat_i} 改为 {args.new_epoches // 2}'
          f'  (原比例 {cur_flat_i}/{cur_ep_i} = {cur_flat_i/cur_ep_i:.2f})')
    print(f'    no_aug_epoch 保持 3(末段关闭增强, 与总轮数无关)')
    print()
    print('  ⚠️ 关键风险:')
    print('     lr 调度通常按 "当前 epoch / 总 epoch" 计算进度。')
    print('     若把总轮数从 %d 改成 %d, 同样的 epoch 索引会对应到更靠前的进度,' % (cur_ep_i, args.new_epoches))
    print('     导致 lr 在续训瞬间**跳回去**(余弦曲线重算), 而不是平滑接续。')

    # ---- 结论 ----
    print()
    print('=' * 76)
    print('结论')
    print('=' * 76)
    missing = [k for k, ok in have.items() if not ok]
    if not missing:
        print('  [OK] checkpoint 含全部关键状态, 理论上可完整恢复')
        print('       但 lr 曲线因总轮数变化会重算, 需确认是否接受这一点。')
    else:
        print(f'  [!] 缺少 {missing}')
        print('      缺 lr_warmup_scheduler 时续训会让 lr 从初始值重新走一遍;')
        print('      缺 ema 时 EMA 权重会丢失, 等于丢掉训练后期的平滑收益。')

    print()
    print('  三种续训方式的取舍:')
    print('    A. -r last.pth + epoches 90   : 权重/优化器接上, 但 lr 曲线按 90 轮重算(会跳)')
    print('    B. 复制目录后用 -t best.pth 新跑 90 轮: 相当于重新调度, 不继承优化器动量')
    print('    C. 停止在 60 轮, 直接进入下一阶段     : 不折腾, 但可能错过后续提升')


if __name__ == '__main__':
    main()
