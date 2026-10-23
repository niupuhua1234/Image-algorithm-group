# -*- coding: utf-8 -*-
"""类增量第一步: 扩展分类头, 老类权重原样保留, 新类随机初始化。

为什么必须单独做这一步:
    增量阶段类别数从 4 变成 6(再变 8)。如果直接改配置里的 num_classes 再用
    `-t` 加载旧权重, 引擎会因分类头形状不匹配而跳过它, 老类权重等于归零 ——
    那才是真正的灾难性遗忘。

流程:
    旧 config(N 类) + 旧 ckpt
        -> 构建 N 类模型并加载权重
        -> 分类头换成 N+M 类: 前 N 行复制旧的, 后 M 行零初始化
        -> 存新 ckpt + 新 config
    之后用 `-t 新ckpt` 训练。

用法:
    python expand_head.py --config <旧配置> --ckpt <旧权重> --new-num-classes 6 \
        --out-ckpt <新权重> --out-config <新配置>
"""
import argparse
import io
import os
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DEIM = Path(r'D:\BaiduNetdiskDownload\deploy_deim_v4\deim')
sys.path.insert(0, str(DEIM))

import torch                     # noqa: E402
import torch.nn as nn            # noqa: E402

# 分类头的候选名(RT-DETR 系: 解码器与编码器各有一个)
SCORE_HEAD_NAMES = ('dec_score_head', 'enc_score_head', 'score_head')


def list_heads(model):
    """返回 [(parent_module, attr_name, head_module, 输出维度)]。"""
    out = []
    for owner_path in ('decoder', 'encoder'):
        owner = getattr(model, owner_path, None)
        if owner is None:
            continue
        for nm in SCORE_HEAD_NAMES:
            h = getattr(owner, nm, None)
            if h is None:
                continue
            dim = _out_dim(h)
            if dim is not None:
                out.append((owner, nm, h, dim))
    return out


def _out_dim(head):
    """取分类头的输出维度。"""
    if isinstance(head, nn.Linear):
        return head.out_features
    if isinstance(head, (nn.Sequential, nn.ModuleList)):
        for m in reversed(list(head)):
            if isinstance(m, nn.Linear):
                return m.out_features
    return None


def expand_one(head, new_out):
    """返回扩展后的 head(不改原对象), 以及旧输出维度。"""
    old_out = _out_dim(head)
    if old_out is None:
        raise RuntimeError('无法确定分类头输出维度')

    def wider(old_lin: nn.Linear):
        new = nn.Linear(old_lin.in_features, new_out, bias=old_lin.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:old_out].copy_(old_lin.weight)
            if old_lin.bias is not None:
                new.bias.zero_()
                new.bias[:old_out].copy_(old_lin.bias)
        return new

    if isinstance(head, nn.Linear):
        return wider(head), old_out

    # Sequential / ModuleList: 复制一份, 替换其中所有 Linear 的输出维度
    new_head = type(head)()
    for m in head:
        if isinstance(m, nn.Linear) and m.out_features == old_out:
            new_head.append(wider(m))
        else:
            new_head.append(m)
    return new_head, old_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--new-num-classes', type=int, required=True)
    ap.add_argument('--out-ckpt', required=True)
    ap.add_argument('--out-config', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.chdir(str(DEIM))
    try:
        import patch_tw
        patch_tw.apply()
    except Exception:
        pass
    from engine.core import YAMLConfig

    torch.manual_seed(args.seed)
    cfg = YAMLConfig(args.config)
    old_num = cfg.yaml_cfg.get('num_classes')
    print('=' * 74)
    print('分类头扩展')
    print('=' * 74)
    print(f'  配置 {args.config}')
    print(f'  权重 {args.ckpt}')
    print(f'  类别 {old_num} -> {args.new_num_classes}  (新增 {args.new_num_classes - old_num})')

    model = cfg.model
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    is_ema = isinstance(ck.get('ema'), dict) and 'module' in ck['ema']
    sd = ck['ema']['module'] if is_ema else ck.get('model', ck)
    print(f'  权重来源: {"ema.module" if is_ema else "model"}  张量 {len(sd)}  \
last_epoch={ck.get("last_epoch")}')

    msd = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    print(f'  加载旧权重: 匹配 {len(matched)}/{len(msd)}')

    # 找分类头并扩展
    heads = list_heads(model)
    if not heads:
        raise RuntimeError('没找到分类头(déc_score_head / enc_score_head)')
    print(f'  找到 {len(heads)} 个分类头:')
    for owner, nm, h, dim in heads:
        print(f'    {type(owner).__name__}.{nm}  类型 {type(h).__name__}  输出 {dim}')
        new_h, old_out = expand_one(h, args.new_num_classes)
        setattr(owner, nm, new_h)
        print(f'      -> 扩展到 {args.new_num_classes}(前 {old_out} 行复制旧权重, 其余置零)')

    # 去噪训练的分类嵌入(denoising_class_embed)行数 = num_classes + 1(含 no-object),
    # 同样要扩展, 否则权重形状不匹配会被跳过。
    dec = getattr(model, 'decoder', None)
    emb = getattr(dec, 'denoising_class_embed', None) if dec is not None else None
    if emb is not None and isinstance(emb, nn.Embedding):
        old_rows = emb.weight.shape[0]
        new_rows = args.new_num_classes + 1
        new_emb = nn.Embedding(new_rows, emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight.zero_()
            new_emb.weight[:old_rows].copy_(emb.weight)
        dec.denoising_class_embed = new_emb
        print(f'    decoder.denoising_class_embed: {old_rows} -> {new_rows} 行'
              f'(前 {old_rows} 行复制, 其余置零会随机初始化)')

    # 组装新 checkpoint(两份权重都换成扩展后的形状)
    new_sd = model.state_dict()
    if is_ema:
        ck['ema']['module'] = new_sd
    else:
        ck['model'] = new_sd
    if isinstance(ck.get('model'), dict) and is_ema:
        ck['model'] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in new_sd.items()}

    out_ckpt = Path(args.out_ckpt)
    if not out_ckpt.is_absolute():
        out_ckpt = DEIM / out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out_ckpt)
    print(f'\n  已保存扩展后权重: {out_ckpt}  ({out_ckpt.stat().st_size/1e6:.0f} MB)')

    # 生成新配置
    out_cfg = None
    if args.out_config:
        src = Path(args.config)
        if not src.is_absolute():
            src = DEIM / src
        txt = src.read_text(encoding='utf-8')
        if re.search(r'^num_classes:\s*\d+', txt, flags=re.M):
            txt = re.sub(r'^num_classes:\s*\d+', f'num_classes: {args.new_num_classes}',
                         txt, count=1, flags=re.M)
        else:
            txt = f'num_classes: {args.new_num_classes}\n' + txt
        out_cfg = Path(args.out_config)
        if not out_cfg.is_absolute():
            out_cfg = DEIM / out_cfg
        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        out_cfg.write_text(txt, encoding='utf-8')
        print(f'  已保存新配置: {out_cfg}')

    # 自检
    print('\n  自检(用新配置重建模型 + 加载新权重)')
    if out_cfg is not None:
        cfg2 = YAMLConfig(str(out_cfg))
        m2 = cfg2.model
        msd2 = m2.state_dict()
        ck2 = torch.load(out_ckpt, map_location='cpu', weights_only=False)
        sd2 = ck2['ema']['module'] if isinstance(ck2.get('ema'), dict) and 'module' in ck2['ema'] else ck2.get('model', ck2)
        bad2 = [k for k in msd2 if k in sd2 and sd2[k].shape != msd2[k].shape]
        miss2 = [k for k in msd2 if k not in sd2]
        print(f'    新模型 {len(msd2)} 张量   新权重 {len(sd2)} 张量   形状不符 {len(bad2)}   缺失 {len(miss2)}')
        ok = len(bad2) == 0 and len(miss2) == 0
        print(f'    => {"完全匹配, 可用于训练" if ok else "仍有不匹配: " + str((bad2[:3], miss2[:3]))}')
        for owner, nm, h, dim in list_heads(m2):
            print(f'    {nm} 输出维度 = {dim}')


if __name__ == '__main__':
    main()
