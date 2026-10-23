# vendored 引擎版本记录

技术路线书 4.1 节要求:

> 正式实施时以官方 RT-DETR v2 PyTorch 实现为基准,**记录源码版本**,
> 并对本地 DEIM 增强版与纯 v2 基线分别管理。

---

## 来源

| 项 | 内容 |
|---|---|
| 引擎 | RT-DETRv2 + DEIM 增强(本地 `deim` 仓库) |
| 原始路径 | `D:\BaiduNetdiskDownload\deploy_deim_v4\deim\engine` |
| vendored 时间 | 2026-10-13 |
| 文件数 | 62 个 Python 文件 |
| 附带补丁 | `third_party/patch_tw.py`(torchvision 0.26 兼容,**必需**) |

## 目录内容

```
deim_engine/
├── backbone/     PResNet / HGNetv2 / CSP 系列骨干
├── core/         YAMLConfig(引擎侧配置加载, 本框架未使用)
├── data/         数据集、transform 算子、collate
├── deim/         **核心**: hybrid_encoder / rtdetrv2_decoder /
│                 deim_criterion(含 MAL) / matcher / denoising / postprocessor
├── misc/         日志、分布式工具
├── optim/        优化器、EMA、LR 调度
└── solver/       训练与评估循环
```

## 已核对的关键实现

| 文件 | 内容 | 路线书对应 |
|---|---|---|
| `deim/hybrid_encoder.py` | HybridEncoder(AIFI + CCFM),`version: rt_detrv2` | v1 的 AIFI/CCFF 结构 |
| `deim/rtdetrv2_decoder.py` | RTDETRTransformerv2,含 `cross_attn_method`(default/discrete)、`num_points` | **v2 主干** |
| `deim/deim_criterion.py` | DEIMCriterion,含 `loss_mal` | **DEIM 的 MAL** |
| `deim/matcher.py` | HungarianMatcher(一对一集合预测) | DETR 族匹配 |
| `backbone/presnet.py` | PResNet-18-d | 主干卷积网络 |

## 与官方实现的关系

本引擎是 **RT-DETRv2 实现 + DEIM 训练改进**,不是纯官方 v2。二者的差异集中在:

| 维度 | 官方 RT-DETRv2 | 本引擎(DEIM 增强) | 本工程如何区分 |
|---|---|---|---|
| 分类损失 | Varifocal (`vfl`) | **MAL** (`mal`) | `configs/models/...::loss.losses` |
| 正样本密度 | 标准一对一 | **Dense O2O**(Mosaic + MixUp) | `configs/training/...::ablation.use_dense_o2o` |
| BN | `freeze_at: 0` / `freeze_norm: True` | **解冻** | `ablation.unfreeze_bn` |
| 查询位置编码 | 默认 | **`as_reg`** | `model.decoder.query_pos_method` |
| 采样方式 | 同 | 同(默认 grid_sample) | `ablation.cross_attn_method` |

**纯 v2 基线**的配置已写在 `configs/models/rtdetrv2_r18_visible.yaml` 的 `baseline_v2` 段,
按其中的 `overrides` 覆盖即可得到官方 v2 的训练方案,与 DEIM 增强版做单因素消融。

## 本工程未使用引擎的部分

| 组件 | 原因 |
|---|---|
| `core/YAMLConfig` | 配置语义不同(引擎的 include 链里 list 是整体替换,易踩坑)。本框架改为四层配置 + 深度合并,适配器直接构造模块 |
| `solver/` 训练循环 | 完整训练(AMP/EMA/flatcosine/增强调度)仍建议**直接用引擎的 `train.py`**,不在框架层重造 |
| `data/` 数据集类 | 评测路径不走 dataloader(直接读 COCO JSON + GPU 预处理);训练路径仍用引擎的 `CocoDetection` |

## 已知坑(来自实测)

| # | 位置 | 现象 | 处置 |
|---|---|---|---|
| 1 | `rtdetrv2_decoder.py` L504 | `eval_spatial_size=None` 时每帧重建 anchors,产生 CPU→GPU 拷贝 | `InferenceRuntime.prepare_static()` 预生成到 GPU |
| 2 | `hybrid_encoder.py` L391 | `pos_embed` 用 `setattr` 存普通属性(`register_buffer` 被注释),`.cuda()` 不会搬 | 同上,手动迁移设备 |
| 3 | `deim_criterion.py` | `num_classes` 默认按 COCO 80 类 | 适配器统一注入 `self.num_classes` |
| 4 | 配置层 | YAML 的 list 整体替换,导致 `optimizer.params` 里的 backbone lr 被覆盖 | 见 `DEIM_v4_超参数与交接说明.md` 坑 4 |

## 升级引擎时注意

1. 保留本文件,记录新版本来源与日期
2. 重新核对上表"已核对的关键实现"的四个文件
3. 重跑三个自检:
   ```bash
   python tools/list_tasks.py
   python tools/train.py --task visible_recognition --dry-run
   python tools/eval.py --task visible_recognition --split test --ckpt <权重> --limit 100 --legacy
   ```
   其中 `--legacy` 的历史口径准确率应与既有结果一致(测试集全量时为 0.9036)。
