# vistarget 可见光目标检测 —— 四阶段训练工程

基于 **DEIM / RT-DETRv2**（PResNet-18 主干）的目标检测训练与类增量学习工程，
在 `vistarget` 数据集上完成「基础训练 → 迁移学习 → 两阶段类增量」的完整链路。

## 一、数据集结构

数据集不随仓库提供，需自行准备为如下结构（图片为 JPEG，标注为 COCO json）：

```
datasets/vistarget_work/
├── base/
│   ├── images/{train,val,test}/*.jpg
│   └── annotations/{rel_train,rel_val,rel_test}.json
├── transfer/
│   ├── images/{train,val,test}/*.jpg
│   └── annotations/{train,val,test}.json
├── inc_p1/
│   ├── images/{train,val,test}/*.jpg
│   └── annotations/{train_replay,rel_val,rel_test}.json
└── inc_p2/
    ├── images/{train,val,test}/*.jpg
    └── annotations/{train_replay,rel_val,rel_test}.json
```

**8 个类别（全局统一 id）**

| id | 类别 | 出现在 |
|---|---|---|
| 0 | car1 | base / transfer |
| 1 | car2 | base / transfer |
| 2 | car3 | base / transfer |
| 3 | car4 | base / transfer |
| 4 | military_fuel_tanker | inc_p1 新增 |
| 5 | radar_vehicle | inc_p1 新增 |
| 6 | military_motorcycle | inc_p2 新增 |
| 7 | bridge_layer | inc_p2 新增 |

**关键约定**

- `rel_*.json` 的 `file_name` 是**相对数据集根目录**的路径（如 `base/images/test/VLO0002.jpg`），
  因此 `img_folder` 统一设为数据集根目录。
- `inc_p1/inc_p2` 的 `train_replay.json` = 当批新类全部样本 + 从 base / inc_p1 抽取的回放样本。

## 二、四阶段流程

```
① 基础训练   base      4 类   train 2,800 图   60 epoch
        ↓ 最优权重
② 迁移学习   transfer  4 类   train   100 图  120 epoch(few-shot: lr 1e-5 + 冻结 backbone 前段)
        ↓ 权重 + 分类头扩展 4→6
③ 类增量 3A  inc_p1    6 类   train 2,100 图   20 epoch(新类 1,050 + 回放 1,050)
        ↓ 权重 + 分类头扩展 6→8
④ 类增量 3B  inc_p2    8 类   train 1,154 图   20 epoch(新类 577 + 回放 577)
```

## 三、关键技术点

| 技术点 | 说明 |
|---|---|
| **模型** | RT-DETRv2 + PResNet-18-d，3 层解码器，300 查询，100 组去噪 |
| **训练配比** | DEIM（Dense O2O + Matchability-Aware Loss） |
| **输入尺寸** | 1280×1280（目标中位仅 29px，960 会压到 14.5px） |
| **Mosaic** | `output_size=640`（默认 320 会把小目标压到约 1/8 面积） |
| **分类头扩展** | 必须**同步扩 3 处**：`dec_score_head`、`enc_score_head`、`denoising_class_embed` |
| **防遗忘** | 回放（按阶段对等配比）+ 冻结 backbone 前段 + 低 lr + 短训练 |
| **干净评估** | 阈值在 val 上定，test 只跑一次；不用 test 调阈值 |

## 四、复现步骤

### 1. 环境

```bash
pip install -r requirements.txt
```

实测环境：Python 3.10.21 / PyTorch 2.11.0+cu128 / RTX 5080 16GB（Windows WDDM）

### 2. 数据预处理

```bash
python scripts/prep_vistarget_base.py
python scripts/prep_vistarget_transfer.py
python scripts/prep_vistarget_inc.py
python scripts/build_inc_replay.py
```

### 3. 数据体检与泄露核查

```bash
python scripts/profile_base.py      # 尺寸/类别/引用一致性
python scripts/audit_leakage.py     # 五级泄露核查
```

### 4. 训练

```bash
# 阶段 1
python train.py -c configs/deim_rtdetrv2/deim_vistarget_base.yml -d cuda:0 --use-amp

# 阶段 2
python train.py -c configs/deim_rtdetrv2/deim_vistarget_transfer.yml -d cuda:0 --use-amp \
    -t output/deim_vistarget_base/best_stg1.pth

# 阶段 3(必须先扩头)
python scripts/expand_head.py \
    --config configs/deim_rtdetrv2/deim_vistarget_transfer.yml \
    --ckpt output/deim_vistarget_transfer/best_stg1.pth \
    --new-num-classes 6 \
    --out-ckpt output/inc_p1_init.pth \
    --out-config configs/deim_rtdetrv2/deim_vistarget_inc_p1_init.yml
python train.py -c configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml -d cuda:0 --use-amp \
    -t output/inc_p1_init.pth

# 阶段 4(同样先扩头)
python scripts/expand_head.py \
    --config configs/deim_rtdetrv2/deim_vistarget_inc_p1.yml \
    --ckpt output/deim_vistarget_inc_p1/best_stg1.pth \
    --new-num-classes 8 \
    --out-ckpt output/inc_p2_init.pth \
    --out-config configs/deim_rtdetrv2/deim_vistarget_inc_p2_init.yml
python train.py -c configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml -d cuda:0 --use-amp \
    -t output/inc_p2_init.pth
```

### 5. 评估

```bash
# 干净评估(val 定阈值 → test 验证),推荐
python scripts/verify_val_to_test.py \
    --config configs/deim_rtdetrv2/deim_vistarget_base.yml \
    --ckpt output/deim_vistarget_base/best_stg1.pth \
    --tag base --grid "0.10:0.90:0.02"

# 全局阈值扫描(在 test 上扫会引入阈值泄漏,仅用于分析)
python tools/sweep_conf.py \
    --config configs/deim_rtdetrv2/deim_vistarget_base.yml \
    --ckpt output/deim_vistarget_base/best_stg1.pth \
    --split test --conf-grid "0.30:0.60:0.02"

# 按类别定阈值
python scripts/per_class_threshold.py \
    --config configs/deim_rtdetrv2/deim_vistarget_inc_p2.yml \
    --ckpt output/deim_vistarget_inc_p2/best_stg1.pth --tag inc_p2 \
    --ann-val <val.json> --ann-test <test.json> --img-dir <root>
```

### 6. 时延测量

```bash
python scripts/bench_latency_fast.py \
    --config configs/deim_rtdetrv2/deim_vistarget_base.yml \
    --ckpt output/deim_vistarget_base/best_stg1.pth \
    --tag base --n-run 200 --workers 8
```

## 五、实验结果

**评估口径**：置信度阈值在 **val** 上确定（选 recall 达标前提下误检最少的点），
再用该阈值在 **test** 上跑一次。test 不参与任何选择。

### 5.1 各阶段主指标

| 阶段 | 类别数 | 训练图 | 评价集 | conf | **recall** | precision | 误检/图 |
|---|---|---|---|---|---|---|---|
| **base** | 4 | 2,800 | 595图/2194框 | 0.640 | **93.25%** | **98.41%** | **0.06** |
| transfer | 4 | 100 | 687图/2084框 | 0.300 | 89.01% | 61.10% | 1.72 |
| 3A | 6 | 2,100 | 224图/887框 | 0.340 | 88.61% | 71.78% | 1.38 |
| **3B** | 8 | 1,154 | 123图/483框 | 0.500 | **92.55%** | **92.74%** | **0.28** |

### 5.2 遗忘检查（老类回归，base 4 类）

| 阶段 | conf | recall | precision | 相对上一阶段 |
|---|---|---|---|---|
| base | 0.640 | **93.25%** | 98.41% | — |
| transfer 后 | 0.660 | 92.53% | 98.26% | −0.72 pp |
| 3A 后 | 0.720 | 90.93% | 98.32% | −1.60 pp |
| 3B 后 | 0.650 | 91.57% | 97.01% | +0.64 pp |

**累计掉点 1.68pp，低于 3pp 门槛。**

### 5.3 时延（RTX 5080，1280 输入）

| 配置 | forward | 预处理 | **端到端 mean** | **p95** |
|---|---|---|---|---|
| HDD + 串行预处理（修复前） | 12.02 | 24.55 | 36.57 | **40.49** |
| **HDD + 并行预处理** | 12.71 | 3.06 | **15.77** | **31.91** |
| 内存预加载 | 12.13 | 1.65 | 13.78 | 14.15 |

**机械硬盘上端到端 p95 = 31.91ms，满足 <40ms 要求。**

## 六、已知限制

| 项 | 说明 |
|---|---|
| **场景级共享** | 数据集仅含 2 个场景（VLP/VLO）+ 4 个采集序列，所有划分均在这些场景内按帧切分，指标可能偏乐观 |
| **transfer 数据量** | 训练集仅 100 张（train:test = 1:6.87），该阶段指标主要受数据量制约 |
| **3A 目标极小** | 新类目标中位 24~26px（96% 小于 32px），是 3A 弱于 3B 的根因 |
| **按类别定阈值** | 仅在类别间置信度差异明显时有效（3B 上 precision +1.35pp，base/3A 反而变差） |

## 七、目录说明

```
.
├── engine/                     DEIM 训练引擎
├── configs/
│   ├── base/                   基础配置(include 依赖)
│   └── deim_rtdetrv2/          本项目 8 个配置
├── scripts/                    数据预处理 / 核查 / 评估 / 流水线
├── tools/                      conf 扫描工具
├── docs/                       实验汇报
├── train.py                    训练入口
└── patch_tw.py                 torchvision.transforms.v2 兼容补丁
```

## 八、参考

- DEIM: *DETR with Improved Matching for Fast Convergence* (CVPR 2025)
- RT-DETRv2: *Bag-of-Freebies* (arXiv 2407.17140)
