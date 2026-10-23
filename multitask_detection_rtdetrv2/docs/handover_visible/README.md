# 多任务图像识别与红外小目标检测 —— 统一工程

按《多任务图像识别与红外小目标检测项目技术路线书》(2026-10-12)搭建。
**代码建设主干:RT-DETRv2**。三个业务任务共享工程接口,各自独立配置、权重与评测口径。

---

## 一、三个任务及其状态

| 任务标识 | 业务内容 | 输出要求 | 原生对照 | 状态 |
|---|---|---|---|---|
| `visible_recognition` | 可见光目标识别 | 目标框、任务类别、分数 | DEIM | **✓ 已实现** |
| `infrared_recognition` | 红外目标识别 | 目标框、任务类别、分数 | YOLOv8 / YOLO11 | ○ 占位 |
| `infrared_small_target` | 红外小目标检测 | 目标框、目标分数 | AA-YOLO、现有 RT-DETR | ○ 占位 |

**本阶段只完整实现可见光任务**,另两个任务按路线书要求"保留位置":
适配器、配置、类别体系、评测口径都已声明,只缺具体实现(见各文件内的接入步骤)。

> **统一代码不等于统一权重**(路线书第一节)。三个任务的类别体系、数据划分、
> 模型权重、阈值、实验输出**相互独立**,不要混用。

---

## 二、目录结构

```
multitask_detection/
├── configs/                        配置(按路线书 5.4 分四层)
│   ├── tasks/                      任务层: 三个任务各一份
│   │   ├── visible_recognition.yaml        ← 完整
│   │   ├── infrared_recognition.yaml       ← 占位
│   │   └── infrared_small_target.yaml      ← 占位
│   ├── models/                     模型层: 主干/编解码器/损失
│   │   └── rtdetrv2_r18_visible.yaml
│   ├── training/                   训练层: 优化器/调度/增强 + 消融开关
│   │   └── deim_v4_45e.yaml
│   └── deployment/                 部署层: 导出与推理后端
├── src/
│   ├── core/                       配置加载 + 组件注册
│   ├── data/                       数据适配、增强、GPU 预处理
│   ├── models/
│   │   ├── backbones/ encoders/ decoders/ heads/     公共网络模块层
│   │   └── adapters/               **算法组件层入口**
│   │       ├── base.py                    TaskAdapter 抽象
│   │       ├── visible_recognition.py     ✓ 完整
│   │       ├── infrared_recognition.py    ○ 占位
│   │       └── infrared_small_target.py   ○ 占位
│   ├── losses/ optim/ engine/     损失、优化、训练调度
│   ├── metrics/                    公共指标(AP / P-R / 类别混淆)
│   └── deployment/                 导出、推理、后处理适配
├── third_party/
│   ├── deim_engine/                vendored RT-DETRv2 / DEIM 引擎
│   └── patch_tw.py                 torchvision 0.26 兼容补丁(必需)
├── tools/                          CLI 入口
├── experiments/                    配置快照、日志、权重、报告
├── docs/                           架构说明与接入指引
└── tests/
```

---

## 三、快速开始

### 3.1 查看任务全貌

```bash
python tools/list_tasks.py                        # 三个任务及状态
python tools/list_tasks.py --task visible_recognition   # 单任务详情 + 最终生效配置
```

### 3.2 评测(可见光)

```bash
python tools/eval.py \
    --task visible_recognition \
    --split test \
    --ckpt /path/to/best_stg1.pth \
    --legacy                 # 同时给出项目历史 recall 口径, 便于对照
```

输出四项**路线书要求的指标**:

| 指标 | 说明 |
|---|---|
| 检测 AP | `mAP@0.5` 与 `mAP@.50:.95` |
| 逐类别精度与召回 | 每类 TP/FP/FN/P/R/F1/AP |
| **匹配目标上的类别混淆** | 匹配成功但类别判断错误的配对统计 |
| 完整业务时延 | 见 `tools/infer.py`(含预处理的端到端) |

快速验证用 `--limit 200` 只跑前 200 张。

### 3.3 推理

```bash
python tools/infer.py --task visible_recognition \
    --ckpt /path/to/best_stg1.pth --image test.jpg --conf 0.3
```

---

## 四、三步接入一个新任务

以 `infrared_recognition` 为例(完整说明见 `docs/ARCHITECTURE.md`):

**① 补适配器** —— `src/models/adapters/infrared_recognition.py`
把 `CLASS_NAMES` 填上红外业务类别,然后:

- 若走 **v2 主线**:直接继承 `VisibleRecognitionAdapter` 的实现,只换类别;
  模型/损失/后处理三件套保持不变
- 若先做 **YOLO 对照**:另建 `yolo_family.py`,**后处理必须带 NMS**,
  且不能把 YOLO 检测头与 DETR 匈牙利匹配混用(路线书 2.3 节)

**② 补任务配置** —— `configs/tasks/infrared_recognition.yaml`
填数据路径、类别映射、评测口径。

**③ 注册** —— 适配器已用 `@register_task` 装饰,无需改注册表;
若新增文件名,在 `src/core/registry.py::load_all_adapters()` 追加一行 import。

---

## 五、与路线书的对应关系

| 路线书要求 | 本工程落点 |
|---|---|
| 2.3 三层复用机制 | 公共工程层 `src/core`+`src/engine`;网络模块层 `src/models/{backbones,encoders,decoders,heads}`;算法组件层 `src/models/adapters` |
| 3.2 配置驱动 + 适配器 | `configs/` 四层 + `TaskAdapter` |
| 4.1 v2 为主干,DEIM 与纯 v2 分别管理 | `configs/models/rtdetrv2_r18_visible.yaml` 内含 `baseline_v2` 段 |
| 5.2 三任务配置 | `configs/tasks/` 三份,yaml 与路线书表格逐项对应 |
| 5.3 统一输出与独立类别 | `Detection` 统一结构;`class_names` 各任务独立 |
| 5.4 建议目录 | 已按此搭建 |
| 6 单因素消融 | `configs/training/deim_v4_45e.yaml` 的 `ablation` 段 |
| 6 分任务评测口径 | `visible_recognition.yaml::evaluation` 与 `infrared_small_target.yaml::evaluation` 分别声明 |

---

## 六、已知风险与坑(交接必读)

| # | 坑 | 影响 | 处置 |
|---|---|---|---|
| 1 | 可见光 train 目录有 **50,024** 张,但 JSON 只引用 **37,847** 张 | 直接读目录会混入 12,177 张旧格式残留 | 评测/训练**一律以 JSON 引用为准** |
| 2 | JSON 的 `path` 字段是绝对路径 | 换机器即失效 | 代码已改为优先 `img_folder + file_name` |
| 3 | 原配置 `mixup_epochs: [4, 64]` 而 epoches 只有 45 | MixUp 全程开启,与 Mosaic 关闭点不一致 | 已在训练配方里修正为 `[4, 24]` |
| 4 | 原配置 backbone 专属 lr(1e-5)被上层整体替换 | backbone 实际用全局 2e-4 | 已在 `optimizer.backbone_lr` 显式暴露 |
| 5 | `weight_dict` 里残留 `loss_vfl` | 误以为同时用了 VFL 和 MAL | **以 `losses` 列表为准** |
| 6 | **Mosaic `output_size: 320`** | 4 张 960 图压到 320,car 目标 32px → 约 10px | 路线书点名要查"是否过度缩小微小目标";消融时试 640 |
| 7 | `eval_spatial_size: null` 时推理每帧重建 anchors | 每帧多花约 2ms,且 CUDA Graph 无法捕获 | 推理时预生成到 GPU(见 `src/deployment/runtime.py`) |
| 8 | val 有 1,121 张、test 有 361 张无标注图 | 属于背景图,计入分母会拉低 mAP | 口径需与此一致,不要当成标注缺失 |

前 5 条来自 `DEIM_v4_超参数与交接说明.md`,后 3 条来自实测。

---

## 七、环境

| 项 | 版本(实测可用) |
|---|---|
| Python | 3.10 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| 其他 | PyYAML, numpy, opencv-python, Pillow |

> `third_party/patch_tw.py` 是**必需**的 —— torchvision 0.26 改了 `Transform.forward`
> 接口而引擎用旧式 `_transform`。适配器在 import 引擎前会自动应用。

---

## 八、性能参考(实测,RTX 5080)

### 8.1 精度(全量 test,3976 张 / 56375 GT)

| 口径 | 数值 |
|---|---|
| 路线书口径 `mAP@0.5` | **0.7415** |
| 路线书口径 `mAP@.50:.95` | **0.4969** |
| 项目历史 recall 口径(IoU≥0.40) | **90.39%**(50958/56375) |
| val 集 mAP@.50:.95(训练日志) | 0.4048(最优 epoch 31) |

逐类别 AP:car 0.7675 / truck 0.6379 / bus 0.8798 / van 0.5201 / pickup 0.6544 / tank 0.9892

### 8.2 时延(200 次压测)

| 口径 | CUDA Graph | PyTorch eager |
|---|---|---|
| 纯模型 forward | **7.71 ms**(p95 7.75) | 40.70 ms(p95 41.65) |
| 预处理(读图+resize+传输) | 17.26 ms(p95 25.09) | 18.05 ms(p95 25.06) |
| **完整业务时延** | **27.27 ms**(p95 **34.79**) | 61.28 ms(p95 67.38) |
| 峰值显存 | 168 MB | 272 MB |

**验收阈值 40 ms/frame:CUDA Graph 后端 mean 与 p95 均达标**(余量 32% / 13%)。

> 路线书要求记录"**完整业务时延**"—— 即上表第三行。只报 forward 会导致对方按端到端
> 测量时对不上账(两者相差 3.5 倍)。eager 后端之所以慢,是因为 600+ 个 CUDA kernel
> 在 Windows WDDM 下的 CPU 提交开销成为瓶颈(GPU 利用率只有约 20%);
> CUDA Graph 把提交压成一次,forward 直接降到 GPU 实际执行时间。
