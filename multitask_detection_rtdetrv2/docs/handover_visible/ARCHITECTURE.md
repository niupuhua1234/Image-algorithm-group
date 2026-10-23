# 工程架构与接入指引

面向接手本项目的人。读完这份文档 + `README.md`,应当能:
① 理解为什么这样分层;② 跑通可见光任务;③ 把另外两个任务接进来。

---

## 一、三层复用机制的落点

技术路线书 2.3 节要求:

| 层级 | 路线书内容 | 复用方式 | 本工程落点 |
|---|---|---|---|
| **公共工程层** | 配置、数据索引、图像与标注读写、日志、随机种子、设备管理、实验记录、训练调度与统一结果管理 | 集中建设,跨任务共享 | `src/core/`(配置+注册)、`src/data/`(IO+增强)、`src/engine/`、`src/metrics/`、`src/deployment/` |
| **公共网络模块层** | 卷积、归一化、上采样、特征拼接、多尺度投影及部分骨干和融合块 | 统一接口后选择性复用 | `src/models/{backbones,encoders,decoders,heads}/` + `third_party/deim_engine/` |
| **算法组件层** | YOLO 检测头、AADH、DETR 解码器、目标匹配、损失、解码与后处理 | 按算法配套配置,通过适配层接入 | **`src/models/adapters/`** |

**关键约束(路线书 2.3 节)**:

> 不能将 YOLO 检测头、AA objectness 或 DETR 匈牙利匹配任意组合。

因此适配器**同时绑定三件套**:`build_model` + `build_criterion` + `postprocess`。
它们必须来自同一算法族,不允许跨族拼接。

---

## 二、数据流

```
configs/tasks/<task>.yaml
        │  (defaults 引用 models / training 两层, 深层合并)
        ▼
   load_config()  ──►  完整配置 dict
        │
        ├─► registry.get_adapter(task_id)  ──►  TaskAdapter 实例
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │  适配器负责:                                          │
 │    build_model(cfg)        模型(算法族绑定)          │
 │    build_criterion(cfg)    损失与匹配(同族)          │
 │    build_dataloader(cfg,s) 数据                       │
 │    postprocess(raw, size)  原始输出 -> Detection      │
 │    evaluate(preds, gts)    指标(默认走公共实现)      │
 └──────────────────────────────────────────────────────┘
        │
        ▼
   统一输出 Detection(boxes 原图 xyxy, scores, labels)
        │
        ├─► src/metrics/detection.py   AP / 逐类别 P-R / 类别混淆
        └─► src/deployment/runtime.py  forward / preprocess / 完整业务时延
```

**"对外结果一致不意味着内部原始输出一致"**(路线书 5.3)——
`Detection` 是对外契约,算法内部的 raw output 不暴露。

---

## 三、接入一个新任务

以 `infrared_recognition` 为例。

### 步骤 1:确定类别体系

编辑 `src/models/adapters/infrared_recognition.py`:

```python
CLASS_NAMES = ['目标A', '目标B']      # 编号必须独立于其他任务
```

**为什么强调独立**:路线书 5.3 节要求"类别数量和类别编号独立配置,避免将不同任务的
类别体系混用"。如果红外任务照抄可见光的 `car=0, truck=1...`,后续做跨任务对比时
`label 0` 的含义会漂移。

### 步骤 2:选择接入路径

**路径 A —— 走 v2 主线(推荐先做这个)**

直接复用可见光的实现,只换类别:

```python
from src.models.adapters.visible_recognition import VisibleRecognitionAdapter

@register_task(task_id='infrared_recognition', status='implemented')
class InfraredRecognitionAdapter(VisibleRecognitionAdapter):
    class_names = CLASS_NAMES          # 只改这一处
    META = TaskMeta(... status='implemented' ...)
```

模型、损失、后处理三件套原样继承 —— 因为路线书把三个任务的框级识别都归到 v2 主干,
差异只在数据、类别和主干/融合模块的选择。

**路径 B —— 先做 YOLO 原生对照**

路线书要求"保留四套原生对照",YOLO 是红外的对照方法。此时:

1. 新建 `src/models/adapters/yolo_family.py`
2. `postprocess` **必须带 NMS**(YOLO 是密集预测 + anchor-based,与 DETR 的集合预测不同)
3. 损失用 YOLO 配套的(`objectness` + 框回归),**不得**混入匈牙利匹配
4. 建议同时保留 yolo 与 v2 两个适配器,通过配置切换,便于同数据同指标对照

### 步骤 3:写任务配置

`configs/tasks/infrared_recognition.yaml`:

```yaml
defaults:
  - ../models/rtdetrv2_r18_visible.yaml    # 或新建红外专用模型配置
  - ../training/deim_v4_45e.yaml

task: {id: infrared_recognition, status: implemented}
classes: {names: [目标A, 目标B]}
data:
  root: /path/to/infrared
  train: {img_folder: images/train, ann_file: ann/train.json, batch_size: 8}
  ...
evaluation:
  protocol: box_level_detection
  metrics: [detection_ap, per_class_precision_recall, class_confusion, end_to_end_latency]
```

### 步骤 4:注册

适配器已用 `@register_task` 装饰。若新建了文件,在
`src/core/registry.py::load_all_adapters()` 里追加一行 import。

### 步骤 5:验证

```bash
python tools/list_tasks.py --task infrared_recognition    # 状态应变 implemented
python tools/eval.py --task infrared_recognition --split val --ckpt ... --limit 100
```

---

## 四、单因素消融怎么做

技术路线书第三阶段要求"每次优先改变一个主要因素"。训练配方里已把 DEIM 的改进
暴露成开关(`configs/training/deim_v4_45e.yaml` 的 `ablation` 段):

| 开关 | 含义 | 关闭后的效果 |
|---|---|---|
| `use_dense_o2o` | 用 Mosaic/MixUp 增加每图目标数 | 回到标准一对一监督 |
| `use_mal` | MAL 替代 VFL 作分类损失 | 用回 `losses: [vfl, boxes]`, alpha 0.75 / gamma 2.0 |
| `unfreeze_bn` | 解冻 BN | `freeze_at: 0`, `freeze_norm: True` |
| `cross_attn_method` | `default`(grid_sample) / `discrete` | discrete 部署友好但**需完整重训** |

**注意**(已写进配置注释):
- 关 Dense O2O 时要把 Mosaic 的 `probability` 一起置 0,否则不是干净的单因素对照
- 关 MAL 时要同步换 `losses` 列表和 alpha/gamma,**不能只删 weight_dict 里的键**

**纯 v2 基线**对照:见 `configs/models/rtdetrv2_r18_visible.yaml` 的 `baseline_v2` 段。

---

## 五、几个设计决策及理由

**① 为什么 vendored 引擎而不是重写**

路线书 4.1 节:"正式实施时以官方 RT-DETR v2 PyTorch 实现为基准,记录源码版本"。
本地 `deim` 仓库本身就是 v2 实现 + DEIM 增强,重写一遍只会引入差异。
因此 `third_party/deim_engine/` 直接 vendored,并在 `VERSION.md` 里记录来源。

**② 为什么适配器不用引擎的 YAMLConfig**

引擎的 `YAMLConfig` 面向训练脚本,配置格式与本工程的四层配置不同。
适配器改为**直接构造模块**,好处是:
- 配置语义清楚(哪层覆盖哪层一目了然)
- 不依赖引擎的 include 链行为(YAML 的 list 是整体替换,容易踩坑 —— 见交接说明坑 4)

**③ 为什么 `prepare_static` 是必需的**

`eval_spatial_size=None` 时,引擎每帧都会重建 anchors 并产生一次 CPU→GPU 拷贝
(`rtdetrv2_decoder.py` L504)。后果有两个:每帧多花约 2ms;**CUDA Graph 无法捕获**
(捕获期禁止 CPU→GPU 拷贝)。`InferenceRuntime.prepare_static()` 一次性把位置相关
张量预生成到 GPU,两个问题同时解决。

**④ 为什么区分三个时延口径**

路线书 3.4 节要求记录"完整业务时延",而业界习惯报的 forward 数字往往不含预处理。
本项目实测:forward 7.7ms(CUDA Graph)/ 端到端 28.8ms,预处理的 18ms 占了大头。
只报 forward 会让对接方按端到端测量时对不上账。

---

## 六、目录职责速查

| 路径 | 职责 | 改动频率 |
|---|---|---|
| `configs/tasks/` | 每个任务一份,含类别映射与评测口径 | **高**(接任务时) |
| `configs/models/` | 网络结构与损失 | 中 |
| `configs/training/` | 优化器/调度/增强/消融开关 | 中 |
| `src/models/adapters/` | **算法组件层,接入新算法的唯一入口** | **高** |
| `src/core/` | 配置加载与注册机制 | 低 |
| `src/data/` | IO、增强、预处理 | 低 |
| `src/metrics/` | 公共指标 | 低 |
| `src/deployment/` | 权重加载、推理运行时 | 低 |
| `third_party/deim_engine/` | vendored 引擎 | **不动** |
| `tools/` | CLI 入口 | 中 |
| `experiments/` | 配置快照、日志、权重、报告 | 自动生成 |
