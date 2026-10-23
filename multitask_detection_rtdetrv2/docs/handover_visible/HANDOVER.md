# 交接说明 —— 给接手的人

这份文档面向三个任务的接手者。**先读 `README.md`,再读本文档,最后按需查 `ARCHITECTURE.md`。**

---

## 一、先搞清楚你接的是哪个任务

| 任务标识 | 业务 | 你的参照物(原生对照) | 交付物 |
|---|---|---|---|
| `visible_recognition` | 可见光目标识别 | DEIM | **已完成,供你参考** |
| `infrared_recognition` | 红外目标识别 | YOLOv8 / YOLO11 | 你的模型 + 评测报告 |
| `infrared_small_target` | 红外小目标检测 | AA-YOLO、现有 RT-DETR | 你的模型 + 评测报告(含 FAR) |

**三个任务彼此独立**:类别体系、数据划分、权重、阈值、实验输出都不共用。
**不要**把可见光任务的 `car/truck/bus/...` 编号照搬到红外任务上。

---

## 二、共同的第一步(所有人)

```bash
pip install -r requirements.txt          # 见文件内注释, torch 需按本机 CUDA 版本装

python tools/list_tasks.py                       # 看清三个任务与状态
python tools/list_tasks.py --task <你的任务>      # 看你的任务详情
python tools/train.py --task visible_recognition --dry-run   # 看已完成的那个长什么样
```

最后一条会打印模型结构、损失项、反向传播情况 —— 这是**接口连通的样板**,
你的任务做完后应该能得到类似的输出。

**目录职责速查**(改哪里):

| 你要改什么 | 改哪个文件 |
|---|---|
| 类别、数据路径、评测口径 | `configs/tasks/<你的任务>.yaml` |
| 网络结构与损失 | `configs/models/*.yaml` |
| 优化器 / 调度 / 增强 / 消融开关 | `configs/training/*.yaml` |
| **算法实现** | `src/models/adapters/<你的任务>.py` |
| 引擎代码 | `third_party/deim_engine/` —— **不要动**,要改请提出来讨论 |

---

## 三、红外目标识别(`infrared_recognition`)

**路线书第 6 节要求"保留四套原生对照"**,所以你的第一步不是写新模型,而是**把对照跑通**。

### 阶段 1:YOLOv8 / YOLO11 原生对照

1. 在 `src/models/adapters/` 下建 `yolo_family.py`
2. 实现 `postprocess` 时**必须带 NMS** —— YOLO 是密集预测 + anchor-free 框回归,
   与 DETR 的集合预测完全不同
3. 损失用 YOLO 配套的(objectness + 框回归),**不得**混入匈牙利匹配

> ⚠️ **路线书 2.3 节明令**:不能将 YOLO 检测头、AA objectness 或 DETR 匈牙利匹配任意组合。
> 框架的适配器设计已经强制了这一点(模型/损失/后处理三件套绑在一起),
> 但你自己新写适配器时仍要守住。

### 阶段 2:v2 框级识别基线

直接继承可见光任务的实现:

```python
from src.models.adapters.visible_recognition import VisibleRecognitionAdapter

@register_task(task_id='infrared_recognition', status='implemented')
class InfraredRecognitionAdapter(VisibleRecognitionAdapter):
    class_names = ['你的', '红外', '类别']        # 只改类别
```

模型、损失、后处理原样继承 —— 路线书把三个任务的框级识别都归到 v2 主干,
差异只在数据、类别与主干/融合模块的选择。

### 阶段 3:YOLO 模块 → v2 的迁移消融

路线书 2.3 节提醒:

> 网络模块迁移时检查特征通道、步长、张量布局、初始化和归一化设置,
> 并移除原框架中无关的导入。共享基础模块不意味着整文件复制后即可运行。

**每次只改一个因素**(路线书第 6 节),记录兼容性问题。

---

## 四、红外小目标检测(`infrared_small_target`)

**这个任务的评价口径和另外两个不一样**,路线书第 6 节单独列了:

| 任务类型 | 指标 |
|---|---|
| 框级识别(可见光/红外) | 检测 AP、逐类别精度与召回、类别混淆、完整业务时延 |
| **红外小目标** | **Pd/Recall、Precision、明确统计口径的 FAR、目标尺寸分组结果、推理资源占用** |

### 你必须先定下来的事:**FAR 的统计口径**

`configs/tasks/infrared_small_target.yaml` 里留了待填项:

```yaml
evaluation:
  far:
    unit: ""          # per_image | per_pixel | per_million_pixels
    definition: ""
```

**为什么这件事必须最先做**:FAR(虚警率)的分母不同,数字会差几个数量级。
"每张图几个虚警"和"每百万像素几个虚警"是完全不同的指标。**口径不写清楚,结果无法比较。**

### 接入顺序

1. **AA-YOLO 原生对照**(异常感知方法)
   - 属 YOLO 族:anchor-based 检测头 + objectness 异常分支
   - 后处理带 NMS,不要套 DETR 的匹配
2. **现有 RT-DETR 基线**(路线书:后续主线升级至 v2)
   - ⚠️ 路线书 2.2 节明确:"本地 `RTDETR_Basic_Source` 为 v1 思路的任务定制实现,
     **不能仅修改名称就视作 v2**。升级须明确替换或核对解码器、配置与训练策略。"
3. **v2 轻量基线**
   - 小目标方向的重点是**浅层特征与高分辨率**
   - 可能需要引入 stride-4 的 P2 尺度(配置里 `model.high_resolution` 留了位置)

### 路线书专门点名的检查项

> 对 DEIM 增强**检查是否过度缩小微小目标**

如果小目标任务想借鉴 DEIM 的 Dense O2O,注意 `Mosaic output_size: 320` 会把
4 张 960 图压到 320 画布 —— **对几像素的目标是灾难性的**。详见 README 第六节坑 6。

---

## 五、可见光任务(已完成,当参考样板)

已实现的内容与实测结果:

| 项 | 结果 |
|---|---|
| 模型 | RT-DETRv2 + PResNet-18-d,20,088,424 参数 |
| 损失 | MAL + L1 + GIoU(DEIM 增强) |
| 后处理 | 集合预测,**无 NMS** |
| **全量 test 评测**(3976 张 / 56375 GT) | `mAP@0.5 = 0.7415` / `mAP@.50:.95 = 0.4969` |
| **历史 recall 口径**(同 test) | **90.39%**(50958/56375) |
| **完整业务时延** | **27.3 ms/帧**(mean)/ **34.8 ms**(p95),GPU 预处理 + CUDA Graph |
| 类别混淆 | car↔truck 双向最多(638 + 470 次) |

**你要参考的是它的三个文件**:

- `src/models/adapters/visible_recognition.py` —— 适配器怎么写
- `configs/tasks/visible_recognition.yaml` —— 任务配置怎么写
- `configs/models/rtdetrv2_r18_visible.yaml` —— 模型配置怎么写(含 `baseline_v2` 纯 v2 对照段)

---

## 六、交付标准(路线书第 6 节)

每个任务交付时要给全这些:

| # | 交付项 | 说明 |
|---|---|---|
| 1 | **配置快照** | 训练用的完整配置(含覆盖后的最终值),放 `experiments/` |
| 2 | **权重** | 训练产出 + 选择依据(在**验证集**上选,不要用测试集) |
| 3 | **指标报告** | 按本任务的评测口径;框级识别给 AP + 逐类别 P/R + 混淆矩阵;小目标给 Pd/Precision/FAR + 尺寸分组 |
| 4 | **时延报告** | 用 `tools/infer.py --bench` 出,区分 forward / 预处理 / **完整业务时延**三档 |
| 5 | **实验记录** | 数据划分、预训练来源、输入尺寸、训练预算、评价规则(路线书第 6 节要求逐项记录) |
| 6 | **消融记录** | 每次只改一个因素;对 DEIM 增强要检查是否过度缩小小目标 |

---

## 七、几条硬规矩

1. **权重与阈值在验证集上选,测试集只用于最终评价一次。** 不要反复用测试集调参。
2. **不要跨任务混用类别编号。** 每个任务的 `label 0` 含义独立。
3. **不要把不同算法族的组件拼在一起。** 适配器的三件套是一个整体。
4. **`third_party/deim_engine/` 不要直接改。** 需要改请提出来,并在 `VERSION.md` 记录。
5. **时延数字必须标口径。** 只报 forward 会导致对接方按端到端测量时对不上账
   (本项目实测:forward 7.7ms,端到端 28.8ms,差 3.7 倍)。

---

## 八、遇到问题先查这三处

| 现象 | 先看 |
|---|---|
| 跑不起来 / 报 `_transform` 错误 | README 第七节环境说明(`patch_tw.py` 是否生效) |
| 数据加载不到 / 结果不可复现 | README 第六节的 8 个坑 |
| 不知道某段代码为什么这么写 | `docs/ARCHITECTURE.md` 第五节(设计决策理由) |
| 不确定某件事路线书怎么要求 | `docs/` 下的路线书原文,或问项目负责人 |
