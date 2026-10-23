# 继承与多态及任务差异

RT-DETR v2是建设主线，红外小目标当前选择v2-S。任务标识与算法配置分别管理，公共入口不把某个算法永久固定到某个业务。

## 实际继承关系

```text
TaskAdapter                         抽象业务契约，统一Detection输出
└─ DetectionTaskAdapter              数据加载、运行时、训练/评估调度、DETR结果解码
   └─ DetrTaskAdapter                从算法注册表构建配套模型与损失
      ├─ InfraredSmallTargetAdapter  单类约束、小目标Pd/Fa/尺寸组、验证工作点策略
      ├─ VisibleRecognitionAdapter  保留DEIM模型/损失/增强的特化，复用公共解码
      └─ InfraredRecognitionAdapter  预留，独立尺寸与类别尚待定义，明确拒绝执行

InferenceBackend                    抽象推理后端
├─ PyTorchBackend                   读取配置构建模型并严格加载权重
└─ TensorRTBackend                  复用引擎、缓存、流，验证来源与形状

DetectionRuntime                    组合TaskAdapter和InferenceBackend
CocoBoxDataset                      统一COCO样本及原图/变换记录
ExperimentStore                     统一记录方式，各任务独立存储
ModelEMA                            公共EMA逻辑，各模型各自实例
```

公共CLI通过注册表取得适配器后，调用 `adapter.run_training()`、`adapter.run_evaluation()`、`adapter.create_runtime()`。调用方式一致，由对象的具体类型决定最终行为。

例如训练验证调度调用 `adapter.evaluate(preds, gts)`：小目标子类计算Pd、Precision、Fa及尺寸组；框级识别继承公共AP、逐类别P/R及混淆统计。未来更换业务评估方式只覆盖该方法，不复制训练循环。

运行时调用 `backend.predict(rgb)`，PyTorch与TensorRT分别实现该抽象接口。上层使用同一个 `Detection`，不用识别各引擎的内部输出。

## 哪些统一

配置合并、任务注册、算法注册、COCO交换格式、图像原始尺寸与变换记录、Detection结构、相容DETR解码、通用推理调度、设备同步、计时边界、训练记录、EMA、优化器分组、断点保存与恢复。

网络组件从 `src/models/backbones`、`encoders`、`decoders` 公开导入；实现保存在固定third_party命名空间。配套匹配和损失位于 `src/losses`，避免将YOLO头与DETR匹配任意拼接。

## 哪些保留差异

| 差异 | 处理 |
|---|---|
| 输入尺寸 | 小目标[1024,1280]，可见光沿用其配置，红外识别留空；公共类读取尺寸，不硬编码为正方形 |
| 类别及原始COCO编号 | 每个任务独立names与category_id_to_label，小目标固定target=0 |
| 模型规模 | v2-S属于当前小目标选择；不强制其他任务都使用S规模 |
| 预处理和增强 | 基础COCO读取可共享；交接版DEIM的Mosaic/MixUp增强仍为其独立策略 |
| 损失及匹配 | 小目标VFL+L1+GIoU+匈牙利匹配；可见光保留DEIM配套；不混合算法族 |
| 后处理 | 公共DETR解码；小目标保留当前可选NMS，可见光保持无NMS策略 |
| 评估 | 小目标Pd/Fa/尺寸；识别任务AP/逐类别/混淆 |
| 权重、EMA、优化器、阈值 | 各任务、各实验独立实例与文件 |

可见光原CUDA Graph运行时保留为交接兼容代码，未强行应用于小目标。当前已验证的公共后端为PyTorch与TensorRT。

## 数据与结果契约

内部标注交换使用原图xyxy、labels及original_size；训练适配为归一化cxcywh；元数据记录resize前后尺寸及水平翻转。模型内部原始输出通过适配器转回原图xyxy。

Detection包括boxes[N,4]、scores[N]、labels[N]。labels只在该任务的类别空间内解释。小目标对外labels恒为0，业务只需框和分数。

## 训练边界

公共监督训练模板通过适配器构建模型、损失、数据和验证选择策略。v2 Criterion内部已经加权，训练总损失直接求和，避免重复加权。完整恢复断点携带任务、类别、模型配置、训练配方、标注哈希和随机状态；不一致则拒绝恢复。

红外小目标基础训练已实测。DEIM训练具有自己的增强调度，保留原配方和接口，未将其强行换成小目标训练配方。其他任务的公共入口需该负责人补齐策略并验证后再启用。
