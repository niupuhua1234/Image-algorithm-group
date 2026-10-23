# 补充红外目标识别与可见光任务

## 红外目标识别

预留位置为 `configs/tasks/infrared_recognition.yaml`、`src/models/adapters/infrared_recognition.py`、`experiments/infrared_recognition/`。当前status为placeholder，公共入口会明确提示尚未实现。

接入时由负责人确定独立的业务类别、COCO编号映射、输入高宽、数据划分和训练配方。模型优先复用DetrTaskAdapter及v2算法工厂，通过configs/models选择规模；相容的v2模型、匹配、损失和后处理一并配置。

实现自己的验证选点策略或明确固定阈值，复用框级AP/P/R/混淆评估。取消占位保护并通过形状、类别映射、训练与推理检查后，才把适配器与配置状态改为implemented。不要从小目标复制target类别或1280×1024假设。

若接入YOLO原生对照，另注册匹配的算法组件；仅共享COCO、Detection、记录和调度契约，保留其检测头、损失、匹配及后处理。

## 可见光任务

原交接版的模型、DEIMCriterion、增强配置、third_party与旧工具入口保留。适配器已继承DetrTaskAdapter，并复用公共DETR输出解码。它的MAL损失、输入尺寸、300queries和Mosaic/MixUp不被小目标参数覆盖。

当前状态为handover_available，表示有交接代码但未完成本机统一入口复验，不冒充已完成集成。交接环境依赖见 `docs/handover_visible/README.md`。在适配其增强数据返回与公共调度、实现相应验证选点策略后，再启用公共训练/评估入口。

同一程序可拥有多个适配器实例；每个对象持有自己的cfg、class_names、模型和运行时。不要用全局单例跨任务共享可变权重或阈值。
