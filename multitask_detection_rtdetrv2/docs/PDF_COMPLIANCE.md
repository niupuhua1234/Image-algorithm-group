# 技术路线书第5与第6页对应

| PDF条目 | 本工程落点 | 验证状态 |
|---|---|---|
| 5.1统一配置、任务入口、训练与记录 | src/core、src/engine、tools | 小目标实际运行；另外任务保留接口 |
| 5.1任务和算法分别配置 | 任务registry与models/factory算法registry | 算法族一致性检查通过 |
| 5.2小目标轻量v2基线 | configs/models/rtdetrv2_s.yaml | R18-vd、100queries、3层decoder、20,073,172参数 |
| 5.2三任务独立规模与类别 | configs/tasks三份、独立适配器 | 小目标1024×1280，可见光原尺寸，红外识别未指定 |
| 5.3图像、框、类别、原尺寸、变换 | src/data/coco.py | 真实图片训练及完整测试集读取通过 |
| 5.3统一结果与算法适配 | Detection、DetrTaskAdapter、postprocess | 框还原、分数筛选、任务可选NMS |
| 5.4建议目录 | configs四层及src对应子目录 | 已建立；组件通过公开模块导入 |
| 5.4独立权重、优化器、EMA、输出、类别 | ExperimentStore及完整checkpoint | 两步训练及下一轮恢复通过 |
| 第6节小目标Pd、Precision、FAR | src/metrics/small_target.py | IoU0.4、Fa=FP/(TP+FP)，边界/空集检查通过 |
| 第6节目标尺寸及资源 | 等效边长分组、runtime benchmark | 测试分组、200帧时间、显存统计范围写入JSON |
| 第6节固定源码 | third_party/rtdetrv2及SOURCE_MANIFEST | 原始模块按SHA256固定，保留许可证 |
| 第6节原生对照 | configs/models/native_baselines.yaml | 原R18保留外部引用；AA-YOLO、YOLO待补；DEIM交接保留 |
| 第6节高分辨率/异常先验/v4研究 | docs/研究边界 | 预留，不声称已实现或已消融 |

本次组织代码并核对既有模型，没有启动72轮正式重训。旧数据、旧权重、原D盘项目及可见光交接原目录保持为只读输入。

## 研究边界

小目标增强、高分辨率特征、异常先验及v4是PDF后续研究阶段，并非当前v2-S整理时自动加入的模块。未来每次选一个主要因素，在固定数据/指标下独立建实验，保留基线。单帧和真实五帧评估也是不同数据协议，当前静态测试不生成五帧成绩。
