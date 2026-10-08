# 迁移后的两阶段域增量代码

## 清理状态（2026-10-02）

按用户要求，已删除阶段1/2全部权重、日志、结果、两个prepared划分、prepared_summary.json及RESULTS.md。本目录只保留代码，不能直接运行已有增量测试。基本项、普通迁移和原始数据未删除。

## 顺序与文件

基本项 → 普通迁移 → 阶段1新风格样本＋旧train回放 → 阶段2新风格样本＋已见train回放。
仍只有target类别，不使用FOMAML更新。

- prepare.py：以后显式重建旧pilot划分，依赖外部V3、D:/导出后和项目内405张低亮数据。本次未执行。正式比赛需按现场样本重新配置，不是通用导入器。
- train.py：调用competition/incremental_trainer.py，阶段1从普通迁移权重开始，阶段2从阶段1最优权重开始。
- test.py：val选confidence/NMS，固定后测试每个域。未见域结果是诊断，不是当前阶段正式分数。
- competition/incremental_trainer.py：有监督微调＋回放。使用meta_learning/common.py的通用工具，但不运行元学习内外循环。

旧pilot默认来源：
outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth

重做实验前核对来源是否与新的基本项/迁移链一致。默认2轮仅为流程试验，不是最终训练建议。已有增量数据和权重均不存在，须以后明确准备并训练才能测试。

没有真实时间序列不能生成五帧指标；学习时长需记录样本下发至提交的全程；Orin三模型并发延迟仍未验证。
