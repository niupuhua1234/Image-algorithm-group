# Image-algorithm-group

## 数据、模型权重与实验结果下载

代码与说明文档通过 GitHub 分支 `Infrared-Small-Target-Detection` 提供；大体积数据、模型权重和实验结果通过同一个网盘资源包提供。

- 网盘下载地址：[百度网盘下载](https://pan.baidu.com/s/1kORKAJxS6XA2yqmHRWq5dg)
- 提取码：`vtgf`
- 压缩包名称：`InfraredSmallTargetDetection-data.zip`
- 2026-10-08 本地核查：未压缩资源合计约 **3.30 GiB**；实际压缩包大小以网盘为准。

### 资源包包含哪些文件

从原项目根目录同时选择以下三个文件夹，完整压缩到同一个 ZIP 中。使用复制或压缩操作，保留原项目中的文件。

| 文件夹 | 内容 | 未压缩大小（约） |
|---|---|---:|
| `datasets/` | 两套 prepared 数据集，以及数据目录说明；包含图片、VOC XML、COCO JSON 和划分清单 | 2.69 GiB |
| `weights/` | 两个历史 RT-DETR 权重：VEDAI 原图/合成图基础权重及 FOMAML 初始化权重，以及用途说明 | 154 MiB |
| `outputs/` | GPT 低亮场景普通迁移和 FOMAML 的 5/10/20-shot 最优权重、训练日志、元数据、评估结果 | 475 MiB |

请保留这些目录下的全部内容及现有 train/val/test 划分，不要只打包图片或单独抽出 best.pth。datasets/README.md、datasets/CATALOG.md 和 weights/README.md 等小型说明文档也应保留在 GitHub 代码仓库中。

### 下载后的放置方法

解压到克隆得到的代码仓库根目录，即 train_basic.py、project_config.yaml 所在目录。最终相对结构应为：

```text
Image-algorithm-group/
├─ train_basic.py
├─ test_basic.py
├─ project_config.yaml
├─ datasets/
│  └─ prepared/
│     ├─ base/
│     └─ rtdetr_gpt_low_contrast_combined_v1/
├─ weights/
│  └─ rtdetr/
│     ├─ vedai_real_synth_best.pth
│     └─ vedai_real_synth_fomaml_best.pth
└─ outputs/
   └─ meta_learning/
      ├─ adaptation/
      │  └─ rtdetr_gpt_low_contrast_combined_v1/
      │     ├─ standard_005shot/
      │     ├─ standard_010shot/
      │     ├─ standard_020shot/
      │     ├─ fomaml_005shot/
      │     ├─ fomaml_010shot/
      │     └─ fomaml_020shot/
      └─ evaluation/
```

若解压软件额外生成了同名资源包文件夹，请将其内部的 datasets、weights、outputs 三个文件夹复制到代码根目录，不要让数据多嵌套一层。与已有文件合并时，先核对同名文件，避免覆盖后来产生的新实验结果。

### 现有资源与运行限制

- 当前资源包保留的是已有数据和历史实验结果，不包含新一轮随机初始化训练产生的 `outputs/rtdetr/base/best.pth`。该文件需完成基本项训练后生成，或在支持指定权重的入口中显式使用已有历史权重。
- `outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/standard_010shot/best.pth` 为已保留的普通迁移权重，也是增量代码使用的初始权重。
- 当前资源包不包含增量阶段 1/2 的数据和训练结果，也不包含真实连续五帧评测数据；不能仅凭本资源包直接完成这些实验。
- 根目录的 `meta_learning/` 是代码目录，要上传 GitHub；`outputs/meta_learning/` 是实验产物，放入网盘资源包。两者不要混淆。
- 上传代码时应通过 .gitignore 排除 `datasets/prepared/`、`outputs/` 和模型权重（如 `*.pth`）；网盘资源 ZIP 也不要加入 Git 提交。
