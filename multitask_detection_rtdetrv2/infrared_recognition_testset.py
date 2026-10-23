# -*- coding: utf-8 -*-
"""红外目标识别（infrared_recognition）测试集推理 —— 单文件总入口。

本文件的交付内容 = **全部配置（内联在 CONFIG，不依赖 configs/ 目录下的 YAML 链）
+ 推理模型 models/model.onnx**。原始权重 best.pth 与 TensorRT 引擎都不随包分发：
前者只有本机 PyTorch 推理才需要，后者与显卡/TRT 版本强绑定。
算法、数据、后处理与指标仍走工程里同一套代码（src/…），结果与 ``tools/eval.py`` 同源。

口径（与 reports/infrared_test_metrics_report.docx 一致）
    任务        infrared_recognition，4 类 car1/car2/car3/car4
    输入        1024×1280（高×宽），RGB，/255 归一化
    后处理      confidence_then_topk_no_nms，最多 200 框
    置信度阈值  0.67
    匹配 IoU    0.40（COCO 类别敏感匹配 → AP / Recall；框级 Pd / Fa 另算）
    测试集      400 张、1563 个真实目标

用法（在工程根目录执行）
    $py = 'D:\software\Anaconda\envs\RT-DETR\python.exe'

    & $py -B infrared_recognition_testset.py --export-assets models   # 导出推理模型到 ./models（默认只带 model.onnx）
    & $py -B infrared_recognition_testset.py --check                  # 校验交付资产（ONNX 必需，权重/引擎可选）
    & $py -B infrared_recognition_testset.py --show-config            # 打印最终生效配置
    & $py -B infrared_recognition_testset.py --ckpt <best.pth>        # 本机跑完整测试集（PyTorch FP32）
    & $py -B infrared_recognition_testset.py --engine <引擎>          # 本机跑完整测试集（TensorRT FP32）
    & $py -B infrared_recognition_testset.py --limit 5 --ckpt <best.pth>   # 冒烟
    & $py -B infrared_recognition_testset.py --out test.json --save-predictions
    & $py -B infrared_recognition_testset.py --export-assets models --with-weights --with-engine   # 需要时连权重/引擎一起打包

资产解析顺序（--参数优先级最高）
    推理模型 model.onnx  同级 models/ > 原工程 finetune/deployment/trt_diagnosis_fp32/（交付必需）
    权重 best.pth        命令行 --ckpt > 同级 models/ > 原工程（仅本机 PyTorch 推理需要）
    TRT 引擎             命令行 --engine > 同级 models/（不回落到原工程；与 GPU 绑定，默认不带）
    数据集不打包，换机器时用 --data-root 或环境变量 INFRARED_RECOGNITION_DATA_ROOT 指定。

复现情况（2026-10-22 本机 RTX 3050 Laptop / TRT 8.6.1 实测，400 张全量）
    带引擎（TensorRT FP32）：TP 1405 / FP 117 / FN 158，AP40 88.29%，Pd 89.89%，Fa 7.69%，
    与存档 reports/test_fp32.json、评估报告 docx 逐项一致。
    带权重（PyTorch FP32）：TP 1406，AP40 88.53%，相差 1 个框——IoU=0.40 边界上的单个框，
    属两个后端的数值差异；mAP 88.5% vs 88.6%，结论不受影响。
    单次全量约 33–50 秒。
"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------- 资产路径与指纹
# 本副本（- 副本）没有 finetune 产物，模型默认从原工程取；用 --export-assets 拷到
# 本文件同级 models/ 之后，入口会自动优先使用本地副本。
SOURCE_PROJECT = Path(r"E:\T\multitask_detection_rtdetrv2_20261016")
FINETUNE = SOURCE_PROJECT / "experiments/infrared_recognition/finetune"
LOCAL_MODELS = Path(__file__).resolve().parent / "models"

# 文件名 -> 原工程路径；导出与回落都按这张表走。
ORIGINAL_ASSETS = {
    "best.pth": FINETUNE / "weights/best.pth",
    "model_fp32.engine": FINETUNE / "deployment/trt_diagnosis_fp32/model_fp32.engine",
    "model_fp32.build.json": FINETUNE / "deployment/trt_diagnosis_fp32/model_fp32.build.json",
    "model.onnx": FINETUNE / "deployment/trt_diagnosis_fp32/model.onnx",
}
DATA_ROOT = SOURCE_PROJECT / "experiments/infrared_recognition/datasets/base"

WEIGHT_SHA256 = "588c04ca57eee7e35e4c8d0bb1e02c4b331c3e3b6c7d45395075855f3bdd1e28"
ENGINE_SHA256 = "51a1363c60a8ec6128e51b3c9b530d3d96a9ef1f3d0bfb40c0da6f3ed99323d6"
ONNX_SHA256 = "264cf543bca78b67429fb502a0342cca4eb10e7cd4962cbc3afb65d3dc7a50f7"
ENGINE_GPU = "NVIDIA GeForce RTX 3050 Laptop GPU"
ENGINE_TRT = "8.6.1"

CLASS_NAMES = ["car1", "car2", "car3", "car4"]
INPUT_SIZE = [1024, 1280]
CONF_THRESHOLD = 0.67
MATCH_IOU = 0.40
MAX_DETECTIONS = 200
SPLIT = "test"

# 存档报告口径，用于跑完后自动对账（reports/test_fp32.json）。
REFERENCE = {
    "tp": 1405, "fp": 117, "fn": 158, "targets": 1563, "detections": 1522,
    "AP40": 0.8829122003954614, "mAP": 0.8851, "pd": 0.8989123480486244, "fa": 0.07687253613666228,
    "images": 400,
}

# ---------------------------------------------------------------- 内联配置
# 取自 finetune/configs/resolved.yaml（该次训练实际生效的快照），只保留推理读取的字段：
# train / optimizer / scheduler / augmentation / experiment 等训练项推理不读，已省略。
CONFIG = {
    "model": {
        "arch": "rtdetrv2",
        "family": "detr",
        "input_size": INPUT_SIZE,
        "backbone": {"depth": 18, "variant": "d", "return_idx": [1, 2, 3], "num_stages": 4,
                     "freeze_at": -1, "freeze_norm": False, "pretrained": False},
        "encoder": {"in_channels": [128, 256, 512], "feat_strides": [8, 16, 32], "hidden_dim": 256,
                    "use_encoder_idx": [2], "num_encoder_layers": 1, "nhead": 8, "dim_feedforward": 1024,
                    "expansion": 0.5, "depth_mult": 1.0, "dropout": 0.0, "enc_act": "gelu", "act": "silu"},
        "decoder": {"feat_channels": [256, 256, 256], "feat_strides": [8, 16, 32], "hidden_dim": 256,
                    "num_levels": 3, "num_layers": 3, "num_queries": 300, "num_denoising": 100,
                    "label_noise_ratio": 0.5, "box_noise_scale": 1.0, "eval_idx": -1,
                    "num_points": [4, 4, 4], "cross_attn_method": "default", "query_select_method": "default"},
    },
    "loss": {
        "family": "detr",
        "losses": ["vfl", "boxes"],
        "weight_dict": {"loss_vfl": 1.5, "loss_bbox": 5, "loss_giou": 2},
        "alpha": 0.75,
        "gamma": 2.0,
        "matcher": {"weight_dict": {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
                    "use_focal_loss": True, "alpha": 0.25, "gamma": 2.0},
    },
    "task": {"id": "infrared_recognition", "business_name": "红外目标识别", "status": "implemented",
             "output_spec": "原图xyxy框、任务类别、分数", "native_baseline": "外部 RGBT RT-DETRv2 R18VD"},
    "classes": {"names": CLASS_NAMES, "category_id_to_label": {0: 0, 1: 1, 2: 2, 3: 3}},
    "data": {
        "root": str(DATA_ROOT),
        "train": {"img_folder": "images/train", "ann_file": "annotations/train.json", "batch_size": 1, "num_workers": 0},
        "val": {"img_folder": "images/val", "ann_file": "annotations/val.json", "batch_size": 1, "num_workers": 0},
        "test": {"img_folder": "images/test", "ann_file": "annotations/test.json", "batch_size": 1, "num_workers": 0},
    },
    "evaluation": {"protocol": "box_level_detection", "primary_iou": MATCH_IOU, "source_max_detections": MAX_DETECTIONS,
                   "metrics": ["detection_ap", "per_class_precision_recall", "class_confusion", "end_to_end_latency"]},
    "deployment": {
        "input_size": INPUT_SIZE,
        "default_backend": "tensorrt",
        "backends": ["pytorch", "tensorrt"],
        "checkpoint": "",                # 由 build_config() 按 --参数 > 本地 models/ > 原工程 解析填入
        "checkpoint_sha256": WEIGHT_SHA256,
        "conf_threshold": CONF_THRESHOLD,
        "postprocess": "confidence_then_topk_no_nms",
        "nms_iou": None,
        "trt_library_root": "",          # 需要独立 TensorRT 依赖目录时，用 --trt-root 覆盖
        "trt_layernorm_fp32": True,
        "engine": "",
        "engine_metadata": "",
    },
}


# ---------------------------------------------------------------- 工具函数
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_asset(name: str, override=None, allow_original=True):
    """定位一个模型资产：--参数 > 本文件同级 models/ >（可选）原工程。"""
    if override:
        return Path(override).resolve(), "命令行"
    local = LOCAL_MODELS / name
    if local.is_file():
        return local, "本地 models/"
    if not allow_original:
        return None, "未提供"
    return Path(ORIGINAL_ASSETS[name]), "原工程"


def build_config(args) -> dict:
    cfg = json.loads(json.dumps(CONFIG))          # 深拷贝，避免多次运行互相污染
    if args.data_root:
        cfg["data"]["root"] = str(Path(args.data_root).resolve())
    # 主模型（权重）允许回落到原工程，方便尚未导出时直接跑；引擎只认命令行与本地 models/，
    # 引擎不随包分发，缺失时自动走 PyTorch 后端。
    weight, w_src = resolve_asset("best.pth", args.ckpt)
    engine, e_src = resolve_asset("model_fp32.engine", args.engine, allow_original=False)
    meta, m_src = resolve_asset("model_fp32.build.json", args.engine_metadata, allow_original=False)
    cfg["deployment"]["checkpoint"] = str(weight)
    cfg["deployment"]["engine"] = str(engine) if engine else ""
    cfg["deployment"]["engine_metadata"] = str(meta) if meta else ""
    if args.ckpt:
        cfg["deployment"]["checkpoint_sha256"] = ""   # 显式换权重时由加载器只校验任务/类别元信息
    if args.trt_root:
        cfg["deployment"]["trt_library_root"] = str(Path(args.trt_root).resolve())
    if args.conf is not None:
        cfg["deployment"]["conf_threshold"] = float(args.conf)
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(Path(__file__).resolve())
    cfg["_asset_sources"] = {"best.pth": w_src, "model_fp32.engine": e_src, "model_fp32.build.json": m_src}
    return cfg


def export_assets(dest, with_weights=False, with_engine=False) -> int:
    """把交付资产拷到 dest；默认只带推理模型 model.onnx，权重与引擎需显式开启。"""
    import shutil

    target = Path(dest)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.mkdir(parents=True, exist_ok=True)
    names = ["model.onnx"]
    if with_weights:
        names += ["best.pth"]
    if with_engine:
        names += ["model_fp32.engine", "model_fp32.build.json"]
    print(f"导出到 {target}\n")
    print(f"{'文件':<24}{'大小':>10}  {'来源':<10}  状态")
    failed = 0
    for name in names:
        src, origin = resolve_asset(name)
        dst = target / name
        if src is None or not src.is_file():
            state, size = "源文件缺失，跳过", 0
            failed += 1
        elif src.resolve() == dst.resolve():
            state, size = "已在目标位置，跳过", src.stat().st_size
        else:
            shutil.copy2(src, dst)
            state, size = "已复制", dst.stat().st_size
        print(f"{name:<24}{size/1024/1024:>9.1f}M  {origin:<10}  {state}")
    lines = [f"{sha256(target / name)}  {name}" for name in names if (target / name).is_file()]
    if lines:
        (target / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n清单已写入 {target / 'SHA256SUMS.txt'}")
    print("交付内容 = 本 .py（全部配置）+ models/（推理模型）；数据集与权重不打包。")
    if not with_weights:
        print("提示：未带 best.pth；本机要跑推理需 --ckpt 指定权重，或加 --with-weights。")
    if not with_engine:
        print("提示：未带 TRT 引擎；需要时加 --with-engine。")
    return 1 if failed else 0


def check_assets(cfg) -> bool:
    """校验交付资产：推理模型 ONNX（必需），权重与引擎（可选，仅本机跑推理时需要）。"""
    dep, ok = cfg["deployment"], True
    sources = cfg.get("_asset_sources", {})
    onnx, onnx_src = resolve_asset("model.onnx")
    weight = Path(dep["checkpoint"]) if dep["checkpoint"] else None
    engine = Path(dep["engine"]) if dep["engine"] else None
    meta_path = Path(dep["engine_metadata"]) if dep["engine_metadata"] else None

    rows = (("推理模型 ONNX", "model.onnx", onnx, onnx_src, True),
            ("权重 best.pth", "best.pth", weight, sources.get("best.pth", "?"), False),
            ("TRT 引擎", "model_fp32.engine", engine, sources.get("model_fp32.engine", "?"), False))
    print(f"{'资产':<14}{'来源':<12}{'大小':>10}  {'状态':<8}")
    for label, _name, path, origin, required in rows:
        exist = bool(path) and Path(path).is_file()
        size = f"{Path(path).stat().st_size/1024/1024:.1f} MiB" if exist else "-"
        state = "OK" if exist else ("缺失" if required else "未提供")
        ok &= exist or not required
        print(f"{label:<14}{origin:<12}{size:>10}  {state:<8}")
    for path in (onnx, weight, engine):
        if path and Path(path).is_file():
            print(f"  {path}")
    if not ok:
        return False

    actual_onnx = sha256(onnx)
    print(f"\n推理模型 SHA256  {'OK' if actual_onnx == ONNX_SHA256 else '与导出记录不一致'}")
    print(f"  {actual_onnx}")
    ok &= actual_onnx == ONNX_SHA256

    if weight and weight.is_file():
        actual_weight = sha256(weight)
        print(f"权重 SHA256      {'OK' if actual_weight == dep['checkpoint_sha256'] else '不一致'}")
        ok &= actual_weight == dep["checkpoint_sha256"]
    else:
        actual_weight = None
        print("权重            未提供（只有 --backend pytorch 需要它）")

    if engine and meta_path and engine.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_engine = sha256(engine)
        print(f"引擎 SHA256      {'OK' if actual_engine == meta['engine_sha256'] else '不一致'}")
        print(f"引擎构建         TensorRT {meta.get('trt')} / {meta.get('gpu')} / {meta.get('precision')}")
        if actual_weight:
            print(f"引擎权重指纹     {'OK' if meta['weight_sha256'] == actual_weight else '与当前权重不匹配'}")
        ok &= actual_engine == meta["engine_sha256"]
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                same = name == meta.get("gpu")
                print(f"本机 GPU         {name}  {'与引擎一致' if same else '与引擎不一致：需在本机重建引擎'}")
                ok &= same
            else:
                print("本机 GPU         不可用（CUDA 未就绪）")
                ok = False
        except Exception as exc:                  # 环境缺 torch 时只报不抛
            print(f"本机 GPU         检查失败：{exc}")
            ok = False
    else:
        print("引擎             未提供")

    can_run = (weight and weight.is_file()) or (engine and engine.is_file())
    print("\n本机可跑推理    " + ("是" if can_run else
                              "否：包内只有 ONNX 交付件，需 --ckpt 指定权重或 --engine 指定引擎"))
    return ok


def print_config(cfg) -> None:
    import yaml
    print(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                         allow_unicode=True, sort_keys=False, default_flow_style=False))


def summarize(result, args) -> None:
    m = result["metrics"]
    print(f"\n任务 {result['task_id']} · 划分 {result['split']} · 后端 {args.backend}"
          f" · 置信度 {result['operating_point']['confidence']} · 匹配 IoU {m['iou_thr']}")
    if args.limit:
        print(f"注意：--limit {args.limit}，仅部分数据，指标不作数")
    print(f"图片 {args.limit or m.get('images') or REFERENCE['images']} · 真实目标 {m['targets']} · 预测框 {m['detections']}")
    print(f"TP {m['tp']} / FP {m['fp']} / FN {m['fn']}")
    print(f"Pd {m['pd']:.4f} · Precision {m['precision']:.4f} · Fa {m['fa']:.4f}")
    print(f"AP40 {m['AP40']:.6f} · mAP {m['mAP']:.4f}")
    print(f"Recall IoU40/50/60 {m['Recall_IoU40']:.4f} / {m['Recall_IoU50']:.4f} / {m['Recall_IoU60']:.4f}")
    for name, row in m["per_class"].items():
        print(f"  {name}: GT {row['num_gt']:>4}  TP {row['tp']:>4}  FP {row['fp']:>3}  FN {row['fn']:>3}"
              f"  P {row['precision']:.4f}  R {row['recall']:.4f}  AP {row['ap']:.4f}")
    print(f"\n权重 {result['checkpoint']}")
    print(f"权重 SHA256 {result['checkpoint_sha256']}")
    print(f"标注 SHA256 {result['annotation_sha256']}")

    if result["split"] == "test" and not args.limit:
        diff_tp = m["tp"] - REFERENCE["tp"]
        diff_ap = m["AP40"] - REFERENCE["AP40"]
        print(f"\n与存档报告对账：TP {REFERENCE['tp']} → {m['tp']}，"
              f"AP40 {REFERENCE['AP40']:.6f} → {m['AP40']:.6f}（{diff_ap:+.4f}）")
        if diff_tp == 0 and abs(diff_ap) < 1e-6:
            print("结论：与存档报告完全一致")
        elif abs(diff_tp) <= 1:
            print("结论：可复现，仅差 1 个框（IoU=0.40 边界上的单个框，属后端数值差异）")
        else:
            print("结论：与存档报告差异偏大，需检查权重、引擎、阈值或数据集版本")


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(description="红外目标识别：测试集推理（单文件总入口）")
    p.add_argument("--backend", choices=["tensorrt", "pytorch"], default="tensorrt",
                   help="默认 tensorrt（与测试集报告口径一致）；pytorch 为不用引擎的兜底路径")
    p.add_argument("--split", choices=["test", "val"], default=SPLIT)
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 张，用于冒烟")
    p.add_argument("--conf", type=float, default=None, help=f"覆盖置信度阈值（默认 {CONF_THRESHOLD}）")
    p.add_argument("--out", default=None, help="把完整结果 JSON 写到该路径")
    p.add_argument("--save-predictions", action="store_true", help="同时写出逐图预测（需 --out）")
    p.add_argument("--bench", type=int, default=0, help="预热后计时 N 次，输出端到端平均耗时")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default=None, help="覆盖数据集根目录（含 images/test、annotations/test.json）")
    p.add_argument("--ckpt", default=None, help="覆盖权重路径")
    p.add_argument("--engine", default=None, help="覆盖 TensorRT 引擎路径")
    p.add_argument("--engine-metadata", default=None, help="覆盖引擎构建记录路径")
    p.add_argument("--trt-root", default=None, help="独立 TensorRT 依赖目录（可选）")
    p.add_argument("--export-assets", metavar="DIR", default=None,
                   help="把推理模型拷到 DIR（相对路径按本文件目录解析；建议 models）")
    p.add_argument("--with-weights", action="store_true", help="导出时一并带上 best.pth（仅本机 PyTorch 推理需要）")
    p.add_argument("--with-engine", action="store_true", help="导出时一并带上 TRT 引擎与构建记录")
    p.add_argument("--check", action="store_true", help="只校验资产与 GPU")
    p.add_argument("--show-config", action="store_true", help="打印最终生效配置后退出")
    args = p.parse_args()

    cfg = build_config(args)
    if args.export_assets:
        return export_assets(args.export_assets, with_weights=args.with_weights, with_engine=args.with_engine)
    if args.show_config:
        print_config(cfg)
        return 0
    if args.check:
        sys.exit(0 if check_assets(cfg) else 1)

    from src.core.registry import get_adapter, load_all_adapters
    load_all_adapters()
    adapter_cls = get_adapter(cfg["task"]["id"])
    adapter = adapter_cls().bind(cfg)            # bind 会复核类别顺序与输入尺寸

    engine_path = cfg["deployment"]["engine"]
    if args.backend == "tensorrt" and not (engine_path and Path(engine_path).is_file()):
        print("未提供 TRT 引擎 → 本次按 PyTorch FP32 运行（需要引擎时加 --engine）")
        args.backend = "pytorch"
    if args.backend == "pytorch" and not Path(cfg["deployment"]["checkpoint"]).is_file():
        sys.exit(f"没有可用的权重：{cfg['deployment']['checkpoint']}\n"
                 f"包内只带推理模型 models/model.onnx，本机跑推理请用 --ckpt 指定 best.pth；\n"
                 f"ONNX 本身用于其它推理框架部署，或 rebuild 引擎。")

    # 走工程同一条评估链路：运行时构建 → 逐图推理 → adapter.evaluate 出指标。
    namespace = SimpleNamespace(split=args.split, limit=args.limit,
                                conf=args.conf if args.conf is not None else cfg["deployment"]["conf_threshold"],
                                bench=args.bench, out=args.out, save_predictions=args.save_predictions,
                                ckpt=None, device=args.device, backend=args.backend)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):     # 评估函数会打印完整 JSON，这里只留自定义摘要
        result = adapter.run_evaluation(cfg, namespace)
    summarize(result, args)
    if args.out:
        print(f"\n完整结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
