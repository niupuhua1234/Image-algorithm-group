# -*- coding: utf-8 -*-
"""Post-training TensorRT validation selection for the imported R18VD task."""
import copy
import csv
import os
import shutil
import tempfile
from pathlib import Path

from src.core.common import save_json
from src.data.coco import CocoBoxDataset
from src.deployment.export import build_engine, export_onnx
from src.deployment.unified_runtime import DetectionRuntime
from src.engine.evaluation import collect


def _candidates(weights):
    names = ['best.pth', 'best_AP40.pth']
    found = [weights / name for name in names if (weights / name).is_file()]
    found.extend(sorted(weights.glob('checkpoint*.pth')))
    if (weights / 'last.pth').is_file():
        found.append(weights / 'last.pth')
    seen, unique = set(), []
    for item in found:
        item = item.resolve()
        if item not in seen:
            seen.add(item); unique.append(item)
    return unique


def _better_recall(current, best):
    if best is None:
        return True
    for key in ('Recall_IoU40', 'Recall_IoU50', 'Recall_IoU60', 'AP40'):
        delta = float(current[key]) - float(best[key])
        if abs(delta) <= 1e-12:
            continue
        return delta > 0.0
    return int(current['candidate_order']) > int(best['candidate_order'])


def _better_ap40(current, best):
    if best is None:
        return True
    delta = float(current['AP40']) - float(best['AP40'])
    if abs(delta) <= 1e-12:
        return int(current['candidate_order']) > int(best['candidate_order'])
    return delta > 0.0


def _replace_selected(best_recall, best_ap40, weights):
    """Snapshot sources first so two selected files can safely cross-reference."""
    temp = Path(tempfile.mkdtemp(prefix='.trt-final-', dir=str(weights)))
    try:
        recall_snapshot = temp / 'best_recall_source.pth'
        ap_snapshot = temp / 'best_ap40_source.pth'
        shutil.copy2(best_recall['source'], recall_snapshot)
        shutil.copy2(best_ap40['source'], ap_snapshot)
        os.replace(str(recall_snapshot), str(weights / 'best.pth'))
        os.replace(str(ap_snapshot), str(weights / 'best_AP40.pth'))
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def select_checkpoints(cfg, adapter, run_dir, device='cuda'):
    """Evaluate retained checkpoints with TensorRT FP16 on the val split.

    The ranking precisely follows the source selection script: R40, then R50,
    then R60, then AP40, with the later candidate breaking exact TensorRT ties.
    Generated ONNX/engine files are discarded unless explicitly retained.
    """
    options = cfg.get('tensorrt_selection') or {}
    if not options.get('enabled', False):
        return None
    run_dir, weights = Path(run_dir), Path(run_dir) / 'weights'
    candidates = _candidates(weights)
    if not candidates:
        raise FileNotFoundError(f'No candidate checkpoints in {weights}')
    selection_dir = run_dir / 'reports' / 'trt_selection'
    work_root = selection_dir / '_work'
    keep_artifacts = bool(options.get('keep_artifacts', False))
    rows = []
    for order, checkpoint in enumerate(candidates):
        work_dir = work_root / f'{order:02d}_{checkpoint.stem}'
        if work_dir.exists():
            shutil.rmtree(work_dir)
        row = {'candidate_order': order, 'checkpoint': checkpoint.name, 'source': checkpoint, 'status': 'failed'}
        try:
            candidate_cfg = copy.deepcopy(cfg)
            candidate_cfg['deployment']['checkpoint'] = str(checkpoint)
            candidate_cfg['deployment']['checkpoint_sha256'] = ''
            candidate_cfg['deployment']['trt_layernorm_fp32'] = bool(options.get('layernorm_fp32', True))
            candidate_cfg['deployment']['trt_workspace_mib'] = int(options.get('workspace_mib', 2048))
            candidate_cfg['evaluation']['source_max_detections'] = int(options.get('candidate_topk', 200))
            export_onnx(candidate_cfg, adapter, checkpoint, work_dir, device)
            engine = build_engine(candidate_cfg, checkpoint, work_dir, options.get('precision', 'fp16'))
            candidate_cfg['deployment']['engine'] = str(engine)
            candidate_cfg['deployment']['engine_metadata'] = str(engine.with_suffix('.build.json'))
            runtime = DetectionRuntime(candidate_cfg, adapter, checkpoint, device, 'tensorrt')
            warmup = int(options.get('warmup', 20))
            if warmup:
                data = CocoBoxDataset(candidate_cfg, 'val')
                for index in range(warmup):
                    runtime.infer_rgb(data.pixels(index % len(data)), conf=0.0)
            predictions, ground_truth, _, _ = collect(runtime, candidate_cfg, 'val', conf=0.0)
            metrics = adapter.evaluate(predictions, ground_truth)
            row.update({key: float(metrics[key]) for key in ('AP40', 'Recall_IoU40', 'Recall_IoU50', 'Recall_IoU60')})
            row['status'] = 'ok'
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            rows.append(row)
            if work_dir.exists() and not keep_artifacts:
                shutil.rmtree(work_dir, ignore_errors=True)

    valid = [row for row in rows if row['status'] == 'ok']
    if not valid:
        selection_dir.mkdir(parents=True, exist_ok=True)
        save_json(selection_dir / 'trt_best_metrics.json', {'rows': _json_rows(rows), 'status': 'all_candidates_failed'})
        raise RuntimeError('TensorRT validation failed for every infrared-recognition checkpoint')
    best_recall = best_ap40 = None
    for row in valid:
        if _better_recall(row, best_recall):
            best_recall = row
        if _better_ap40(row, best_ap40):
            best_ap40 = row
    if options.get('replace_best', False):
        _replace_selected(best_recall, best_ap40, weights)
    selection_dir.mkdir(parents=True, exist_ok=True)
    with (selection_dir / 'trt_selection_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['candidate_order', 'checkpoint', 'status', 'AP40', 'Recall_IoU40', 'Recall_IoU50', 'Recall_IoU60', 'error']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in _json_rows(rows): writer.writerow(row)
    summary = {
        'selection_backend': 'TensorRT FP16 validation',
        'layernorm_fp32': bool(options.get('layernorm_fp32', True)),
        'candidate_topk': int(options.get('candidate_topk', 200)),
        'workspace_mib': int(options.get('workspace_mib', 2048)),
        'replaced_training_best_files': bool(options.get('replace_best', False)),
        'selection_order': ['Recall_IoU40', 'Recall_IoU50', 'Recall_IoU60', 'AP40', 'later_candidate_on_tie'],
        'best_recall': _json_row(best_recall), 'best_AP40': _json_row(best_ap40),
        'rows': _json_rows(rows),
    }
    save_json(selection_dir / 'trt_best_metrics.json', summary)
    return summary


def _json_row(row):
    return {key: value for key, value in row.items() if key != 'source'}


def _json_rows(rows):
    return [_json_row(row) for row in rows]
