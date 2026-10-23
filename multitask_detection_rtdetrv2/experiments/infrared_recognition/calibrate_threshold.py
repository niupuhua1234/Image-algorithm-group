"""Calibrate only the infrared-recognition deployment threshold on validation data.

Training checkpoint selection remains source-compatible (R40 -> R50 -> R60 ->
AP40 with all RT-DETR candidates).  This script is deliberately separate: it
chooses the business inference score threshold after a final checkpoint exists.
"""
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.common import digest, save_json  # noqa: E402
from src.core.config import resolve_path  # noqa: E402
from src.engine.application import parser, setup  # noqa: E402
from src.engine.evaluation import collect  # noqa: E402
from src.metrics.detection import evaluate_detection  # noqa: E402


def thresholds(spec):
    """Return an inclusive, reproducible threshold grid from task configuration."""
    low = float(spec['min'])
    high = float(spec['max'])
    step = float(spec['step'])
    if not 0.0 <= low <= high <= 1.0 or step <= 0.0:
        raise ValueError('threshold_calibration.candidates must satisfy 0 <= min <= max <= 1 and step > 0')
    values = []
    value = low
    while value <= high + step * 1e-9:
        values.append(round(min(value, high), 10))
        value += step
    return values


def summary_at_threshold(predictions, ground_truth, class_names, iou_thr, threshold):
    """Score cached zero-threshold predictions without rerunning the network."""
    filtered = [prediction.filter_by_score(threshold) for prediction in predictions]
    metrics = evaluate_detection(filtered, ground_truth, class_names, iou_thr=iou_thr)
    rows = [row for row in metrics['per_class'].values() if int(row['num_gt']) > 0]
    if not rows:
        raise ValueError('Validation split has no labelled targets')

    def mean(name):
        return sum(float(row[name]) for row in rows) / len(rows)

    tp = sum(int(row['tp']) for row in rows)
    fp = sum(int(row['fp']) for row in rows)
    fn = sum(int(row['fn']) for row in rows)
    return {
        'threshold': float(threshold),
        'macro_f1': mean('f1'),
        'macro_recall': mean('recall'),
        'macro_precision': mean('precision'),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'fa': fp / (tp + fp) if tp + fp else None,
        'metrics': metrics,
    }


def select_best(rows):
    """Maximise class-balanced F1, then preserve recall and suppress extra detections."""
    return max(
        rows,
        key=lambda row: (
            row['macro_f1'], row['macro_recall'], row['macro_precision'], row['threshold'],
        ),
    )


def relative_to_project(cfg, path):
    return os.path.relpath(str(Path(path).resolve()), cfg['_project_root']).replace('\\', '/')


def calibrate(cfg, adapter, checkpoint, run_dir, device='cuda', backend='pytorch', output_dir=None, force=False):
    """Write a frozen deployment config for one completed infrared run."""
    settings = cfg['deployment'].get('threshold_calibration') or {}
    if settings.get('objective') != 'macro_f1':
        raise ValueError('This infrared calibration entry supports objective: macro_f1')

    run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    output_dir = Path(output_dir).resolve() if output_dir else run_dir / 'deployment'
    report_path = output_dir / 'threshold_calibration.json'
    frozen_path = output_dir / 'frozen_task.yaml'
    if not force and (report_path.exists() or frozen_path.exists()):
        raise FileExistsError(f'Calibration output already exists: {output_dir}; use --force to replace it')

    # Generate all candidate detections once. Threshold scoring below only filters
    # cached Detection objects, so this is 99 metric passes rather than 99 model runs.
    runtime = adapter.create_runtime(cfg, checkpoint, device, backend)
    split = settings.get('split', 'val')
    predictions, ground_truth, _, dataset = collect(runtime, cfg, split, conf=0.0)
    iou_thr = float(settings.get('iou_thr', cfg['evaluation']['primary_iou']))
    rows = [
        summary_at_threshold(predictions, ground_truth, adapter.class_names, iou_thr, candidate)
        for candidate in thresholds(settings['candidates'])
    ]
    best = select_best(rows)
    report = {
        'task_id': cfg['task']['id'],
        'selection_role': 'deployment confidence calibration; independent of training checkpoint selection',
        'split': split,
        'partial_dataset': False,
        'backend': backend,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': digest(checkpoint),
        'annotation_sha256': digest(dataset.annotation_path),
        'class_names': adapter.class_names,
        'iou_thr': iou_thr,
        'objective': settings['objective'],
        'tie_breakers': settings.get('tie_breakers', []),
        'selected': best,
        'candidates': rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_path, report)
    frozen = {
        'defaults': [os.path.relpath(cfg['_config_path'], output_dir).replace('\\', '/')],
        'deployment': {
            'checkpoint': relative_to_project(cfg, checkpoint),
            'checkpoint_sha256': digest(checkpoint),
            'conf_threshold': best['threshold'],
        },
    }
    frozen_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return {
        'threshold': best['threshold'], 'macro_f1': best['macro_f1'],
        'macro_recall': best['macro_recall'], 'macro_precision': best['macro_precision'],
        'frozen_config': str(frozen_path),
    }


def main():
    command = parser('Calibrate infrared-recognition deployment confidence on validation data')
    command.set_defaults(task='infrared_recognition')
    command.add_argument('--run-dir', required=True, help='Completed run directory, not its weights subdirectory')
    command.add_argument('--output-dir', default=None, help='Defaults to <run-dir>/deployment')
    command.add_argument('--force', action='store_true', help='Replace existing calibration files in output-dir')
    args = command.parse_args()
    cfg, adapter = setup(args)
    run_dir = resolve_path(cfg, args.run_dir)
    checkpoint = resolve_path(cfg, args.ckpt) if args.ckpt else run_dir / 'weights' / 'best.pth'
    output_dir = resolve_path(cfg, args.output_dir) if args.output_dir else None
    print(json.dumps(calibrate(
        cfg, adapter, checkpoint, run_dir, args.device, args.backend, output_dir, args.force,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
