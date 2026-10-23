"""Train the infrared branch and automatically create its deployment config."""
import json
import signal
import shutil
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.infrared_recognition.bootstrap import install  # noqa: E402
from src.core.config import dump_resolved, resolve_path  # noqa: E402
from src.engine.application import parser, setup  # noqa: E402


def _staging_run_dir(cfg, name):
    return (
        resolve_path(cfg, cfg['experiment']['root']) / cfg['task']['id'] /
        cfg['model']['arch'] / cfg['experiment']['stage'] / name
    )


def _flat_run_dir(cfg, name):
    return resolve_path(cfg, cfg['experiment']['root']) / cfg['task']['id'] / name


def _remove_empty_parents(path, stop):
    """Remove only known, empty staging directories; never recurse."""
    current, stop = Path(path), Path(stop)
    while current != stop and current.exists() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent


def _move_run(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Final infrared run already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    _remove_empty_parents(source.parent, destination.parent)


def _stage_existing_run(cfg, args):
    """Resume a flat run or an interrupted run already in the staging directory."""
    final_run = _flat_run_dir(cfg, args.name)
    staging_run = _staging_run_dir(cfg, args.name)
    if not args.resume:
        if staging_run.exists() or final_run.exists():
            raise FileExistsError(f'Run already exists: {final_run} or {staging_run}; use --resume')
        return final_run, staging_run
    if staging_run.exists() and final_run.exists():
        raise FileExistsError(f'Both run directories exist; resolve the conflict: {final_run}, {staging_run}')
    if not staging_run.exists() and not final_run.exists():
        raise FileNotFoundError(f'Cannot resume: no run directory at {final_run} or {staging_run}')

    current_run = staging_run if staging_run.exists() else final_run
    if args.resume == 'auto':
        resume = current_run / 'weights' / 'last.pth'
    else:
        resume = resolve_path(cfg, args.resume)
        # Existing commands may point at the final path even if an abrupt stop
        # left that same run in staging (or vice versa).
        for other_run in (final_run, staging_run):
            try:
                relative = resume.resolve().relative_to(other_run.resolve())
            except ValueError:
                continue
            resume = current_run / relative
            break
    if not resume.is_file():
        raise FileNotFoundError(f'Resume checkpoint does not exist: {resume}')

    if current_run == final_run:
        try:
            relative = resume.resolve().relative_to(final_run.resolve())
        except ValueError:
            relative = None
        staging_run.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(final_run), str(staging_run))
        if relative is not None:
            resume = staging_run / relative
    args.resume = str(resume)
    return final_run, staging_run


def _install_interrupt_guard():
    """Require a second Ctrl+C within five seconds to interrupt training."""
    previous = signal.getsignal(signal.SIGINT)
    last_press = None

    def on_interrupt(signum, frame):
        nonlocal last_press
        now = time.monotonic()
        if last_press is not None and now - last_press <= 5:
            raise KeyboardInterrupt
        last_press = now
        print('\nCtrl+C detected. Training continues; press Ctrl+C again within '
              '5 seconds to stop.', flush=True)

    signal.signal(signal.SIGINT, on_interrupt)
    return previous


def main():
    install()
    command = parser('Infrared recognition training with automatic deployment calibration')
    command.set_defaults(task='infrared_recognition')
    command.add_argument('--name', default='finetune')
    command.add_argument('--resume', nargs='?', const='auto', default=None,
                         help='Resume this run from weights/last.pth, or specify a checkpoint path')
    command.add_argument('--epochs', type=int, default=None)
    command.add_argument('--max-steps', type=int, default=None, help='SMOKE ONLY: limit steps per epoch')
    command.add_argument('--val-limit', type=int, default=None, help='SMOKE ONLY: limit validation')
    command.add_argument('--dry-run', action='store_true')
    command.add_argument('--show-config', action='store_true')
    command.add_argument('--initialize', default=None)
    args = command.parse_args()
    cfg, adapter = setup(args)
    if args.show_config:
        print(dump_resolved(cfg))
        return None
    if args.initialize:
        cfg['train']['initialize'] = args.initialize
    if args.dry_run:
        return adapter.run_training(cfg, args)

    final_run, staging_run = _stage_existing_run(cfg, args)
    previous_interrupt_handler = _install_interrupt_guard()
    try:
        result = adapter.run_training(cfg, args)
    except (Exception, KeyboardInterrupt):
        # Keep even a failed/interrupted run at its documented flat location.
        if staging_run.exists() and not final_run.exists():
            _move_run(staging_run, final_run)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_interrupt_handler)
    _move_run(staging_run, final_run)
    result['path'] = str(final_run)

    is_smoke = bool(args.max_steps or args.val_limit)
    calibration = cfg['deployment'].get('threshold_calibration') or {}
    if calibration.get('run_after_training', False) and not is_smoke:
        from experiments.infrared_recognition.calibrate_threshold import calibrate
        result['deployment_calibration'] = calibrate(
            cfg, adapter, final_run / 'weights' / 'best.pth', final_run,
            args.device, args.backend, force=True,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == '__main__':
    main()
