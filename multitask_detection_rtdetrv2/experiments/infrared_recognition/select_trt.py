"""Re-run source-style TensorRT checkpoint selection for a completed infrared run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.engine.application import parser, setup  # noqa: E402
from src.engine.infrared_trt_selection import select_checkpoints  # noqa: E402


if __name__ == '__main__':
    command = parser('Select infrared RT-DETR checkpoints with TensorRT validation')
    command.set_defaults(task='infrared_recognition')
    command.add_argument('--run-dir', required=True, help='Completed experiment directory, not its weights subdirectory')
    command.add_argument('--force-enable', action='store_true', help='Run optional TensorRT validation even when the task config disables it')
    args = command.parse_args()
    cfg, adapter = setup(args)
    if args.force_enable:
        cfg.setdefault('tensorrt_selection', {})['enabled'] = True
    print(select_checkpoints(cfg, adapter, args.run_dir, args.device))
