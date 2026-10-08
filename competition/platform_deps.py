"""Platform-specific dependency fallbacks for the copied research code."""

import importlib
import os
import sys

from .paths import ROOT


def configure_rtdetr_imports():
    """Make RT-DETR and a compatible pycocotools importable."""
    engine = ROOT / "engines" / "rtdetr"
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    try:
        importlib.import_module("pycocotools._mask")
    except ImportError as error:
        vendor = ROOT / "vendor" / "windows_py38"
        compatible = os.name == "nt" and sys.version_info[:2] == (3, 8) and vendor.is_dir()
        if not compatible:
            raise ImportError(
                "Install pycocotools for this Python/platform before using RT-DETR"
            ) from error
        sys.path.insert(0, str(vendor))
        importlib.import_module("pycocotools._mask")
    return engine
