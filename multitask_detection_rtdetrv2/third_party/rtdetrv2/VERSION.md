# RT-DETRv2 source pin

Source repository: https://github.com/lyuwenyu/RT-DETR , rtdetrv2_pytorch/src.

This integration vendors the exact local source snapshot used by the completed v2-S experiment. Per-file SHA256 and original file locations are recorded in docs/SOURCE_MANIFEST.json. No upstream commit SHA is inferred from an unversioned archive.

Network, matcher and criterion source files are unchanged. Package __init__ files, a no-op registration shim and distributed utility shim are local integration code. They avoid loading a second global `src` package or unrelated optional CLI dependencies. Distributed utility uses torch.distributed when initialized; this trainer is validated for one GPU.

License: Apache 2.0, retained in LICENSE. Original source copyright notices remain.
