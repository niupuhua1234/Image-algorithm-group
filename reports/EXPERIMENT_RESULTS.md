# Infrared Small-Target Experiment Summary

Evaluation is class agnostic. A prediction is a true positive only when it is
matched one-to-one to a ground-truth box at `IoU >= 0.40`.

`Pd = TP / (TP + FN)` and `FAR = FP / (TP + FP)`.

| Model and run | Training data | Test TP/FP/FN | Pd | FAR | F1 | Model / pipeline mean | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| RT-DETR-R18 zero shot | VEDAI real + traditional synthetic | 27/374/35 | 43.55% | 93.27% | 11.66% | 111.12 / 148.13 ms, 1650 Ti | Diagnostic only |
| AA-YOLO Tiny zero shot | VEDAI real + traditional synthetic | 5/286/57 | 8.06% | 98.28% | 2.83% | 106.24 / 123.09 ms, 1650 Ti | Rejected |
| RT-DETR HIT transfer | HIT-UAV 75 images / 1046 targets | 33/438/29 | 53.23% | 92.99% | 12.38% | 103.28 / 140.14 ms, 1650 Ti | Rejected |
| RT-DETR road V1 | 840 images / 3099 targets | 45/483/17 | 72.58% | 91.48% | 15.25% | 12.20 / 24.86 ms, 5090 | Rejected |
| RT-DETR P2-Lite V1 | 840 images / 3099 targets | 32/553/30 | 51.61% | 94.53% | 9.89% | 9.59 / 20.92 ms, 5090 | Rejected |
| RT-DETR road V2 | 1044 images / 2940 targets | 32/108/30 | 51.61% | 77.14% | 31.68% | 8.68 / 25.59 ms, 5090 | Low FAR, insufficient Pd |
| RT-DETR road V3 | 2954 images / 15466 targets | 50/890/12 | 80.65% | 94.68% | 9.98% | 8.93 / 26.82 ms, 5090 | Highest retained single-model Pd |
| RT-DETR P2-Lite V4 | V3 + 800 hard-negative images | 49/646/13 | 79.03% | 92.95% | 12.95% | 9.17 / 21.19 ms, 5090 | Fast baseline, below Pd target |
| RT-DETR V3 + V4 ensemble | Two-model ensemble | 51/659/11 | 82.26% | 92.82% | 13.21% | Two passes | Rejected for latency |
| AA-YOLO Tiny V4 | 3754 images / 15466 targets | Not formally threshold-tested | <5% default recall | - | - | - | Stop this training line |

## Interpretation

None of these HIT-UAV proxy-domain experiments reaches the internal target of
`Pd >= 90%` with low FAR. V3 has the best retained single-model test Pd, while
V4 is the best speed-oriented baseline. The V3+V4 ensemble gains only 1 target
over V3 and requires two model passes, so it is unsuitable for the `45 ms`
competition budget.

The sub-45-ms measurements above were made on one RTX 5090. They prove only
that the implementation is lightweight on that server. Final compliance must
include transfer, preprocessing, model, postprocessing, and output time on a
Jetson AGX Orin 64GB while all three competition models are running.

The next useful training round should wait for genuinely competition-like
thermal images with correct VOC boxes. Repeating epochs on the present proxy
set is unlikely to fix the domain mismatch.

The V4 structural audit found no missing images, corrupt files, invalid boxes,
or dimension errors. Its target geometry is nevertheless mismatched: median
box sizes are `28x40` for train, `34x44` for validation, and `22x32` for test,
whereas the competition target is approximately `20x12`. This is a data-domain
limitation, not a reason to run more epochs.
