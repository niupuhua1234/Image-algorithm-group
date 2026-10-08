# RT-DETR Low-Contrast Transfer Experiment

## 1. Experiment purpose

This experiment compares two transfer strategies for the same RT-DETR-R18
base model on low-contrast infrared targets:

1. **Ordinary transfer (`standard`)**: directly fine-tune the best weight
   trained on real VEDAI plus synthetic VEDAI.
2. **FOMAML transfer (`fomaml`)**: start with the same VEDAI weight, perform
   first-order MAML meta-training on the real and synthetic VEDAI domains, and
   then fine-tune with exactly the same low-contrast support images.

FOMAML is the first-order approximation of MAML. It is used here so the
experiment can run on a GTX 1650 Ti with 4 GB VRAM.

## 2. Dataset and leakage controls

- Target profile: `rtdetr_gpt_low_contrast_combined_v1`
- Image size: 1280 x 1024
- Class handling: every object is collapsed into one `target` class
- Train: 283 images / 770 targets
- Validation: 60 images / 158 targets
- Test: 62 images / 163 targets
- Matching: class-agnostic one-to-one matching at IoU >= 0.40
- Confidence and class-agnostic NMS thresholds are selected on validation only
- The fixed validation thresholds are applied once to test
- Test is not used for training, epoch selection, or threshold selection

Support sets are sampled from target-domain train with seed `20260924`. They
are nested, so the 5-shot images are contained in 10-shot, and 10-shot is
contained in 20-shot.

| Support | Images | Targets |
|---|---:|---:|
| 5-shot | 5 | 14 |
| 10-shot | 10 | 28 |
| 20-shot | 20 | 56 |

Support files are in:

`meta_learning/support/rtdetr_gpt_low_contrast_combined_v1/`

## 3. Initial weights

Ordinary-transfer initialization:

`weights/rtdetr/vedai_real_synth_best.pth`

FOMAML initialization:

`weights/rtdetr/vedai_real_synth_fomaml_best.pth`

FOMAML meta-training used only `VEDAI_real_638` and `VEDAI_synthetic_253`.
It did not use low-contrast train, validation, or test data. Its settings were
20 meta epochs, 10 episodes per epoch, 2/2 support/query, inner LR `1e-4`, and
meta LR `1e-5`.

## 4. Adaptation settings and commands

- 20 adaptation epochs
- Learning rate: `2e-5`
- Trainable modules: HybridEncoder and RT-DETR decoder
- Frozen module: backbone
- Best epoch selected with validation mAP50:95

Create deterministic support sets:

```powershell
D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe meta_learning\03_prepare_fewshot_support.py --profile rtdetr_gpt_low_contrast_combined_v1 --shots 5 10 20 --seed 20260924
```

Ordinary transfer example:

```powershell
D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe meta_learning\04_adapt_fewshot.py --method standard --profile rtdetr_gpt_low_contrast_combined_v1 --shots 5 --gpu 0 --epochs 20 --lr 0.00002 --seed 20260924 --initial-weight weights\rtdetr\vedai_real_synth_best.pth
```

FOMAML transfer example:

```powershell
D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe meta_learning\04_adapt_fewshot.py --method fomaml --profile rtdetr_gpt_low_contrast_combined_v1 --shots 5 --gpu 0 --epochs 20 --lr 0.00002 --seed 20260924 --fomaml-weight weights\rtdetr\vedai_real_synth_fomaml_best.pth
```

Evaluate all adapted weights:

```powershell
D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe meta_learning\05_compare_fewshot.py --profile rtdetr_gpt_low_contrast_combined_v1 --shots 5 10 20 --gpu 0 --iou 0.4 --target-pd 0.9
```

Evaluate the zero-shot baseline:

```powershell
D:\MySoftware\Anaconda3\envs\aa_yolo\python.exe meta_learning\06_evaluate_zero_shot.py --profile rtdetr_gpt_low_contrast_combined_v1 --weight weights\rtdetr\vedai_real_synth_best.pth --gpu 0 --iou 0.4 --target-pd 0.9
```

## 5. Test results

| Method | Support | TP/FP/FN | Pd | FAR | F1 | Pd >= 90% |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 144/29/19 | 88.34% | 16.76% | 85.71% | No |
| Ordinary transfer | 5 | 152/42/11 | **93.25%** | 21.65% | 85.15% | Yes |
| FOMAML | 5 | 150/32/13 | 92.02% | **17.58%** | **86.96%** | Yes |
| Ordinary transfer | 10 | 148/25/15 | **90.80%** | 14.45% | 88.10% | Yes |
| FOMAML | 10 | 145/19/18 | 88.96% | **11.59%** | **88.69%** | No |
| Ordinary transfer | 20 | 141/10/22 | 86.50% | 6.62% | 89.81% | No |
| FOMAML | 20 | 143/6/20 | **87.73%** | **4.03%** | **91.67%** | No |

FAR is `FP / (TP + FP)`. Bold values compare the main metrics within the same
shot count.

## 6. Conclusions

- When `Pd >= 90%` is mandatory, **FOMAML 5-shot** is the recommended balance:
  Pd 92.02%, FAR 17.58%, and F1 86.96%.
- **Ordinary 5-shot** has the highest Pd at 93.25%, but FAR rises to 21.65%.
- **FOMAML 20-shot** has the best FAR and F1 at 4.03% and 91.67%, but its Pd
  is 87.73%, so it does not satisfy the current Pd requirement.
- Pd is not monotonic with support size because the best epoch is selected by
  generic mAP. A useful next improvement is to select the epoch by the
  competition objective: first satisfy validation Pd, then minimize FAR.

Local GTX 1650 Ti full-precision model time is about 102-114 ms. This is not
the official Jetson AGX Orin TensorRT FP16 latency. Re-measure the 45 ms limit
on the competition device while all three required models are running.

## 7. Outputs

Complete table including zero-shot:

`outputs/meta_learning/evaluation/rtdetr_gpt_low_contrast_combined_v1/comparison_with_zero_shot.csv`

Adapted best weights:

`outputs/meta_learning/adaptation/rtdetr_gpt_low_contrast_combined_v1/<method>_<shots>shot/best.pth`

Each completed run retains `best.pth`, `training_complete.json`, a JSONL
training log, and TensorBoard logs. Redundant `last.pth` files were removed
during the 2026-10-03 cleanup. Validation/test predictions, threshold grids,
and metrics remain under the matching evaluation directory.
