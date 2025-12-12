# Coronary Vessel Segmentation using UMamba-UNet

This repository implements a complete coronary angiography (CAG) vessel segmentation pipeline using the UMamba-UNet architecture. It includes baseline training, topology-aware clDice learning, semi-supervised pseudo-label training, 5-fold cross-validation, and evaluation with Dice, IoU, Precision, Recall, and clDice metrics.

## Repository Structure

```
workspace/
├── train_umamba.py
├── evaluate.py
├── losses_cldice.py
├── umamba_unet.py
├── command.sh
└── ARCADE/
    ├── images/
    ├── masks/
    └── pseudo/
```

## Dataset

The ARCADE coronary angiography dataset is required.  
It is not included in this repository and must be downloaded separately.

Expected folder structure:

```
ARCADE/
├── images/
├── masks/
└── pseudo/
```

## Experiments (Milestone 4)

| Experiment | Folder | Description |
|-----------|--------|-------------|
| Baseline | `exp_baseline/` | GT only, Dice+BCE |
| Pseudo | `exp_A/` | Pseudo-labels + GT |
| clDice | `exp_B/` | GT only, Dice+clDice |
| Mixed | `exp_AB/` | GT + pseudo + clDice |
| Final | `exp_ABC/` | Mixed + TTA |

## Training Example

```bash
python train_umamba.py     --experiment baseline     --data_root ARCADE     --out_dir exp_baseline
```

Available modes:

```
baseline
pseudo
cldice
pseudo_cldice
final
```

## Evaluation Example

```bash
python evaluate.py     --ckpt exp_ABC/fold4_best.pth     --img_dir ARCADE/images     --gt_dir ARCADE/masks     --save_dir eval_ABC     --experiment final
```

Outputs:

- Dice, IoU, Precision, Recall  
- clDice  
- Predicted masks  

## Loss Functions

Implemented in `losses_cldice.py`:

- Dice + BCE  
- Dice + clDice  
- Mixed loss  

## Model Architecture

UMamba-UNet includes:

- UNet encoder–decoder  
- Mamba-style lightweight blocks  
- Depthwise separable convolutions  
- Skip connections  

Attention modules are disabled.

## TTA (Test-Time Augmentation)

Small-angle rotations and flips are used for stable predictions.

## File Descriptions

| File | Description |
|------|-------------|
| train_umamba.py | Training pipeline |
| evaluate.py | Evaluation with TTA and clDice |
| losses_cldice.py | Loss functions including clDice |
| umamba_unet.py | UMamba-UNet model |
| command.sh | Example script |
| Dockerfile | Environment setup |

## Notes

- ARCADE dataset cannot be redistributed; download from official source.
- All experiments follow Milestone 4 requirements.
- This repository provides training, evaluation, and model code only.
