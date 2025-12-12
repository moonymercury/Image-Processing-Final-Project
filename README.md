# Coronary Vessel Segmentation using Mamba-Inspired UNet (Mi-UNet)

This repository implements a complete coronary angiography (CAG) vessel segmentation pipeline using a lightweight **Mamba-inspired UNet (Mi-UNet)** architecture.  
The pipeline includes baseline training, topology-aware clDice learning, semi-supervised pseudo-label training, 5-fold cross-validation, and evaluation with Dice, IoU, Precision, Recall, and clDice metrics.

---

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

---

## Dataset

The ARCADE coronary angiography dataset is required for the experiments.  
It is **not included in this repository** and must be downloaded separately.

Expected dataset directory structure:

```
ARCADE/
├── images/
├── masks/
└── pseudo/
```

---

# Model Architecture: Mamba-Inspired UNet (Mi-UNet)

Although the source code uses the filename `umamba_unet.py`, the actual design is a **Mamba-inspired UNet**, not a full Mamba-SSM implementation.

### Key characteristics

- No Mamba-SSM  
- No Selective Scan  
- No recurrent SSM kernels  
- No sequence-style Mamba operations  

Instead, Mi-UNet employs:

- Lightweight depthwise-separable convolutions (DSConv)  
- Gated mixing layers inspired by the Mamba architecture  
- Efficient convolution-based feature extraction suitable for CAG segmentation  

### Rationale

This design preserves Mamba's *gating and mixing spirit*, while avoiding:

- High VRAM usage  
- Training instability  
- Heavy SSM kernel dependencies  

It ensures full compatibility with:

- clDice topology-aware loss  
- pseudo-label training  
- TTA  
- 5-fold cross-validation  
- Existing segmentation pipelines  

---

## Experiments (Milestone 4)

| Experiment | Folder | Description |
|-----------|--------|-------------|
| Baseline | `exp_baseline/` | GT-only, Dice+BCE |
| Pseudo | `exp_A/` | GT + pseudo-labels |
| clDice | `exp_B/` | GT + Dice+clDice |
| Mixed | `exp_AB/` | GT + pseudo + clDice |
| Final | `exp_ABC/` | Mixed + TTA |

---

## Training Example

```
python train_umamba.py     --experiment baseline     --data_root ARCADE     --out_dir exp_baseline
```

Available experiment modes:

```
baseline
pseudo
cldice
pseudo_cldice
final
```

---

## Evaluation Example

```
python evaluate.py     --ckpt exp_ABC/fold4_best.pth     --img_dir ARCADE/images     --gt_dir ARCADE/masks     --save_dir eval_ABC     --experiment final
```

Outputs include:

- Dice  
- IoU  
- Precision  
- Recall  
- clDice  
- Predicted masks  

---

## Loss Functions

Defined in `losses_cldice.py`:

- Dice + BCE loss  
- Dice + clDice loss  
- Mixed loss variants  

clDice enhances the continuity of thin vessel structures, improving segmentation topology.

---

## Test-Time Augmentation (TTA)

The final experiment applies:

- Small-angle rotations  
- Horizontal flips  

TTA stabilizes predictions in challenging low-contrast coronary angiography images.

---

## File Descriptions

| File | Description |
|------|-------------|
| `train_umamba.py` | Training script for all experiment modes |
| `evaluate.py` | Evaluation script with clDice and TTA |
| `losses_cldice.py` | Loss functions including Dice, BCE, clDice |
| `umamba_unet.py` | Implementation of the Mamba-inspired UNet (Mi-UNet) |
| `command.sh` | Example script for running experiment pipelines |
| `Dockerfile` | Optional environment configuration |

---

## Notes

- The ARCADE dataset must be downloaded from its official source and **cannot be redistributed**.
- This repository contains the full training and evaluation pipeline for Milestone 4.
- Mi-UNet is a **Mamba-inspired lightweight architecture**, not a full SSM-based Mamba model.

