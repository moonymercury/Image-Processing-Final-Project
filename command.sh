"""

# Training
python train_umamba.py \
    --experiment baseline \
    --data_root /workspace/ARCADE \
    --out_dir exp_baseline

python train_umamba.py \
    --experiment pseudo_cldice \
    --data_root /workspace/ARCADE \
    --out_dir exp_AB

python train_umamba.py \
    --experiment pseudo \
    --data_root /workspace/ARCADE \
    --out_dir exp_A

python train_umamba.py \
    --experiment cldice \
    --data_root /workspace/ARCADE \
    --out_dir exp_B

"""

# baseline 
python evaluate.py \
    --ckpt exp_baseline/fold4_best.pth \
    --img_dir /workspace/val/images \
    --gt_dir /workspace/val/mask \
    --save_dir eval_baseline \
    --experiment baseline

# pseudo labels (No TTA)
python evaluate.py \
    --ckpt exp_A/fold4_best.pth \
    --img_dir /workspace/val/images \
    --gt_dir /workspace/val/mask \
    --save_dir eval_A \
    --experiment A

# GT + clDice
python evaluate.py \
    --ckpt exp_B/fold4_best.pth \
    --img_dir /workspace/val/images \
    --gt_dir /workspace/val/mask \
    --save_dir eval_B \
    --experiment B

# pseudo labels + clDice
python evaluate.py \
    --ckpt exp_AB/fold4_best.pth \
    --img_dir /workspace/val/images \
    --gt_dir /workspace/val/mask \
    --save_dir eval_AB \
    --experiment A+B

# pseudo labels + clDice + TTA
python evaluate.py \
    --ckpt exp_AB/fold4_best.pth \
    --img_dir /workspace/val/images \
    --gt_dir /workspace/val/mask \
    --save_dir eval_ABC \
    --experiment final

