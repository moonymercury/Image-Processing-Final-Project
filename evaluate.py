import os
import cv2
import numpy as np
import torch
from glob import glob
from sklearn.metrics import precision_score, recall_score

from train_umamba import UMambaUNet


# ------------------------------
# Utility Functions
# ------------------------------

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"[ERROR] Failed to load image: {path}")
    return img.astype(np.float32) / 255.0


def dice_score(gt, pred):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    inter = np.sum(gt * pred)
    denom = np.sum(gt) + np.sum(pred)
    if denom == 0:
        return 1.0
    return 2 * inter / denom


def iou_score(gt, pred):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    inter = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - inter
    if union == 0:
        return 1.0
    return inter / union

# ------------------------------
# clDice Functions (fixed version)
# ------------------------------

import scipy.ndimage as ndi

def soft_skeletonize(prob, thresh=0.5, iters=50):
    """Binary skeletonization for clDice (fixed boolean ops)."""
    img = (prob > thresh).astype(np.uint8)

    skel = np.zeros_like(img, dtype=np.uint8)
    eroded = img.copy()

    for _ in range(iters):
        opened = ndi.binary_opening(eroded)

        # XOR = pixels removed during opening (skeleton contribution)
        temp = eroded ^ opened  
        skel = skel | temp

        # Continue thinning
        eroded = ndi.binary_erosion(eroded)

        # Stop if fully eroded
        if not eroded.any():
            break

    return skel.astype(np.uint8)


def cldice_metric(pred, gt):
    """Compute clDice metric for binary masks (0/1)."""

    pred_skel = soft_skeletonize(pred)
    gt_skel = soft_skeletonize(gt)

    inter = np.sum(pred_skel * gt_skel)
    denom = np.sum(pred_skel) + np.sum(gt_skel)

    if denom == 0:
        return 1.0

    return (2 * inter) / denom

# ------------------------------
# TTA Inference
# ------------------------------

def infer_one(model, img_np):
    """ img_np: H,W float32 """
    x = torch.from_numpy(img_np.copy()).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        out = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    return out


def predict_tta(model, img_np):
    preds = []

    # small rotation angles
    angles = [0, -3, 3, -5, 5]

    def rotate(img, angle):
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return rotated

    def rotate_back(pred, angle):
        # invert rotation
        h, w = pred.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
        restored = cv2.warpAffine(pred, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return restored

    for ang in angles:
        img_rot = rotate(img_np, ang)
        out = infer_one(model, img_rot)
        out = rotate_back(out, ang)
        preds.append(out)

    merged = np.mean(preds, axis=0)
    return (merged > 0.5).astype(np.uint8)

def predict_single(model, img_np, use_tta=False):
    if use_tta:
        return predict_tta(model, img_np)

    out = infer_one(model, img_np)
    return (out > 0.5).astype(np.uint8)


# ------------------------------
# Evaluation Pipeline
# ------------------------------

def evaluate_model(model_path, img_dir, gt_dir, save_dir, experiment):
    os.makedirs(save_dir, exist_ok=True)

    # Determine if attention & TTA should be used
    use_tta = experiment.lower() in ["final", "abc", "a+b+c"]

    print(f"[INFO] Loading model: {model_path}")
    model = UMambaUNet(in_channels=1, num_classes=1, use_attention=False)
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.cuda()
    model.eval()

    img_files = sorted(glob(os.path.join(img_dir, "*.png")))
    print(f"[INFO] Found {len(img_files)} images")

    dice_list, iou_list, pre_list, rec_list, cl_list = [], [], [], [], []

    debug_counter = 0

    for img_path in img_files:
        name = os.path.basename(img_path).replace(".png", "")
        gt_path = os.path.join(gt_dir, f"{name}_mask.png")

        if not os.path.exists(gt_path):
            print(f"[WARN] Missing GT for {name}, skipping.")
            continue

        # ---- Load image ----
        img_np = load_gray(img_path)

        # ---- Predict ----
        pred = predict_single(model, img_np, use_tta=use_tta)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt > 0).astype(np.uint8)

        # ---- Size alignment (safety) ----
        if pred.shape != gt_bin.shape:
            print(f"[WARN] Pred shape {pred.shape} != GT shape {gt_bin.shape}, resizing")
            pred = cv2.resize(pred.astype(np.uint8), (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ---- Metrics ----
        d = dice_score(gt_bin, pred)
        i = iou_score(gt_bin, pred)
        p = precision_score(gt_bin.flatten(), pred.flatten(), zero_division=1)
        r = recall_score(gt_bin.flatten(), pred.flatten(), zero_division=1)
        c = cldice_metric(pred, gt_bin)

        dice_list.append(d)
        iou_list.append(i)
        pre_list.append(p)
        rec_list.append(r)
        cl_list.append(c)

        # ---- Save prediction ----
        out_path = os.path.join(save_dir, f"{name}_pred.png")
        cv2.imwrite(out_path, pred * 255)

        # Debug first few images
        if debug_counter < 3:
            print(f"[DEBUG] {name} Dice={d:.4f}, IoU={i:.4f}")
            debug_counter += 1

    # ---- Summary ----
    print("\n================= FINAL METRICS =================")
    print(f"Mean Dice      : {np.mean(dice_list):.4f}")
    print(f"Mean IoU       : {np.mean(iou_list):.4f}")
    print(f"Mean Precision : {np.mean(pre_list):.4f}")
    print(f"Mean Recall    : {np.mean(rec_list):.4f}")
    print(f"Mean clDice    : {np.mean(cl_list):.4f}")
    print("=================================================")

    np.savetxt(os.path.join(save_dir, "metrics_summary.txt"),
    [np.mean(dice_list),
     np.mean(iou_list),
     np.mean(pre_list),
     np.mean(rec_list),
     np.mean(cl_list)],
    header="Dice IoU Precision Recall clDice")

    print(f"[DONE] All predictions saved to {save_dir}")


# ------------------------------
# Script Entry
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--save_dir", default="eval_out")
    parser.add_argument("--experiment", default="final")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.ckpt,
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        save_dir=args.save_dir,
        experiment=args.experiment
    )
