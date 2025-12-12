import os
import argparse
import json
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from umamba_unet import UMambaUNet
from losses_cldice import DiceBCELoss, DiceClDiceLoss, DiceLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
# ----------------- Dataset -----------------
class MixedArcadeDataset(Dataset):
    """
    不使用 alpha，不做 pseudo filtering。
    pseudo label = {stem}_mask.png (png 格式)
    """
    def __init__(self, img_dir, gt_dir, pseudo_dir, ids, use_pseudo=False, augment=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.pseudo_dir = pseudo_dir
        self.ids = ids
        self.use_pseudo = use_pseudo
        self.augment = augment

    def __len__(self):
        return len(self.ids)     # <── 這行是你缺少的

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{stem}.png")

        if self.use_pseudo:
            mask_path = os.path.join(self.pseudo_dir, f"{stem}_mask.png")
        else:
            mask_path = os.path.join(self.gt_dir, f"{stem}_mask.png")

        img = cv2.imread(img_path, 0).astype(np.float32) / 255.0
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 127).astype(np.float32)

        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)

        return torch.from_numpy(img), torch.from_numpy(mask)

class ArcadeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = img.astype(np.float32) / 255.0  # [0,1]
        img = np.expand_dims(img, axis=0)     # (1,H,W)
        return img

    def _load_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        m = (m > 127).astype(np.float32)      # 0/1
        m = np.expand_dims(m, axis=0)
        return m

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{stem}.png")
        mask_path = os.path.join(self.mask_dir, f"{stem}_mask.png")

        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

        img = torch.from_numpy(img)   # (1,H,W)
        mask = torch.from_numpy(mask) # (1,H,W)
        return img, mask


# ----------------- Metrics & TTA -----------------
def dice_coeff(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean().item()


def tta_predict(model, imgs):
    """imgs: (B,1,H,W)"""
    with torch.no_grad():
        preds = []

        # identity
        logits = model(imgs)
        preds.append(torch.sigmoid(logits))

        # hflip
        h = torch.flip(imgs, dims=[3])
        logits_h = model(h)
        logits_h = torch.flip(logits_h, dims=[3])
        preds.append(torch.sigmoid(logits_h))

        # vflip
        v = torch.flip(imgs, dims=[2])
        logits_v = model(v)
        logits_v = torch.flip(logits_v, dims=[2])
        preds.append(torch.sigmoid(logits_v))

        # rot90
        r = torch.rot90(imgs, k=1, dims=[2, 3])
        logits_r = model(r)
        logits_r = torch.rot90(logits_r, k=-1, dims=[2, 3])
        preds.append(torch.sigmoid(logits_r))

        prob = torch.stack(preds, dim=0).mean(dim=0)
    return prob


# ----------------- Training / Eval loops -----------------
def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0

    progress = tqdm(loader, desc="Validating", ncols=100)

    with torch.no_grad():
        for imgs, masks in progress:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)

            val_loss += loss.item()
            progress.set_postfix(val_loss=f"{loss.item():.4f}")

    return val_loss / len(loader)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0

    progress = tqdm(loader, desc="Training", ncols=100)

    for imgs, masks in progress:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 更新進度條文字
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(loader)

def eval_one_epoch(model, loader, loss_fn, device, use_tta=False):
    model.eval()
    running = 0.0
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            if use_tta:
                probs = tta_predict(model, imgs)
                logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
            else:
                logits = model(imgs)
            loss = loss_fn(logits, masks)
            running += loss.item() * imgs.size(0)
            dices.append(dice_coeff(logits, masks))
    return running / len(loader.dataset), float(np.mean(dices)) if dices else 0.0


# ----------------- Pseudo-label generation -----------------
def generate_pseudo_labels(model, img_dir, pseudo_dir, device):
    os.makedirs(pseudo_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for img_path in sorted(glob(os.path.join(img_dir, "*.png"))):
            img_id = os.path.splitext(os.path.basename(img_path))[0]

            img = cv2.imread(img_path, 0).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

            pred = torch.sigmoid(model(img_tensor))[0,0].cpu().numpy()

            # save soft mask
            np.save(os.path.join(pseudo_dir, f"{img_id}_mask_prob.npy"), pred)

            # For visualization only
            mask = (pred > 0.5).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(pseudo_dir, f"{img_id}_mask.png"), mask)

        print(f"[INFO] Pseudo labels saved to {pseudo_dir}")

class SimpleImageDataset(Dataset):
    """用於產生 pseudo-label 時，只需要影像不需要 GT。"""
    def __init__(self, img_dir, ids):
        self.img_dir = img_dir
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        path = os.path.join(self.img_dir, f"{stem}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read: {path}")
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        return img, stem


# ----------------- Main -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="ARCADE 資料夾根目錄，例如 /workspace/ARCADE")
    p.add_argument("--out_dir", type=str, default="./umamba_output")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--experiment", type=str, default="baseline",
                   choices=["baseline", "pseudo", "cldice", "pseudo_cldice", "final"],
                   help="對應 I~V 五組實驗")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_fold", type=int, default=0)
    p.add_argument("--baseline_ckpt", type=str,
                   help="已訓練好的 baseline 權重，用於產生 pseudo-label 或 final 模型微調")
    return p.parse_args()


def main():
    args = parse_args()
    # NEW ↓↓↓ 保證 exp_A / exp_B / exp_AB / exp_ABC 都會事先建立
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    img_dir = os.path.join(args.data_root, "images")
    gt_dir = os.path.join(args.data_root, "masks")
    pseudo_dir = os.path.join(args.data_root, "pseudo")

    # --------- 收集資料 ID ---------
    img_files = sorted(glob(os.path.join(img_dir, "*.png")))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    if len(ids) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    # --------- Loss / Attention 設定 ---------
    if args.experiment in ["baseline", "pseudo"]:
        loss_fn = DiceBCELoss()
        use_cldice = False
    elif args.experiment in ["cldice", "pseudo_cldice", "final"]:
        loss_fn = DiceClDiceLoss()
        use_cldice = True
    else:
        loss_fn = DiceBCELoss()
        use_cldice = False

    use_tta = (args.experiment == "final")

    print(f"[INFO] Experiment = {args.experiment}")
    print(f"       Loss = {'Dice+BCE' if not use_cldice else 'Dice+clDice'}")
    
    # --------- 決定使用 GT 或 pseudo ---------
    if args.experiment in ["baseline", "cldice"]:
        train_mask_source = "GT"
        print("[INFO] Using original GT masks.")
    elif args.experiment in ["pseudo_cldice", "pseudo", "final"]:
        train_mask_source = "MIXED"   # GT + pseudo 混合
    else:
        train_mask_source = "PSEUDO"

    # ========================================================
    # 🔥 5-FOLD CROSS VALIDATION
    # ========================================================
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    all_fold_dice = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(ids)):
        if fold < args.start_fold:
            continue
        
        print("\n" + "=" * 60)
        print(f"🔥 Starting Fold {fold+1}/5")
        print("=" * 60)

        train_ids = [ids[i] for i in train_idx]
        val_ids = [ids[i] for i in val_idx]

        # --------------------------------------------
        # 建立 dataset：根據 mask_source 決定使用 GT / pseudo / MIXED
        # --------------------------------------------
        if train_mask_source == "GT":
            train_set = ArcadeDataset(img_dir, gt_dir, train_ids, augment=True)

        elif train_mask_source == "PSEUDO":
            train_set = ArcadeDataset(img_dir, pseudo_dir, train_ids, augment=True)

        elif train_mask_source == "MIXED":
            # 這裡選擇 50% pseudo, 50% GT，可自行調整 alpha
            train_set = MixedArcadeDataset(img_dir, gt_dir, pseudo_dir, train_ids, use_pseudo=True, augment=True)

        # Validation 一律用 GT，避免 pseudo 影響評估
        val_set = ArcadeDataset(img_dir, gt_dir, val_ids, augment=False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

        # -------- 建立模型 --------
        model = UMambaUNet(in_channels=1, num_classes=1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # -------- Early Stopping --------
        best_dice = 0.0
        patience = 15
        patience_counter = 0

        ckpt_path = os.path.join(args.out_dir, f"fold{fold}_best.pth")
        history = {"train_loss": [], "val_loss": []}

        # ========================================================
        # 🔥 Training Loop (with early stopping)
        # ========================================================
        for epoch in range(1, args.epochs + 1):
            print(f"\n===== Fold {fold+1} | Epoch {epoch}/{args.epochs} =====")

            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_dice = eval_one_epoch(model, val_loader, loss_fn, device, use_tta=use_tta)

            print(f"[F{fold}][Epoch {epoch}] Train {train_loss:.4f} | Val {val_loss:.4f} | Dice {val_dice:.4f}")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # --------- 判斷是否更新 best model ---------
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"[INFO] New best Dice {best_dice:.4f} → saved to {ckpt_path}")
            else:
                patience_counter += 1
                print(f"[INFO] No improvement ({patience_counter}/{patience})")

            # --------- Early stopping trigger ---------
            if patience_counter >= patience:
                print(f"⛔ Early stopping at epoch {epoch}")
                break

        all_fold_dice.append(best_dice)

        # --------- 畫 loss curve ---------
        plt.figure(figsize=(7,5))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.title(f"Fold {fold} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(args.out_dir, f"loss_curve_fold{fold}.png"), dpi=200)
        plt.close()

    # ========================================================
    # 🔥 全部 fold 結果
    # ========================================================
    print("\n===============================")
    print("🔥 5-FOLD RESULTS")
    print("===============================")
    for f, dice in enumerate(all_fold_dice):
        print(f"Fold {f}: Dice = {dice:.4f}")
    print(f"\n🔥 Mean Dice = {np.mean(all_fold_dice):.4f}")

    print("[DONE] 5-fold cross validation finished.")

if __name__ == "__main__":
    main()
