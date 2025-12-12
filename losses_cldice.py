import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B,1,H,W), targets: (B,1,H,W)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class BCEWithLogitsLoss2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets)


# --------- clDice (soft version, r = 1) ---------
def soft_erode(img):
    # img: (B,1,H,W)
    p1 = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    return p1


def soft_dilate(img):
    p1 = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    return p1


def soft_skeletonize(img, iters=10):
    img1 = soft_erode(img)
    skel = F.relu(img - soft_dilate(img1))
    for _ in range(iters - 1):
        img = img1
        img1 = soft_erode(img)
        skel = skel + F.relu(img - soft_dilate(img1))
    return skel


def soft_cldice(pred, target, smooth=1.0):
    """
    clDice: measure of centerline overlap
    pred, target: probabilities in [0,1], shape (B,1,H,W)
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    skel_pred = soft_skeletonize(pred)
    skel_targ = soft_skeletonize(target)

    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + smooth)
    tsens = (skel_targ * pred).sum(dim=(1, 2, 3)) / (skel_targ.sum(dim=(1, 2, 3)) + smooth)

    cl = (2 * tprec * tsens) / (tprec + tsens + smooth)
    return 1 - cl.mean()


class DiceBCELoss(nn.Module):
    """標準損失：Dice + BCE"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = BCEWithLogitsLoss2D()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.bw * self.bce(logits, targets)

class DiceClDiceLoss(nn.Module):
    def __init__(self, w_dice=0.4, w_bce=0.4, w_cl=0.2):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = BCEWithLogitsLoss2D()
        self.wd = w_dice
        self.wb = w_bce
        self.wc = w_cl

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dice = self.dice(logits, targets)
        bce = self.bce(logits, targets)
        cld = soft_cldice(probs, targets)
        return self.wd * dice + self.wb * bce + self.wc * cld
