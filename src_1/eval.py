# eval.py
import torch
import numpy as np

def dice_np(pred, target, eps=1e-6):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    inter = (pred & target).sum()
    denom = pred.sum() + target.sum()
    return float((2*inter + eps) / (denom + eps))

def iou_np(pred, target, eps=1e-6):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return float((inter + eps) / (union + eps))

def evaluate_binary(model, dataloader, device, threshold=0.5):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)  # logits
            probs = torch.sigmoid(out).squeeze(1)  # [B,H,W]
            preds = (probs > threshold).cpu().numpy().astype(np.uint8)
            targets = masks.squeeze(1).cpu().numpy().astype(np.uint8)
            for p, t in zip(preds, targets):
                total_iou += iou_np(p, t)
                total_dice += dice_np(p, t)
                n += 1
    return {"IoU": total_iou / max(1,n), "Dice": total_dice / max(1,n)}
