import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm

def dice_score(preds, targets, num_classes=1, epsilon=1e-6):
    """Compute Dice coefficient for binary or multi-class segmentation."""
    if num_classes > 1:
        # preds: [B,H,W] (already class indices)
        if preds.ndim == 4:  # safety fallback
            preds = torch.argmax(preds, dim=1)
        # one-hot encode
        targets_onehot = torch.nn.functional.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2)
        preds_onehot = torch.nn.functional.one_hot(preds.long(), num_classes).permute(0, 3, 1, 2)
        intersection = (preds_onehot * targets_onehot).sum(dim=(0, 2, 3))
        union = preds_onehot.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return dice.mean()
    else:
        preds = (preds > 0.5).float()
        targets = targets.float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return dice.mean()

def evaluate(model, dataloader, device):
    """
    Evaluate segmentation model using IoU (Jaccard) and Dice score.
    Supports both binary and multi-class segmentation.
    """
    model.eval()
    total_iou, total_dice = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            # Detect number of classes
            if outputs.shape[1] > 1:
                num_classes = outputs.shape[1]
                preds = torch.argmax(outputs, dim=1)
                iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
                total_iou += iou_metric(preds, masks).item()
                total_dice += dice_score(preds, masks, num_classes=num_classes).item()
            else:
                preds = torch.sigmoid(outputs)
                preds_bin = (preds > 0.5).float()
                iou_metric = JaccardIndex(task="binary").to(device)
                total_iou += iou_metric(preds_bin, masks.int()).item()
                total_dice += dice_score(preds_bin, masks).item()

            num_batches += 1

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    print(f"\n✅ Evaluation Results:")
    print(f"   IoU (Jaccard Index): {avg_iou:.4f}")
    print(f"   Dice Coefficient:    {avg_dice:.4f}")

    return {"IoU": avg_iou, "Dice": avg_dice}
