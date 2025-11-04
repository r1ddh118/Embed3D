# visualisations.py
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch
from utils import ensure_dir
import cv2

ensure_dir("results/visuals")

def save_sample_predictions(model, dataloader, device, num_samples=5, threshold=0.5):
    model.eval()
    saved = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)
            probs = torch.sigmoid(out).squeeze(1).cpu().numpy()  # [B,H,W]
            preds = (probs > threshold).astype(np.uint8)
            imgs_np = imgs.cpu().permute(0,2,3,1).numpy()
            masks_np = masks.squeeze(1).cpu().numpy()

            for i in range(len(imgs_np)):
                if saved >= num_samples:
                    return
                img = imgs_np[i]
                img_vis = np.clip((img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])), 0, 1)
                mask = masks_np[i]
                pred = preds[i]

                # heatmap of prob
                heat = cv2.applyColorMap((probs[i]*255).astype(np.uint8), cv2.COLORMAP_JET)
                heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                heat = cv2.resize(heat, (img_vis.shape[1], img_vis.shape[0]))

                overlay = (0.6 * (img_vis*255).astype(np.uint8) + 0.4 * heat).astype(np.uint8)

                # save side-by-side
                fig, axs = plt.subplots(1,4, figsize=(16,4))
                axs[0].imshow(img_vis); axs[0].set_title("Image"); axs[0].axis("off")
                axs[1].imshow(mask, cmap="gray"); axs[1].set_title("GT Mask"); axs[1].axis("off")
                axs[2].imshow(pred, cmap="gray"); axs[2].set_title("Pred Mask"); axs[2].axis("off")
                axs[3].imshow(overlay); axs[3].set_title("Prob Overlay"); axs[3].axis("off")
                save_path = f"results/visuals/sample_{saved}.png"
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                saved += 1

def plot_loss_curves(logs, out_path="results/visuals/loss_curve.png"):
    # logs: dict with 'train' and 'val' lists
    ensure_dir(os.path.dirname(out_path))
    epochs = range(1, len(logs["train"])+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, logs["train"], label="train")
    plt.plot(epochs, logs["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
