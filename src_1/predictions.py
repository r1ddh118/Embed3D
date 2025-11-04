# predictions.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import ensure_dir

ensure_dir("results/predictions")

def load_pickled_model(pkl_path, device):
    model = torch.load(pkl_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def predict_on_image(model, image_path, device, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    img = Image.open(image_path).convert("RGB")
    input_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_t)  # logits
        prob = torch.sigmoid(out).squeeze().cpu().numpy()  # HxW
    return np.array(img.resize((prob.shape[1], prob.shape[0]))), prob

def save_prediction_overlay(image_np, prob_map, out_path, thresh=0.5):
    # image_np: HxWx3 (uint8)
    ensure_dir(os.path.dirname(out_path))
    heat = (prob_map * 255).astype(np.uint8)
    import cv2
    heat_col = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
    overlay = ((image_np * 0.7) + (heat_col * 0.3)).astype(np.uint8)
    # mask contour
    mask = (prob_map > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = overlay.copy()
    cv2.drawContours(vis, contours, -1, (255,255,255), 2)

    # Save
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1); plt.imshow(image_np); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(prob_map, cmap="jet"); plt.title("Prob Map"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(vis); plt.title("Overlay + Contours"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
