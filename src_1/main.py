import os
import pickle
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from dataset_class import CustomDataset
from model_unet import UNetBinary as UNet
from trainer import Trainer
from eval import evaluate_binary as evaluate
from visualisations import save_sample_predictions, plot_loss_curves
from predictions import load_pickled_model, predict_on_image, save_prediction_overlay
from utils import ensure_dir


def main():
    CACHE_FILE = "adam_5pct_subset.pkl"
    IMG_SIZE = 256
    BATCH_SIZE = 4
    EPOCHS = 3
    LR = 1e-4

    # === Load or create subset ===
    if os.path.exists(CACHE_FILE):
        print(f"Loaded cached subset from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            data_list = pickle.load(f)
    else:
        print("Loading dataset: pmchard/3D-ADAM in streaming mode...")
        dataset = load_dataset("pmchard/3D-ADAM", split="train", streaming=True)
        sample_size = int(14120 * 0.05)
        data_list = list(dataset.take(sample_size))
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data_list, f)
        print("Subset cached.")

    # === Dataset and splits ===
    full_ds = CustomDataset(data_list, img_size=IMG_SIZE, augment=True)
    total = len(full_ds)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(full_ds, [train_size, val_size, test_size])
    val_set.dataset.augment = False
    test_set.dataset.augment = False

    # === Dataloaders ===
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # === Model, optimizer, loss ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = "checkpoints"
    ensure_dir(checkpoint_dir)
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir)

    # === Training ===
    print("\nStarting training...")
    logs = trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    model_pkl_path = os.path.join(checkpoint_dir, "trained_unet.pkl")
    torch.save(model, model_pkl_path)
    print(f"\n✅ Model pickled at: {model_pkl_path}")

    # === Evaluation ===
    print("\nStarting evaluation...")
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nValidation Metrics: IoU={val_metrics['IoU']:.4f}, Dice={val_metrics['Dice']:.4f}")
    print(f"Test Metrics: IoU={test_metrics['IoU']:.4f}, Dice={test_metrics['Dice']:.4f}")

    # === Visualization ===
    print("\nGenerating visualizations...")
    save_sample_predictions(model, val_loader, device, num_samples=5)
    plot_loss_curves(logs, out_path="results/visuals/loss_curve.png")

    # === Fault prediction ===
    print("\nRunning fault prediction on a sample image...")
    sample_image_path = "/home/r1ddh1/3rd_year/project_papers/dl_project/demo_img.png"
    ensure_dir("results/predictions")

    if not os.path.exists(sample_image_path):
        print("⚠️ sample_image.png not found. Using one image from validation set instead.")
        imgs, _ = next(iter(val_loader))
        from torchvision.utils import save_image
        save_image(imgs[0], sample_image_path)

    loaded_model = load_pickled_model(model_pkl_path, device)
    img_np, prob_map = predict_on_image(loaded_model, sample_image_path, device)
    out_path = "results/predictions/fault_overlay.png"
    save_prediction_overlay(img_np, prob_map, out_path=out_path)
    print(f"✅ Fault overlay saved to {out_path}")

    print("\n✅ Full training, evaluation, and prediction pipeline completed successfully.")


if __name__ == "__main__":
    main()
