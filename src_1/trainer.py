import torch
from tqdm import tqdm
import os
import json
from utils import ensure_dir

class Trainer:
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir="checkpoints",
                 patience=5, clip_grad=2.0):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        ensure_dir(checkpoint_dir)
        self.best_val = float("inf")
        self.patience = patience
        self.clip_grad = clip_grad
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(loader, desc="Training", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)  # logits [B,1,H,W]
            loss = self.criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc="Validating", leave=False):
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs=20):
        logs = {"train": [], "val": []}
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            logs["train"].append(train_loss)
            logs["val"].append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Scheduler step
            self.scheduler.step(val_loss)

            # Checkpointing
            if val_loss < self.best_val:
                self.best_val = val_loss
                no_improve_epochs = 0
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Validation loss improved. Model saved to {checkpoint_path}")
            else:
                no_improve_epochs += 1
                print(f"No improvement in validation loss for {no_improve_epochs} epoch(s).")

            if no_improve_epochs >= self.patience:
                print("Early stopping triggered.")
                break

        # ✅ Save logs automatically for visualization
        logs_path = os.path.join("results", "logs", "training_log.json")
        ensure_dir(os.path.dirname(logs_path))
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=4)
        print(f"Training logs saved to {logs_path}")

        return logs
