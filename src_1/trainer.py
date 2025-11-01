# trainer.py

import torch
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir):
        # Move model to device here (important!)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(loader, desc="Training"):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc="Validating"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_dir, f"unet_epoch{epoch+1}.pth")
            )
        print("Training complete.")
