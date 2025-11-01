# main.py

import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from data_module import AdamDataModule
from model_unet import UNet
from trainer import Trainer
from utils import seed_everything

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed_everything(42)
    data_module = AdamDataModule(
    dataset_name=config["dataset_name"],
    subset_percent=config["subset_percent"],
    batch_size=config["batch_size"]
)

    train_loader, val_loader = data_module.get_dataloaders()
    subset = data_module.load_subset()

# Example access
    print(subset[0].keys())
    print(subset[0]['image'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=config["num_classes"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, device, config["checkpoint_dir"])
    trainer.fit(train_loader, val_loader, config["epochs"])

if __name__ == "__main__":
    main()
