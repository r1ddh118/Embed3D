import os
import pickle
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from dataset_class import CustomDataset
from trainer import Trainer
from model_unet import UNet
import torch

def main():
    CACHE_FILE = "adam_5pct_subset.pkl"

    # Check if cached subset exists
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached 5% subset from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            small_dataset = pickle.load(f)
    else:
        print("Loading dataset: pmchard/3D-ADAM in streaming mode (no full download)...")
        dataset = load_dataset("pmchard/3D-ADAM", split="train", streaming=True)
        total_samples = 14120
        sample_size = int(total_samples * 0.05)
        print(f"Taking only 5% subset → {sample_size} samples out of {total_samples}")

        small_dataset = list(dataset.take(sample_size))
        print("Successfully extracted 5% subset.")
        
        # Cache for next time
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(small_dataset, f)
        print(f"Cached subset saved at {CACHE_FILE}")

    # Create PyTorch dataset
    full_dataset = CustomDataset(small_dataset)

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize UNet (adjust if your UNet requires in/out channels)
    model = UNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Ensure checkpoint directory exists
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pass checkpoint_dir to Trainer
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir)

    trainer.fit(train_loader, val_loader, epochs=10)

if __name__ == "__main__":
    main()
