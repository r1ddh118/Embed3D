# data_module.py

import random
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from dataset_class import AdamDataset
from torchvision import transforms

class AdamDataModule:
    def __init__(self, dataset_name="pmchard/3D-ADAM", subset_percent=5, batch_size=8, img_size=512):
        self.dataset_name = dataset_name
        self.subset_percent = subset_percent
        self.batch_size = batch_size
        self.img_size = img_size 
        
    def load_subset(self):
        print(f"📦 Loading dataset: {self.dataset_name} in streaming mode (no full download)...")

        # ✅ Load only the train split as a stream
        dataset = load_dataset(self.dataset_name, split="train", streaming=True)

        # ✅ Known total dataset size (from paper/documentation)
        total_size = 14120
        sample_size = int(total_size * self.subset_percent / 100)

        print(f"Taking only {self.subset_percent}% subset → {sample_size} samples out of {total_size}")

        # ✅ Take the first 'sample_size' samples from the stream
        subset_iter = dataset.take(sample_size)

        # ✅ Convert the streamed subset to a list (or optionally to a Dataset)
        data_list = list(subset_iter)

        print(f"✅ Successfully extracted {len(data_list)} samples.")
        return data_list

    def get_dataloaders(self):
        ds_subset = self.load_subset()

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        dataset = AdamDataset(ds_subset, transform)
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader
