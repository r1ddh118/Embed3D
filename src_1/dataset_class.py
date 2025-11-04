# dataset_class.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO

class CustomDataset(Dataset):
    """
    hf_dataset: list-like of dataset items from HuggingFace streaming (each item contains 'image' and 'mask')
    augment: boolean, apply train augmentations
    Returns: image Tensor [C,H,W], mask Tensor [1,H,W] with values {0,1} float32
    """
    def __init__(self, hf_dataset, img_size=256, augment=False):
        self.items = hf_dataset
        self.img_size = img_size
        self.augment = augment

        # train augmentations
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

        # val/test transforms
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.items)

    def _open(self, field):
        if isinstance(field, dict) and "bytes" in field:
            return Image.open(BytesIO(field["bytes"]))
        elif isinstance(field, Image.Image):
            return field
        else:
            return Image.open(field)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_field = item["image"]
        mask_field = item.get("mask", None)

        img = self._open(img_field).convert("RGB")
        if mask_field is not None:
            mask_img = self._open(mask_field).convert("L")
        else:
            mask_img = Image.fromarray(np.zeros((self.img_size, self.img_size), dtype=np.uint8))

        if self.augment:
            image_tensor = self.train_transform(img)
        else:
            image_tensor = self.val_transform(img)

        mask_pil = self.mask_transform(mask_img)
        mask_np = np.array(mask_pil, dtype=np.uint8)
        # ensure binary (0 or 1)
        mask_np = (mask_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1,H,W] float32

        return image_tensor, mask_tensor
