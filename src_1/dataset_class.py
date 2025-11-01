from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from io import BytesIO

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, img_size=256, num_classes=13):
        self.dataset = hf_dataset
        self.img_size = img_size
        self.num_classes = num_classes

        # Resize & normalize images
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        # Use nearest-neighbor resize for masks (no interpolation blur)
        self.mask_transform = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # ✅ Load image
        image = item["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        image = self.transform(image)

        # Load mask
        mask = item.get("mask")
        if mask is not None:
            if isinstance(mask, dict) and "bytes" in mask:
                mask = Image.open(BytesIO(mask["bytes"]))
            elif not isinstance(mask, Image.Image):
                mask = Image.open(mask)

            mask = mask.convert("L")  # grayscale mask
            mask = self.mask_transform(mask)

            # Convert to integer class tensor
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        else:
            # Fallback: empty mask
            mask = torch.zeros((self.img_size, self.img_size), dtype=torch.int64)

        return image, mask
