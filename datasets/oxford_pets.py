import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class OxfordPetsSegmentation(Dataset):
    """
    Binary semantic segmentation dataset for Oxford-IIIT Pet.

    - Downloads the dataset if missing
    - Resizes all images and masks to (image_size x image_size)
    - Converts trimap mask into binary mask (pet=1, background=0)
    - Supports 'train' and 'val' split (80/20)
    """

    def __init__(self, root, split="train", image_size=128, split_indices=None):
        self.root = root
        self.image_size = image_size
        self.split = split

        # Use torchvision Resize for RGB images
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BILINEAR
        )

        # Load dataset with segmentation masks
        self.dataset = OxfordIIITPet(
            root=root, target_types="segmentation", download=True
        )

        # Split: 80% train, 20% val
        n = len(self.dataset)
        split_idx = int(0.8 * n)
        self.indices = (
            list(range(0, split_idx)) if split == "train" else list(range(split_idx, n))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.dataset[self.indices[idx]]

        # Resize RGB image
        img = self.resize(img)
        img = self.to_tensor(img)

        # Resize mask using OpenCV to preserve class labels
        mask = np.array(mask, dtype=np.uint8)
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )

        # Convert trimap to binary: pet (1 or 3) → 1.0, background (2) → 0.0
        mask = (mask != 2).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
