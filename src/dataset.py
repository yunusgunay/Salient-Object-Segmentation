# src/dataset.py
import random
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[Path, Path]]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])
    mask_paths = sorted([p for p in masks_dir.iterdir() if p.suffix.lower() in valid_exts])

    if len(image_paths) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    if len(mask_paths) == 0:
        raise ValueError(f"No mask files found in {masks_dir}")

    mask_map = {p.stem: p for p in mask_paths}

    pairs = []
    missing_masks = []

    for image_path in image_paths:
        stem = image_path.stem
        if stem in mask_map:
            pairs.append((image_path, mask_map[stem]))
        else:
            missing_masks.append(image_path.name)

    if len(pairs) == 0:
        raise ValueError("No matched image-mask pairs found. Check filenames.")

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images do not have matching masks.")

    print(f"Matched pairs: {len(pairs)}")
    return pairs


class ECSSDDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: Tuple[int, int] = (256, 256),
        normalize_mean: Optional[Tuple[float, float, float]] = None,
        normalize_std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size

        image_transform_steps = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]

        if normalize_mean is not None and normalize_std is not None:
            image_transform_steps.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))
        
        self.image_transform = transforms.Compose(image_transform_steps)

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask


# [FINAL REPORT CHANGE] Added AugmentedECSSDDataset for training-time augmentation.
# Original dataset had no augmentation. With only 700 training images, both CNN and U-Net
# underfit/overfit because they see the same pixels every epoch.
# Augmentations applied (geometric transforms mirror-applied to mask to keep labels valid):
#   - Random horizontal flip (p=0.5)
#   - Random rotation ±15° (small enough to not break object structure)
#   - Color jitter on image only (brightness/contrast/saturation variation)
# augment=False by default so val/test loaders stay deterministic.
class AugmentedECSSDDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        normalize_mean: Optional[Tuple[float, float, float]] = None,
        normalize_std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

        self.resize_img = transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST)
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        self.to_tensor = transforms.ToTensor()

        self.normalize = None
        if normalize_mean is not None and normalize_std is not None:
            self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.resize_img(image)
        mask = self.resize_mask(mask)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            image = self.color_jitter(image)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        if self.normalize is not None:
            image = self.normalize(image)

        mask = (mask > 0.5).float()
        return image, mask
