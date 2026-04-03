# src/dataset.py
from pathlib import Path
from typing import List, Tuple
from PIL import Image
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
    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: Tuple[int, int] = (256, 256)) -> None:
        self.pairs = pairs
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

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
