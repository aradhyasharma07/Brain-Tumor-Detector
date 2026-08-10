import os
import random
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Class definitions and mappings
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Brain Tumor MRI images.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform: transforms.Compose = None):
        """
        Args:
            samples (List[Tuple[str, int]]): List of (file_path, class_idx) tuples.
            transform (transforms.Compose, optional): torchvision transforms to apply.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns image transformation pipelines for training and validation/testing.
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transforms, val_transforms)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, BrainTumorDataset, BrainTumorDataset]:
    """
    Scans the dataset directory, performs a stratified 80/20 train/validation split,
    and returns PyTorch DataLoaders.
    
    Args:
        data_dir (str): Path to dataset directory (root directory or Training directory).
        batch_size (int): Batch size for DataLoaders (default: 32).
        val_split (float): Fraction of images to reserve for validation (default: 0.2).
        seed (int): Random seed for reproducible stratified split.
        num_workers (int): Number of subprocesses for data loading.
        
    Returns:
        Tuple[DataLoader, DataLoader, BrainTumorDataset, BrainTumorDataset]:
            (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Locate Training directory
    training_dir = data_dir
    if not os.path.basename(data_dir.rstrip("/\\")).lower() == "training":
        potential_training = os.path.join(data_dir, "Training")
        if os.path.isdir(potential_training):
            training_dir = potential_training

    if not os.path.exists(training_dir):
        raise FileNotFoundError(f"Training directory not found at path: {training_dir}")

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []

    # Deterministic stratified split per class folder
    random_gen = random.Random(seed)

    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(training_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class folder '{cls_name}' missing in {training_dir}")

        cls_idx = CLASS_TO_IDX[cls_name]
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        
        # Shuffle deterministically per class
        random_gen.shuffle(files)

        num_val = int(len(files) * val_split)
        num_train = len(files) - num_val

        val_files = files[:num_val]
        train_files = files[num_val:]

        train_samples.extend([(f, cls_idx) for f in train_files])
        val_samples.extend([(f, cls_idx) for f in val_files])

    train_transform, val_transform = get_transforms()

    train_dataset = BrainTumorDataset(train_samples, transform=train_transform)
    val_dataset = BrainTumorDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_dataset, val_dataset
