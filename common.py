"""Shared utilities for the laboratory-to-field plant disease evaluation."""
import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
ARCHITECTURES = ("resnet50", "efficientnet_b0", "swin_t")
REPO_ROOT = Path(__file__).resolve().parent.parent


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_classes(path=None):
    """The 38 PlantVillage class names in the order used by ImageFolder (sorted)."""
    path = Path(path) if path else REPO_ROOT / "mappings" / "plantvillage_classes.txt"
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_mapping(csv_path):
    """Return {field_dataset_class: plantvillage_class} from a mapping CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["field_class"]: row["plantvillage_class"] for row in csv.DictReader(f)}


def make_plantvillage_splits(data_dir, classes):
    """70/15/15 image-level split with a fixed seed, identical for every model and run."""
    train_view = datasets.ImageFolder(data_dir, transform=train_transform())
    eval_view = datasets.ImageFolder(data_dir, transform=eval_transform())
    if train_view.classes != classes:
        raise ValueError("Class folders in data_dir do not match mappings/plantvillage_classes.txt")
    n = len(train_view)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_ds, val_ds, test_ds = random_split(
        train_view, [n_train, n_val, n - n_train - n_val], generator=generator)
    # Validation and test subsets use the non-augmenting view of the same images.
    val_ds.dataset = eval_view
    test_ds.dataset = eval_view
    return train_ds, val_ds, test_ds


def _list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def collect_plantdoc(root, mapping, classes):
    """PlantDoc classification folders: train and test are merged and used only for testing."""
    items = []
    for split in ("train", "test"):
        for field_cls, pv_cls in mapping.items():
            folder = Path(root) / split / field_cls
            if folder.is_dir():
                items += [(str(p), classes.index(pv_cls)) for p in _list_images(folder)]
    return items


def collect_plantwild(root, mapping, classes):
    """PlantWild v1: <root>/images/<class name>/*.jpg (all shared-class images used for testing)."""
    items = []
    for field_cls, pv_cls in mapping.items():
        folder = Path(root) / "images" / field_cls
        if folder.is_dir():
            items += [(str(p), classes.index(pv_cls)) for p in _list_images(folder)]
    return items


class FieldImageDataset(Dataset):
    """Images from a list of (path, plantvillage_class_index)."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:  # unreadable file: use a blank image so the run is not interrupted
            print(f"warning: could not read {path}")
            image = Image.new("RGB", (224, 224))
        return self.transform(image), label


def build_model(arch, num_classes, pretrained=True):
    """Return (model, head) where head holds the new classification layer."""
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier
    elif arch == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        head = model.head
    else:
        raise ValueError(f"unknown architecture: {arch}")
    return model, head


def arch_from_tag(tag):
    """'resnet50_s1' -> 'resnet50'; 'resnet50' -> 'resnet50'."""
    return re.sub(r"_s\d+$", "", tag)


@torch.no_grad()
def predict(model, loader, device, restricted_idx=None):
    """Return labels, unrestricted predictions and (optionally) predictions restricted to shared classes."""
    model.eval()
    ys, p_all, p_restricted = [], [], []
    for x, y in loader:
        out = model(x.to(device)).float().cpu()
        ys += y.tolist()
        p_all += out.argmax(1).tolist()
        if restricted_idx is not None:
            masked = torch.full_like(out, float("-inf"))
            masked[:, restricted_idx] = out[:, restricted_idx]
            p_restricted += masked.argmax(1).tolist()
    return np.array(ys), np.array(p_all), np.array(p_restricted)
