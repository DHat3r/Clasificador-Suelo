"""Carga de datos: listado de imágenes, partición train/val/test y Dataset.

Idea clave: la partición se hace sobre RUTAS de archivo, antes de cargar ninguna
imagen. Así cada archivo cae en una sola partición (no hay fuga de datos) y la
partición es reproducible gracias a la semilla fija.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Residential", "SeaLake"]
IMAGE_SIZE = 64
EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Media y desviación de ImageNet: las exige ResNet18 preentrenada y no perjudica a las CNN propias.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def list_images(data_dir):
    """Devuelve una lista ordenada de (ruta, índice_de_clase) leyendo una subcarpeta por clase."""
    data_dir = Path(data_dir)
    samples = []
    for label, name in enumerate(CLASSES):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Falta la carpeta de la clase '{name}' en {data_dir}")
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        samples.extend((p, label) for p in files)
    return samples


def split_samples(samples, fractions=(0.70, 0.15, 0.15), seed=42):
    """Partición estratificada: cada clase se reparte por separado en las mismas proporciones.

    Devuelve {"train": [...], "val": [...], "test": [...]} con tuplas (ruta, clase).
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Las fracciones deben sumar 1")
    rng = random.Random(seed)
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_class):
        items = sorted(by_class[label])  # orden estable antes de barajar -> reproducible
        rng.shuffle(items)
        n_train = int(len(items) * fractions[0])
        n_val = int(len(items) * fractions[1])
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]
    return splits


def to_tensor(image):
    """Imagen PIL -> tensor float (3, 64, 64) normalizado."""
    array = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def augment(tensor):
    """Volteos y rotaciones de 90°. Una imagen satelital no tiene 'arriba', así que son transformaciones válidas."""
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=[2])
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=[1])
    return torch.rot90(tensor, k=random.randint(0, 3), dims=[1, 2])


class EuroSatDataset(Dataset):
    """Dataset perezoso: guarda rutas y abre cada imagen solo cuando se pide."""

    def __init__(self, samples, train=False):
        self.samples = list(samples)
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = to_tensor(image)
        if self.train:
            tensor = augment(tensor)
        return tensor, label
