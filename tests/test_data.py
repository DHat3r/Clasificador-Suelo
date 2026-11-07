import random

import pytest
import torch
from PIL import Image

from landuse.data import CLASSES, EuroSatDataset, list_images, split_samples

IMAGES_PER_CLASS = 20


@pytest.fixture
def data_dir(tmp_path):
    """Dataset falso: una carpeta por clase con 20 PNG 64x64 de color aleatorio."""
    rng = random.Random(0)
    root = tmp_path / "data"
    for name in CLASSES:
        class_dir = root / name
        class_dir.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            color = tuple(rng.randint(0, 255) for _ in range(3))
            Image.new("RGB", (64, 64), color).save(class_dir / f"img_{i:02d}.png")
    return root


@pytest.fixture
def samples(data_dir):
    return list_images(data_dir)


def _paths(split):
    return {path for path, _ in split}


def test_split_no_tiene_fuga_y_cubre_todo(samples):
    splits = split_samples(samples)
    train, val, test = (_paths(splits[k]) for k in ("train", "val", "test"))

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == _paths(samples)


def test_split_es_reproducible_con_la_misma_semilla(samples):
    first = split_samples(samples, seed=42)
    second = split_samples(samples, seed=42)
    assert first == second


def test_split_cambia_con_otra_semilla(samples):
    first = split_samples(samples, seed=1)
    other = split_samples(samples, seed=2)
    assert _paths(first["train"]) != _paths(other["train"])


def test_split_es_estratificado_14_3_3_por_clase(samples):
    splits = split_samples(samples)
    for label in range(len(CLASSES)):
        counts = {
            name: sum(1 for _, lbl in splits[name] if lbl == label)
            for name in ("train", "val", "test")
        }
        assert counts == {"train": 14, "val": 3, "test": 3}, CLASSES[label]


def test_dataset_devuelve_tensor_y_etiqueta(samples):
    dataset = EuroSatDataset(samples)
    tensor, label = dataset[0]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (3, 64, 64)
    assert isinstance(label, int)


def test_dataset_con_train_true_mantiene_la_forma(samples):
    dataset = EuroSatDataset(samples, train=True)
    tensor, label = dataset[0]

    assert tuple(tensor.shape) == (3, 64, 64)
    assert isinstance(label, int)
