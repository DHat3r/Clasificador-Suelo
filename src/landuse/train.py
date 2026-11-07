"""Entrenamiento: python -m landuse.train --arch resnet18 --epochs 5 --out models/best.pt"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from landuse.data import EuroSatDataset, list_images, split_samples
from landuse.model import ARCHS, build_model, save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_predictions(model, loader):
    """Devuelve (etiquetas_reales, predicciones) de todo el loader, sin calcular gradientes."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            y_pred.extend(model(images).argmax(dim=1).tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def accuracy(model, loader):
    y_true, y_pred = collect_predictions(model, loader)
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def make_loaders(data_dir, batch_size, seed):
    splits = split_samples(list_images(data_dir), seed=seed)
    train = DataLoader(EuroSatDataset(splits["train"], train=True), batch_size=batch_size, shuffle=True, num_workers=2)
    val = DataLoader(EuroSatDataset(splits["val"]), batch_size=batch_size, num_workers=2)
    test = DataLoader(EuroSatDataset(splits["test"]), batch_size=batch_size, num_workers=2)
    return train, val, test


def train(arch, data_dir, out, epochs, batch_size, lr, seed, reports_dir):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(data_dir, batch_size, seed)
    model = build_model(arch, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history, best_val = [], -1.0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_loader.dataset)
        val_acc = accuracy(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"[{arch}] época {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        # Se guarda solo si mejora en validación: nos quedamos con el mejor, no el último.
        if val_acc > best_val:
            best_val = val_acc
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, arch, out)

    # El test set se toca UNA vez, al final, con el mejor modelo según validación.
    model.load_state_dict(torch.load(out, map_location="cpu", weights_only=True)["state_dict"])
    test_acc = accuracy(model, test_loader)
    metrics = {
        "arch": arch, "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset), "n_test": len(test_loader.dataset),
        "best_val_acc": best_val, "test_acc": test_acc,
        "train_seconds": round(time.time() - start, 1), "history": history,
    }
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    metrics_path = reports / f"metrics_{Path(out).stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{arch}] val={best_val:.4f} test={test_acc:.4f} -> {out}, {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Entrena un clasificador EuroSAT")
    parser.add_argument("--arch", choices=ARCHS, default="resnet18")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--out", default="models/best.pt")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.arch, args.data_dir, args.out, args.epochs, args.batch_size, args.lr, args.seed, args.reports_dir)


if __name__ == "__main__":
    main()
