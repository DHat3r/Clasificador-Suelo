"""Evaluación sobre el test set: python -m landuse.evaluate --model models/best.pt"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: funciona dentro de Docker
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from landuse.data import CLASSES, EuroSatDataset, list_images, split_samples
from landuse.model import load_checkpoint
from landuse.train import collect_predictions


def save_confusion_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_misclassified(samples, y_pred, path, limit=12):
    """Cuadrícula con imágenes mal clasificadas (real -> predicha)."""
    from PIL import Image

    wrong = [(p, CLASSES[t], CLASSES[y]) for (p, t), y in zip(samples, y_pred) if t != y]
    wrong = wrong[:limit]
    if not wrong:
        return 0
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (file, real, pred) in zip(axes.ravel(), wrong):
        with Image.open(file) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"real: {real}\npred: {pred}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return len(wrong)


def evaluate(model_path, data_dir, reports_dir, seed):
    model, _ = load_checkpoint(model_path)
    # Misma semilla que en el entrenamiento -> misma partición -> este es el test set real.
    test_samples = split_samples(list_images(data_dir), seed=seed)["test"]
    loader = DataLoader(EuroSatDataset(test_samples), batch_size=64, num_workers=2)
    y_true, y_pred = collect_predictions(model, loader)

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    report["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    (reports / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix, reports / "confusion_matrix.png")
    shown = save_misclassified(test_samples, y_pred, reports / "misclassified.png")

    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    print(f"Precisión global (test): {report['accuracy']:.4f}. Ejemplos errados guardados: {shown}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evalúa un modelo en el test set")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "models/best.pt"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.reports_dir, args.seed)


if __name__ == "__main__":
    main()
