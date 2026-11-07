"""Arquitecturas: CNN base, CNN con dropout y ResNet18 con transfer learning."""
import torch
from torch import nn

from landuse.data import CLASSES

ARCHS = ("base", "dropout", "resnet18")


class SmallCNN(nn.Module):
    """Tres bloques conv+ReLU+pool y un clasificador denso.

    dropout=0 es la CNN base. Con dropout>0 se apaga al azar esa fracción de neuronas
    en cada paso de entrenamiento, lo que evita que la red memorice el train set
    (sobreajuste). En evaluación (model.eval()) el dropout se desactiva solo.
    """

    def __init__(self, num_classes=len(CLASSES), dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch, pretrained=False):
    """Crea el modelo. `pretrained` solo aplica a resnet18 (descarga pesos de ImageNet)."""
    if arch == "base":
        return SmallCNN(dropout=0.0)
    if arch == "dropout":
        return SmallCNN(dropout=0.5)
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # Transfer learning: se conserva todo lo aprendido en ImageNet y solo se
        # reemplaza la última capa para que dé 5 clases en vez de 1000.
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        return model
    raise ValueError(f"Arquitectura desconocida: {arch}. Opciones: {ARCHS}")


def save_checkpoint(model, arch, path):
    torch.save({"arch": arch, "classes": CLASSES, "state_dict": model.state_dict()}, path)


def load_checkpoint(path):
    """Carga un modelo guardado por save_checkpoint, listo para inferencia."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["arch"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["classes"]
