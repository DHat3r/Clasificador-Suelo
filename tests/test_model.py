import pytest
import torch

from landuse.data import CLASSES
from landuse.model import build_model, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("arch", ["base", "dropout", "resnet18"])
def test_build_model_devuelve_salida_de_5_clases(arch):
    model = build_model(arch, pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    assert tuple(output.shape) == (2, len(CLASSES))


def test_checkpoint_guarda_y_carga_el_mismo_modelo(tmp_path):
    torch.manual_seed(0)
    original = build_model("base").eval()
    path = tmp_path / "model.pt"
    save_checkpoint(original, "base", path)

    loaded, classes = load_checkpoint(path)

    assert classes == CLASSES
    assert not loaded.training
    entrada = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(original(entrada), loaded(entrada))
