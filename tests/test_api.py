import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import landuse.api
from landuse.data import CLASSES
from landuse.model import build_model, save_checkpoint

MAX_BYTES = 5 * 1024 * 1024


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Cliente de la API con un checkpoint 'base' sin entrenar y el estado del modelo limpio."""
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(build_model("base"), "base", checkpoint)
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    landuse.api._state.clear()
    yield TestClient(landuse.api.app)
    landuse.api._state.clear()


def test_health_responde_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_con_png_valido(client):
    response = client.post("/predict", files={"file": ("foto.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_rechaza_tipo_no_imagen(client):
    response = client.post("/predict", files={"file": ("nota.txt", b"hola", "text/plain")})
    assert response.status_code == 415


def test_predict_rechaza_archivo_vacio(client):
    response = client.post("/predict", files={"file": ("vacio.png", b"", "image/png")})
    assert response.status_code == 400


def test_predict_rechaza_bytes_que_no_son_imagen(client):
    response = client.post(
        "/predict", files={"file": ("basura.png", b"no soy una imagen", "image/png")}
    )
    assert response.status_code == 400


def test_predict_rechaza_archivo_demasiado_grande(client):
    data = b"0" * (MAX_BYTES + 1)
    response = client.post("/predict", files={"file": ("grande.png", data, "image/png")})
    assert response.status_code == 413


def test_predict_sin_modelo_devuelve_503(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "no_existe.pt"))
    response = client.post("/predict", files={"file": ("foto.png", _png_bytes(), "image/png")})
    assert response.status_code == 503
