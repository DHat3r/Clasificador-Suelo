"""API de inferencia: POST /predict, GET /health y la web en /."""
import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from landuse.data import to_tensor
from landuse.model import load_checkpoint

MAX_BYTES = 5 * 1024 * 1024  # una imagen de 64x64 pesa unos KB; 5 MB es holgado
ALLOWED_TYPES = {"image/jpeg", "image/png"}
WEB_INDEX = Path(__file__).resolve().parents[2] / "web" / "index.html"

app = FastAPI(title="Clasificador de cobertura del suelo (EuroSAT)")
_state = {}


def get_model():
    """Carga el modelo la primera vez que se necesita (no al importar el módulo)."""
    if "model" not in _state:
        path = Path(os.environ.get("MODEL_PATH", "models/best.pt"))
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Modelo no disponible: entrena uno primero")
        _state["model"], _state["classes"] = load_checkpoint(path)
    return _state["model"], _state["classes"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/", include_in_schema=False)
def index():
    if not WEB_INDEX.is_file():
        raise HTTPException(status_code=404, detail="Web no encontrada")
    return FileResponse(WEB_INDEX)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Solo se aceptan imágenes JPEG o PNG")
    # Se lee un byte de más para detectar archivos demasiado grandes sin cargarlos enteros.
    data = await file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera los 5 MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # fuerza la decodificación: detecta archivos corruptos o truncados
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="La imagen está corrupta o no es válida")

    model, classes = get_model()
    with torch.no_grad():
        probs = torch.softmax(model(to_tensor(image).unsqueeze(0)), dim=1)[0]
    ranking = sorted(zip(classes, probs.tolist()), key=lambda item: -item[1])
    return {
        "prediction": ranking[0][0],
        "confidence": ranking[0][1],
        "probabilities": {name: p for name, p in ranking},
    }
