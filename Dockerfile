# Imagen base ligera con Python 3.12
FROM python:3.12-slim

WORKDIR /app

# Torch se instala desde el índice CPU: la versión por defecto de PyPI trae CUDA
# (cientos de MB extra) que un clasificador de inferencia en CPU no necesita.
# Va antes de requirements.txt para que esta capa se cachee entre builds.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copiar solo requirements primero: si el código cambia, no se reinstalan las dependencias.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación
COPY src/ ./src/
COPY web/ ./web/

ENV PYTHONPATH=/app/src

# Usuario no root: si alguien compromete la app, no tiene permisos de root dentro del contenedor.
RUN useradd -m app
USER app

CMD ["uvicorn", "landuse.api:app", "--host", "0.0.0.0", "--port", "8000"]
