FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e .

ENV CLE_MODELS_DIR=models/video_physio_regression_z01_geom

EXPOSE 8000

CMD ["uvicorn", "src.cle.server:app", "--host", "0.0.0.0", "--port", "8000"]
