# syntax=docker/dockerfile:1.6
# ======================================================================
# MONAI + Ray + MLflow (CUDA devel image)
# Suitable for local development, training, ONNX export, or serving.
# ======================================================================
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
# apt with cache (faster rebuilds)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-setuptools \
    wget curl git ca-certificates build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
# Install torch separately so it caches
RUN pip install --upgrade pip
RUN pip install torch==2.3.0 torchvision==0.18.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# cache pip wheels for requirements
# COPY requirements.txt /tmp/requirements.txt
COPY requirements.txt constraints.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt --constraint /tmp/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    && rm -f /tmp/requirements.txt /tmp/constraints.txt

# Application setup
COPY . /app/app

# Environment
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
    # TRITON_URL=triton:8001

# EXPOSE 8000

# Entrypoint
# CMD ["python", "-m", "ray.serve", "run", "app.main:deployment"]
