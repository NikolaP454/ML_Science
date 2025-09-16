FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Keep Python output unbuffered
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive


# Install system dependencies commonly needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install uv
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir uv


# Create and use an app directory
WORKDIR /app

COPY ./src /app
COPY ./pyproject.toml /app
COPY ./uv.lock /app
COPY ./.python-version /app

# Sync dependencies
RUN uv sync --no-dev