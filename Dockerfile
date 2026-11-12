FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    FLASH_ATTN_CUDA_ARCHS="120" \
    FORCE_CUDA=1 \
    NVCC_FLAGS="-Xcompiler -Wno-register" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml .python-version ./

# Install python and project dependencies into virtual environment
RUN uv sync --no-install-project && uv cache clean

# Copy project files
COPY . ./

# Simplify container startup banner to only print CUDA version
RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh \
          /opt/nvidia/entrypoint.d/12-banner.sh \
          /opt/nvidia/entrypoint.d/15-container-copyright.txt \
          /opt/nvidia/entrypoint.d/30-container-license.txt \
    && echo "CUDA Version ${CUDA_VERSION:-12.8.0}" > /opt/nvidia/entrypoint.d/10-banner.txt

CMD ["bash"]
