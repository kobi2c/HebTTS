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
COPY docker/hebtts/pyproject.toml ./pyproject.toml
COPY docker/hebtts/uv.lock ./uv.lock
COPY docker/hebtts/.python-version ./

# Install python and project dependencies into virtual environment
RUN uv sync --frozen --no-install-project && uv cache clean

# ----------------------------------------------------------------------------
# Zonos-specific build tools, packages, and compiled CUDA extensions
# ----------------------------------------------------------------------------

# Install build tools and extra system dependencies needed for Zonos CUDA compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    wget \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Build environment flags for compiling causal-conv1d and mamba
ENV MAX_JOBS=4 \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE

# Install Zonos-specific Python dependencies on top of HebTTS base via uv pip
RUN uv pip install --no-cache \
      "soundfile>=0.13.1" \
      "transformers>=4.48.1" \
      "huggingface-hub>=0.28.1" \
      "inflect>=7.5.0" \
      "kanjize>=1.5.0" \
      "sudachipy>=0.6.10" \
      "sudachidict-full>=20241021" \
      "phonikud-onnx>=1.0.6" \
      "gradio>=5.15.0"

# Build and install CUDA acceleration extensions for Zonos hybrid backbone via uv pip
RUN uv pip install --no-cache --no-build-isolation \
      --no-binary flash-attn 'flash-attn==2.7.4.post1' && \
    uv pip install --no-cache --no-build-isolation \
      git+https://github.com/Dao-AILab/causal-conv1d.git && \
    uv pip install --no-cache --no-build-isolation \
      git+https://github.com/state-spaces/mamba.git && \
    uv pip install --no-cache "apache-tvm-ffi<=0.1.9"

# Copy project files
COPY . ./

# Simplify container startup banner to only print CUDA version
RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh \
          /opt/nvidia/entrypoint.d/12-banner.sh \
          /opt/nvidia/entrypoint.d/15-container-copyright.txt \
          /opt/nvidia/entrypoint.d/30-container-license.txt \
    && echo "CUDA Version ${CUDA_VERSION:-12.8.0}" > /opt/nvidia/entrypoint.d/10-banner.txt

EXPOSE 7860

CMD ["python", "engine/zonos/zonos/gradio_interface.py"]
