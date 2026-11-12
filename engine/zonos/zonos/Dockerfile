FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    FLASH_ATTN_CUDA_ARCHS="120" \
    FORCE_CUDA=1 \
    NVCC_FLAGS="-Xcompiler -Wno-register"

# System dependencies (build + audio + espeak-ng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY pyproject.toml ./

# Torch triplet pinned for CUDA 12.8 (supports sm_120 / RTX 50 series) + build dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision torchaudio && \
    python3 -m pip install --no-cache-dir psutil ninja packaging

# Limit concurrent compile jobs to prevent Docker OOM
ENV MAX_JOBS=4

# Step 1: Install flash-attn from source with sm_120 support
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
      --no-binary flash-attn 'flash-attn==2.7.4.post1'

# Step 2: Install latest causal-conv1d and mamba-ssm from git with sm_120 compilation support
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE

RUN python3 -m pip install --no-cache-dir --no-build-isolation \
      git+https://github.com/Dao-AILab/causal-conv1d.git && \
    python3 -m pip install --no-cache-dir --no-build-isolation \
      git+https://github.com/state-spaces/mamba.git && \
    python3 -m pip install --no-cache-dir "apache-tvm-ffi<=0.1.9"

# Copy project files and install in editable mode
COPY . ./
RUN python3 -m pip install --no-cache-dir -e .
# Expose Gradio default port
EXPOSE 7860

# Default entrypoint
CMD ["python3", "gradio_interface.py"]
