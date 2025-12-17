# DMD2 10-Step ImageNet Training Container
# Build (for vast.ai/cloud): docker buildx build --platform linux/amd64 -t mckellcarter/dmd2-train:latest --push .
# Build (local ARM Mac):     docker build -t mckellcarter/dmd2-train:latest .
#
# For vast.ai: Use devel image for NCCL/multi-GPU support
# Valid tags: https://hub.docker.com/r/pytorch/pytorch/tags
# Use PyTorch 2.5+ for compatibility with newer NVIDIA drivers (580+)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Ports for SSH and tensorboard/wandb
EXPOSE 22 6006

# Install system dependencies for multi-GPU training
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-server \
    openssh-client \
    rsync \
    wget \
    curl \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH for vast.ai
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install visualizer deps if present
COPY visualizer/requirements.txt ./vis_requirements.txt
RUN pip install --no-cache-dir -r vis_requirements.txt || true

# Install additional training deps
# Pin numpy<2 for PyTorch compatibility
RUN pip install --no-cache-dir \
    "numpy<2" \
    ninja \
    packaging \
    pytest \
    huggingface_hub[cli]

# Create directories for data and checkpoints
RUN mkdir -p /data /checkpoints /outputs

# Copy source code
COPY . /workspace/DMD2
WORKDIR /workspace/DMD2

# Environment for multi-GPU training
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL
ENV PYTHONPATH=/workspace/DMD2:$PYTHONPATH

# Make scripts executable
RUN chmod +x experiments/imagenet/*.sh scripts/*.sh scripts/*.py || true

# Default: keep running for vast.ai attach
CMD ["sleep", "infinity"]



