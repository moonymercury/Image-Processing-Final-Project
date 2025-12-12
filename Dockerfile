# ==========================
# 基底：CUDA 12.8 + nvcc + Ubuntu 22.04
# ==========================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ==========================
# 基本套件
# ==========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    git wget curl build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ==========================
# Python / PyTorch（nightly）
# ==========================
RUN pip install --upgrade pip

# 安裝支援 CUDA 12.8 的 PyTorch nightly
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ==========================
# nnUNet v2 依賴
# ==========================
RUN pip install \
    SimpleITK \
    scikit-image \
    scikit-learn \
    nibabel \
    pandas \
    matplotlib \
    batchgenerators==0.25 \
    acvl-utils \
    dynamic-network-architectures \
    tiktoken

RUN pip install nnunetv2==2.4.2

# ==========================
# 安裝 U-Mamba 依賴：causal-conv1d
# ==========================
WORKDIR /workspace
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git
WORKDIR /workspace/causal-conv1d

# 修改 setup.py 以支援 RTX 5070 Ti（sm_120）
RUN sed -i 's/arch=compute_90,code=sm_90/arch=compute_120,code=sm_120/g' setup.py

# 編譯 CUDA kernel
RUN TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda pip install -e .

# ==========================
# 安裝 U-Mamba：mamba-ssm
# ==========================
WORKDIR /workspace
RUN git clone https://github.com/state-spaces/mamba.git
WORKDIR /workspace/mamba

# 修改 setup.py 以支援 sm_120
RUN sed -i 's/arch=compute_90,code=sm_90/arch=compute_120,code=sm_120/g' setup.py

# 編譯 Mamba CUDA kernel
RUN TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda pip install -e .

# ==========================
# 預設資料夾
# ==========================
RUN mkdir -p /workspace/nnUNet_raw \
    /workspace/nnUNet_preprocessed \
    /workspace/nnUNet_results

ENV nnUNet_raw="/workspace/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"

WORKDIR /workspace

CMD ["/bin/bash"]
