ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND="noninteractive" 
# non debian frontend variable set to noninteractive to avoid prompt

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631

# /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && rm apt-key del 7fa2af80 \
#     && 

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git ninja-build python3 python3-pip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade
RUN apt-get install ffmpeg libglib2.0-0 libsm6 libxrender-dev libxext6 -y

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
# Install MMEngine, MMCV and MMDetection
RUN pip3 install openmim
RUN mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"

RUN pip3 install --no-cache-dir -e .
RUN pip3 install "zod[all]" wandb

WORKDIR '/mmdetection3d'

ENTRYPOINT ["/bin/bash"]