# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TRITON_VERSION=22.08
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE}

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=Asia/Shanghai


RUN sed -i -e 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list &&\
    apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    autogen \
    clangd \
    gdb \
    git-lfs \
    libb64-dev \
    libz-dev \
    locales-all \
    mosh \
    openssh-server \
    python3-dev \
    rapidjson-dev \
    sudo \
    tmux \
    unzip \
    zstd \
    zip \
    zsh \
    nano tmux htop software-properties-common locales tzdata \
    && locale-gen "$LANG" && \
    update-locale LANG="$LANG" LC_ALL="$LC_ALL"

RUN pip3 install -i https://mirrors.ustc.edu.cn/pypi/web/simple --no-cache-dir torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install --no-cache-dir  regex fire tritonclient[all] --extra-index-url https://pypi.ngc.nvidia.com && \
    pip3 install -i https://mirrors.ustc.edu.cn/pypi/web/simple --no-cache-dir transformers huggingface_hub tokenizers \
    uvicorn fastapi deepspeed peft==0.2.0 rich more_itertools ipdb ipython \
    gpustat loguru SentencePiece sacrebleu datasets tqdm omegaconf rouge_score && \
    pip3 install --no-cache-dir cmake==3.24.3
    
# RUN apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# backend build
WORKDIR /workspace/build/fastertransformer_backend/

COPY all_models all_models
COPY cmake cmake
COPY docs docs
COPY src src
COPY tools tools
COPY CMakeLists.txt CMakeLists.txt

WORKDIR /workspace/build/fastertransformer_backend/build
ARG FORCE_BACKEND_REBUILD=0
RUN https_proxy=http://172.18.178.13:3129 cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D SM=86 \
      -D ENABLE_FP8=OFF \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      ..

RUN cd _deps/repo-ft-src/ && \
    git log | head -n 3 2>&1 | tee /workspace/build/fastertransformer_backend/FT_version.txt && \
    cd /workspace/build/fastertransformer_backend/build && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install && \
    rm /workspace/build/fastertransformer_backend/build/bin/*_example -rf && \
    rm /workspace/build/fastertransformer_backend/build/lib/lib*Backend.so -rf


RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd -p && \
    git clone https://github.com/huggingface/peft.git && \
    cd peft && git checkout 13e53fc7ee5d89d59b16523051006dddf0fb7a49 && \
    pip install . && cd ../ && rm -rf peft

ENV NCCL_LAUNCH_MODE=GROUP
ENV WORKSPACE /workspace
WORKDIR /workspace
