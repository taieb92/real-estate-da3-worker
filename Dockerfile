FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DA3_CODE_REVISION=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

LABEL org.opencontainers.image.source="https://github.com/taieb92/real-estate-da3-worker" \
      org.opencontainers.image.description="Pinned Depth Anything 3 monocular RunPod worker" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONHASHSEED=0 \
    DA3_MODEL_DIR=/models/DA3MONO-LARGE

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY python/runpod_handler.py python/download_pinned_da3.py python/da3-inference-only.patch ./

RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/depth-anything-3 \
    && git -C /opt/depth-anything-3 checkout --detach "$DA3_CODE_REVISION" \
    && git -C /opt/depth-anything-3 apply --check /app/da3-inference-only.patch \
    && git -C /opt/depth-anything-3 apply /app/da3-inference-only.patch \
    && python -m pip install --no-cache-dir --no-deps -e /opt/depth-anything-3 \
    && python -m pip install --no-cache-dir runpod==1.8.1 addict==2.4.0 \
       einops==0.8.1 huggingface-hub==0.34.4 imageio==2.37.0 numpy==1.26.4 \
       omegaconf==2.3.0 opencv-python-headless==4.11.0.86 safetensors==0.5.3 \
       tqdm==4.67.1 xformers==0.0.31.post1

RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python download_pinned_da3.py \
    && rm download_pinned_da3.py da3-inference-only.patch \
    && python -m pip cache purge

USER 1000:1000
CMD ["python", "-u", "runpod_handler.py"]
