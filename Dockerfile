FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_STAGE=full
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG TORCHAUDIO_VERSION=2.9.1
ARG FLASH_ATTN_VERSION=2.8.3
ARG VLLM_VERSION=0.16.0

ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    openssh-server \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && python --version \
    && pip --version

RUN python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY . /workspace

RUN if [ "$INSTALL_STAGE" = "torch" ] || [ "$INSTALL_STAGE" = "flash-attn" ] || [ "$INSTALL_STAGE" = "vllm" ]; then \
        python -m pip install --no-cache-dir \
            torch==${TORCH_VERSION} \
            torchvision==${TORCHVISION_VERSION} \
            torchaudio==${TORCHAUDIO_VERSION}; \
    elif [ "$INSTALL_STAGE" = "full" ]; then \
        python -m pip install --no-cache-dir -r /workspace/requirements.runpod.lock.txt; \
    else \
        echo "Unsupported INSTALL_STAGE: $INSTALL_STAGE" >&2; \
        exit 1; \
    fi

RUN if [ "$INSTALL_STAGE" = "flash-attn" ] || [ "$INSTALL_STAGE" = "vllm" ] || [ "$INSTALL_STAGE" = "full" ]; then \
        python -m pip install --no-cache-dir psutil packaging ninja \
        && \
        python -m pip install --no-cache-dir --no-build-isolation flash-attn==${FLASH_ATTN_VERSION}; \
    fi

RUN if [ "$INSTALL_STAGE" = "vllm" ]; then \
        python -m pip install --no-cache-dir vllm==${VLLM_VERSION}; \
    fi

RUN python -m pip check \
    && INSTALL_STAGE="$INSTALL_STAGE" python - <<'PY'
import os
import sys
import torch

install_stage = os.environ["INSTALL_STAGE"]

print("install stage:", install_stage)
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if install_stage in {"flash-attn", "vllm", "full"}:
    import flash_attn
    print("flash_attn:", getattr(flash_attn, "__version__", "imported"))

if install_stage in {"vllm", "full"}:
    import vllm
    print("vllm:", getattr(vllm, "__version__", "imported"))
PY

RUN mkdir -p /run/sshd \
    && ssh-keygen -A \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D", "-e"]
