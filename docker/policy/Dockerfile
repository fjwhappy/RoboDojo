# syntax=docker/dockerfile:1
# XPolicyLab policy-server image (default policy: Pi_05 / OpenPI).
#
# Contains: RoboDojo launcher scripts + XPolicyLab + the policy's Python env.
# Does NOT contain checkpoints or model caches; mount them at runtime:
#   -v <ckpt cache>:/workspace/RoboDojo/XPolicyLab/policy/<POLICY>/checkpoints:ro
#   -v <data cache>:/root/.cache/openpi
# Build (from the repo root, submodules checked out):
#   docker build -f docker/policy/Dockerfile --build-arg POLICY=Pi_05 -t robodojo-policy-pi_05:latest .
# Used by .github/workflows/robodojo-policy-server.yml (see docs/CI_EVAL.md).

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG POLICY=Pi_05
ARG UBUNTU_MIRROR=""
ARG PIP_INDEX_URL="https://pypi.org/simple"
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
# Optional mirror for python-build-standalone (uv-managed Python downloads).
ARG UV_PYTHON_INSTALL_MIRROR=""

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/root/.local/bin:/root/miniconda3/bin:${PATH} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    UV_INDEX_URL=${PIP_INDEX_URL} \
    UV_DEFAULT_INDEX=${PIP_INDEX_URL} \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    GIT_LFS_SKIP_SMUDGE=1 \
    OPENPI_DATA_HOME=/root/.cache/openpi \
    POLICY=${POLICY}

RUN if [ -n "${UBUNTU_MIRROR}" ]; then \
        sed -i "s@http://archive.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g; s@http://security.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g" /etc/apt/sources.list; \
        rm -f /etc/apt/sources.list.d/cuda*.list; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git build-essential ffmpeg libgl1 libglib2.0-0 netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Miniconda: setup_eval_policy_server.sh uses `conda info --base` and the base python (pyyaml).
# A conda py3.11 env is used as the uv interpreter (openpi needs >=3.11,<3.14), so uv
# never has to download python-build-standalone from GitHub.
ARG CONDA_CHANNEL_MIRROR=""
RUN wget -q "${MINICONDA_URL}" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && rm -f /tmp/miniconda.sh && \
    if [ -n "${CONDA_CHANNEL_MIRROR}" ]; then \
        conda config --add default_channels "${CONDA_CHANNEL_MIRROR}/pkgs/main" && \
        conda config --set channel_alias "${CONDA_CHANNEL_MIRROR}/cloud"; \
    fi && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true) && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true) && \
    conda create -y -q -n py311 python=3.11 && conda clean -afy && \
    /root/miniconda3/bin/python -m pip install --no-cache-dir pyyaml && \
    /root/miniconda3/bin/python -m pip install --no-cache-dir uv && \
    ln -sf /root/miniconda3/bin/uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /workspace/RoboDojo

# Dependency layer: only the policy's lockfiles + XPolicyLab packaging, so source edits
# do not invalidate the (slow) env build.
COPY XPolicyLab/pyproject.toml XPolicyLab/pyproject.toml
COPY XPolicyLab/policy/${POLICY}/ XPolicyLab/policy/${POLICY}/
RUN if [ -n "${UV_PYTHON_INSTALL_MIRROR}" ]; then export UV_PYTHON_INSTALL_MIRROR; fi && \
    cd "XPolicyLab/policy/${POLICY}" && \
    if [ -f openpi/pyproject.toml ]; then \
        cd openpi && \
        py=/root/miniconda3/envs/py311/bin/python && \
        uv sync --frozen --group lerobot --python "${py}" --no-install-project && \
        uv sync --frozen --group lerobot --python "${py}" ; \
    elif [ -f requirements.txt ]; then \
        uv venv .venv --python /root/miniconda3/envs/py311/bin/python && uv pip install --python .venv/bin/python -r requirements.txt ; \
    fi && \
    rm -rf /root/.cache/uv

# Full sources (launcher scripts, XPolicyLab, robot metadata used for action_dim).
COPY scripts/ scripts/
COPY env_cfg/ env_cfg/
COPY XPolicyLab/ XPolicyLab/
RUN cd "XPolicyLab/policy/${POLICY}" && \
    venv="" && for d in openpi/.venv .venv; do [ -d "$d" ] && { venv="$d"; break; }; done; \
    if [ -n "${venv}" ]; then \
        uv pip install --python "${venv}/bin/python" --no-deps -e /workspace/RoboDojo/XPolicyLab && \
        "${venv}/bin/python" -c "import XPolicyLab; print('XPolicyLab ok')" ; \
    fi && \
    rm -rf /root/.cache/uv

COPY docker/policy/entrypoint.sh /usr/local/bin/policy-entrypoint.sh
RUN chmod +x /usr/local/bin/policy-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/policy-entrypoint.sh"]
