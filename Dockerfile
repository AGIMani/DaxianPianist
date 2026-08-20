# CUDA 12.8 + Ubuntu 22.04 (Python 3.10). Matches this machine's toolkit
# and gives JAX a recent enough stack for Blackwell (sm_120).
# Pull from NVIDIA NGC instead of Docker Hub (registry-1.docker.io often times out here).
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HOME=/home/dev \
    PYTHONPATH=/workspace/robopianist

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        build-essential \
        ca-certificates \
        wget \
        git \
        fluidsynth \
        portaudio19-dev \
        ffmpeg \
        libgl1 \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
        libosmesa6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/dev /opt/robopianist \
    && chmod 777 /home/dev

# Default TimGM6mb soundfont (required at `import robopianist`). Copied into
# the bind-mounted tree by docker/entrypoint.sh if the host copy is missing.
RUN wget -q --tries=3 -O /opt/robopianist/TimGM6mb.sf2 \
        "https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw"

COPY robopianist/setup.py robopianist/README.md /tmp/robopianist/
COPY robopianist/robopianist/__init__.py /tmp/robopianist/robopianist/
COPY robopianist-rl/requirements.txt /tmp/robopianist-rl/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip wheel \
    && python3 -m pip install --no-cache-dir \
        "dm_control>=1.0.16" \
        "dm_env_wrappers>=0.0.11" \
        "mujoco>=3.1.1" \
        "mujoco_utils>=0.0.6" \
        "note_seq>=0.0.5" \
        "pretty_midi>=0.2.10" \
        "pyaudio>=0.2.12" \
        "pyfluidsynth>=1.3.2" \
        "scikit-learn==1.4.2" \
        termcolor \
        tqdm \
        "jax[cuda12]" \
    && python3 -m pip install --no-cache-dir -r /tmp/robopianist-rl/requirements.txt \
    && python3 -m pip install --no-cache-dir \
        black ruff mypy absl-py pytest-xdist \
    && rm -rf /tmp/robopianist /tmp/robopianist-rl

# Cursor / VS Code server needs these. Keep as a late layer so the JAX
# install above stays cached when only this list changes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        socat \
        libatomic1 \
        procps \
        sudo \
        gpg \
        netcat-openbsd \
        openssh-client \
        locales \
        libvulkan1 \
        libglvnd0 \
        libxkbcommon0 \
        libxrandr2 \
        libxi6 \
        libxxf86vm1 \
        libxcursor1 \
        libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# SAPIEN 3 viewer (达显 V2 预览). Late layer so JAX stay cached.
# Tsinghua index: this machine times out on pypi.org wheel downloads.
RUN python3 -m pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        sapien transforms3d

WORKDIR /workspace
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
