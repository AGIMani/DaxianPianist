# Docker 训练

仓库目前没有预构建镜像。下面在带 NVIDIA GPU 的 Linux 宿主机上，用官方 CUDA 镜像挂载本仓库训练。本机已验证：Docker 29、`nvidia` runtime、RTX 3090。

必须满足：

- 使用本仓库的 `robopianist/`，不要 `pip install robopianist`（PyPI 原版没有 `robot=`）
- `MUJOCO_GL=egl`（无显示器渲染评测视频）
- `--gpus all`，让 JAX 和 EGL 都能用到 GPU
- 把仓库挂进容器，checkpoint / 视频才会写回宿主机

## 1. 宿主机检查

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

能列出 GPU 再继续。

## 2. 启动容器

在**项目根目录**执行（路径按你的克隆位置改）：

```bash
cd /path/to/DaxianPianist   # 例如 /home/houjue/pianist_daxian_v2

docker run --gpus all -it \
  --name daxian-train \
  -e MUJOCO_GL=egl \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e SWANLAB_API_KEY="你的key" \
  -v "$PWD":/workspace \
  -w /workspace \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash
```

说明：

- `-v "$PWD":/workspace` 把代码和产物目录一起挂进去
- `SWANLAB_API_KEY` 可选；不设也能训，指标只写本地 `swanlog/`
- 容器退出后还要用：`docker start -ai daxian-train`

## 3. 容器内安装依赖（每个容器做一次）

```bash
apt-get update && apt-get install -y \
  python3.10 python3.10-venv python3-pip \
  build-essential wget \
  libegl1 libgl1 libgles2 libosmesa6 \
  fluidsynth portaudio19-dev ffmpeg git

python3.10 -m venv /opt/venv
source /opt/venv/bin/activate

pip install -U pip
pip install "jax[cuda12]==0.6.2"
pip install -e /workspace/robopianist
pip install -r /workspace/robopianist-rl/requirements.txt
```

`robopianist` 必须 editable install 本仓库路径，这样 `--robot daxian` / `daxian_v2` 才可用。

确认 GPU：

```bash
python -c "import jax; print(jax.devices())"
```

应看到 `CudaDevice(id=0)`。

## 4. 开始训练

每次进容器先激活环境：

```bash
source /opt/venv/bin/activate
cd /workspace/robopianist-rl
```

| 实验 | 命令 | `--robot` | 产物目录 |
| --- | --- | --- | --- |
| 达显 V2（当前主推） | `bash run_daxian_v2.sh` | `daxian_v2` | `eval_daxian_v2/` |
| 达显 V3，锁定 PIP/DIP | `bash run_daxian.sh` | `daxian` | `eval_daxian/` |
| 达显 V3，解锁四指 PIP/DIP | `bash run_daxian_unlock_pip_dip.sh` | `daxian` | `eval_daxian_unlock_pip_dip/` |
| Shadow 手 | `bash run.sh` | `shadow` | `/tmp/robopianist/rl/`（容器内，建议改 `--root-dir`） |

默认 5M 步、曲目 `RoboPianist-debug-TwinkleTwinkleRousseau-v0`、评测相机 `piano/back`。产物在挂载目录里，宿主机可直接看 `checkpoints/`、`videos/`、`metrics/`、`swanlog/`。

附加参数接在脚本后面，例如缩短试跑：

```bash
bash run_daxian_v2.sh --max-steps 20000 --eval-interval 5000
```

Shadow 脚本默认写 `/tmp`，容器删除会丢。可改成：

```bash
bash run.sh
# 或显式：
python train.py --robot shadow --root-dir /workspace/eval_shadow --artifact-dir /workspace/eval_shadow ...
```

## 5. 后台跑 / 再进入

另开终端：

```bash
docker exec -it daxian-train bash
source /opt/venv/bin/activate
```

启动时直接开训（依赖已装好）：

```bash
docker start daxian-train
docker exec -it daxian-train bash -lc \
  "source /opt/venv/bin/activate && cd /workspace/robopianist-rl && bash run_daxian_v2.sh"
```

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| `suite.load()` 没有 `robot=` | 装成了 PyPI 原版；改用 `pip install -e /workspace/robopianist` |
| `Failed to initialize EGL` / 评测视频黑 | 加 `--gpus all`，并确认 `MUJOCO_GL=egl` |
| JAX 只有 CPU | 用 `nvidia/cuda:12.4.1-devel`，装 `jax[cuda12]==0.6.2` |
| SwanLab 网页没有曲线 | 设置 `SWANLAB_API_KEY`；或之后对本地 `swanlog/run-...` 执行 `swanlab sync` |
| 容器删了结果没了 | 必须 `-v` 挂仓库；不要把产物只写在容器内 `/tmp` |

训练产物已被 `.gitignore` 排除，不要 `git add -f` 把 checkpoint / 视频推到 GitHub。
