# Docker 开发环境

在 GPU 容器里开发 / 训练 DaxianPianist（MuJoCo + JAX SAC）。源码挂载进容器，改代码不用重建镜像。

相关文件：

- `Dockerfile`
- `docker-compose.yml`
- `docker/entrypoint.sh`
- `.devcontainer/devcontainer.json`

容器内工作目录固定为 `/workspace`，对应宿主机上的本仓库根目录。

---

## 要替换的字段

先改这些，再执行后面的命令。本机（`houjue@FC500T-K`）的填法已写在「本机示例」列。


| 字段                  | 在哪改                                                                    | 含义                             | 本机示例                                            | 别的机器怎么填                    |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------- | -------------------------- |
| `<PROJECT_DIR>`     | 终端命令里的 `cd`                                                            | 仓库在宿主机上的绝对路径                   | `/home/houjue/DaxianPianist`                    | `pwd` 在仓库根目录的输出            |
| `<HOST_UID>`        | 启动前的环境变量，或写入 `.env`                                                    | 容器进程的 Linux uid，避免写出的文件变成 root | `$(id -u)` → 当前是 `2102`                         | 在目标机器上执行 `id -u`           |
| `<HOST_GID>`        | 同上                                                                     | 容器进程的 Linux gid                | `$(id -g)` → 当前是 `2102`                         | 在目标机器上执行 `id -g`           |
| `<GPU_ID>`          | `docker-compose.yml` 的 `CUDA_VISIBLE_DEVICES` 和 `MUJOCO_EGL_DEVICE_ID` | 用哪张 GPU                        | `0`（只有一张卡时）                                     | `nvidia-smi -L` 里的编号       |
| `<SWANLAB_API_KEY>` | 启动前的环境变量，或写入 `.env`                                                    | SwanLab 云端日志。不用云端可留空           | 你的 SwanLab key，没有就空着                            | 从 SwanLab 网站复制             |
| `<CKPT>`            | `eval_song.py` 的 `--ckpt`                                              | 评测用的 checkpoint                | `../eval_daxian/checkpoints/latest.pkl`         | 实际 `.pkl` 路径               |
| `<ENV_NAME>`        | `eval_song.py` 的 `--environment-name`                                  | 评测曲子                           | `RoboPianist-debug-TwinkleTwinkleLittleStar-v0` | 见 `train.py` / suite 里的环境名 |


可选、一般不用改：


| 字段                                             | 默认                               | 何时改                         |
| ---------------------------------------------- | -------------------------------- | --------------------------- |
| `Dockerfile` 里的 `FROM nvcr.io/nvidia/cuda:...` | `12.8.1-cudnn-devel-ubuntu22.04` | 换了驱动/CUDA 大版本，或 JAX 装不上 GPU |
| `docker-compose.yml` 的 `image:`                | `daxian-pianist:dev`             | 想换本地镜像名                     |
| `shm_size`                                     | `8gb`                            | 共享内存不够、渲染/DataLoader 报错时再加大 |


不要改（容器内部约定）：

- 容器工作目录 `/workspace`
- `PYTHONPATH=/workspace/robopianist`
- `MUJOCO_GL=egl`（无头训练）

推荐在仓库根目录建 `.env`（已被 git 忽略的话可自行加入 `.gitignore`；**不要提交 API key**）：

```bash
# <PROJECT_DIR>/ .env
HOST_UID=2102          # 换成 id -u 的结果
HOST_GID=2102          # 换成 id -g 的结果
SWANLAB_API_KEY=       # 换成你的 key，不用就留空
```

`CUDA_VISIBLE_DEVICES` 目前写在 `docker-compose.yml` 里。要换 GPU，把下面两处的 `0` 改成 `<GPU_ID>`：

```yaml
CUDA_VISIBLE_DEVICES: "0"
MUJOCO_EGL_DEVICE_ID: "0"
```

---



## 前提（这台 FC500T-K 已满足）

- Docker 与 `docker compose`
- NVIDIA 驱动 + `nvidia-container-toolkit`
- `nvidia-smi` 能看到 GPU

自检：

```bash
docker --version
docker compose version
nvidia-smi
```

---



## 1. 第一次构建并启动

把 `<PROJECT_DIR>` 换成仓库路径。若已写好 `.env`，不必再 `export HOST_UID`。

```bash
cd <PROJECT_DIR>

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
# 需要 SwanLab 云端时再设：
export SWANLAB_API_KEY=<SWANLAB_API_KEY>

docker compose build
docker compose up -d
```

本机可直接：

```bash
cd /home/houjue/DaxianPianist
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
docker compose build
docker compose up -d
```

第一次会拉 CUDA 镜像并安装 JAX，耗时十几分钟、体积数 GB。之后改 Python 代码不用重建；只有改了 `Dockerfile` / 依赖才需要再 `docker compose build`。

---



## 2. 进容器并自检

```bash
docker compose exec dev bash
```

在容器里：

```bash
nvidia-smi
python -c "import jax; print(jax.devices())"
python -c "from robopianist import suite; print('ok')"
```

预期：

- `nvidia-smi` 能看到 GPU（本机是 RTX PRO 6000 Blackwell）
- `jax.devices()` 里有 `GpuDevice`，不要只有 `CpuDevice`
- `from robopianist import suite` 成功。必须用本仓库的 `robopianist`（支持 `--robot daxian`），不要 `pip install robopianist`

---



## 3. 日常命令

容器默认 `sleep infinity`，源码挂载为 `/workspace`。


| 目的            | 命令                                                                  |
| ------------- | ------------------------------------------------------------------- |
| 开 shell       | `docker compose exec dev bash`                                      |
| 达显 V2 SAPIEN 预览 | 宿主机执行 `bash docker/run_sapien_v2.sh`（不要在训练容器里跑）。窗口出现在 xrdp XFCE（`:10`），不是 Cursor。 |
| 达显 V2 训练      | `cd /workspace/robopianist-rl && bash run_daxian_v2.sh`（说明见 `docs/daxian_v2.md`） |
| 达显 V3 训练      | `cd /workspace/robopianist-rl && bash run_daxian.sh`                |
| 解锁 PIP/DIP 对照 | `cd /workspace/robopianist-rl && bash run_daxian_unlock_pip_dip.sh` |
| 换曲评测          | 见下方                                                                 |
| 停容器           | `docker compose stop`                                               |
| 删容器（镜像保留）     | `docker compose down`                                               |


换曲评测（在容器内、`/workspace/robopianist-rl`）：

```bash
python eval_song.py \
  --ckpt <CKPT> \
  --environment-name <ENV_NAME>
```

本机示例：

```bash
python eval_song.py \
  --ckpt ../eval_daxian/checkpoints/latest.pkl \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0
```

训练产物写在宿主机的 `eval_daxian/`（V3）、`eval_daxian_v2_base/`（V2 当前基线）与对照目录，不进镜像，也不应 `git add -f`。

### 达显 SAPIEN 预览（必须 Docker + xrdp）

当前训练容器 `daxianpianist-dev-1` **没有** `/tmp/.X11-unix`，在里面 `python examples/sapien_left_hand_v2_tune.py` 打不开窗口。不要 `docker compose up -d --force-recreate` 去补挂载——那会停掉正在跑的训练。详见 `question.md`。

先连上这台机器的远程桌面（xrdp **3391**，会话是 `DISPLAY=:10`），再在**宿主机**终端（不要 `exec` 进训练容器）执行：

```bash
cd /home/houjue/DaxianPianist
bash docker/run_sapien_v2.sh
```

sidecar `daxianpianist-sapien` 已在跑时，该脚本会 `docker exec` 复用，不会重建容器。窗口出现在 XFCE 桌面上，不在 Cursor 里。
---



## 4. 用 Cursor 开发（推荐做法）

本机终端里 `docker info` 是正常的。Cursor 报

`[docker info] Command failed with exit code 401`

**不是 Docker 没装**，而是 Cursor 的 Dev Containers 在已经 Remote 过的窗口里再套一层容器：它把 `docker info` / `printenv` 发到一条 HTTP 通道上，返回了 **401 Unauthorized**。先前的端口转发失败、`printenv 401` 是同一件事。

当前这个 Cursor 窗口已经连在 `FC500T-K` 上，**不要再 Attach / Reopen into Container**。

### 推荐：SSH/本机窗口里写代码，容器里跑训练

容器已在跑。在当前 Cursor 终端：

```bash
cd /home/houjue/DaxianPianist
docker compose exec dev bash
```

之后在容器里训练、评测：

```bash
cd /workspace/robopianist-rl
bash run_daxian.sh
```

改代码仍在宿主机仓库，挂载进 `/workspace`，保存即生效。

### 如果一定要把编辑器开进容器

必须满足：Cursor 是 **这台 Linux 本机直接打开的窗口**（不是另一台电脑 Remote SSH 再套容器）。

1. 只装 **Anysphere 的 Dev Containers**。卸掉 Microsoft Dev Containers、Azure Container Tools
2. 设置里把 Docker 路径指到本机：`dev.containers.dockerPath` = `/usr/bin/docker`
3. 若本机装了 Docker Desktop，先在 Desktop 里 **Sign out**（Hub 登录过期也会变成 401）
4. `docker compose up -d`
5. **Dev Containers: Attach to Running Container** → `daxianpianist-dev-1`
6. **File → Open Folder** → `/workspace`

---



## 5. 注意

- 训练用 EGL，不需要显示器。达显 SAPIEN 预览需要 xrdp 桌面（本机 `:10`），在**宿主机**跑 `bash docker/run_sapien_v2.sh`。当前训练容器 `daxianpianist-dev-1` 启动时没有挂 X11，不要在那个 exec 里开 viewer。
- 本机 GPU 是 Blackwell（算力 12.0），需要较新的 `jax[cuda12]`。若 `jax.devices()` 只有 CPU，把完整报错留下再改 JAX / CUDA 镜像标签。
- `import robopianist` 需要 `robopianist/robopianist/soundfonts/TimGM6mb.sf2`。镜像里有一份，entrypoint 会在宿主机缺失时拷进去。
- 不要在容器里再装一份 PyPI 的 `robopianist`。
- 本机访问 Docker Hub（`registry-1.docker.io`）会超时。基础镜像已改为 NVIDIA NGC：`nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`。若仍拉不下，可把 `FROM` 换成 DaoCloud 前缀：`docker.m.daocloud.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`。
- 构建时若访问 GitHub / SourceForge / PyPI 失败（TLS、代理），需要先解决宿主机出网，再 `docker compose build`。

