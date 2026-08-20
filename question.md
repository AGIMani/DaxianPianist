# 训练问题记录

## 开头第一拍食指键按不下去

**现象：** `eval_daxian_v2` 评测视频（如 `videos/00011.mp4`）里，曲子一开始右手食指对应的键已经按指法上色，但食指没有压下去。后面段落可以有按键，开头这一下经常空掉。

**曲子：** `RoboPianist-debug-TwinkleTwinkleRousseau-v0`（trim silence 之后）。

**第一拍实际是什么：**

- 量化到 `control_timestep=0.05` 后，第一响在 **t = 1.0 s**（`--initial-buffer-time 1.0` 的静音之后）。
- 同时两音：右手食指 **C5**（指法 1，白键 51）+ 左手小指 **C3**（指法 9，白键 27）。
- 食指这颗 C5 只持续约 **0.3 s（6 步）**，错过窗口就没有 `key_press`。

**原因：**

1. Rest 时右手食指大约在 `y ≈ 0.29`，C5 在 `y ≈ 0.11`，沿琴键差约 **18 cm**，必须靠 `forearm_tx` 滑过去再下压 MCP。
2. 原先 `--n-steps-lookahead 10` 只看未来 **0.5 s**。静音 1.0 s，所以 **前 0.5 s 的 `goal` 全空**，策略看不见即将到来的 C5；从 t=0.5 s 才看见，只剩 0.5 s 去滑 18 cm，再赶上 0.3 s 的音符。
3. `fingering` 奖励只在**当前音符步**给，1 s 预备里没有“先把食指移到 C5”的 shaping。

**修改（2026-08-20）：**

三个 V2 脚本把 `--n-steps-lookahead` 从 10 改成 **20**（= `1.0 / 0.05`），从第 0 帧就能在 `goal` 里看到第一拍：

- `robopianist-rl/run_daxian_v2.sh`
- `robopianist-rl/run_daxian_v2_unlock.sh`
- `robopianist-rl/run_daxian_v2_unlock_ty.sh`

约束：`n_steps_lookahead >= initial_buffer_time / control_timestep`。

**注意：** lookahead 变长会改变观测维，旧 checkpoint 不能接着训，需要重新开跑。正在跑的任务也要重启脚本才生效。

## 训练容器里打不开 SAPIEN 预览

**现象：** 提示符是 `root@4bd279b03a9d:/workspace/robopianist` 时执行

```bash
python examples/sapien_left_hand_v2_tune.py
```

会先打 Vulkan ICD 警告（`Failed to find Vulkan ICD file`），然后退出：

```
SAPIEN viewer needs X11. This training container has no /tmp/.X11-unix.
```

**原因：** `4bd279b03a9d` 是训练容器 `daxianpianist-dev-1`，不是预览容器。它 2026-08-19 启动时只挂了仓库，**没有** `/tmp/.X11-unix`，也没有 `DISPLAY`。`docker-compose.yml` 后来虽然加了 X11 挂载，但**不会**自动进已经在跑的容器；`docker compose up -d --force-recreate` 会停掉里面的训练，不要用。

SAPIEN viewer 必须连 X11（这台机器是 xrdp XFCE，`DISPLAY=:10`，端口 **3391**）。训练本身用 EGL，不需要显示器。Vulkan 警告是 `import sapien` 时还没设 `VK_ICD_FILENAMES`，发生在 X11 检查之前，**不是**打不开窗口的原因。

**修改（2026-08-20）：**

1. `sapien_left_hand_v2_tune.py` 在 `import sapien` 之前设置 `VK_ICD_FILENAMES=.../docker/nvidia_icd.json`，训练容器里不应再刷 ICD 警告；没有 X11 时提示去宿主机跑脚本。
2. `docker/run_sapien_v2.sh`：在容器内直接拒绝；sidecar 已在跑则 `docker exec` 复用，**不再** `docker rm -f`（会误杀 sidecar 里的进程）。

**正确做法：** 先连远程桌面 **xrdp 3391**，再 `exit` 出训练容器，在**宿主机**仓库根目录：

```bash
bash docker/run_sapien_v2.sh
```

窗口出现在 XFCE 桌面上，不会出现在 Cursor 里。若报 X11 鉴权失败，在宿主机执行 `DISPLAY=:10 xhost +local:root`。

同日已在 sidecar 里核实：`scene.create_viewer()` 成功，并已拉起 `sapien_left_hand_v2_tune.py`（连上 3391 即可看到）。
