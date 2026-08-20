# 达显 V2

Tsensei **达显**手当前训练 / 预览用的是 **V2**。安装位姿、rest、关节范围只改 `daxian_v2_hand_constants.py`，不要写进 V3 的 `daxian_hand_constants.py`。

`docs/technical_report.md`、`docs/report.md` 里的 **25 维** 是上一套 **V3**（`thumb_rota_block` / `rotaback`）。不要拿那些数字套到 V2。

奖励和 SAC 算法与 V3 共用，见 `docs/technical_report.md` §6–§7。真机迁移见 `docs/sim2real.md`。

---

## 1. 和 V3 的差别

| | V2（当前） | V3（上一套） |
| --- | --- | --- |
| 代码 | `daxian_v2_hand.py`、`daxian_v2_hand_constants.py` | `daxian_hand.py`、`daxian_hand_constants.py` |
| URDF | `daxian_V2/` | `daxian_V3/` |
| `--robot` | `daxian_v2` | `daxian` |
| 拇指 | `rota` + `swing` + MCP + PIP，全由策略控 | `rota_block` + `rotaback`；MCP/PIP 钉在 rest |
| 小指 | `pinky_rota` **焊死在 0**，不是关节 | 无此焊死 |
| SAPIEN | `examples/sapien_left_hand_v2_tune.py` | `examples/sapien_left_hand_tune.py` |
| 训练脚本 | `run_daxian_v2.sh` | `run_daxian.sh` |
| 产物 | `eval_daxian_v2_base/` | `eval_daxian/` |

预览：宿主机 `bash docker/run_sapien_v2.sh`（xrdp **3391**，`DISPLAY=:10`）。不要在训练容器里开 GUI。

---

## 2. 安装位姿

常量：`LEFT_HAND_POSITION` / `RIGHT_HAND_POSITION`。右手是左手的 **Y 镜像**。

| | 左手 | 右手 |
| --- | --- | --- |
| 世界位置 (m) | `(0.205, -0.30, 0.131)` | `(0.205, 0.30, 0.131)` |
| 基座欧拉 | `(0, -90°, 0)`，再绕前臂 +Z 转 180° | 同基座，−180° |
| 手掌上仰 | `PALM_PITCH_UP_DEG = 0` | 同 |

指尖碰撞是钢琴垫 **mesh**（`FINGERTIP_COLLISION_TYPE = "mesh"`），不是 8 mm 球。

---

## 3. 基线动作空间：33 维

`--reduced-action-space`，**不**设 `--unlock-four-finger-pip-dip`。

每只手 **16** 维：

| 关节 | 策略 | CanonicalSpec 0 |
| --- | --- | --- |
| `thumb_rota` / `swing` / `MCP` / `PIP` | 是 | `THUMB_REST_CTRL` 全 0 |
| 四指 `swing` | 是 | 0 |
| 四指 `MCP` | 是 | 0（伸直）。训练范围 `[0, 0.6]` rad（约 34°），不是 URDF 90° |
| 四指 PIP / DIP | 否 | 见 §4 |
| `pinky_rota` | 否 | 焊死 0 |
| `forearm_tx` | 是 | 区间中点（沿键盘，范围按钢琴宽度 − 安装 y 重写） |
| `forearm_ty` | 是 | 0 = 安装 z |
| `forearm_tz` | 是 | 0 = 安装 x |
| `forearm_yaw` | 是 | 0 = 不偏航 |

双手 32 + 延音 1 = **33**。SAC target entropy = `−0.5 × 33 = −16.5`。

**没有** `forearm_roll`。

前臂范围（`daxian_v2_hand_constants.py`，改这里即可）：

| DOF | 轴 | 范围 | 策略 −1 / +1 |
| --- | --- | --- | --- |
| `forearm_tx` | 世界 +Y，沿键盘 | 按钢琴宽度 | 滑到左右行程端 |
| `forearm_tz` | 世界 +X（SAPIEN `pos_x`） | `FOREARM_TZ_RANGE = [-0.05, 0]` m | −1：伸进键盘 5 cm；+1：夹回 0 |
| `forearm_ty` | 世界 +Z（SAPIEN `pos_z`） | `FOREARM_TY_RANGE = [-0.04, 0.06]` m | −1：压低 4 cm；+1：抬高 6 cm |
| `forearm_yaw` | 世界 +Z 转。轴已 reflect，`+ctrl` = 向外 | 左右手都是 `[-0.6, 0]` | −1：向内 0.6 rad；+1：夹回 0 |

`ty` / `tz` / `yaw` 经 `PipRestAtZeroWrapper` 把 CanonicalSpec 0 对齐到 rest（区间端点或内部的 0），不是物理范围中点。

---

## 4. 四指 PIP / DIP（不进策略）

`COUPLE_PIP_DIP_TO_MCP = True`。按**当前帧 MIDI 指法**门控，不是「MCP≠0 才耦合」。

| 该指这帧有 MIDI 指法 | PIP | DIP |
| --- | --- | --- |
| 有（食指=1 … 小指=4，每手） | IK：指尖远端轴朝世界 −Z（垂直落键） | 同 IK，与 PIP 按 rest 比例拆 |
| 无（空闲） | `FINGER_REST_CTRL`（耦合前的 rest） | **0**（伸直） |

空闲 PIP rest（rad）：

| 手指 | PIP | （表里的 DIP 只用于 IK 比例，空闲时 DIP=0） |
| --- | --- | --- |
| 食指 | 0.549 | 0.874 |
| 中指 | 0.531 | 1.285 |
| 无名指 | 0.531 | 1.047 |
| 小指 | 0.472 | 0.920 |

拇指不走这套四指耦合。MIDI：0–4 右手拇→小，5–9 左手拇→小。

对照：`--unlock-four-finger-pip-dip` 把四指 PIP/DIP 交给策略（每手再 +8），约 **49** 维，产物目录必须分开。

---

## 5. 训练

必须在 Docker 训练容器里跑（见 `docs/docker.md`）。

```bash
docker compose exec dev bash
cd /workspace/robopianist-rl
bash run_daxian_v2.sh
```

`run_daxian_v2.sh` 当前写入 **`eval_daxian_v2_base/`**（不要和旧的 `eval_daxian_v2/` 混）。要点：

- `--robot daxian_v2`
- `--reduced-action-space`
- `--forearm-dofs forearm_tx forearm_ty forearm_tz forearm_yaw`
- lookahead **20**（`initial-buffer-time 1.0` / `control-timestep 0.05`）
- 5M 步，曲 `RoboPianist-debug-TwinkleTwinkleRousseau-v0`

对照（不要覆盖基线目录）：

| 脚本 | 产物 | 和基线的差别 |
| --- | --- | --- |
| `run_daxian_v2_unlock.sh` | `eval_daxian_v2_unlock/` | 四指 PIP/DIP 进策略 |
| `run_daxian_v2_unlock_ty.sh` | `eval_daxian_v2_unlock_ty/` | 解锁手指（前臂与基线相同） |

`--resume` 会按 checkpoint 里保存的 `forearm_dofs` / 动作空间还原。**29 维、31 维、带 roll 的 33 维、当前 33 维不能混。** 维数相同但 DOF 集合不同（例如 roll 换成 ty）也不能接着训。

---

## 6. 改参数改哪里

| 要改的 | 文件 |
| --- | --- |
| 安装 xyz / 欧拉 / rest / 前臂范围 / MCP 上限 | `robopianist/robopianist/models/hands/daxian_v2_hand_constants.py` |
| 基线 DOF 列表、步数、lookahead | `robopianist-rl/run_daxian_v2.sh` |
| CanonicalSpec 把 0 对齐到 rest | `robopianist/robopianist/wrappers/pip_rest.py` |
| MIDI 门控耦合 | `daxian_hand.py` 的 `_pin_fixed_fingers`，任务里 `_sync_pip_dip_midi_gate` |
