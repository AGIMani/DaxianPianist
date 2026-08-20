# 达显钢琴 Sim2Real

Tsensei 达显手（Daxian V2）从 MuJoCo 策略迁到真机的笔记、阅读清单和工程计划。仿真主路径仍是本仓库的 `robopianist/` + `robopianist-rl/`。安装位姿和 rest 只改 `daxian_v2_hand_constants.py`。

当前仿真动作空间、前臂范围、训练命令以 **`docs/daxian_v2.md`** 为准。

本文会随真机标定和实验更新。先对齐几何与驱动，再谈随机化和残差 RL。

---

## 1. 当前仿真里已经定死的约定

这些会直接变成真机接口约束。对不上，策略再好也按不准。

| 项 | 现状 | 真机必须对齐 |
| --- | --- | --- |
| 手 | 达显 V2，`--robot daxian_v2` | 同一套 URDF / 关节名 / 方向 |
| 基线动作 | `run_daxian_v2.sh`：33 维，`reduced_action_space`，四指 PIP/DIP 钉住；MIDI 指到的手指 IK 耦合到 MCP | 真机 PIP/DIP 用同一套门控 IK，或固件里做同样耦合 |
| `pinky_rota` | 钉在 0，不进策略 | 真机锁 0 |
| 拇指 | `rota` / `swing` / `MCP` / `PIP` 全由策略控 | 四维都要能跟 |
| 前臂默认 | `tx` + `ty` + `tz` + `yaw`（无 `roll`） | 沿琴键、高度、前后、偏航；范围见 `docs/daxian_v2.md` |
| 对照实验 | `run_daxian_v2_unlock.sh` 解锁四指 PIP/DIP（约 49 维） | 未选作主策略前不要上真机 |
| CanonicalSpec 0 | MCP 伸直、ty/tz/yaw=0、拇指 rest 全 0 | 硬件 `q_zero` 约定为全 0 |
| 控制周期 | `control-timestep 0.05`（20 Hz） | 真机伺服周期、指令延迟要量出来 |
| 安装 | `LEFT/RIGHT_HAND_POSITION`、`HAND_BASE_RPY_DEG`、Z-spin | 手相对钢琴的位姿（含高度 z） |
| 指尖 | 钢琴垫 mesh 碰撞，`FINGERTIP_COLLISION_TYPE = "mesh"` | 真机指垫厚度、触点位置 |
| 四指 MCP 训练范围 | `(0, 0.6)` rad，约 34°，不是 URDF 90° | 真机行程和力限不要按满行程去跟 |

V2 耦合开关：`COUPLE_PIP_DIP_TO_MCP = True`。策略只出四指 `swing`+`MCP`。当前帧 MIDI 指到的手指才 IK（指尖朝键盘竖直向下）；空闲手指 PIP 用 rest、DIP=0。详情 `docs/daxian_v2.md`。

---

## 2. 知识：这条任务的 gap 在哪

钢琴 sim2real 不是视觉域随机就能过。主 gap 是 **接触、驱动、几何**。

### 2.1 几何

- 手相对钢琴的 6D：仿真 attach 位姿必须等于实测安装。差几毫米就会按到键缝或按空。
- 指尖垫相对 DIP 的偏移：mesh 换过 `A2_tip` 之后，真机垫厚必须再量。
- 键平面高度、键间距、白键/黑键高度差。
- `forearm_tx` 的零位：仿真 0 = 安装位置，不是键盘中点。

### 2.2 驱动与时间

- 仿真是位置伺服 + `kp` / damping / `forcerange`；真机是电机 + 减速 + 通信延迟。
- 20 Hz 策略 vs 内环 1 kHz：中间要有插值，不能一步跳目标角。
- 关节零位、方向、限位、最大角速度。硬件 `q_zero` 已假定全 0。
- 耦合关节：若真机 PIP/DIP 是腱或从动，不要让策略再出独立 PIP/DIP（所以基线比 unlock 更接近真机）。

### 2.3 接触

- 键的行程、回弹、摩擦、是否允许“凿进键盘”（仿真里 `forearm_ty` 最容易学坏这个）。
- 指–键、指–指、掌–键。掌碰撞在仿真里已改成 visual-only，真机掌会碰到琴的话要补回来。
- MuJoCo `solref` / `impratio` 再硬，也不等于真机接触刚度。

### 2.4 观测

- 仿真策略吃的是特权状态（关节、键位置、MIDI lookahead），不是相机。
- 真机若没有逐键开合传感器，要用：关节编码器 + 可选 MIDI/拾音，或教师–学生把特权信息蒸馏掉。
- 双手 + sustain，动作维以基线为准（约 12 指 + 2 前臂）×2 + sustain。

### 2.5 常用手法（按优先级）

1. **系统辨识**：把仿真伺服和几何调到接近真机，比加随机化更有效。
2. **动作空间与真机同构**：钉住/耦合与固件一致。
3. **Domain randomization**：质量、摩擦、`kp`、延迟、安装误差；幅度从小到大。
4. **残差 RL**：IK / PD / 已有仿真策略当底座，RL 只学修正（PianoMime、ManipTrans）。
5. **非对称 actor-critic / 蒸馏**：训练时用按键真值，部署时只用编码器。
6. **不要先上 `forearm_ty`**：多一个往下钻的自由度，真机力限对不上就会把琴砸穿或学不会。

---

## 3. 开源项目与论文

只列和达显钢琴相关、能对照代码的。

### 3.1 钢琴 / 双手

| 资源 | 为什么看 |
| --- | --- |
| [RoboPianist](https://github.com/google-research/robopianist)（Zakka et al., CoRL 2023） | 当前仿真基准与奖励 |
| [robopianist-rl](https://github.com/kevinzakka/robopianist-rl) | SAC + DroQ 式 dropout/LN |
| [PianoMime](https://github.com/sNiper-Qian/pianomime)（Qian et al., 2024） | 演示 → 残差 RL → diffusion；最接近“先有轨迹再上真机” |
| [Privileged Sensing Scaffolds RL](https://penn-pal-lab.github.io/scaffolder/) | 教师用特权观测，学生用真机观测 |

### 3.2 灵巧手 sim2real

| 资源 | 为什么看 |
| --- | --- |
| OpenAI Dactyl / Rubik’s Cube（2018–2019） | 大规模 DR 的经典 |
| [DeXtreme](https://dextreme.org/)（Handa et al., CoRL 2022） | Isaac + Allegro，随机化质量/摩擦/驱动 |
| [LEAP Hand](https://leaphand.com/)（Shaw et al., RSS 2023） | 开源低成本灵巧手 + 训练配方 |
| HORA / AnyRotate（Qi et al.） | 手内旋转，teacher-student |
| DAPG / Adroit（Rajeswaran et al., RSS 2018） | 演示 + RL |
| [ManipTrans](https://github.com/ManipTrans/ManipTrans)（CVPR 2025） | 双臂灵巧残差迁移，可换 URDF |

### 3.3 残差与随机化

- Residual Policy Learning（Johannink / Silver 等）：底座控制器 + RL 残差。
- Tobin et al. 2017 domain randomization；CAD2RL。
- DroQ / REDQ：高 UTD 提升样本效率。本仓库已有 critic dropout + layer-norm；**不要**用 `n-envs=8` 配 `utd=8` 假装高 UTD（二者相除仍是每步 1 次更新）。

### 3.4 仿真栈

- MuJoCo + 本仓库 MJCF（当前训练）。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) + RSL-RL：以后若要大规模 DR 再考虑，不是第一阶段。

---

## 4. 学习计划

按周读，读完要能回答右边的问题。不必全文精读，对着达显差距看方法部分即可。

### 第 1 周：把仿真和真机接口画成同一张表

- 读本文第 1–2 节 + `daxian_v2_hand_constants.py`、`daxian_hand.py` 的 `_pin_fixed_fingers` / 耦合 IK。
- 读 RoboPianist 论文的任务定义、动作空间、控制频率。
- **产出**：关节名、方向、限位、零位、20 Hz 指令、PIP/DIP 耦合公式的对照表（仿真列 / 真机列 / 未知列）。

### 第 2 周：接触与驱动

- DeXtreme 论文的 randomization 表（抄可迁移的项：质量、摩擦、执行器延迟，不要抄视觉）。
- 本仓库指尖 mesh、`solref`、`FINGER_FORCE_LIMIT`、MCP 范围 0.6。
- **产出**：真机上要测的清单（见第 5.1 节），开始记数。

### 第 3 周：残差与蒸馏

- PianoMime：残差是加在关节还是任务空间；单曲 specialist vs 通才。
- Scaffolder：特权观测有哪些、学生观测怎么定义。
- **产出**：达显第一版真机控制器草图——PD 跟踪仿真关节轨迹，还是策略直接出目标角。

### 第 4 周：选一条主路径并冻结

- 基线（耦合 PIP/DIP，无 `ty`）确定为 **唯一上真机的动作空间**，直到几何对上。
- unlock / `forearm_ty` 只留在仿真对照，写进实验日志，不上台。
- **产出**：下面第 5 节的阶段闸门（gate）签字：测完哪些量才能进下一阶段。

补充阅读（有余力）：LEAP Hand 开源仓库的 sim2real 脚本；ManipTrans 的残差模块；OpenAI Rubik 的 DR 附录。

---

## 5. 实现计划

原则：**先让开环轨迹在真机上按到键，再闭环 RL。** 不要一上来把 SAC 策略接进电机。

### 阶段 0 — 记录与安全（约 3–5 天）

- [ ] 真机急停、电流/力矩限幅、关节限位软件再包一层（比 URDF 更紧）。
- [ ] 日志：指令角、实测角、时间戳、若有则 MIDI 或拾音。
- [ ] 明确左右手坐标系与仿真 `lh_` / `rh_` 一致。
- [ ] `pinky_rota` 固件锁 0。

闸门：能 20 Hz 发关节目标，编码器能回读，限幅生效。

### 阶段 1 — 几何标定（约 1–2 周）

只改 V2 constants，或单独的 `hardware/` 标定文件，**不要**改 V3。

- [ ] 测手基座相对钢琴：位置、RPY、左右镜像。回写 `LEFT/RIGHT_HAND_POSITION` 和 RPY/Z-spin。
- [ ] 测 z：指尖轻触未按下的白键表面时的高度，对比仿真 rest + 耦合后的指尖高度。
- [ ] 测键宽、相邻键中心距，核对 `forearm_tx` 零位和行程。
- [ ] 单指：固定 `tx`，只动一个 MCP（PIP/DIP 走仿真同一套 IK），看触点是否在键中央、是否竖直向下。
- [ ] 指垫厚度：若真机比 mesh 厚，优先改仿真 collision 或 rest，而不是在真机里盲加偏移。

闸门：四指各能稳定按下一颗指定白键，不凿进键缝、不擦邻键。

### 阶段 2 — 驱动辨识（与阶段 1 可部分并行）

- [ ] 阶跃/斜坡响应：延迟、超调、稳态误差；用来设仿真 `kp`、damping，或真机侧插值。
- [ ] 确认策略 0 对应真机全 0（MCP 伸直，不是行程中点）。
- [ ] 把仿真 `FINGER_FORCE_LIMIT` / MCP 0.6 rad 映射到真机电流或速度限。
- [ ] 20 Hz 目标角之间做插值（三次或最短时间），避免方波冲击。

闸门：回放仿真里一段“只按 C 大调音阶”的开环关节轨迹，时值大致对、能出声。

### 阶段 3 — 开环回放仿真策略（约 1 周）

- [ ] 用当前基线 checkpoint 在仿真 dump 关节目标（含耦合后的 PIP/DIP）。
- [ ] 真机纯跟踪，**不**接奖励、不在线 RL。
- [ ] 对比：按对的键、错键、延迟、力度（键到底程度）。
- [ ] 记下系统误差：整体偏左/偏高/偏慢，优先用标定消掉，不要用残差硬补大偏差。

闸门：《小星星》片段能听出旋律，错键可统计。若几何仍漂，停在阶段 1。

### 阶段 4 — 仿真侧为真机做的最小改动

仍在 `robopianist/`，为阶段 5 做准备，不换 Isaac。

- [ ] Domain randomization（小范围）：安装 xy/z ±几毫米、yaw 小扰动、`kp`、摩擦、指令延迟 0–2 步。
- [ ] 观测噪声：关节角加编码器量级噪声。
- [ ] 可选：非对称 critic（训练见真键位置，actor 只见关节 + MIDI）。
- [ ] 继续 **单曲 specialist**；通才和 diffusion 放到更后。
- [ ] 训练配置保持轻量：`n-envs=1`、`batch=256`、`utd=1`（或以后单独加高 UTD，不要跟 n-envs 绑死）。评测视频仍可每 10 万步。

闸门：DR 策略在仿真扰动下 F1/按键不崩；无 DR 的旧策略当对照。

### 阶段 5 — 真机闭环（残差优先）

- [ ] 底座：阶段 3 的开环轨迹或 PD 跟踪。
- [ ] 残差策略：输出 Δq 或 Δxyz，限幅要小（几度 / 几毫米）。
- [ ] 奖励尽量用真机测得到的量：错键（MIDI 或拾音）、关节跟踪误差；不要假设有仿真里的 fingering 特权。
- [ ] 先单手、单小节，再双手。
- [ ] 不把 `forearm_ty`、unlock PIP/DIP 带进这一阶段。

闸门：残差比纯开环少错键，且力矩不顶死。失败则减小残差范围或回到标定。

### 阶段 6 — 以后才做（现在不排期）

- 多曲、转调、PianoMime 式 diffusion。
- Isaac Lab 大规模 DR。
- 视觉 servoing / 无 MIDI 听音闭环。
- `forearm_ty`、完全独立 PIP/DIP：等基线真机稳定再开仿真对照。

---

## 6. 真机测量清单（阶段 1–2 填数）

复制到实验记录里填。未知就写未知，不要用仿真值冒充。

```
日期 / 手（左/右） / 固件版本：

关节零位与方向（与仿真同名关节）：
延迟（指令到编码器 90%，ms）：
最大跟踪误差（慢斜坡，deg）：

基座相对钢琴 origin（m, deg）：
白键表面高度（m）：
指尖触白键时的 MCP/PIP/DIP（deg）：

pinky_rota 是否机械锁 0：
PIP/DIP 真机是独立伺服还是腱/从动：
20 Hz 接口是否丢包：
```

---

## 7. 仓库内相关文件

| 文件 | 用途 |
| --- | --- |
| `robopianist/robopianist/models/hands/daxian_v2_hand_constants.py` | V2 安装、rest、耦合、MCP 范围 |
| `robopianist/robopianist/models/hands/daxian_hand.py` | 钉住关节、IK 耦合、前臂 DOF |
| `robopianist/robopianist/wrappers/pip_rest.py` | 策略 0 = rest |
| `robopianist/examples/sapien_left_hand_v2_tune.py` | 安装与 MCP/IK 预览（xrdp 上看窗口） |
| `robopianist-rl/run_daxian_v2.sh` | 基线训练（应对准真机动作空间） |
| `docs/docker_train.md` | Docker 里怎么训 |
| `docs/technical_report.md` | 奖励与 SAC 说明（文中旧称“达仙”以代码为准） |

训练产物目录：`eval_daxian_v2_base/`（当前基线）、旧目录 `eval_daxian_v2/`（历史 29/31 维，不要混）、`eval_daxian_v2_unlock/`、`eval_daxian_v2_unlock_ty/`。上真机只考虑当前基线 checkpoint。
