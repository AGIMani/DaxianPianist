# DaxianPianist 技术汇报

本文档根据当前仓库代码整理，用于内部汇报。覆盖仿真任务、达仙手动作空间、**奖励函数**、**SAC 算法**、训练协议与现有实验结果。实现以本仓库为准，不依赖口头约定。

**达显 V2（当前训练/预览）** 的安装、33 维动作空间、PIP/DIP 门控和训练命令见 **`docs/daxian_v2.md`**。下文 §4 / §8 的 25 维数字是 **V3** 主实验，不要套到 V2。

- 仿真与任务：`robopianist/`
- 训练：`robopianist-rl/`（JAX / Flax SAC）
- 达仙手 CAD/URDF：V2 `daxian_V2/`（当前）；V3 `daxian_V3/`
- 远程仓库：<https://github.com/AGIMani/DaxianPianist>
- 原论文：Zakka et al., *RoboPianist: A Benchmark for High-Dimensional Robot Control*, 2023

---

## 1. 项目在做什么

DaxianPianist 把 Google Research 的 RoboPianist 从 **Shadow Hand** 迁到 **达仙仿人双手**，在 MuJoCo 里用强化学习学会按 MIDI 指定的琴键。

任务可表述为：每 50 ms 给出一次关节指令，双手在 88 键钢琴上跟踪当前时刻的目标键位（以及未来 10 步的 lookahead），使按下的键与 MIDI 一致，并尽量用 MIDI 标注的手指去按。

这不是“听音频弹琴”，也不是多曲通才。当前主实验是 **单曲 specialist**：只在 Rousseau 版《小星星》片段 `RoboPianist-debug-TwinkleTwinkleRousseau-v0` 上训练 5M 步。观测里虽有 10 步 lookahead，理论上可以支持多曲，但训练数据没有混 MIDI、没有转调/变速增强，所以策略会过拟合这一首。

---

## 2. 仓库结构

| 路径 | 作用 |
| --- | --- |
| `robopianist/robopianist/models/hands/daxian_hand.py` | 达仙手 Composer 实体：关节、执行器、指尖碰撞球、rest 姿态 |
| `robopianist/robopianist/models/hands/daxian_hand_constants.py` | 关节分组、rest 角、动作空间、指尖颜色、安装位姿 |
| `robopianist/robopianist/suite/tasks/piano_with_shadow_hands.py` | 双手弹钢琴任务：奖励、观测、指法着色 |
| `robopianist/robopianist/suite/tasks/base.py` | `PianoTask`：装左手/右手、重力补偿、前臂滑轨范围 |
| `robopianist/robopianist/wrappers/pip_rest.py` | CanonicalSpec 的 0 对应 rest（伸直 MCP），而不是关节中点 |
| `robopianist-rl/sac.py` | Soft Actor-Critic |
| `robopianist-rl/train.py` | 训练循环、checkpoint、SwanLab、评测视频 |
| `robopianist-rl/run_daxian.sh` | 主实验：锁定四指 PIP/DIP，产物写到 `eval_daxian/` |
| `robopianist-rl/run_daxian_unlock_pip_dip.sh` | 对照：解锁 PIP/DIP，产物写到 `eval_daxian_unlock_pip_dip/` |
| `robopianist-rl/eval_song.py` | 用已有 checkpoint 换曲评测 |

`PYTHONPATH` 必须包含本仓库的 `robopianist/`。conda 里若装着原版 RoboPianist，`suite.load` 没有 `robot=` 参数，达仙手跑不起来。

训练产物（checkpoint、mp4、SwanLab）在 `.gitignore` 中，不进 GitHub。本机目录约 19 GB。

---

## 3. 仿真环境

### 3.1 物理与控制频率

- 物理步长 `physics_timestep = 0.005 s`（200 Hz）
- 策略步长 `control_timestep = 0.05 s`（20 Hz）
- 每个动作在 MuJoCo 里积分 10 个物理步
- 钢琴键 `solref = (0.01, 1)`，比默认更硬，减少“按下去像海绵”

### 3.2 钢琴

全尺寸 88 键数字钢琴。键激活由关节角相对行程判断。延音踏板是任务动作的最后一维，范围 `[0, 1]`。

评测相机 `piano/back`：垂直视场 36°，覆盖约 **1.05 倍键盘宽度**，从琴后上方看，避免默认 45° 广角裁掉谱架、手指接触看不清。

### 3.3 达仙手安装

双手是 URDF 导出的仿人灵巧手（左右 Y 镜像），再加虚拟前臂。

| | 左手 | 右手 |
| --- | --- | --- |
| 世界位置 (m) | `(0.205, -0.30, 0.045)` | `(0.205, 0.30, 0.045)` |
| 基座欧拉 | `(0, -90°, 0)`，再绕前臂 +Z 转 180° | 同基座，-180° |
| 手掌上仰 | 绕世界 +Y 再转 22°，指尖侧高于腕部 | 同 |

默认前臂自由度：

- `forearm_tx`：沿键盘左右平移（范围按钢琴宽度重写）
- `forearm_tz`：世界 +X（SAPIEN `pos_x`）。范围见 `FOREARM_TZ_RANGE`，负值伸进键盘；CanonicalSpec 的 0 表示**安装 x、不额外前后移**
- `forearm_ty`：世界 +Z（SAPIEN `pos_z`）。范围见 `FOREARM_TY_RANGE`，负值压低手掌；CanonicalSpec 的 0 表示**安装 z、不额外升降**
- `forearm_yaw`：绕世界 Z。左右手都是 `[-0.6, 0]`（轴已 reflect，`+ctrl` 向外；CanonicalSpec 的 0 表示**不偏航**）

默认训练不再命令 `forearm_roll`。下压力主要靠四指 MCP / swing，安装高度已经压低；`forearm_ty` 只做小范围高度微调。

手指位置伺服力矩上限 `±2 N·m`；前臂平移 `±20 N`、滚转 `±5 N·m`。不加力矩上限时，未训练策略会用前臂顶穿琴键。

`gravity_compensation=True` 时对整只手补重力，避免手臂因自重下垂。

指尖碰撞用半径 8 mm 的球，而不是网格。奖励 site 放在球的掌侧表面，使“site 到键面”和“指腹碰到键”是同一件事。

### 3.4 MIDI 与 episode

当前训练曲：`TwinkleTwinkleRousseau`，Rousseau YouTube 版《小星星》裁剪片段，带逐音指法（`part` 字段）。`--trim-silence` 把第一音对齐到 t=0。

指法编号：0–4 右手拇指到小指，5–9 左手拇指到小指。

Episode 长度等于 MIDI 按 `control_timestep` 离散化后的步数。默认 **不会** 因为按错键提前结束（`wrong_press_termination=False`）。失败终止若打开，折扣会被置 0。

每步目标 `goal` 是长度为 `n_keys+1` 的 0/1 向量（88 键 + 延音），再沿时间堆 `n_steps_lookahead+1` 帧。训练用 lookahead = 10，即看到当前加未来 10 步，共 11 帧。

---

## 4. 动作空间

策略输出经 `CanonicalSpecWrapper` 映射到 `[-1, 1]`，再线性还原到各执行器 `ctrlrange`。达仙手额外套 `PipRestAtZeroWrapper`：对 MCP、拇指 rest、前臂滚转，**策略 0 对应 rest，不是区间中点**。

这一点很关键。四指 MCP 的物理范围是 `[0, 1.57]` rad。若不平移，CanonicalSpec 的 0 会变成约 0.785 rad，手一开局就已经半握、压在键上，策略几乎只能学“别按错”，很难学“何时按”。

### 4.1 主实验：锁定 PIP/DIP（25 维）

`--reduced-action-space`，不设 `--unlock-four-finger-pip-dip`。

每只手 12 维：

| 关节 | 策略是否控制 | 锁定时的值 |
| --- | --- | --- |
| `thumb_rota_block` | 是 | — |
| `thumb_rotaback` | 是 | — |
| `index/mid/ring/pinky_swing` | 是 | — |
| `index/mid/ring/pinky_MCP` | 是 | idle = **0 rad（伸直）** |
| `thumb_MCP`, `thumb_PIP` | 否 | rest：MCP=0, PIP=0 |
| 四指 PIP/DIP | 否 | 弯曲 rest（见下表） |
| `forearm_tx`, `forearm_roll` | 是 | roll rest = 0 |

四指 PIP/DIP 锁定角（rad，左右手关节空间相同）：

| 手指 | PIP | DIP |
| --- | --- | --- |
| 食指 | 0.549 | 0.874 |
| 中指 | 0.531 | 1.285 |
| 无名指 | 0.531 | 1.047 |
| 小指 | 0.472 | 0.920 |

拇指 rest：`rota=1.184`，`rotaback=0.488`。

双手 24 维 + 延音 1 维 = **25**。

锁定 PIP/DIP、MCP=0 时，指尖大约悬在键面上 3 cm，idle **不会**误触。按键靠学到的少量 MCP 屈曲（以及 fingering 奖励把指尖拉向目标键）。

### 4.2 对照：解锁 PIP/DIP（41 维）

`--unlock-four-finger-pip-dip`。四指 PIP/DIP 交给策略；CanonicalSpec 0 仍是伸直（范围下限）。拇指 MCP/PIP 仍然钉死。每只手 20 维 + 延音 = **41**。

产物目录必须分开，避免覆盖 `eval_daxian/`。

### 4.3 每步如何施加动作

```text
action  →  对半拆成右手 / 左手（去掉最后一维 sustain）
        →  写入对应执行器 ctrl
        →  把 pinned 关节的 ctrl 强制写回 rest
        →  sustain 写入钢琴踏板
```

因此锁定关节即使出现在观测的 `joints_pos` 里，策略也改不了它们。

---

## 5. 观测空间

`ConcatObservationWrapper` 把下列向量拼成一条：

| 字段 | 内容 | 典型维度 |
| --- | --- | --- |
| 左右手 `joints_pos` | 全部关节角（含锁定关节和前臂） | 每手约 22，双手约 44 |
| `piano.state` | 88 键归一化行程 | 88 |
| `piano.sustain_state` | 踏板状态 | 1 |
| `goal` | `(lookahead+1) × (88+1)` 展平 | 11 × 89 = 979 |
| `fingering` | 当前步左右手 5 指是否该用 | 10 |
| （可选）上一步 action、reward | `--action-reward-observation` | 25 + 1（主实验） |

主实验打开了 action-reward 观测。`frame_stack=1`，不叠历史帧。lookahead 已经提供未来目标，所以这是“看谱”而不是纯 MDP 无记忆。

---

## 6. 奖励函数（逐步、逐项）

总奖励是若干标量项之和，实现见 `CompositeReward`：每步按添加顺序计算，写入 `reward_terms`，训练时会记三项日志：

- `reward/<name>`：该项本 episode 求和（训练）或平均（评测）
- `reward_frac/<name>`：`|该项| / Σ|各项|`，看谁在“绝对值上占主导”
- `reward_contrib/<name>`：`该项 / 有符号总和`，看谁在推高/拉低回报

主实验启用的项（按代码添加顺序）：

1. `key_press_reward`
2. `sustain_reward`
3. `energy_reward`
4. `fingering_reward`（MIDI 有指法时）
5. `forearm_reward`
6. `key_center_reward`（本仓库新增）
7. `false_press_penalty`（本仓库新增）

本仓库相对原版 RoboPianist **新增了 6、7**，并让 fingering 被误按键数除权。

下面用的 `tolerance(x; bounds=(lo, hi), margin=m, sigmoid=gaussian)` 来自 `dm_control.utils.rewards`：

- `x ∈ [lo, hi]` → 1
- 偏离区间后按高斯衰减
- `|x|` 超出 `margin` → 约 0

### 6.1 `key_press_reward` ∈ [0, 1]

这是“弹对音”的主项，拆成两半，各最多 0.5。

**该按的键（true positive 形状奖励）**

对当前 MIDI 目标为 1 的键，比较“目标按下”与“实际归一化行程”：

```text
actual[k] = piano.state[k] / qpos_range_max[k]   # 0 = 完全抬起, 1 = 按到底
err[k]    = goal[k] - actual[k]
```

```text
r_on = 0.5 * mean_k∈on  tolerance(err[k]; bounds=(0, 0.05), margin=0.50, gaussian)
```

键行程误差在 5% 以内得满分；margin 为 0.5，大约半程按下还能拿到可观的 shaping。当前步没有任何目标键时，这一半为 0。

**不该按的键（false positive 开关）**

```text
r_off = 0.5 * (0  if 任意非目标键被 activation 判为按下
               1  otherwise)
```

这是硬开关，不是按误按数量平滑。只要有一个错键处于激活态，这 0.5 整段拿不到。误按的**数量**由第 7 项再罚。

### 6.2 `sustain_reward` ∈ [0, 1]

```text
tolerance(goal_sustain - actual_sustain;
          bounds=(0, 0.05), margin=0.50, gaussian)
```

与按键同一套阈值。训练曲若踏板几乎不用，这项长期接近 1，对梯度贡献很小。

### 6.3 `energy_reward` ≤ 0

```text
r_energy = - 5e-3 * Σ_{双手, 全部执行器} |force| · |velocity|
```

系数 `_ENERGY_PENALTY_COEF = 5e-3`。抑制猛甩和顶键。它没有上界为 0 以外的下限，剧烈动作可以把总回报打到明显为负。功率传感器覆盖手部执行器（含被 pin 但仍有伺服维持 rest 的关节）。

### 6.4 `fingering_reward` ∈ [0, 1]（再被误按除权）

对当前步 MIDI 指法指定的每一个 (键, 手指)：

1. 取该手指奖励 site 的世界坐标
2. 取该键 geom 顶面、略靠玩家一侧的目标点（键中心 z 加上半高，x 方向再偏 0.35×键长）
3. 欧氏距离 `d`

```text
r_raw = mean  tolerance(d; bounds=(0, 0.01 m), margin=0.10 m, gaussian)
```

1 cm 内满分，10 cm 外接近 0。没有需要按的音时该项为 0（不是 1）。

本仓库加了除权，避免“粘在错键上仍拿 fingering”：

```text
r_fingering = r_raw / (1 + n_false)
```

`n_false` = 当前被激活、但不在 MIDI 目标里的键的个数。

若 MIDI 没有指法（`--disable-fingering-reward` 或文件无 fingering），则改用 **OT fingering**（RP1M，arXiv:2408.11048）：对 10 个指尖和当前目标键做匈牙利算法最小匹配，再用同样的距离 `tolerance`，同样除以 `(1+n_false)`。无目标键时 OT 项返回 1。

主实验曲有标注指法，走的是普通 fingering，不是 OT。

### 6.5 `forearm_reward` ∈ {0, 0.5}

检测左右手前臂根 body 的几何是否碰撞。碰撞则 0，否则 0.5。这是常数偏置加碰撞门控，不是连续距离奖励。`--disable-forearm-reward` 可关掉。

### 6.6 `key_center_reward` ∈ [0, 1]（新增）

只对 **fingering 指定的那些 (键, 手指)** 计算。用世界系 Y（沿键盘左右）衡量指尖是否压在键的中线上：

```text
offset = |y_tip - y_key| / key_half_width
r_center = mean  tolerance(offset; bounds=(0, 0.25), margin=1.0, gaussian)
```

相对半宽 0.25 以内视为在键中；到 1.0（键边缘以外）衰减到 0。没有指法目标时为 0。

设计意图：fingering 只拉“靠近键”，不区分按在键缝还是键心。键心奖励鼓励落在白键中线，减少擦到邻键。

### 6.7 `false_press_penalty` ≤ 0（新增）

```text
r_false = -0.15 * n_false
```

每个非目标激活键罚 0.15。与 `key_press` 的 0.5 开关不同：这里按**个数**线性加严。idle 时若左手一直压着 A3，每步 `n_false≥1`，该项持续为负，并且 fingering 被除权。

### 6.8 单步量级（便于读曲线）

无误按、指法到位、键按对、前臂不碰、耗能很小的“好步”大约：

```text
1.0 (key) + 1.0 (sustain) + 1.0 (fingering) + 0.5 (forearm) + 1.0 (center) + 0 (false) + ε(energy)
≈ 4.5 + ε
```

一次误按立刻：

- `key_press` 最多掉 0.5
- `false_press_penalty` −0.15（更多错键则更多）
- `fingering` 变成原来的 1/2、1/3、…

所以误按会同时打三项，这是刻意的：原先只有 `key_press` 的 0/1 开关，策略可以“粘在键上混 fingering”。

训练脚本把每项的 episode 累积和占比打到 SwanLab 的 `train/reward/*`、`eval/reward/*`。

---

## 7. 强化学习算法：SAC（必须按实现读）

算法是 **Soft Actor-Critic**（Haarnoja et al., 2018），代码在 `robopianist-rl/sac.py`，与原版 `kevinzakka/robopianist-rl` 同结构。离线缓冲 + 随机策略 + 双 Q + 可学习温度。

记号：状态 `s`，动作 `a`，奖励 `r`，折扣 `γ`，下一状态 `s'`，bootstrap 折扣 `d`（episode 结束为 0，否则为 1）。

### 7.1 策略（Actor）

三层 MLP，`hidden = (256, 256, 256)`，GELU，Xavier 初始化，最后一层激活后再出头。

输出高斯的均值与 **状态依赖** 的 `log σ`，`log σ` clip 到 `[-20, 2]`。分布是对角高斯再经 **tanh 双射**（`TanhNormal` / distrax），动作成到 `(-1, 1)`，对应 CanonicalSpec。

训练时从分布采样；评测用 `mode()`，即 tanh(均值)，确定性。

### 7.2 批评家（Critic）

两个 Q 网络（`num_qs=2`），结构同 actor 主干，输入 `[s, a]` 拼接，输出标量。主实验额外：

- Dropout `0.01`（仅 critic，训练时开启）
- LayerNorm（每层激活前）

这是原论文 specialist 设定里用来稳住高维连续控制的正则，不是标准 SAC 必选项。

目标网络 `Q̄` 用 critic 参数的 Polyak 平均：

```text
θ̄ ← (1 - τ) θ̄ + τ θ ,   τ = 0.005
```

实现是 `optax.incremental_update(critic, target, tau)`。支持 REDQ 式随机抽子集 `num_min_qs`；当前配置为 `None`，两个 Q 都用。

### 7.3 温度 α

`α = exp(log_temp)`，`log_temp` 是标量参数，初值 `α_0 = 1`。

目标熵：

```text
H̄ = -0.5 × action_dim
```

主实验 `action_dim = 25`，故 `H̄ = -12.5`。动作维变 41 时自动变成 `-20.5`，无需手调。

### 7.4 一次 `update()` 的顺序

每个环境步（warmstart 之后、缓冲不少于 batch）采样 256 条转移，**先更 critic，再 actor，再温度**。三者都是 Adam，`lr = 3e-4`。

#### Critic：TD 目标带熵备份

下一动作从**当前** actor 采样（不是行为策略），取目标双 Q 的最小值：

```text
a' ~ π_θ(·|s')
q̄(s', a') = min_i Q̄_i(s', a')
y = r + γ d q̄(s', a')  -  γ d α log π_θ(a'|s')     # backup_entropy=True
```

`backup_entropy=True` 时减去熵项，使 Q 估计软价值 `Q^π`，与最大熵目标一致。

Critic 损失（对 ensemble 所有 Q 做均方）：

```text
L_Q = mean_i  ( Q_i(s, a) - y )²
```

`y` 对 critic 参数停止梯度。Dropout 的 RNG 在前向里传入。

#### Actor：最大化 Q − α log π

```text
ã ~ π_θ(·|s)
L_π = mean[  α log π_θ(ã|s)  −  mean_i Q_i(s, ã)  ]
```

Q 对 actor 取 ensemble **均值**（不是 min）。这与 Haarnoja 原文用 min 略有不同，与本仓库 / jaxrl 常见实现一致：actor 用平均 Q，bootstrap 用 min Q，减轻过估计同时给 actor 稍稳的梯度。

日志里的 `entropy` 是 `-log π` 的均值，即采样熵的估计。

#### 温度：把熵推到 H̄

```text
L_α = α · ( Ĥ - H̄ )
```

`Ĥ` 用刚刚 actor 损失里算出的熵（`−log π` 均值）。`α` 变大则更随机，变小则更利用。

### 7.5 折扣 γ = 0.8（不是默认 0.99）

`train.py` 默认 `discount=0.99`，但 `run_daxian.sh` 显式 `--discount 0.8`。这与原 RoboPianist 论文 specialist 一致。

20 Hz 下：

```text
有效时间尺度 ≈ 0.05 / (1 - 0.8) = 0.25 s
```

策略主要优化约 0.25 s（5 个控制步）内的回报，再往后的信用分配衰减很快。lookahead 观测补偿了“看更远的谱”；价值函数本身偏短视。对按键这种密集奖励是合理的，对需要提前移动前臂跨越大跨度的乐句会吃亏。

### 7.6 探索与缓冲

| 项 | 值 |
| --- | --- |
| 总环境步 | 5,000,000 |
| Warmstart | 前 5,000 步均匀随机动作，不更新网络 |
| Replay | 容量 1,000,000，均匀采样，batch 256 |
| 之后每步 | 1 次环境交互 + 1 次梯度更新（1:1） |
| 评测间隔 | 每 10,000 步，1 个 episode，动作用 `eval_actions` |

缓冲是 numpy 环形数组，存 `(s, a, r, d, s')`。Episode 边界：`last()` 之后 `reset()`，并向缓冲插入 `action=None` 的新初始状态，避免把终止和下一条轨迹的第一步拼成一条转移。

### 7.7 为什么用 SAC 而不是 PPO / TD3

原基准就是 SAC。理由在本任务上仍然成立：

- 动作 25–41 维连续，需要随机策略覆盖指法组合
- 奖励稠密（每步都有 shaping），off-policy 样本效率比 on-policy 更合适
- 最大熵有助于避免过早塌缩到“左手粘在某颗键上”的差局部解（当前实验说明熵正则**不够**单独解决粘键，所以才加了 `false_press_penalty`）

TD3 是确定性策略，探索靠动作噪声，在这套指法离散组合上通常更难。PPO 也能做，但 5M 步、单环境、无向量化时样本效率会差一截。

---

## 8. 训练协议（主实验）

`robopianist-rl/run_daxian.sh` 的完整语义：

```text
机器人          daxian
环境            RoboPianist-debug-TwinkleTwinkleRousseau-v0
步数            5e6
γ               0.8
warmstart       5e3
动作空间        reduced，PIP/DIP 锁定，25 维
lookahead       10
控制周期        0.05 s
trim_silence    开
gravity_comp    开
primitive tips  开（指尖球）
action-reward 观测  开
评测            每 1e4 步 1 episode，相机 piano/back
日志            SwanLab project=robopianist，本地目录 eval_daxian/swanlog
产物            eval_daxian/{checkpoints,videos,metrics}
```

网络与优化（`SACConfig` + 脚本覆盖）：

```text
actor/critic/temp lr     3e-4
hidden                   256×3, GELU
critic dropout           0.01
critic layer norm        True
τ                        0.005
init α                   1.0
target entropy           -0.5 × 25 = -12.5
batch                    256
replay                   1e6
```

Checkpoint 含 actor/critic/target/温度的参数与优化器状态、RNG、γ、τ、target_entropy 和完整 `args`。`latest.pkl` 每轮评测覆盖；`step_XXXXXXXX.pkl` 按步保留。

评测视频复制为 `eval_step_{step}.mp4`，**不删除**旧视频。另转 4 fps GIF 给 SwanLab（云端只收 GIF）。

必须用本仓库的 `robopianist`：

```bash
export PYTHONPATH="/path/to/pianist_daxian_v2/robopianist:$PYTHONPATH"
```

---

## 9. 评价指标

`MidiEvaluationWrapper` 在每个控制步记录 88 键的 `activation` 0/1，episode 结束时与 MIDI 离散化后的目标逐帧做 **binary precision / recall / F1**（sklearn，`zero_division=1`），再对时间取平均。踏板同样算一套 `sustain_*`。

含义：

- **Precision**：这一帧手里按下的键，有多少是谱上要的。粘键、误按会打低 precision。
- **Recall**：谱上要的键，有多少真的亮了。够不到、按太轻会打低 recall。
- **F1**：二者调和。论文主指标。

这是**逐帧键位分类**，不是 MIDI 音符起始对齐（onset F1）。按对了但早/晚一个控制步，仍可能被算错。`zero_division=1`：某一帧谱上全 0 且预测也全 0 时，该帧 P/R/F1 记为 1。

训练还记录 episode `return`、`length`、动作饱和比例 `action_sat_frac` 等。

---

## 10. 现有实验结果（5M specialist）

主实验目录 `eval_daxian/`，checkpoint `latest.pkl` / `step_05000000.pkl`。

训练曲是 Rousseau《小星星》双手片段，音区大致在 C3 / C5 / G5 一带，有指法标注。

**换曲零样本**（同一策略，不微调）：

| 曲 | 环境名 | 现象 | 约略 F1 |
| --- | --- | --- | --- |
| 训练曲 Rousseau 小星星 | `TwinkleTwinkleRousseau` | 应当相对最好（以训练曲线为准） | 训练过程中评测 |
| Rousseau Nocturne 片段 | `NocturneRousseau` | 几乎不会弹 | F1 ≈ 0.008，recall ≈ 0.007 |
| 单手简易小星星（C4–A4） | `TwinkleTwinkleLittleStar` | 右手几乎不屈；左手食指 MCP 均值约 +0.51，长时间停在 **A3** | F1 ≈ 0.006，recall ≈ 0.043；161 步里约 122 步误按 |

换曲失败的原因不是“RL 只能学一首”，而是 **这一次 run 的数据就是一首**。原论文的 150 曲 repertoire 是每曲一个独立 specialist，或另训通才。本 run：

- 没有 MIDI 混合
- 没有 `stretch` / `shift` 增强（脚本里因子分别为 1.0 和 0）
- γ=0.8 短视
- 左手“停在键上”在训练曲上可能仍能混到 fingering / 部分 key_press，于是泛化成默认姿势

`false_press_penalty` 和 fingering 除权是针对粘键加的；它们改变了奖励景观，但 **5M 这个已训好的 checkpoint 是在旧奖励或当前奖励下训的，换曲数字反映的是该 checkpoint，不是尚未训完的解锁 PIP 对照**。

---

## 11. 相对原版 RoboPianist 改了什么

| 项目 | 原版 | 本仓库 |
| --- | --- | --- |
| 默认机器人 | Shadow Hand | 达仙手 |
| 锁定关节 rest | Shadow 论文的 wrist/finger 子集 | 四指 PIP/DIP 弯曲 + 拇指 MCP/PIP；MCP idle=0 |
| CanonicalSpec 0 | 关节范围中点 | MCP/拇指/roll 的 rest（`PipRestAtZeroWrapper`） |
| 奖励 | key / sustain / energy / fingering / forearm | 加上 key_center、false_press，fingering 除以 `(1+n_false)` |
| 指尖着色 | Shadow 视觉网格 | 达仙视觉网格 + 碰撞球，按拇指紫 / 食指橙 / 中指青 / 无名蓝 / 小指黄 |
| 相机 | 默认视场易裁切、显广角 | 36°，1.05× 键盘宽 |
| 日志 | wandb | SwanLab，并拆 reward 分项 |
| 产物 | `/tmp` | `eval_daxian/` 与 `eval_daxian_unlock_pip_dip/` 隔离 |

原版奖励设计仍然构成骨架：`key_press` 的 0.5+0.5 结构、fingering 的 1 cm 高斯、energy `5e-3`、forearm 0.5，都来自论文任务。

---

## 12. 局限与后续（汇报时可讲的结论）

1. **当前策略是单曲专家，不是通用钢琴家。** 要通才需要训练时混合多 MIDI，并做转调/时间拉伸；只加 lookahead 不够。
2. **粘键是当前最可见的失败模式。** 奖励侧已经加重误按成本；是否够，要看带新奖励的完整 5M 曲线，而不是只看旧 ckpt 换曲。
3. **锁定 PIP/DIP 降低了维度（25 vs 41），也限制了触键姿态。** 解锁对照在 `run_daxian_unlock_pip_dip.sh`，不要和主实验写进同一目录。
4. **γ=0.8 使价值短视。** 大跨度移位、提前准备手指，更多靠观测里的谱，不靠长期价值。
5. **F1 是逐帧 88 键分类**，对时值误差敏感；听感/节奏要用视频和 MIDI onset 指标另评。

建议的下一组实验（按信息量）：

- 同一 25 维动作空间，确认新奖励下训练曲 F1 与 `n_false` 是否同时改善
- 训练曲上做 `stretch∈[0.9,1.1]`、`shift∈[-2,2]` 的 episode 级增强，再测简易小星星
- 小规模多曲混合（debug 集几首）再谈 repertoire-150

---

## 13. 复现命令

达仙手主实验：

```bash
cd robopianist-rl
export PYTHONPATH="../robopianist:${PYTHONPATH}"
bash run_daxian.sh
```

解锁 PIP/DIP 对照：

```bash
bash run_daxian_unlock_pip_dip.sh
```

换曲评测：

```bash
python eval_song.py \
  --ckpt ../eval_daxian/checkpoints/latest.pkl \
  --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0
```

Shadow 原版对照仍可用 `run.sh`（`--robot shadow`，产物在 `/tmp`）。
