# DaxianPianist 汇报

## 1. 仿真与控制（已落地）

- 把 RoboPianist 从 Shadow Hand 接到达显左右手（URDF → MJCF，镜像安装，指尖碰撞球）。
- 主实验动作空间 25 维：每手拇指 `rota`/`rotaback`、四指 swing+MCP、前臂左右平移和滚转，再加延音。四指 PIP/DIP 和拇指 MCP/PIP 钉在 rest。
- 对照 41 维：四指 PIP/DIP 交给策略，产物目录分开，不会覆盖主实验。
- CanonicalSpec 的 0 对齐 rest：四指 MCP idle = 伸直 0 rad，不是区间中点，idle 时指尖大约悬在键上 3 cm，不会一开局就压键。
- 评测相机拉远、36° 视场，覆盖约 1.05 倍键盘宽度；指尖按拇指/食指/中指/无名/小指着色，方便看指法。

## 2. 奖励与训练代码（已落地）

在原版 key / sustain / energy / fingering / forearm 上加了：

- **键心奖励**：鼓励按在键中线，减少擦邻键。
- **误按惩罚**：每个非目标激活键 −0.15，fingering 再除以 (1 + 误按数)，避免粘在错键上仍拿 shaping。

算法是 JAX SAC（γ=0.8，256×3，critic dropout + LayerNorm），日志改到 SwanLab，分项记录 `reward/*`。训练脚本、checkpoint、评测视频、`eval_song.py` 换曲都已打通。

## 3. 实验（两套都已跑满 5M）

训练曲都是 `RoboPianist-debug-TwinkleTwinkleRousseau-v0`。

| 实验 | 动作维 | 5M 训练曲 F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| 锁定 PIP/DIP（`eval_daxian/`） | 25 | 0.923 | 1.00 | 0.914 |
| 解锁 PIP/DIP（`eval_daxian_unlock_pip_dip/`） | 41 | 0.941 | 0.998 | 0.931 |

主实验 5M 评测上 `false_press_penalty = 0`，训练曲上几乎不误按。从 1 万步 F1≈0.06 拉到 5M 的 0.92，学习是成立的。解锁版在训练曲上略好一点，能耗略高。

### 换曲（同一 25 维 5M 策略，不微调）

- Nocturne：F1 ≈ 0.008
- 简易小星星（右手 C4–A4）：F1 ≈ 0.006；左手食指常停在 A3，右手几乎不屈

原因已经核对过：这次 run 只见过一首 MIDI，没有转调/变速/多曲混合。lookahead 不能代替多曲数据。
