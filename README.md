# DaxianPianist（Tsensei）

基于 [RoboPianist](https://github.com/google-research/robopianist) 的达显手弹钢琴项目。

远程仓库：[https://github.com/AGIMani/DaxianPianist.git](https://github.com/AGIMani/DaxianPianist.git)  
默认分支：`main`

详细技术汇报（奖励项、SAC 更新公式、动作空间、训练与评测）：[docs/technical_report.md](docs/technical_report.md)

Docker 训练（GPU 容器、依赖安装、`run_daxian_v2.sh` 等入口）：[docs/docker_train.md](docs/docker_train.md)

训练产物（checkpoint、视频、SwanLab 日志）已被 `.gitignore` 排除，**不要**用 `git add -f` 强行加入。

## 下次提交

在项目根目录执行：

```bash
cd /home/houjue/pianist_daxian_v2

git config user.name "你的名字"
git config user.email "你的邮箱"

git status
git add -A
git commit -m "用一句话说明这次为什么改"
git push
```

只提交部分文件时，把 `git add -A` 换成具体路径，例如：

```bash
git add robopianist-rl/train.py robopianist/robopianist/models/hands/daxian_hand.py
```

