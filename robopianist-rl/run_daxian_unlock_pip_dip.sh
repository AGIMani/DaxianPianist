#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer this workspace's robopianist (supports --robot shadow|daxian).
export PYTHONPATH="${SCRIPT_DIR}/../robopianist${PYTHONPATH:+:$PYTHONPATH}"

# Comparison vs run_daxian.sh: same reduced space, but four-finger PIP/DIP are
# unlocked (policy-controlled). Thumb MCP/PIP stay pinned. Writes to a separate
# folder so it does not overwrite the locked-joint run in eval_daxian/.
EVAL_DIR="${SCRIPT_DIR}/../eval_daxian_unlock_pip_dip"
mkdir -p "${EVAL_DIR}/videos" "${EVAL_DIR}/checkpoints" "${EVAL_DIR}/metrics"

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir "${EVAL_DIR}" \
    --artifact-dir "${EVAL_DIR}" \
    --robot daxian \
    --mode online \
    --warmstart-steps 5000 \
    --max-steps 5000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --initial-buffer-time 1.0 \
    --gravity-compensation \
    --reduced-action-space \
    --unlock-four-finger-pip-dip \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    "$@"
