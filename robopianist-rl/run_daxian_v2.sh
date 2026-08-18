#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer this workspace's robopianist (supports --robot shadow|daxian|daxian_v2).
export PYTHONPATH="${SCRIPT_DIR}/../robopianist${PYTHONPATH:+:$PYTHONPATH}"

# V2 only. Does not write eval_daxian/ (V3) or eval_daxian_unlock_pip_dip/.
EVAL_DIR="${SCRIPT_DIR}/../eval_daxian_v2"
mkdir -p "${EVAL_DIR}/videos" "${EVAL_DIR}/checkpoints" "${EVAL_DIR}/metrics"

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir "${EVAL_DIR}" \
    --artifact-dir "${EVAL_DIR}" \
    --robot daxian_v2 \
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
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    "$@"
