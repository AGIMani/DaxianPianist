#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer this workspace's robopianist (supports --robot shadow|daxian|daxian_v2).
export PYTHONPATH="${SCRIPT_DIR}/../robopianist${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../docker/jax_cuda_libs.sh"

# V2 comparison vs run_daxian_v2.sh: same reduced space, but four-finger
# PIP/DIP are policy-controlled. pinky_rota is welded at 0 (not a joint).
# Writes to a separate folder so it does not overwrite eval_daxian_v2/.
# lookahead 20 = 1.0s buffer / 0.05s dt so the first note is visible at t=0.
EVAL_DIR="${SCRIPT_DIR}/../eval_daxian_v2_unlock"
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
    --unlock-four-finger-pip-dip \
    --forearm-dofs forearm_tx forearm_ty forearm_tz forearm_yaw \
    --control-timestep 0.05 \
    --n-steps-lookahead 20 \
    --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
    --action-reward-observation \
    --eval-interval 10000 \
    --video-interval 100000 \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    "$@"
