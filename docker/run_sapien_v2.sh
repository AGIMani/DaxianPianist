#!/usr/bin/env bash
# Run the 达显 V2 SAPIEN viewer with X11.
# Start this from the Linux host (or an SSH session), not from inside
# `docker compose exec dev` — that training container has no X11 socket.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f /.dockerenv ]]; then
  echo "Do not run this inside the training container (no X11, no docker)." >&2
  echo "Exit, then on the Linux host:" >&2
  echo "  bash docker/run_sapien_v2.sh" >&2
  exit 1
fi

DISPLAY_NUM="${SAPIEN_DISPLAY:-:10}"
XAUTH="${XAUTHORITY:-${HOME}/.Xauthority}"
NAME="${SAPIEN_CONTAINER:-daxianpianist-sapien}"

if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM#:}" ]]; then
  DISPLAY="${DISPLAY_NUM}" XAUTHORITY="${XAUTH}" xhost +SI:localuser:root >/dev/null 2>&1 || true
  DISPLAY="${DISPLAY_NUM}" XAUTHORITY="${XAUTH}" xhost +local:root >/dev/null 2>&1 || true
fi

IT=()
if [[ -t 0 && -t 1 ]]; then
  IT=(-it)
fi

run_preview() {
  docker exec "${IT[@]}" "${NAME}" bash -lc \
    'cd /workspace/robopianist && python3 examples/sapien_left_hand_v2_tune.py'
}

if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "reusing running ${NAME} (not recreating — that would kill processes inside)"
  run_preview
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker rm "${NAME}" >/dev/null
fi

docker run --rm "${IT[@]}" \
  --name "${NAME}" \
  --gpus all \
  --shm-size 8g \
  --security-opt seccomp=unconfined \
  -v "${ROOT}:/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${XAUTH}:/root/.Xauthority:ro" \
  -e DISPLAY="${DISPLAY_NUM}" \
  -e XAUTHORITY=/root/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -e PYTHONPATH=/workspace/.docker-python:/workspace/robopianist \
  -e VK_ICD_FILENAMES=/workspace/docker/nvidia_icd.json \
  -w /workspace/robopianist \
  --entrypoint bash \
  daxian-pianist:dev \
  -lc 'if ! python3 -c "import sapien" >/dev/null 2>&1; then
         python3 -m pip install -q sapien transforms3d \
           -i https://pypi.tuna.tsinghua.edu.cn/simple \
           --trusted-host pypi.tuna.tsinghua.edu.cn
       fi
       python3 examples/sapien_left_hand_v2_tune.py'
