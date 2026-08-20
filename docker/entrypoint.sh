#!/usr/bin/env bash
set -euo pipefail

# Bind-mount hides files baked into the image, so restore the default
# soundfont into the workspace copy when it is missing.
sf_dir="/workspace/robopianist/robopianist/soundfonts"
mkdir -p "${sf_dir}"
if [[ ! -f "${sf_dir}/TimGM6mb.sf2" && -f /opt/robopianist/TimGM6mb.sf2 ]]; then
  cp /opt/robopianist/TimGM6mb.sf2 "${sf_dir}/TimGM6mb.sf2"
fi

# Numeric uid without a passwd entry breaks Cursor (printenv / server install).
# Container still runs as root so the IDE can attach; this only creates the user.
host_uid="${HOST_UID:-0}"
host_gid="${HOST_GID:-0}"
if [[ "${host_uid}" != "0" ]] && ! getent passwd "${host_uid}" >/dev/null 2>&1; then
  groupadd -o -g "${host_gid}" pianist 2>/dev/null || true
  useradd -o -M -u "${host_uid}" -g "${host_gid}" -s /bin/bash -d /home/dev pianist 2>/dev/null || true
  chown "${host_uid}:${host_gid}" /home/dev || true
fi

export PYTHONPATH="/workspace/robopianist${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Bind-mounted; also sourced by run_daxian*.sh because `docker compose exec`
# does not re-run this entrypoint.
if [[ -f /workspace/docker/jax_cuda_libs.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/docker/jax_cuda_libs.sh
fi

exec "$@"
