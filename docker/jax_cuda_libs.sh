# JAX 0.6 pip CUDA wheels look up cuSPARSE/cuBLAS in nvidia-*-cu12, not only
# /usr/local/cuda/lib64 from the NGC image. If the image path wins, plugin
# init fails and training silently falls back to CPU.
_jax_nv="/usr/local/lib/python3.10/dist-packages/nvidia"
if [[ -d "${_jax_nv}" ]]; then
  _jax_libs=""
  for _d in "${_jax_nv}"/*/lib; do
    [[ -d "${_d}" ]] && _jax_libs="${_jax_libs}${_d}:"
  done
  export LD_LIBRARY_PATH="${_jax_libs}${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
  unset _d _jax_libs
fi
unset _jax_nv
