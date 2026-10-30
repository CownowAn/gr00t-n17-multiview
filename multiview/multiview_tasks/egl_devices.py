"""Map physical GPU indices (nvidia-smi order) to EGL device indices.

robosuite renders on the EGL device whose index equals CUDA_VISIBLE_DEVICES (or
MUJOCO_EGL_DEVICE_ID). EGL enumerates GPUs in its own order, which on multi-GPU
machines need not match nvidia-smi: on this host CUDA_VISIBLE_DEVICES=3 rendered on
GPU 1. Query each EGL device's CUDA ordinal instead (EGL_CUDA_DEVICE_NV), in a clean
subprocess with CUDA_DEVICE_ORDER=PCI_BUS_ID so that ordinals follow nvidia-smi.
"""

import json
import os
import subprocess
import sys

_QUERY = r"""
import ctypes, json, os
os.environ["MUJOCO_GL"] = "egl"
from mujoco.egl import egl_ext as EGL
lib = ctypes.CDLL("libEGL.so.1")
lib.eglGetProcAddress.restype = ctypes.c_void_p
lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long))(
    lib.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
)
EGL_CUDA_DEVICE_NV = 0x323A
mapping = {}
for egl_index, device in enumerate(EGL.eglQueryDevicesEXT()):
    handle = ctypes.cast(device, ctypes.c_void_p).value
    ordinal = ctypes.c_long(-1)
    if query(handle, EGL_CUDA_DEVICE_NV, ctypes.byref(ordinal)):
        mapping[ordinal.value] = egl_index
print(json.dumps(mapping))
"""


def gpu_to_egl_index() -> dict:
    """{physical GPU index (nvidia-smi order): EGL device index}."""
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")}
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    out = subprocess.run(
        [sys.executable, "-c", _QUERY], env=env, check=True, capture_output=True, text=True
    ).stdout
    return {int(k): v for k, v in json.loads(out.strip().splitlines()[-1]).items()}


def render_env_for_gpu(gpu: int, base_env=None) -> dict:
    """Environment variables that make robosuite render on physical GPU ``gpu``.

    CUDA_VISIBLE_DEVICES is set to the EGL index, because robosuite reads it as one;
    the simulation workers do not use CUDA otherwise.
    """
    env = dict(os.environ if base_env is None else base_env)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_to_egl_index()[gpu])
    return env
