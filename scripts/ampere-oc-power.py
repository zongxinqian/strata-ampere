#!/usr/bin/env python3
"""
Headless NVML VRAM/Core Overclock & Power-Limit Tuner for Ampere GPUs (e.g. Dual RTX 3080 20GB).
Works on Linux without X11 or Coolbits via NVIDIA R550+ NVML VF offset APIs.

Usage:
  sudo python3 ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240
  sudo python3 ampere-oc-power.py --reset
  python3 ampere-oc-power.py --status
"""
import argparse
import ctypes
import sys


def load_nvml():
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    rc = nvml.nvmlInit_v2()
    if rc != 0:
        raise RuntimeError(f"nvmlInit_v2 failed with rc={rc}")
    return nvml


def gpu_indices(nvml, gpus_arg: str):
    cnt = ctypes.c_uint()
    assert nvml.nvmlDeviceGetCount_v2(ctypes.byref(cnt)) == 0
    if not gpus_arg:
        return list(range(cnt.value))
    return [int(x.strip()) for x in gpus_arg.split(",") if x.strip()]


def show_status(nvml, indices):
    for idx in indices:
        h = ctypes.c_void_p()
        assert nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) == 0
        mem_off, gpc_off = ctypes.c_int(), ctypes.c_int()
        pl_mw, pw_mw, temp = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        nvml.nvmlDeviceGetMemClkVfOffset(h, ctypes.byref(mem_off))
        nvml.nvmlDeviceGetGpcClkVfOffset(h, ctypes.byref(gpc_off))
        nvml.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(pl_mw))
        nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(pw_mw))
        nvml.nvmlDeviceGetTemperature(h, 0, ctypes.byref(temp))
        print(
            f"GPU {idx}: mem_offset={mem_off.value:+d} MHz (eff) | "
            f"gpc_offset={gpc_off.value:+d} MHz | "
            f"power_limit={pl_mw.value / 1000:.0f} W | "
            f"draw={pw_mw.value / 1000:.1f} W | temp={temp.value} C"
        )


def apply_settings(nvml, indices, mem_off, gpc_off, pl_w):
    for idx in indices:
        h = ctypes.c_void_p()
        assert nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) == 0
        if mem_off is not None:
            lo, hi = ctypes.c_int(), ctypes.c_int()
            if nvml.nvmlDeviceGetMemClkMinMaxVfOffset(h, ctypes.byref(lo), ctypes.byref(hi)) == 0:
                mem_off = max(lo.value, min(hi.value, mem_off))
            rc = nvml.nvmlDeviceSetMemClkVfOffset(h, ctypes.c_int(mem_off))
            if rc != 0:
                raise RuntimeError(f"GPU {idx}: nvmlDeviceSetMemClkVfOffset({mem_off}) failed rc={rc}")
        if gpc_off is not None:
            lo, hi = ctypes.c_int(), ctypes.c_int()
            if nvml.nvmlDeviceGetGpcClkMinMaxVfOffset(h, ctypes.byref(lo), ctypes.byref(hi)) == 0:
                gpc_off = max(lo.value, min(hi.value, gpc_off))
            rc = nvml.nvmlDeviceSetGpcClkVfOffset(h, ctypes.c_int(gpc_off))
            if rc != 0:
                raise RuntimeError(f"GPU {idx}: nvmlDeviceSetGpcClkVfOffset({gpc_off}) failed rc={rc}")
        if pl_w is not None:
            pmin, pmax = ctypes.c_uint(), ctypes.c_uint()
            if nvml.nvmlDeviceGetPowerManagementLimitConstraints(h, ctypes.byref(pmin), ctypes.byref(pmax)) == 0:
                want_mw = max(pmin.value, min(pmax.value, int(pl_w * 1000)))
            else:
                want_mw = int(pl_w * 1000)
            rc = nvml.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(want_mw))
            if rc != 0:
                raise RuntimeError(f"GPU {idx}: nvmlDeviceSetPowerManagementLimit({want_mw}) failed rc={rc}")


def main():
    ap = argparse.ArgumentParser(description="Headless NVML VRAM/Core OC & Power-Limit Tuner")
    ap.add_argument("--gpus", default="", help="Comma-separated GPU indices (default: all)")
    ap.add_argument("--mem-offset", type=int, default=None, help="Effective mem clock offset in MHz (e.g. 1800 -> +900 MHz real)")
    ap.add_argument("--gpc-offset", type=int, default=None, help="Core clock offset in MHz (e.g. 120)")
    ap.add_argument("--power-limit", type=int, default=None, help="Power limit in Watts per GPU (e.g. 240)")
    ap.add_argument("--reset", action="store_true", help="Reset offsets to 0 and power limit to card default")
    ap.add_argument("--status", action="store_true", help="Show current OC and power state")
    args = ap.parse_args()

    nvml = load_nvml()
    try:
        idxs = gpu_indices(nvml, args.gpus)
        if args.reset:
            for idx in idxs:
                h = ctypes.c_void_p()
                assert nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) == 0
                def_mw = ctypes.c_uint()
                nvml.nvmlDeviceGetPowerManagementDefaultLimit(h, ctypes.byref(def_mw))
                apply_settings(nvml, [idx], 0, 0, int(def_mw.value // 1000))
        elif args.mem_offset is not None or args.gpc_offset is not None or args.power_limit is not None:
            apply_settings(nvml, idxs, args.mem_offset, args.gpc_offset, args.power_limit)
        show_status(nvml, idxs)
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
