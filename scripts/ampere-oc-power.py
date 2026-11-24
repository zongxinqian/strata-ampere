#!/usr/bin/env python3
"""
Headless NVML VRAM/Core Overclock, Power-Limit, PCIe MaxReadReq & THP Tuner for Linux Ampere GPUs.
Works without X11 or Coolbits via NVIDIA R550+ NVML VF offset APIs + setpci + sysfs.

Usage:
  sudo python3 ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240 --pcie-mrrs 4096 --thp-defrag defer+madvise
  sudo python3 ampere-oc-power.py --reset
  python3 ampere-oc-power.py --status
"""
import argparse
import ctypes
import os
import subprocess
import sys

MRRS_MAP = {128: 0, 256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5}
MRRS_REV = {v: k for k, v in MRRS_MAP.items()}


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


def nvidia_bdfs():
    try:
        out = subprocess.check_output(["lspci", "-D"], text=True)
    except Exception:
        return []
    bdfs = []
    for ln in out.splitlines():
        if "NVIDIA" in ln and ("VGA" in ln or "3D" in ln):
            bdfs.append(ln.split()[0])
    return bdfs


def get_mrrs_bytes(bdf: str):
    try:
        val = int(subprocess.check_output(["setpci", "-s", bdf, "CAP_EXP+0x08.w"], text=True).strip(), 16)
        return MRRS_REV.get((val >> 12) & 7, -1)
    except Exception:
        return -1


def set_mrrs_bytes(bdf: str, mrrs_bytes: int):
    if mrrs_bytes not in MRRS_MAP:
        raise ValueError(f"Unsupported --pcie-mrrs {mrrs_bytes}; valid: {sorted(MRRS_MAP)}")
    cur = int(subprocess.check_output(["setpci", "-s", bdf, "CAP_EXP+0x08.w"], text=True).strip(), 16)
    nxt = (cur & ~0x7000) | (MRRS_MAP[mrrs_bytes] << 12)
    subprocess.run(["setpci", "-s", bdf, f"CAP_EXP+0x08.w={nxt:04x}"], check=True)
    after = get_mrrs_bytes(bdf)
    if after != mrrs_bytes:
        raise RuntimeError(f"{bdf}: MRRS readback mismatch (wanted {mrrs_bytes}B, got {after}B)")


def set_thp_defrag(mode: str):
    p = "/sys/kernel/mm/transparent_hugepage/defrag"
    if not os.path.exists(p):
        return
    with open(p, "w", encoding="utf-8") as f:
        f.write(mode + "\n")


def show_status(nvml, indices):
    bdfs = nvidia_bdfs()
    for pos, idx in enumerate(indices):
        h = ctypes.c_void_p()
        assert nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) == 0
        mem_off, gpc_off = ctypes.c_int(), ctypes.c_int()
        pl_mw, pw_mw, temp = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        nvml.nvmlDeviceGetMemClkVfOffset(h, ctypes.byref(mem_off))
        nvml.nvmlDeviceGetGpcClkVfOffset(h, ctypes.byref(gpc_off))
        nvml.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(pl_mw))
        nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(pw_mw))
        nvml.nvmlDeviceGetTemperature(h, 0, ctypes.byref(temp))
        mrrs = get_mrrs_bytes(bdfs[pos]) if pos < len(bdfs) else -1
        mrrs_s = f"{mrrs}B" if mrrs > 0 else "n/a"
        print(
            f"GPU {idx}: mem_offset={mem_off.value:+d} MHz (eff) | "
            f"gpc_offset={gpc_off.value:+d} MHz | "
            f"power_limit={pl_mw.value / 1000:.0f} W | "
            f"MaxReadReq={mrrs_s} | "
            f"draw={pw_mw.value / 1000:.1f} W | temp={temp.value} C"
        )
    p = "/sys/kernel/mm/transparent_hugepage/defrag"
    if os.path.exists(p):
        print(f"THP defrag: {open(p).read().strip()}")


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
    ap = argparse.ArgumentParser(description="Headless NVML VRAM/Core OC, Power-Limit, PCIe MRRS & THP Tuner")
    ap.add_argument("--gpus", default="", help="Comma-separated GPU indices (default: all)")
    ap.add_argument("--mem-offset", type=int, default=None, help="Effective mem clock offset in MHz (e.g. 1800 -> +900 MHz real)")
    ap.add_argument("--gpc-offset", type=int, default=None, help="Core clock offset in MHz (e.g. 120)")
    ap.add_argument("--power-limit", type=int, default=None, help="Power limit in Watts per GPU (e.g. 240)")
    ap.add_argument("--pcie-mrrs", type=int, default=None, help="PCIe MaxReadRequestSize in bytes (e.g. 4096)")
    ap.add_argument("--thp-defrag", default=None, help="THP defrag policy (e.g. defer+madvise)")
    ap.add_argument("--reset", action="store_true", help="Reset offsets to 0 and power limit to card default")
    ap.add_argument("--status", action="store_true", help="Show current OC, PCIe MRRS, THP, and power state")
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
        else:
            if args.mem_offset is not None or args.gpc_offset is not None or args.power_limit is not None:
                apply_settings(nvml, idxs, args.mem_offset, args.gpc_offset, args.power_limit)
            if args.pcie_mrrs is not None:
                for bdf in nvidia_bdfs():
                    set_mrrs_bytes(bdf, args.pcie_mrrs)
            if args.thp_defrag is not None:
                set_thp_defrag(args.thp_defrag)
        show_status(nvml, idxs)
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
