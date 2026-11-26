"""Lightweight, read-only telemetry collector for the Strata-Ampere port 8080 dashboard.

Zero external dependencies (Python stdlib only), TTL-cached (1.5s GPU/log, 15s dmesg, 30s PCIe)
so multiple open browser tabs add negligible CPU load and zero GPU/inference overhead.
Never reads or exposes API keys, prompt text, or model output text.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}
_CACHE_TS: dict[str, float] = {}

_MRRS_REV = {0: 128, 1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096}

_RE_PROMPT = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)"
)
_RE_EXPERT = re.compile(
    r"strata serve: decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)"
    r"(?:; \d+ more read by the GPU over PCIe or from another GPU \(([\d.]+)%)?"
)
_RE_KV = re.compile(
    r"strata serve: KV streaming: ([\d.]+)% of \d+ block reads hit VRAM, ([\d.]+) MiB read from RAM"
)
_RE_SUFFIX = re.compile(
    r"strata serve: suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted"
)
_RE_CHAIN = re.compile(
    r"strata serve: lookup chain: (\d+) of (\d+) windows, (\d+) of (\d+) chained drafts accepted"
)
_RE_PARKED = re.compile(
    r"strata serve: conversation cache: parked (\d+) tokens.*? in ([\d.]+) ms; "
    r"parked=(\d+) bytes=(\d+) evictions=(\d+) snapshot_bytes=(\d+) reused_kv_bytes=(\d+)"
)
_RE_RESTORED = re.compile(
    r"strata serve: conversation cache: restored (\d+) tokens \(([^)]+)\) in ([\d.]+) ms; "
    r"parked=(\d+) bytes=(\d+)"
)
_RE_ROLLING = re.compile(
    r"strata serve: persistent rolling suffix cache ON \(capacity (\d+) tokens, min_match (\d+)\)"
)
_RE_LOOKUP_MAX = re.compile(
    r"strata serve: --lookup-max-t (\d+): MTP windows up to (\d+) tokens, window sizes ([0-9,]+)"
)


def _cached(key: str, ttl: float, fn):
    now = time.monotonic()
    with _LOCK:
        if key in _CACHE and (now - _CACHE_TS.get(key, 0.0)) < ttl:
            return _CACHE[key]
    val = fn()
    with _LOCK:
        _CACHE[key] = val
        _CACHE_TS[key] = time.monotonic()
    return val


def _read_pcie_mrrs() -> list[dict[str, Any]]:
    out_list: list[dict[str, Any]] = []
    try:
        lspci = subprocess.check_output(["lspci", "-D"], text=True, timeout=2.0)
    except Exception:
        return out_list
    for ln in lspci.splitlines():
        if "NVIDIA" in ln and ("VGA" in ln or "3D" in ln):
            bdf = ln.split()[0]
            mrrs = -1
            devctl = ""
            try:
                devctl = subprocess.check_output(
                    ["setpci", "-s", bdf, "CAP_EXP+0x08.w"], text=True, timeout=1.5
                ).strip()
                mrrs = _MRRS_REV.get((int(devctl, 16) >> 12) & 7, -1)
            except Exception:
                pass
            out_list.append({"bdf": bdf, "devctl": devctl, "mrrs_bytes": mrrs})
    return out_list


def _read_gpus() -> list[dict[str, Any]]:
    mrrs_info = _cached("pcie_mrrs", 30.0, _read_pcie_mrrs)
    fields = (
        "index,name,temperature.gpu,power.draw,power.limit,"
        "clocks.current.graphics,clocks.current.memory,utilization.gpu,"
        "memory.used,memory.free,memory.total,pcie.link.gen.current,pcie.link.width.current"
    )
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.5,
        )
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    for pos, ln in enumerate(raw.strip().splitlines()):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 13:
            continue
        def _flt(s, d=0.0):
            try:
                return float(s)
            except Exception:
                return d
        def _int(s, d=0):
            try:
                return int(float(s))
            except Exception:
                return d
        m = mrrs_info[pos] if pos < len(mrrs_info) else {}
        mem_used = _int(parts[8])
        mem_free = _int(parts[9])
        mem_tot = max(_int(parts[10]), 1)
        gpus.append({
            "index": _int(parts[0], pos),
            "name": parts[1],
            "temp_c": _int(parts[2]),
            "power_w": round(_flt(parts[3]), 1),
            "power_limit_w": round(_flt(parts[4]), 0),
            "clock_sm_mhz": _int(parts[5]),
            "clock_mem_mhz": _int(parts[6]),
            "util_pct": _int(parts[7]),
            "vram_used_mib": mem_used,
            "vram_free_mib": mem_free,
            "vram_total_mib": mem_tot,
            "vram_pct": round(100.0 * mem_used / mem_tot, 1),
            "pcie_gen": _int(parts[11]),
            "pcie_width": _int(parts[12]),
            "pcie_bdf": m.get("bdf", ""),
            "pcie_mrrs_bytes": m.get("mrrs_bytes", -1),
        })
    return gpus


def _read_dmesg_health() -> dict[str, Any]:
    xid = 0
    aer = 0
    oom = 0
    try:
        out = subprocess.check_output(["dmesg"], text=True, timeout=2.0, errors="ignore")
        for ln in out.splitlines():
            low = ln.lower()
            if "nvrm: xid" in low:
                xid += 1
            if "pcie bus error" in low or "aer: uncorrected" in low or "aer: corrected" in low:
                aer += 1
            if "out of memory: killed process" in low:
                oom += 1
    except Exception:
        return {"xid": -1, "aer": -1, "oom": -1, "healthy": True}
    return {"xid": xid, "aer": aer, "oom": oom, "healthy": (xid == 0 and aer == 0 and oom == 0)}


def _read_host() -> dict[str, Any]:
    mem_total_kb = 0
    mem_avail_kb = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    mem_total_kb = int(ln.split()[1])
                elif ln.startswith("MemAvailable:"):
                    mem_avail_kb = int(ln.split()[1])
    except Exception:
        pass
    mem_total_gib = round(mem_total_kb / (1024 * 1024), 1)
    mem_avail_gib = round(mem_avail_kb / (1024 * 1024), 1)
    mem_used_gib = round(max(mem_total_gib - mem_avail_gib, 0.0), 1)

    load1, load5, load15 = 0.0, 0.0, 0.0
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as f:
            ps = f.read().split()
            load1, load5, load15 = float(ps[0]), float(ps[1]), float(ps[2])
    except Exception:
        pass

    uptime_s = 0.0
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            uptime_s = float(f.read().split()[0])
    except Exception:
        pass

    thp = "unknown"
    try:
        raw = Path("/sys/kernel/mm/transparent_hugepage/defrag").read_text(encoding="utf-8").strip()
        m = re.search(r"\[([^\]]+)\]", raw)
        thp = m.group(1) if m else raw
    except Exception:
        pass

    return {
        "cpu_count": os.cpu_count() or 0,
        "load": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "ram_total_gib": mem_total_gib,
        "ram_used_gib": mem_used_gib,
        "ram_avail_gib": mem_avail_gib,
        "ram_pct": round(100.0 * mem_used_gib / mem_total_gib, 1) if mem_total_gib > 0 else 0.0,
        "uptime_s": int(uptime_s),
        "thp_defrag": thp,
    }


def _parse_engine_log(log_path: str | None) -> dict[str, Any]:
    res: dict[str, Any] = {
        "requests": [],
        "conv_cache": {
            "parked_count": 0,
            "parked_mib": 0.0,
            "last_park_ms": None,
            "last_restore_ms": None,
            "last_restore_tokens": 0,
            "last_restore_mode": "",
            "last_reused_kv_mib": 0.0,
            "evictions": 0,
        },
        "features": {
            "rolling_suffix_cap": 0,
            "lookup_max_t": 0,
            "window_sizes": "",
        },
        "summary": {
            "latest_decode_tps": 0.0,
            "avg_decode_tps": 0.0,
            "peak_decode_tps": 0.0,
            "latest_prefill_tps": 0.0,
            "avg_expert_hit_pct": 0.0,
            "avg_draft_accept_pct": 0.0,
            "avg_suffix_accept_pct": 0.0,
            "avg_kv_hit_pct": 0.0,
            "avg_reuse_pct": 0.0,
            "total_generated": 0,
            "request_count": 0,
        },
    }
    if not log_path or not os.path.isfile(log_path):
        return res

    try:
        sz = os.path.getsize(log_path)
        read_bytes = min(sz, 320 * 1024)
        with open(log_path, "rb") as f:
            if sz > read_bytes:
                f.seek(sz - read_bytes)
                f.readline()
            text = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return res

    reqs: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    conv = res["conv_cache"]
    feats = res["features"]

    for ln in text.splitlines():
        if "strata serve:" not in ln:
            continue
        m = _RE_ROLLING.search(ln)
        if m:
            feats["rolling_suffix_cap"] = int(m.group(1))
            continue
        m = _RE_LOOKUP_MAX.search(ln)
        if m:
            feats["lookup_max_t"] = int(m.group(1))
            feats["window_sizes"] = m.group(3)
            continue
        m = _RE_PARKED.search(ln)
        if m:
            conv["parked_count"] = int(m.group(3))
            conv["parked_mib"] = round(int(m.group(4)) / (1024 * 1024), 1)
            conv["last_park_ms"] = round(float(m.group(2)), 1)
            conv["evictions"] = int(m.group(5))
            conv["last_reused_kv_mib"] = round(int(m.group(7)) / (1024 * 1024), 1)
            continue
        m = _RE_RESTORED.search(ln)
        if m:
            conv["last_restore_tokens"] = int(m.group(1))
            conv["last_restore_mode"] = m.group(2)
            conv["last_restore_ms"] = round(float(m.group(3)), 1)
            conv["parked_count"] = int(m.group(4))
            conv["parked_mib"] = round(int(m.group(5)) / (1024 * 1024), 1)
            continue
        m = _RE_PROMPT.search(ln)
        if m:
            pt = int(m.group(1))
            reused = int(m.group(2))
            read_n = int(m.group(3))
            p_ms = float(m.group(4))
            p_tps = float(m.group(5))
            gen = int(m.group(6))
            d_ms = float(m.group(7))
            d_tps = float(m.group(8))
            d_acc = int(m.group(9))
            d_off = int(m.group(10))
            cur = {
                "prompt_tokens": pt,
                "reused_tokens": reused,
                "reuse_pct": round(100.0 * reused / pt, 1) if pt > 0 else 0.0,
                "read_tokens": read_n,
                "prompt_ms": round(p_ms, 1),
                "prefill_tps": round(p_tps, 1),
                "generated": gen,
                "decode_ms": round(d_ms, 1),
                "decode_tps": round(d_tps, 1),
                "draft_accept_pct": round(100.0 * d_acc / d_off, 1) if d_off > 0 else 0.0,
                "expert_hit_pct": None,
                "pcie_offload_pct": None,
                "kv_hit_pct": None,
                "suffix_windows": 0,
                "suffix_ok": 0,
                "suffix_drafts": 0,
                "chain_windows": 0,
                "chain_ok": 0,
                "chain_drafts": 0,
            }
            reqs.append(cur)
            continue
        if cur is None:
            continue
        m = _RE_EXPERT.search(ln)
        if m:
            cur["expert_hit_pct"] = round(float(m.group(1)), 1)
            cur["pcie_offload_pct"] = round(float(m.group(4)), 1) if m.group(4) else 0.0
            continue
        m = _RE_KV.search(ln)
        if m:
            cur["kv_hit_pct"] = round(float(m.group(1)), 2)
            continue
        m = _RE_SUFFIX.search(ln)
        if m:
            cur["suffix_windows"] = int(m.group(1))
            cur["suffix_ok"] = int(m.group(2))
            cur["suffix_drafts"] = int(m.group(3))
            continue
        m = _RE_CHAIN.search(ln)
        if m:
            cur["chain_windows"] = int(m.group(1))
            cur["chain_ok"] = int(m.group(3))
            cur["chain_drafts"] = int(m.group(4))
            continue

    recent = reqs[-30:]
    res["requests"] = list(reversed(recent[-16:]))
    res["chart"] = [
        {
            "idx": i + 1,
            "decode_tps": r["decode_tps"],
            "prefill_tps": r["prefill_tps"],
            "expert_hit_pct": r["expert_hit_pct"] or 0.0,
            "reuse_pct": r["reuse_pct"],
            "generated": r["generated"],
            "prompt_tokens": r["prompt_tokens"],
        }
        for i, r in enumerate(recent[-24:])
    ]

    if recent:
        dec_vals = [r["decode_tps"] for r in recent if r["generated"] >= 16]
        if not dec_vals:
            dec_vals = [r["decode_tps"] for r in recent]
        exp_vals = [r["expert_hit_pct"] for r in recent if r["expert_hit_pct"] is not None]
        kv_vals = [r["kv_hit_pct"] for r in recent if r["kv_hit_pct"] is not None]
        sfx_ok = sum(r["suffix_ok"] for r in recent)
        sfx_tot = sum(r["suffix_drafts"] for r in recent)
        reused_req = [r["reuse_pct"] for r in recent if r["reused_tokens"] > 0]
        pf_heavy = [r["prefill_tps"] for r in recent if r["read_tokens"] >= 512]
        res["summary"] = {
            "latest_decode_tps": recent[-1]["decode_tps"],
            "avg_decode_tps": round(sum(dec_vals) / len(dec_vals), 1) if dec_vals else 0.0,
            "peak_decode_tps": round(max(dec_vals), 1) if dec_vals else 0.0,
            "latest_prefill_tps": pf_heavy[-1] if pf_heavy else recent[-1]["prefill_tps"],
            "avg_expert_hit_pct": round(sum(exp_vals) / len(exp_vals), 1) if exp_vals else 0.0,
            "avg_draft_accept_pct": round(sum(r["draft_accept_pct"] for r in recent) / len(recent), 1),
            "avg_suffix_accept_pct": round(100.0 * sfx_ok / sfx_tot, 1) if sfx_tot > 0 else 0.0,
            "avg_kv_hit_pct": round(sum(kv_vals) / len(kv_vals), 2) if kv_vals else 100.0,
            "avg_reuse_pct": round(sum(reused_req) / len(reused_req), 1) if reused_req else 0.0,
            "total_generated": sum(r["generated"] for r in recent),
            "request_count": len(recent),
        }
    return res


def collect_metrics(svc: Any, log_path: str | None = None) -> dict[str, Any]:
    # Live engine status (busy/idle + current streaming tok/s) — never touches api_key or text
    live_status: dict[str, Any] = {"busy": False, "phase": "idle", "tokens": 0, "tokens_per_s": 0.0}
    try:
        with svc.status_lock:
            s = dict(svc.status)
        live_status["busy"] = bool(s.get("busy", False))
        live_status["phase"] = s.get("phase") or ("generating" if live_status["busy"] else "idle")
        live_status["tokens"] = int(s.get("tokens") or 0)
        if live_status["busy"] and s.get("first_token"):
            live_status["tokens_per_s"] = round(float(svc._tok_s()), 1)
            live_status["elapsed_s"] = round(time.time() - float(s.get("started", time.time())), 1)
    except Exception:
        pass

    # Avoid spawning nvidia-smi / dmesg during active generation (#905 kernel nv_lock PCIe BAR jitter)
    gpu_ttl = 20.0 if live_status["busy"] else 1.5
    dmesg_ttl = 60.0 if live_status["busy"] else 15.0
    gpus = _cached("gpus", gpu_ttl, _read_gpus)
    host = _cached("host", 2.0, _read_host)
    health = _cached("dmesg_health", dmesg_ttl, _read_dmesg_health)

    # Resolve log path from svc config if not explicitly passed
    if not log_path:
        cfg = getattr(svc, "config", None) or getattr(svc, "cfg", None)
        if isinstance(cfg, dict):
            log_path = cfg.get("log")
        if not log_path:
            eng = getattr(svc, "engine", None)
            lp = getattr(eng, "log_path", None) or getattr(eng, "log", None)
            if isinstance(lp, (str, Path)):
                log_path = str(lp)
    if not log_path:
        for cand in ("/root/Strata-0.1.34/strata-iq3_s.log", "strata-iq3_s.log"):
            if os.path.isfile(cand):
                log_path = cand
                break

    log_data = _cached(f"log:{log_path}", 1.5, lambda: _parse_engine_log(log_path))

    return {
        "ts": int(time.time()),
        "service": {
            "brand": "strata-ampere",
            "model": getattr(svc, "model", "unknown"),
            "max_context": svc.reported_ctx() if hasattr(svc, "reported_ctx") else 0,
            "loaded": bool(svc.loaded()) if hasattr(svc, "loaded") else True,
            "vision": getattr(svc, "vision", None) is not None,
            "live": live_status,
        },
        "gpus": gpus,
        "host": host,
        "health": health,
        "engine": log_data,
    }
