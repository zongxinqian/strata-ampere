#!/usr/bin/env python3
"""
Universal 1-Click Domain MoE Expert Heatmap Profiler for Strata / strata-ampere.

Runs a representative suite of domain prompts (coding / writing / research / balanced,
or your own --prompts-file) with STRATA_PROFILE_PURE_HEAT=1 and --expert-profile-save,
ranking all MoE (layer, expert) pairs strictly by real activation frequency and writing
a ready-to-use STRP v1 `expert-profile-<domain>.bin`.

Examples:
  # 1. Profile via an active systemd unit (recommended on production hosts; zero RAM conflict):
  sudo python3 scripts/profile-experts.py \
      --config /path/to/strata-config.json \
      --unit strata@iq3_s \
      --domain coding \
      --out /path/to/expert-profile-coding.bin

  # 2. Standalone mode (spawns a temporary server on --port 18090):
  python3 scripts/profile-experts.py \
      --config /path/to/strata-config.json \
      --domain research \
      --out /path/to/expert-profile-research.bin

  # 3. Inspect & compare two STRP v1 profiles (top-K VRAM residency overlap):
  python3 scripts/profile-experts.py --compare old.bin new.bin --top-k 10600
"""
import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

DOMAIN_SUITES = {
    "coding": [
        "Write a thread-safe lock-free ring buffer in C++20 using std::atomic with acquire/release semantics, plus a unit test.",
        "用 Python 实现一个支持异步超时重试、连接池限制和指数退避的 HTTP 客户端封装，并附上类型注解与 pytest 用例。",
        "Write a CUDA kernel for row-wise RMSNorm + residual add in FP32 with BF16 output using warp shuffle reductions, and explain memory coalescing.",
        "排查一段 Next.js + TypeScript 前端导航栏在移动端缩放后左上角 Logo 消失的 CSS Flex/Grid 布局问题，给出最小复现与修复补丁。",
        "Implement a clean SQL query planner optimization pass in Rust that pushes filter predicates below inner joins.",
        "编写一个 Linux Bash + Python 巡检脚本，采集双卡 GPU 功耗、显存温度、PCIe 链路状态并输出 JSON 告警报告。",
        "Design an LRU+LFU hybrid cache (ARC-style) in Go with O(1) operations and explain how ghost lists prevent scan pollution.",
        "重构一段包含回调地狱的 Node.js 文件批处理脚本，改为并发受控的 async/await worker pool 并处理异常回滚。",
    ],
    "writing": [
        "写一篇硬核又通俗易懂的小红书技术笔记，分享如何在家里的消费级双显卡主机上无损部署本地大模型。",
        "Write a vivid, atmospheric sci-fi short story set aboard an orbital solar-sail relay station when an anomalous signal arrives from Jupiter.",
        "以深沉克制的笔触写一段关于江南冬夜旧书店与归乡旅人的散文，注重感官细节与节奏留白。",
        "Draft a crisp, persuasive product launch keynote script introducing an open-source local AI workstation tool for indie developers.",
        "为一部关于深海科考队发现远古生物群落的纪录片撰写三分钟中英双语解说词，兼顾科学质感与悬念张力。",
        "Write a thoughtful essay comparing craftsmanship in mechanical watchmaking with low-level systems programming.",
    ],
    "research": [
        "Derive the online softmax recurrence used in FlashAttention and prove its numerical equivalence to two-pass softmax under exact arithmetic.",
        "详细推导混合专家模型（MoE）在半显存驻留架构下，草稿模型接受率、冷专家 PCIe 带宽与端到端 Decode 吞吐之间的定量关系。",
        "Prove by induction and generating functions the closed-form solution to T(n) = 3T(n-1) - 2T(n-2) + n, given T(0)=1, T(1)=2.",
        "Analyze the thermodynamic and queueing-theory trade-offs of dynamic power capping on memory-bandwidth-bound GPU workloads.",
        "从信息论与统计力学角度解释为什么高维稀疏激活网络在特定子领域语料上会呈现出强烈的幂律（Power-Law）专家路由分布。",
        "Formulate a mixed-integer linear program for static + adaptive expert placement across two asymmetric GPUs connected over non-P2P PCIe.",
    ],
}
DOMAIN_SUITES["balanced"] = (
    DOMAIN_SUITES["coding"][:3] + DOMAIN_SUITES["writing"][:2] + DOMAIN_SUITES["research"][:3]
)


def read_strp(path: str):
    with open(path, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            raise ValueError(f"{path}: file too short (<24 bytes)")
        if hdr[:4] != b"STRP":
            raise ValueError(f"{path}: invalid magic {hdr[:4]!r} (expected b'STRP')")
        ver, nl, ne, want, n_ranked = struct.unpack("<5I", hdr[4:24])
        raw = f.read(n_ranked * 4)
        if len(raw) < n_ranked * 4:
            raise ValueError(f"{path}: truncated body")
        pairs = list(struct.iter_unpack("<2H", raw[: n_ranked * 4]))
    return {"version": ver, "n_layers": nl, "n_expert": ne, "want": want, "n_ranked": n_ranked, "pairs": pairs}


def compare_profiles(p_a: str, p_b: str, top_k: int):
    a = read_strp(p_a)
    b = read_strp(p_b)
    k = min(top_k, len(a["pairs"]), len(b["pairs"]))
    set_a = set(a["pairs"][:k])
    set_b = set(b["pairs"][:k])
    ov = len(set_a & set_b)
    pct = 100.0 * ov / max(k, 1)
    swapped = k - ov
    print(f"Profile A: {p_a} ({a['n_layers']}x{a['n_expert']}, ranked={a['n_ranked']})")
    print(f"Profile B: {p_b} ({b['n_layers']}x{b['n_expert']}, ranked={b['n_ranked']})")
    print(f"Top-{k} VRAM tier overlap: {ov}/{k} ({pct:.1f}%) -> {swapped} domain-specialized experts promoted into VRAM!")


def load_custom_prompts(path: str):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("{"):
                obj = json.loads(ln)
                p = obj.get("prompt") or obj.get("content") or ""
                if p:
                    prompts.append(p)
            else:
                prompts.append(ln)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def wait_ready(url: str, api_key: str, timeout: int = 240):
    req = urllib.request.Request(f"{url}/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError(f"Server at {url} did not become ready within {timeout}s")


def stream_prompt(url: str, api_key: str, model: str, prompt: str, max_tokens: int):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    t0 = time.perf_counter()
    tf = tl = None
    ctoks = nev = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            ln = raw.decode("utf-8", errors="ignore").strip()
            if not ln.startswith("data:"):
                continue
            d = ln[5:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except Exception:
                continue
            u = obj.get("usage")
            if u:
                ctoks = u.get("completion_tokens", ctoks)
            ch = obj.get("choices") or []
            if ch:
                dl = ch[0].get("delta") or {}
                if (dl.get("content") or "") + (dl.get("reasoning_content") or ""):
                    now = time.perf_counter()
                    if tf is None:
                        tf = now
                    tl = now
                    nev += 1
    ctoks = ctoks if ctoks > 0 else nev
    dt = (tl - tf) if (tf and tl and tl > tf) else max(time.perf_counter() - t0, 1e-6)
    return ctoks, max(ctoks - 1, 1) / dt


def set_or_append_arg(args_list, flag, val):
    for i, a in enumerate(args_list):
        if a == flag and i + 1 < len(args_list):
            args_list[i + 1] = str(val)
            return
    args_list.extend([flag, str(val)])


def main():
    ap = argparse.ArgumentParser(description="Universal 1-Click Domain MoE Expert Heatmap Profiler")
    ap.add_argument("--config", help="Path to existing Strata server JSON config")
    ap.add_argument("--unit", default="", help="Systemd unit name (e.g. strata@iq3_s) to profile in-place with zero RAM conflict")
    ap.add_argument("--domain", choices=sorted(DOMAIN_SUITES.keys()), default="coding", help="Built-in domain prompt suite")
    ap.add_argument("--prompts-file", help="Optional custom prompts file (.txt or .jsonl)")
    ap.add_argument("--out", help="Output STRP v1 binary path (e.g. expert-profile-coding.bin)")
    ap.add_argument("--port", type=int, default=18090, help="Temporary HTTP port for standalone mode")
    ap.add_argument("--max-tokens", type=int, default=256, help="Tokens to generate per prompt (default: 256)")
    ap.add_argument("--prior-blend", type=float, default=0.45, help="RRF weight on base profile (0.0=raw heat, 0.45=default anti-overfit blend)")
    ap.add_argument("--activate", action="store_true", help="Automatically switch --expert-profile in --config to --out upon success")
    ap.add_argument("--top-k", type=int, default=10600, help="Top-K VRAM slots to compare against base profile")
    ap.add_argument("--compare", nargs=2, metavar=("OLD_BIN", "NEW_BIN"), help="Compare two STRP profile files and exit")
    args = ap.parse_args()

    if args.compare:
        compare_profiles(args.compare[0], args.compare[1], args.top_k)
        return

    if not args.config or not args.out:
        ap.error("--config and --out are required unless --compare is used")

    with open(args.config, "r", encoding="utf-8") as f:
        orig_text = f.read()
    cfg = json.loads(orig_text)

    prompts = load_custom_prompts(args.prompts_file) if args.prompts_file else DOMAIN_SUITES[args.domain]
    eng_args = list(cfg.get("args", []))
    base_prof = None
    for i, a in enumerate(eng_args):
        if a == "--expert-profile" and i + 1 < len(eng_args):
            base_prof = eng_args[i + 1]
            break

    out_abs = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    if os.path.exists(out_abs):
        os.remove(out_abs)

    api_key = cfg.get("api_key", "")
    model = cfg.get("model_name", "qwen3.8")

    if args.unit:
        # In-place systemd mode: avoids spawning a second 100GB+ engine or fighting watchdog timers
        prof_cfg = json.loads(orig_text)
        p_args = list(prof_cfg.get("args", []))
        set_or_append_arg(p_args, "--adapt-every", "1")
        set_or_append_arg(p_args, "--adapt-swaps", "4")
        set_or_append_arg(p_args, "--expert-profile-save", out_abs)
        set_or_append_arg(p_args, "--expert-profile-save-every", "0")
        prof_cfg["args"] = p_args
        prof_cfg.setdefault("env", {})["STRATA_PROFILE_PURE_HEAT"] = "1"
        live_port = prof_cfg.get("port", 8080)
        url = f"http://127.0.0.1:{live_port}/v1"
        try:
            print(f"[1/4] Restarting {args.unit} in pure-heat profiling mode (domain={args.domain}, prompts={len(prompts)})...", flush=True)
            with open(args.config, "w", encoding="utf-8") as f:
                json.dump(prof_cfg, f, indent=2)
            subprocess.run(["systemctl", "restart", args.unit], check=True)
            wait_ready(url, api_key, timeout=240)

            print("[2/4] Collecting real MoE routing heatmap across prompts...", flush=True)
            for idx, pr in enumerate(prompts, 1):
                ctoks, tps = stream_prompt(url, api_key, model, pr, args.max_tokens)
                print(f"  Prompt {idx:2d}/{len(prompts)}: {ctoks:4d} tokens @ {tps:6.1f} tok/s", flush=True)
        finally:
            print("[3/4] Restoring config & restarting service to flush STRP v1 profile...", flush=True)
            restore_cfg = json.loads(orig_text)
            if args.activate:
                r_args = list(restore_cfg.get("args", []))
                set_or_append_arg(r_args, "--expert-profile", out_abs)
                restore_cfg["args"] = r_args
                with open(args.config, "w", encoding="utf-8") as f:
                    json.dump(restore_cfg, f, indent=2)
            else:
                with open(args.config, "w", encoding="utf-8") as f:
                    f.write(orig_text)
            subprocess.run(["systemctl", "restart", args.unit], check=True)
    else:
        # Standalone mode
        cwd = cfg.get("cwd") or os.path.dirname(os.path.abspath(args.config))
        venv_py = os.path.join(cwd, ".venv", "bin", "python")
        py_exe = venv_py if os.path.exists(venv_py) else sys.executable
        server_py = os.path.join(cwd, "serve", "server.py")
        set_or_append_arg(eng_args, "--adapt-every", "1")
        set_or_append_arg(eng_args, "--adapt-swaps", "4")
        set_or_append_arg(eng_args, "--expert-profile-save", out_abs)
        set_or_append_arg(eng_args, "--expert-profile-save-every", "0")
        cfg["args"] = eng_args
        cfg["port"] = args.port
        fd_log, tmp_log = tempfile.mkstemp(prefix="strata_prof_", suffix=".log")
        os.close(fd_log)
        cfg["log"] = tmp_log
        cfg.setdefault("env", {})["STRATA_PROFILE_PURE_HEAT"] = "1"
        fd_cfg, tmp_cfg = tempfile.mkstemp(prefix="strata_prof_cfg_", suffix=".json")
        os.close(fd_cfg)
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        url = f"http://127.0.0.1:{args.port}/v1"
        srv_out = open(tmp_log + ".srv", "w")
        proc = subprocess.Popen(
            [py_exe, server_py, "--engine", "strata", "--config", tmp_cfg, "--port", str(args.port)],
            cwd=cwd,
            stdout=srv_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(url, api_key, timeout=240)
            for idx, pr in enumerate(prompts, 1):
                ctoks, tps = stream_prompt(url, api_key, model, pr, args.max_tokens)
                print(f"  Prompt {idx:2d}/{len(prompts)}: {ctoks:4d} tokens @ {tps:6.1f} tok/s", flush=True)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                pass
            srv_out.close()
            for p in (tmp_cfg, tmp_log, tmp_log + ".srv"):
                if os.path.exists(p):
                    os.remove(p)

    if not os.path.exists(out_abs) or os.path.getsize(out_abs) < 24:
        raise RuntimeError(f"Profile was not written to {out_abs}")

    if base_prof and os.path.exists(base_prof) and 0.0 < args.prior_blend < 1.0:
        b_info = read_strp(base_prof)
        d_info = read_strp(out_abs)
        if (b_info["n_layers"], b_info["n_expert"]) == (d_info["n_layers"], d_info["n_expert"]):
            scores = {}
            w = args.prior_blend
            for r, p in enumerate(b_info["pairs"]):
                scores[p] = scores.get(p, 0.0) + w / (1000.0 + r)
            for r, p in enumerate(d_info["pairs"]):
                scores[p] = scores.get(p, 0.0) + (1.0 - w) / (1000.0 + r)
            merged = sorted(scores.keys(), key=lambda p: -scores[p])
            with open(out_abs, "wb") as f:
                f.write(b"STRP" + struct.pack("<5I", d_info["version"], d_info["n_layers"], d_info["n_expert"], d_info["want"], len(merged)))
                for l, e in merged:
                    f.write(struct.pack("<2H", l, e))

    info = read_strp(out_abs)
    print(f"\n[4/4] SUCCESS: Saved {out_abs} ({info['n_layers']} layers x {info['n_expert']} experts, {info['n_ranked']} ranked pairs, prior_blend={args.prior_blend})")
    if base_prof and os.path.exists(base_prof):
        compare_profiles(base_prof, out_abs, args.top_k)


if __name__ == "__main__":
    main()
