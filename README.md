# strata-ampere

<p align="center">
  <b>High-throughput lossless TP=2 + EP=2 fork of <a href="https://github.com/Niko1221/Strata">Strata</a> tuned for Ampere (<code>sm_86</code>) dual-GPU MoE inference.</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.zh-CN.md">简体中文</a>
</p>

`strata-ampere` is a performance-focused fork of [Strata](https://github.com/Niko1221/Strata) (`0.1.60.1`) engineered to maximize **strictly lossless** greedy decode throughput on dual consumer Ampere GPUs (`sm_86`, e.g., 2× RTX 3080 20 GB over non-P2P PCIe Gen4 x16) paired with a multi-channel DDR4 host CPU expert pool.

On `Qwen3.8-Flash-Next-UD-Q4_K_XL` (48 layers, 512 routed experts/layer, 256K context, `int8` KV cache with 32K resident in VRAM), `strata-ampere` achieves **159.68 tok/s** 6-workload median average decode (**+74.7%** over the baseline `91.42 tok/s`, **+43.8%** over the earlier TP=2 Surgery 5 milestone `111.02 tok/s`) with **zero quantization degradation**, **zero dropped routed experts** (`STRATA_ROUTE_TAIL_SKIP=0`), and **BF16** TP2 AllReduce precision.

---

## Key Optimizations

1. **Dual-GPU Non-P2P `TP=2 + EP=2` Architecture**
   - Full-layer 50/50 Tensor Parallelism for attention/GDN/QSA/shared-MLP/LM-head plus non-overlapping 50/50 Expert Parallelism across both GPUs (`STRATA_TP2=1`) with TP=2 checkpoint/prefix-cache support (`--prompt-cache 6`).
   - Duplex Copy-Engine DMA BF16 `TpComm` for prefill and low-latency ring-buffer mapped-host BF16 AllReduce for decode, with polling `__nanosleep(40)` removed from `ld.acquire.sys` loops (`src/kernels/cuda/tp_comm.cu`, saving ~80–91 us per 96× AllReduce).
   - 50/50 even/odd pinned EP prefill split and balanced static expert sharding.

2. **Ampere (`sm_86`) Dense & Expert CUDA Microkernels**
   - **`Q8_0` / `Q4_0` `sm_86` kernels** (`src/kernels/cuda/q8_sm86_fast.cuh`, `src/kernels/cuda/q4_sm86.cuh`, `src/kernels/cuda/native_mmvq.cu`): tuned `RPB/TPR` tiles, `subwarp_kernel`, `row1_direct_kernel`, explicit `__launch_bounds__`, and `native_mmvq_pair` (`ncols >= 1`) support for `Q8_0`/`Q4_0`.
   - **RTX 3080 (`sm_86`, 68 SMs) `kIlRows_86` table** (`src/kernels/cuda/native_mmvq.cu`): re-measured via `mmvq_il_parity --bench --emit-table` for `Q4_K`, `Q6_K`, `IQ4_XS`, and `Q3_K` (replacing legacy RTX A4000 cells; e.g., `Q4_K 6144x2560 T=2` `24.8 -> 16.0 us`).
   - **36-layer GDN Split with Tree Support** (`src/kernels/cuda/verify_kernels.cu`): `gdn_step_split_kernel` extended with `bool TREE` (`tree_fork`/`tree_parent`) and enabled by default on `sm_86` (`STRATA_GDN_SPLIT=1`).
   - **`S27` Native Grouped Experts** (`src/kernels/cuda/iq_kernels.cu`): `__ldg` read-only cache loads in `S27<13>::load` and `sm_86` defaults `gy_gu=16`, `gy_dn=16` (`gth=128`).

3. **MTP TP=2 Speculative Drafting & Tree Verification**
   - `--mtp-q4 all` compact Q4_0 draft head reallocation freeing ~130 MiB on GPU0 for additional resident experts.
   - Vectorized `uint4` TP2 head publish/wait, fused `bump_round`, `(double)` warp-parallel `sumexp` reduction, fused `dvocab` mapping inside `mtp_tp2_merge_head`, and fused `k_proj + v_proj` via `native_mmvq_pair` (`src/core/mtp.cpp`, `src/kernels/cuda/verify_kernels.cu`).
   - Calibrated `STRATA_SPEC_FIXED_TABLE=10.5,12.2,14.0,16.1,20.2,23.6`, `STRATA_SPEC_R_OP=0.130`, `STRATA_SPEC_NO_HICONF=0`, and `--spec-min-p 0.22` with `STRATA_TREE_FORK=1`.

4. **CPU Expert Pool Alignment**
   - `--pool-workers 15` pinned across 8 CCXs (`15 workers + 1 host` = 16 active threads) saturates 4-channel DDR4-3200 at ~77.8 GB/s without cross-CCX contention.

---

## Verified Benchmark Results (`REPS=3`, Medians)

**Testbed**: 2× RTX 3080 20 GB (`sm_86`, 320 W, non-P2P PCIe Gen4 x16, CUDA 13.1), AMD Threadripper 3960X (24c/48t, `performance` governor), 128 GB 4-ch DDR4-3200.  
**Model**: `Qwen3.8-Flash-Next-UD-Q4_K_XL` (`temperature=0`, `presence_penalty=0`, `enable_thinking=False`, `--kv int8 --kv-resident 32768`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_TP_AR_MODE=bf16`).

| Workload | Baseline (11-20) | TP=2 Surgery 5 (11-21) | **`strata-ampere` Client (`srv_decode`)** | Peak (3-rep) | `e2e_out_tok_s` |
|---|---:|---:|---:|---:|---:|
| `ctx512` | 91.6 | 103.2 (`103.1`) | **155.2** (`154.9`) | **160.2** | 101.6 |
| `ctx2048` | 97.2 | 118.5 (`118.3`) | **168.4** (`168.0`) | **171.7** | 87.9 |
| `ctx8192` | 91.9 | 115.9 (`115.8`) | **169.1** (`168.7`) | **169.7** | 56.3 |
| `ctx32768` | 101.1 | 111.6 (`111.4`) | **171.9** (`171.3`) | **176.4** | 25.9 (`3,407.1` srv prefill) |
| `creative` | 77.4 | 93.1 (`93.0`) | **131.5** (`131.3`) | 135.3 | 114.0 |
| `code` | 89.3 | 123.8 (`123.6`) | **162.0** (`161.5`) | 165.0* | 136.4 |
| **6-Workload Mean** | **91.42** | **111.02** (`110.87`) | **`159.68 tok/s`** (`159.28`) | — | **`87.02 tok/s`** |

> \*Single-run `code` peak of `189.1 tok/s` was also observed on the same binary with `STRATA_SPEC_R_OP=0.120`. In `bench.py`, `creative` runs interleaved immediately after `code`, lowering transient expert cache hit rate from `84.1%` to `78.4–79.1%`.

### Accuracy & Parity Verification

- **CUDA/CPU Parity Suite**: 19/19 core parity and microbench targets pass with `0` failures (`verify_parity`, `native_multi_parity`, `gdn_parity` default/`SPLIT=0`/`SPLIT=1`, `router_top10_parity`, `mmvq_il_parity`, `sampler_parity`, `gr_parity` default & `Q8_PACKED+HC_Q8+QFUSE_HC`, `native_expert_parity` `q4_K/q5_1` & `q5_K/q8_0`, `q8_gemv_bench`, `router_chain_bench`, `mmvf_rows_parity`, `spec_proposal_parity`, `spec_verify_parity`, `qsa_parity`, `shared_expert_parity`). In `fused_gr_bench`, production `Q8` and `Q8-PK` paths align (`rel rms mixed 0.0012–0.0016`), while 12 pre-existing legacy BF16 `apply 1` cases remain (`apply 0` bitwise equal).
- **Long-Context Needle Test**: 3/3 exact hits at depth 0.5 (~150K chars / ~87K prompt tokens).
- **VRAM Stability**: `0` `#2065` / `min_free` warnings across startup, `REPS=3` benchmark, sanity, greedy, and needle runs (`GPU0` post-bench free `536 MiB >= 512 MiB`).

---

## Building (`sm_86`)

Requires CUDA Toolkit 13.1 (or 12.x+ compatible with `sm_86`), CMake, Ninja, and a C++20 compiler.

```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)
```

---

## Quick Start

Choose the drafter preset that best matches your workload (same `build86/strata` binary, zero recompilation):

### Choosing a Drafter: `MTP-6` vs `DFlash2 Q4_0`

Both presets run strictly lossless (`temperature=0`, `STRATA_ROUTE_TAIL_SKIP=0`) on `2× RTX 3080 20 GB` (`TP=2 + EP=2`, `Qwen3.8-Flash-Next-UD-Q4_K_XL`, `REPS=2` head-to-head 6-workload suite):

| Workload | `MTP-6` (Balanced) | `DFlash2 Q4_0` (`conf=0.35`, Code/Reasoning) | Delta |
|---|---:|---:|---:|
| `code` (Python LRU + tests) | 156.76 tok/s | **171.42 tok/s** | **+9.4%** |
| `math` (Proofs & recurrence) | 158.14 tok/s | **166.87 tok/s** | **+5.5%** |
| `reasoning` (Multi-step logic) | 183.94 tok/s | **197.95 tok/s** | **+7.6%** |
| `en_tech` (CUDA architecture) | 142.48 tok/s | **144.98 tok/s** | +1.8% |
| `zh_qa` (Systems Q&A) | **142.54 tok/s** | 140.87 tok/s | -1.2% |
| `creative` (Chinese sci-fi) | **135.11 tok/s** | 122.64 tok/s | -9.2% |
| **6-Workload Mean (decode / e2e)** | `153.16 / 137.48 tok/s` | **`157.45 / 141.01 tok/s`** | **+2.80%** |

- **Pick `DFlash2 Q4_0`** ([examples/strata-2x3080-20g-dflash2.example.json](examples/strata-2x3080-20g-dflash2.example.json)) if you mainly do **coding, math, or structured reasoning** (keep `--draft-conf 0.35` and `--vram-reserve-mib 1050`; lowering `--draft-conf` below `0.35` increases unique cold-expert DDR4 traffic on partial-VRAM setups and slows decode down).
- **Pick `MTP-6`** ([examples/strata-2x3080-20g.example.json](examples/strata-2x3080-20g.example.json)) if you want the most **balanced all-rounder** or do heavy **creative writing** (`+12.5 tok/s` faster on high-entropy creative text).

### Launch Steps

1. Copy your chosen template (`examples/strata-2x3080-20g-dflash2.example.json` or `examples/strata-2x3080-20g.example.json`) to your deployment directory and replace `/path/to/...` and `CHANGE_ME_API_KEY`.
2. Adjust `--pool-workers` and `--pool-cores` to match your CPU topology (the examples pin 15 workers across 8 CCXs on a 24-core Threadripper 3960X).
3. Launch the server using [examples/start-strata-ampere.sh](examples/start-strata-ampere.sh) or `serve/server.py`:

```bash
python3 serve/server.py --config /path/to/strata-2x3080-20g.json
```

> **Note for 20 GB RTX 3080 / Ampere cards**: Keep `--vram-reserve-mib 760` (MTP-6) or `1050` (DFlash2 `Q4_0`) with `--vram-reserve-later-mib 544` (or higher if driving a display) to maintain `>= 512 MiB` free VRAM headroom on GPU0.

---

## Latest Lossless Updates: 4,267 tok/s Prefill, DFlash2 BF16 TC & Headless VRAM OC

### 1. Strictly Lossless Prefill & DFlash2 BF16 Tensor Core Attention
- **Async Pre-Pinned CPU Share (`ad89a3b`)**: Dedicated `cpu_x_stream` D2H + `ExpertPool::run_range` parallel quantization + `STRATA_PREFILL_CPU_SHARE_NT=32, MAX=6144` (`+8.9%` on 4K–5K prefill).
- **14.3× Faster Parallel RAM-Backed PLE (`4fda8ca`)**: 8-thread `PleTable::gather_batch` skipping redundant `madvise()` syscalls when `--ple-io ram` is faulted/locked — slashing GPU PLE wait from **`945 ms -> 66 ms`**, lifting 32K single-chunk prefill from **`3,823 -> 4,267 tok/s` (`+11.6%`)**, 40K E2E from **`3,143 -> 3,468 tok/s` (`+10.3%`)**, and 10K E2E to **`2,843 tok/s` (`+13.7%`)**.
- **DFlash2 `sm_80+` BF16 Tensor Core Attention (`fe965b1`)**: PTX `mma.sync.aligned.m16n8k16` BF16 attention with exact FP32 online softmax — **`+4.25%` (`+6.3 tok/s`)** on multi-step reasoning decode and **`+4.06%`** on warm 16K-context decode with zero prefill regression.

### 2. Headless Linux NVML VRAM/Core OC & Power-Efficiency Pareto Sweep
NVIDIA's CUDA P2 state downclocks RTX 3080 GDDR6X from `9,501 MHz` to `9,251 MHz`. Using [scripts/ampere-oc-power.py](scripts/ampere-oc-power.py) (headless R550+ NVML VF offsets, no X11/Coolbits required), we swept GDDR6X clocks and power limits on `2× RTX 3080 20 GB` (`DFlash2 Q4_0`, `decode_code` 384 tok + 16K prefill):

| Profile (per GPU) | Real Mem / Core | `decode_code` | Decode Draw | **Perf / 100W** | `prefill_16k` | Prefill Draw | Peak Temp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock (`+0 / +0 / 320 W`) | `9,251 / 1,995 MHz` | `189.42 tok/s` | `389.2 W` | `48.66` | `3,399.4 tok/s` | `546.7 W` | `46 °C` |
| **Pareto Optimal (`+1800 / +120 / 240 W`)** | **`10,151 / 2,100 MHz`** | **`207.52 tok/s` (+9.6%)** | **`389.0 W`** | **`53.34` (+9.6% vs stock, +19.3% vs 320W OC)** | **`3,426.5 tok/s`** | **`453.5 W` (-93 W)** | **`41 °C`** |
| Peak Speed (`+1800 / +120 / 320 W`) | `10,151 / 2,100 MHz` | **`209.80 tok/s` (+10.8%)** | `469.3 W` | `44.71` | **`3,530.6 tok/s`** | `563.3 W` | `48 °C` |
| Over-Limit (`+2200 / +135 / 320 W`) | `10,351 / 2,115 MHz` | `194.22 tok/s` (ECC drop) | `468.8 W` | `41.43` | `3,545.1 tok/s` | `569.9 W` | `49 °C` |

```bash
# Apply the Pareto-optimal 240 W + GDDR6X 10,151 MHz profile (no engine restart needed):
sudo python3 scripts/ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240
```

---

## Upstream Attribution & License

`strata-ampere` is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) and is distributed under the original **MIT License** — see [LICENSE](LICENSE). All credit for the base Strata engine, pack format, KV streaming, and verification framework belongs to Niko1221 and the Strata contributors.
