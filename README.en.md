# strata-ampere

<p align="center">
  <b>Strictly lossless, high-throughput TP=2 + EP=2 fork of <a href="https://github.com/Niko1221/Strata">Strata</a> for dual modded Ampere (<code>sm_86</code> RTX 3080 20G · 50.86 GB/s Direct PCIe 4.0 P2P on native 256 MiB BAR1).</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/语言-简体中文-00b894.svg" alt="简体中文"></a>
  <img src="https://img.shields.io/badge/v1.4.0-236.1%20Warm%20%7C%20210.1%20All%20%7C%20178.3%20%40%20117K-00d4ff.svg" alt="v1.4.0 Decode">
  <img src="https://img.shields.io/badge/Peak-287.7%E2%80%93388.3%20tok%2Fs%20%40%20240W-7c4dff.svg" alt="Peak 287.7-388.3 tok/s">
  <img src="https://img.shields.io/badge/PCIe%204.0%20P2P-50.86%20GB%2Fs%20Full--Duplex-00e676.svg" alt="50.86 GB/s P2P">
  <img src="https://img.shields.io/badge/Precision-21%2F21%20Bit--Exact%20Lossless-ffbe28.svg" alt="21/21 Bit-Exact Lossless">
</p>

<p align="center">
  <img src="docs/assets/dashboard-8080-live.jpg" alt="strata-ampere Port 8080 Live Telemetry Dashboard" width="100%">
  <br>
  <sub><b>Built-in Port 8080 Live Telemetry Dashboard</b>: 2× Modded RTX 3080 20GB running <b>v1.4.0 Unlocked-P2P Deep Re-Optimization (Zero-Extra-BAR1 VMM Direct Push + In-Graph 512 Penalized Sampler + 122 MiB Verifier VRAM Trim + DFlash2 P2P Cache-Coherency Fix)</b> — hitting <code>210.14 tok/s</code> 3-round canonical all-mean (<code>236.10 tok/s</code> warm mean, <code>287.70 tok/s</code> task peak, <code>388.3 tok/s</code> burst peak), <b><code>3,842.3 tok/s</code> 117K cold prefill + <code>178.3 tok/s</code> 117K decode (<code>197.0 tok/s</code> @ 173.8K extreme stress)</b>, <b><code>9,982</code> resident experts (<code>4,873 + 5,109</code>)</b>, <code>199.5 tok/s</code> (<code>0.640 s</code>) 4.1K cross-GPU session restore, <code>0.00%</code> idle CPU, <code>51 / 256 MiB</code> BAR1 used, <code>0 Xid · 0 AER · 0 OOM</code></sub>
</p>

`strata-ampere` is a systems-engineered fork of [Strata](https://github.com/Niko1221/Strata) (`0.1.60.1`) built specifically for **2× Modded RTX 3080 20 GB (`sm_86`, 40 GB total VRAM)** to deliver **100% strictly lossless (17/17 DFlash2 + 4/4 MTP = 21/21 SHA256 bit-exact)** MoE **Decode & Prefill** performance:
- **v1.4.0 Unlocked-P2P Full-Stack Re-Optimization (`240W` cap, 21/21 Bit-Exact SHA256 verified)**: Deep second-pass optimization tailored to `50.86 GB/s` native `256 MiB` Dynamic BAR1 VMM P2P — **① DFlash2 TP=2 BAR1 P2P Memory-Ordering & Cache-Coherency Fix (`dflash2.cu` + dedicated `d_p2p_df2_buf_`)**: eliminates stale L1 `LDG.CONSTANT` prefetching and premature device-scope flag resets in `df2_p2p_wait_merge_top16_kernel` via `st.global.wt` + `st.release.sys.global.u32` + `ld.acquire.sys.global.u32` + `ld.global.cv` + post-read system-scope ack, achieving **17/17 deterministic parity even across `DF2_HEAD_TP2=0` vs `=1`** while lifting warm draft acceptance from `77.90%` to **`84.96%` (`+7.06%`)**; **② `-122.0 MiB` Dead `Verifier::staging_` Trim + P2P-Aware `49:51` Auto-Cache Sizing (`STRATA_TP2_VRAM_TRIM=1`)**: skips unused `61.0 MiB/GPU` `staging_` when `--pcie-frac 0` or `tp_rank=1` (`Verifier` device buffers drop from `103.3 → 42.3 MiB/GPU`) and reclaims double-counted GPU0 draft-head reserves, boosting resident hot experts from `9,870` to **`9,982` (`4,873 + 5,109`, `+112` experts, `83.8%` 30-req average hit rate)** while simultaneously *increasing* `CUDA1` free VRAM from `96 MiB` to **`124 MiB`** (`94 MiB` post-117K, `197.0 tok/s` @ 173.8K with `0` OOM); **③ Zero-Extra-BAR1 VMM P2P for MTP Split Draft Head (`STRATA_MTP_TP2_P2P=1`)**: carves `xq / hp / go / done / logits` inside `TpComm`'s existing `2 MiB` VMM tail (`0 MiB` extra BAR1, holding `51 / 256 MiB`), replacing 5 `cudaHostAllocMapped` host-bounce buffers (`4/4` SHA256 bit-exact); **④ `kTpHeadMaxHistLen=512` In-Graph Penalized Sampler + Unioned Shared Memory**: moves `penalty_last_n=256` (up to `512`) 100% inside the Verify CUDA Graph and cuts `tp2_head_part_kernel` shared memory from `~22.2 KiB` to `16.4 KiB` (`6 CTAs/SM` = `100%` `sm_86` occupancy); **⑤ Persistent Pinned `Prefill` `tp_worker` (`STRATA_PF_TP_WORKER=1`)**: replaces per-chunk `std::async` thread creation with a persistent `HandWorker` pinned to `aux_cpus`, sustaining **`3,793.4 / 3,842.3 tok/s`** at 32K/117K prefill!
- **v1.3.0 Full-Repository Post-P2P Architectural Overhaul + DFlash2 Speculative Tree Verification (`240W` cap, 17/17 Bit-Exact SHA256 verified)**: Eliminates pre-P2P legacy bottlenecks across the entire codebase — **① DFlash2 Frontier Speculative Tree Verification (`STRATA_DF2_TREE=1`) + `sm_86` Fast Q8/MMVQ/AVX2 Microkernels**; **② In-Graph Unified TP=2 BAR1 P2P Top-64 Shortlist Sampler (`STRATA_TP_HEAD_FAST=1`)**: moves penalized real-chat sampling (`presence_penalty=1.5`, `hist_len <= 128`) directly inside the Verify CUDA Graph via coherent mapped `TpHeadSampleCfg`, eliminating legacy-stream `cudaMemcpy` syncs and post-graph re-sampling; **③ Direct `Q4_0` TP=2 BAR1 P2P Draft Head + Native BF16 KV Ring (`STRATA_DF2_KV_BF16=1`)**: unlocks `STRATA_DF2_HEAD_TP2=1` under `STRATA_DF2_HEAD_Q4=1` (eliminating the `98.7 MiB` GPU0 startup spike and cutting DFlash2 draft latency from `1.32 → 0.87–1.17 ms/window`, **`-34%`**), while native BF16 KV rings + `22 MiB` VMM slabs add **`+40` resident experts → `9,870`**; **④ Fused BAR1 P2P `peer_bo_bf16` Prefill AllReduce (`STRATA_TP_FUSE_PF_AR=1`) + Compacted Async PLE Upload**: fuses peer BF16 AllReduce directly into `gr_write_norm_rs_kernel`, driving **117K (`116,800` tok) cold prefill to `3,830.9–3,842.6 tok/s` and 117K decode from `119.70 → 173.60 tok/s` (`+45.0%`)**; **⑤ Per-Device Async `ConversationCache` (`32.9 ms` live restore of 4.1K tokens) + Lock-Free `0.00%` Idle CPU `tp1` Worker + Telemetry `nv_lock` Debounce & SSE `TCP_NODELAY`**!
- **v1.2.0 P2P-Native Deep Architecture Campaign (`240W` cap, 4/4 Bit-Exact Greedy SHA256 verified)**: Exploits our 50.86 GB/s VMM Push-Write P2P link beyond MoE AllReduce — **① Direct P2P ArgHalf & Full-Logits Gather (`STRATA_TP_HEAD_P2P=1`)**: eliminates the `16 MiB` GPU1→Host→GPU0 PCIe bounce per verify step on `temp > 0`, boosting 4-task real-chat sampling (`temp=0.6`) from `164.57 → 185.50 tok/s` (**`+20.93 tok/s / +12.7%`**); **② DFlash2 106K CJK Vocab-Parallel P2P Draft Head (`STRATA_DF2_HEAD_TP2=1`)**: shards the `146.0 MiB` (`106,299`-row) Q4_0 draft head 50/50 across GPU0 & GPU1 (`53,152 + 53,147` rows, `73.0 MiB/GPU`), pushing `20 KiB` `xq_` over BAR1 to wake GPU1's captured graph and merging `896 B` peer Top-16 in a single 32-lane bitonic warp — **freeing `73.0 MiB` on GPU0** (`--vram-reserve-mib 608` adds **`+24` resident experts → `9,831`**); **③ Ampere `sm_86` `STRATA_DF2_GEMM4=26`** (auto-selected on `sm_86`), raising canonical 4-task warm mean to **`203.22–208.27 tok/s`** (`+8.4% ~ +11.1%`) with **4/4 100% bit-exact SHA256** across 2,560 tokens!
- **v1.1.0 6-Direction Deep Optimization Campaign**: Adds **GPU1 Sparse-Hit MoE Combine**, **Capacity-Weighted `48:52` Bresenham Expert Partitioning**, **Ampere `sm_86` Hardware `__nanosleep(64)` P2P Spin-Barrier + Warp-Rounded `K=1` Blocks**, **Zen 2 AVX2 Weight Prefetch + PCIe Root Port ACS Disable + `/dev/cpu_dma_latency=0`**, and **`STRATA_DF2_HEAD_Q4=1` + true `--spec 8 --draft-conf 0.50` (`user_args`) + recalibrated `draft-costs.json`**.
- **50.86 GB/s full-duplex card-to-card PCIe 4.0 x16 P2P on the card's native `256 MiB BAR1` (ZERO BAR1 resize / ZERO PCI `0x610` hacking!)** via our version-matched dynamic BAR1 open-gpu kernel patch ([patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)), CUDA Driver VMM selective peer mapping (`39 / 256 MiB` BAR1 used), 1-hop Push-Write + VRAM-Flag AllReduce, and `72 × 1 GiB` Static HugePages + `iommu=pt`.

> **Strict Losslessness Guarantee**: Every headline number below is a multi-run average or median under `temperature=0`, original target `UD-Q4_K_XL` quantization, `STRATA_ROUTE_TAIL_SKIP=0` (zero dropped experts), full `BF16` AllReduce, and bit-exact greedy SHA256 verification (`STRATA_DF2_HEAD_Q4=1` applies only to the DFlash2 drafter; target model verification remains 100% lossless).

---

## 1. Verified Benchmarks (Strictly Lossless, `Qwen3.8-Flash-Next-UD-Q4_K_XL`)

### Latest Milestone — v1.4.0 Unlocked-P2P Deep Re-Optimization (`240W` Pareto, 21/21 Bit-Exact)

Back-to-back calibrated production A/B on the same dual modded RTX 3080 20 GB host (`240W` cap, `10,151 MHz` GDDR6X, `50.86 GB/s` native `256 MiB` BAR1 P2P, `72 × 1 GiB` HugePages):

| Metric (`240W` Strictly Lossless `UD-Q4_K_XL`) | v1.3.0 Report | v1.3.0 Back-to-Back | **v1.4.0 P2P Champion** | Net Gain & Key Mechanism |
|---|---:|---:|---:|---|
| **3-Round Canonical `bench_6dir` All-Mean** | `204.67 tok/s` | `196.59 tok/s` | **`210.14 tok/s`** | **`+13.55 tok/s` (`+6.9%`)** back-to-back, **`+5.47`** vs v1.3.0 report |
| **3-Round Warm Mean (`r3`) · Single-Task Peak** | `214.00` · `276.60` | `209.10` · `281.70` | **`236.10` · `287.70 tok/s`** | **`+27.00 tok/s` (`+12.9%`)** warm gain, **`388.3 tok/s`** burst peak |
| **Warm Expert Hit Rate / Draft Acceptance** | `80.92%` / `70.1%` | `79.70%` / `77.90%` | **`81.55%` (`83.8%` 30-req) / `84.96%`** | **`+112` resident experts** + `ld.global.cv` DFlash2 P2P fix |
| **Real-Chat `temp=0.6` (`presence_penalty=1.5`)** | `179.10 tok/s` | `181.21` (`r2 180.25`) | **`181.01` (`r2 182.10`, `206.4` peak)** | Unioned smem (`16.4 KiB`, `6 CTAs/SM` `100%` occupancy) |
| **Long-Window `penalty_last_n=256` Sampled** | Post-graph P2P | `156.55–166.55` | **`161.40–166.15 tok/s` (`181.3` peak)** | `kTpHeadMaxHistLen=512` **100% inside Verify CUDA Graph** |
| **32K / 116.8K Cold Prefill** | `3,765.0` / `3,830.9` | `3,775.5` (32K) | **`3,793.4` (32K) / `3,842.3 tok/s` (117K)** | Persistent pinned `Prefill` `tp_worker` (`aux_cpus`) |
| **116.8K / 173.8K Long-Context Decode** | `173.60` / — | — | **`178.30` (117K) / `197.00 tok/s` (174K)** | **`+4.70 tok/s`** @ 117K; `197.0 tok/s` @ 173.8K (`0` OOM) |
| **Resident Hot Experts / `Verifier` Buffers** | `9,870` / `103.3M×2` | `9,870 (4766+5104)` | **`9,982 (4873+5109)` / `42.3 MiB×2`** | `-122.0 MiB` dead `staging_` + `49:51` split (`124 MiB` GPU1 free) |
| **4.1K Session Restore / Idle CPU / BAR1** | `0.673s` (`186.6`) / `0%` | `0.25%` / `51 MiB` | **`0.640s` (`199.5 tok/s`) / `0.000%` / `51 MiB`** | `0 MiB` extra BAR1 (`205 MiB` free), `0 Xid · 0 AER · 0 OOM` |

- **21/21 Strict Bit-Exact SHA256 Gate**:
  - **17/17 DFlash2 Canonical Suite** (`CUDA0 4500 + CUDA1 4500 = 9000` symmetric resident experts, `--adapt-every 999999`): `4/4` Pure Greedy, `4/4` Unpenalized `temp=0.6`, `4/4` Real-Chat `presence_penalty=1.5`, `2/2` Penalized Greedy, `2/2` `penalty_last_n=256` In-Graph 512 Sampling, and `1/1` `9.7K` QSA Sparse-Block Prefill+Decode — **17/17 IDENTICAL SHA256 even across `DF2_HEAD_TP2=0` vs `=1`**!
  - **4/4 MTP VMM BAR1 P2P Suite** (`--draft mtp --mtp-q4 all`, `STRATA_MTP_TP2_P2P=0` vs `=1`): **4/4 IDENTICAL SHA256**!
- **Honest v1.4.0 Ablation Disclosure**:
  - ✅ **Enabled ON by default**: DFlash2 BAR1 P2P strict ordering + isolated `d_p2p_df2_buf_`, `-122 MiB` `Verifier::staging_` trim, `STRATA_TP2_VRAM_TRIM=1` + `49:51` split (`9,982` experts), `kTpHeadMaxHistLen=512` + unioned shared memory, `STRATA_MTP_TP2_P2P=1`, and `STRATA_PF_TP_WORKER=1`.
  - ⚙️ **Opt-in (`=0` by default)**: `STRATA_DF2_TREE_PEN=0` (branch-exact `penalty_rows_tree` unit-tested `30/30` and speeds up CJK penalized greedy by `+9.8%`, gated off by default to preserve 17/17 fixed-batch-size bit-exactness); `STRATA_DF2_L45_OVERLAP=0` (isolated to low-priority `ctx_pre_cs_`, kept off due to L47 `final_norm` buffer reuse); `STRATA_TP_SHARE_INPUTS=0` (cross-root PCIe DMA reads of a single NUMA-0 `h_emb_raw_` buffer contend on host memory channels, `-6%` slower than per-rank host staging).

### v1.3.0 Milestone — Full-Repo Post-P2P Overhaul + Speculative Tree Suite (`240W` Pareto, 17/17 Bit-Exact)

| Metric (`240W` Strictly Lossless `UD-Q4_K_XL`) | Pre-P2P Baseline | Tree Baseline | **v1.3.0 Post-P2P Champion** | Net Gain & Key Mechanism |
|---|---:|---:|---:|---|
| **3-Round Canonical `bench_6dir` All-Mean** | `165.30 tok/s` | `200.03 tok/s` | **`204.67 tok/s`** | **`+23.8%`** vs pre-P2P (`80.92%` expert hit rate) |
| **Warm Mean (`r2 / r3`) · Single-Task Peak** | `176.2` / `~210` | `215.30` / `272.80` | **`217.00 / 214.00` · `276.60 tok/s`** | Live telemetry burst peak **`373.0 tok/s`** |
| **Real-Chat `temp=0.6` (`presence_penalty=1.5`)** | `~142 tok/s` | `172.15 / 172.75` | **`180.38 / 177.83 tok/s` (`201.2` peak)** | Penalized sampling **100% inside Verify CUDA Graph** (**`+3.9%`**) |
| **117K (`116,800` tok) Cold Prefill** | `~3,150 tok/s` | `3,798.8 tok/s` | **`3,830.9–3,842.6 tok/s`** | Fused BAR1 P2P `peer_bo_bf16` AllReduce + Compact Async PLE |
| **117K (`116,800` tok) Long-Context Decode** | `~92 tok/s` | `119.70 tok/s` | **`173.60 tok/s`** | **`+53.90 tok/s` (`+45.0%` surge!)** |
| **Resident Hot Experts / DFlash2 Draft Latency** | `9,656` / `1.74 ms` | `9,830` / `1.32 ms` | **`9,870 (4766+5104)` / `0.87–1.17 ms`** | **`+40` experts**, up to **`-34%`** draft latency |
| **4.1K Cross-GPU Session Restore / Idle CPU** | Blocking / High | `~115 ms` / Spin | **`32.9 ms` (+`17.6 ms` rebuild) / `0.00%`** | Per-device async `ParkAsync` + lock-free sleep (`0 Xid / 0 AER`) |

- **17/17 Strict Bit-Exact SHA256 Gate (`CUDA0 4500 + CUDA1 4500 = 9000` symmetric resident experts, `--adapt-every 999999`)**:
  - **Pure Greedy (`4/4 EXACT`)**, **Unpenalized `temp=0.6` (`4/4 EXACT`)**, **Real-Chat `presence_penalty=1.5` In-Graph Sampling (`4/4 EXACT`)**, **Penalized Greedy (`2/2 EXACT`)**, **`penalty_last_n=256` Post-Graph P2P Merge (`2/2 EXACT`)**, and **`9.7K` Sparse QSA Block-Selection Prefill+Decode (`1/1 EXACT`)** — **17/17 IDENTICAL SHA256** with across-the-board throughput gains!
- **Honest v1.3.0 Ablation Disclosure**:
  - ✅ **Enabled ON by default**: `STRATA_DF2_TREE=1`, `STRATA_TP_HEAD_FAST=1`, `STRATA_DF2_KV_BF16=1`, `STRATA_DF2_HEAD_TP2=1` + `STRATA_DF2_HEAD_Q4=1`, `STRATA_TP_FUSE_PF_AR=1`, `STRATA_PLE_COMPACT=1` + `STRATA_PLE_COPY_STREAM=1`, `STRATA_TP_VMM_PIPE_MB=22`.
  - ⚙️ **Opt-in (`=0` by default)**: `STRATA_TP_QSA_IDX_SPLIT=1` (6+6 cross-GPU BAR1 P2P QSA Indexer split pushes residency to `9,998` experts, but exhibits a 1-ulp BF16 half-order divergence on 1/17 long-prompt test and `-4.5%` prefill barrier wait); `STRATA_ADAPT_SMART=0` (unthrottled `29.6 swaps/win` live adaptation beats static-floor throttling by `+13.9%` hit rate).

### v1.2.0 Milestone — P2P-Native Deep Architecture Suite (`240W` Pareto, Strict Controlled A/B)

Same `build86_w16` (`STRATA_VERIFY_MAX_T=16`) binary, canonical 4-task suite (`moe_arch`, `code_cuda`, `agent_json`, `kv_tech`, 640 tok, `enable_thinking=False`), `240W` cap, `10,151 MHz` GDDR6X, `MRRS 4096B`, `50.86 GB/s` PCIe 4.0 P2P:

| Controlled A/B Arm (Same Binary, 240W) | Greedy r1 Mean | Greedy Warm Mean | **Real-Chat `temp=0.6` Mean** | Resident Experts / Free VRAM | 84K / 28K Cold Prefill | Parity |
|---|---:|---:|---:|---:|---:|:---:|
| **v1.1.0 Reference (all v1.2 flags OFF)** | `175.10 tok/s` | `187.45 tok/s` | **`164.57 tok/s`** | `9,807` (`548 / 230 MiB`) | `3,692.9 tok/s` (28K) | Baseline |
| **+ #1 `STRATA_TP_HEAD_P2P=1` (P2P Logits)** | `177.80 tok/s` | `189.45 tok/s` | **`180.78 tok/s` (`+9.9%`)** | `9,807` (`548 / 230 MiB`) | `3,699.5 tok/s` (28K) | ✅ Bit-Exact |
| **+ #1 + #2 `STRATA_DF2_HEAD_TP2=1` (TP=2 Draft)** | `178.68 tok/s` | **`203.03 tok/s` (`+8.3%`)** | **`179.43 tok/s` (`+9.0%`)** | `9,807` (**`620 / 152 MiB`**, -73M GPU0) | `3,705.9 tok/s` (28K) | ✅ **4/4 SHA256** |
| **v1.2.0 Champion (+`GEMM4=26` +`reserve=608`, Live)** | **`179.20–185.82`** | **`203.22–208.27` (`+8.4% ~ +11.1%`)** | **`185.50 tok/s` (`+12.7%`!!)** | **`9,831 (+24)`** (`530 / 152 MiB`) | **`3,836.2 tok/s` (84K, 0 OOM)** | ✅ **4/4 100% Exact** |

- **Strict 4/4 Bit-Exact Greedy SHA256 Proof (`--adapt-every 999999`, 2,560 tokens across all 4 tasks)**:
  - `moe_arch` (`+3.6%`), `code_cuda` (`+3.4%`), `agent_json` (`+2.5%`), `kv_tech` (`+4.1%`) — **4/4 IDENTICAL SHA256** against v1.1.0.
- **Honest 6-Direction Disclosure**:
  - ✅ **Shipped ON by default**: `#1 STRATA_TP_HEAD_P2P=1`, `#2 STRATA_DF2_HEAD_TP2=1`, and Ampere `sm_86` `STRATA_DF2_GEMM4=26`.
  - ❌ **Gated OFF by default (opt-in env vars)**: `#3 STRATA_TP_AR_GR_NORM=0` (`-17~30 tok/s` due to low CTA count `n_tok<=8` vs split `gr_norm`), `#4 STRATA_TP2_GPU1_CPU_MISS=0` / `HEAT_W0=0.96` (GPU1 CPU-miss sync regression + `0.82` bias regressed `temp=0.6`), `#5 STRATA_TP_P2P_PIPE=0` (4 MiB prefill micro-chunks `-165 ms` vs 16 MiB double-buffered P2P), `#6 STRATA_TP2_P2P_REBALANCE=0` (neutral/noisy in short runs).

### v1.1.0 Milestone — 6-Direction Deep Kernel & System Suite (`240W` Pareto, Strict A/B)

Canonical 4-task suite (`moe_arch`, `code_cuda`, `agent_json`, `kv_tech`), `temperature=0`, `240W` power cap, `10,151 MHz` GDDR6X, `MRRS 4096B`, `50.86 GB/s` PCIe 4.0 P2P:

| Strict A/B Arm (Same Hardware, 240W, Same 4-Task Suite) | 4-Task Warm Mean | `moe_arch` | `code_cuda` | `agent_json` | `kv_tech` | Resident Experts / GPU0 Free | Greedy SHA256 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **v1.0.0 Baseline (fresh-start r2 warm)** | `186.22 tok/s` | `204.2 tok/s` | `234.6 tok/s` | `177.0 tok/s` | `129.1 tok/s` | `9,764` / `508 MiB` | ✅ Bit-Exact |
| **v1.1.0 Champion (fresh-start r2 warm)** | **`202.50 tok/s` (`+8.7%`)** | `200.6 tok/s` | **`254.1 tok/s`** | **`222.6 tok/s` (`+25.8%`)** | `132.7 tok/s` | **`9,807 (+43)` / `528 MiB`** | ✅ Bit-Exact |
| **v1.0.0 Hot-Warm Baseline** | `199.68 tok/s` | — | — | — | — | `9,764` / `508 MiB` | ✅ Bit-Exact |
| **v1.1.0 Hot-Warm Champion (r3, Live)** | **`231.35 tok/s` (`+15.9%`)** | **`287.20 tok/s` (97.5% acc)** | **`256.70 tok/s` (92.1% acc)** | **`198.90 tok/s`** | **`182.60 tok/s`** | **`9,807 (+43)` / `528 MiB`** | ✅ **100% Bit-Exact** |

- **Key v1.1.0 Runtime & Hardware Telemetry**:
  - Recalibrated post-P2P `draft-costs.json`: `T=5` **`20.43 ms`**, `T=6` **`20.39 ms`** (warm `T=6` down to **`18.75 ms`**, fixing pre-P2P `42.19 ms` inversion).
  - DFlash2 drafter footprint: `862 MiB → 733 MiB` (`-129 MiB`), **`1.737 ms/round`** (head `0.374 ms`), GPU0 free VRAM **`528–532 MiB`** (above `512 MiB` warning threshold), GPU1 `228 MiB`.
  - Hardware P2P microbenchmarks (native `256 MiB BAR1`): `26.32 GB/s` uni / **`50.86 GB/s`** full-duplex / `50.75 GB/s` SM Push-Pack / **`39 / 256 MiB`** BAR1 used (**0 bytes** of weights/KV in BAR1).
  - Thermals & stability: **`43 °C / 38 °C`** under `238.6W / 238.3W` load, **`4.8 W`** idle (`210/405 MHz`, `23–28 °C`), **`0 Xid · 0 AER · 0 DMAR`**.

### v1.1.0 6-Direction Engineering Breakdown (Including Honest Negative Results)

All 6 directions were implemented, verified bit-exact, and A/B benchmarked. Winners are enabled by default; negative/neutral experiments remain cleanly gated OFF via env vars:

| Direction | Technique | Deployed Default | Verified Outcome |
|---|---|:---:|---|
| **Dir 1** | GPU1 Zero-Copy SM Read of 72×1G Portable HugePages (`STRATA_TP2_GPU1_MISS_256`) | **OFF (`=0`)** | Worked 100% bit-exact & 0 Xid/AER (hit rate `79%→89%`), but uncoalesced 144B/176B Q4_K/Q5_K PCIe reads achieved only `4–6 GB/s` and contended with CPU on 4-ch DDR4-3200 (`143.3 tok/s`) |
| **Dir 2** | **GPU1 Sparse-Hit MoE Combine (`STRATA_TP2_GPU1_SPARSE_COMBINE`)** + Fused Push (`STRATA_TP_FUSE_MOE_PUSH`) | **Sparse ON (`=1`)** · Fused Push OFF (`=0`) | Sparse Combine removes 48× `cudaMemsetAsync` + >50% zero-row reads/step; multi-block fused push lost `~0.4 ms` due to per-block `__threadfence_system()` + `local_buf` reloads |
| **Dir 3** | **Capacity-Weighted `48:52` Bresenham EP (`STRATA_TP2_EXPERT_SPLIT=48:52`)** + **Ampere `__nanosleep(64)` P2P Barrier & Warp-Rounded `K=1`** | **ON (Default)** | Matches actual `4,702 : 5,105` VRAM capacity (`+43` hot experts → `9,807`, zero mid-rank eviction gap); `__nanosleep(64)` trims wasted GPU1 spin power under the 240W cap |
| **Dir 4** | **`STRATA_DF2_HEAD_Q4=1` + `--lookup-max-t 12 --lookup-sizes 12` + `user_args --spec 8 --draft-conf 0.50` + Recalibrated `draft-costs.json`** | **ON (Config)** | `user_args` bypasses `drafting.py`'s automatic `--spec 6` override, unlocking all 7 native DFlash2 drafts while pruning low-confidence prose drafts (`0.50`) and freeing `129 MiB` VRAM |
| **Dir 5** | Ampere `sm_86` `STRATA_EXPDEC=1` | **Dropped** | Regressed `sm_86` by `~1.5 ms/window`; not recommended on Ampere |
| **Dir 6** | **Zen 2 AVX2 Weight Prefetch (`kq_avx2.cpp`) + PCIe Root Port ACS Disable + `/dev/cpu_dma_latency=0`** (vs 22 cores / `L3_PREWARM=2`) | **Prefetch + ACS + DMA Latency ON** · 15+1 Cores | 15+1 cores already saturate 4-ch DDR4-3200 (22 cores `<0.2 ms` neutral); AVX2 prefetch + ACS disable + `cpu_dma_latency=0` cut tail jitter |

### Full Evolution Summary (`91.4 tok/s` → `231.4 / 287.2 tok/s`)

| Stage | Before | After | Gain | Highlights |
|---|---:|---:|---:|---|
| Layer-Split → True `TP=2 + EP=2` (6-task median) | `91.42 tok/s` | **`159.68 tok/s`** | **`+74.7%`** | 50/50 full-layer TP + disjoint 24,576 EP + 68-SM `sm_86` kernels |
| `DFlash2 Q4_0` + `sm_80+` BF16 TC Attention | `148.30 tok/s` | **`154.61 tok/s`** | **`+4.25%`** | `mma.sync.m16n8k16` BF16 attention |
| Headless NVML OC + `240W` Pareto + `MRRS 4096B` | `189.42 tok/s` (320W) | **`207.52 tok/s`** (240W) | **`+9.6%` / `-93W`** | `10,151 MHz` GDDR6X, `41 °C`, `+19.3%` perf/W |
| Hybrid Drafting + 256K Rolling Suffix Tree | `185.46 tok/s` | **`229.98 tok/s`** | **`+24.0%`** | Cross-request suffix tree, bit-exact SHA |
| TP=2 Multi-Tab RAM Parking (`ConversationCache`) | `4,869.8 ms` (`0%`) | **`499.3 ms` (`99.7%`)** | **`9.75×` TTFT** | GPU 1 integrated into RAM parking (`55 ms` restore) |
| Deep Kernel Suite P0–P4 (`W16 G=1`, P1, P2, P4 INT8 TC) | `186.72 tok/s` | **`194.55 tok/s`** | **`+4.2%`** | Halved AR barriers, 84K cold hit `+9.2%`, `m16n8k32` INT8 TC |
| **v1.0.0: 1G HugePages + `iommu=pt` + 50.86 GB/s VMM Push-P2P** | `187.28` / `3,780` | **`201.40` warm / `3,870`** | **Breaks 200 tok/s** | 72×1 GiB dTLB fit + native 256 MiB BAR1 direct P2P |
| **v1.1.0: 6-Dir Deep Opt + 48:52 EP + `user_args --spec 8` + ACS/DMA** | `186.22` / `199.68` | **`202.50` fresh / `231.35` warm (`287.2` peak)** | **`+8.7% ~ +15.9%`** | Sparse Combine + 9,807 experts + `__nanosleep` + `DF2_HEAD_Q4` |
| **v1.2.0: P2P Logits + DFlash2 106K Head TP=2 + `sm_86` GEMM4=26** | `164.6` chat / `187.5` greedy | **`185.5` real chat / `203.2–208.3` warm (`273` peak)** | **`+12.7%` chat / `+8.4%~11.1%` greedy** | Zero 16M host bounce + 73M GPU0 saved (`9,831` experts) + 4/4 SHA256 exact |
| **v1.3.0: Full-Repo Post-P2P Overhaul + In-Graph Penalized Sampler + BF16 KV + Fused Prefill AR + Tree** | `200.0` all / `119.7` @117K | **`204.67` all (`217` warm / `276.6` peak) / `173.6` @117K** | **`+45.0%` @117K / `3,842.6` Prefill** | 17/17 SHA256 exact, `9,870` experts, `-34%` draft, `32.9 ms` restore, `0.00%` idle CPU |
| **v1.4.0: DFlash2 P2P Coherency Fix + `-122M` Verifier Trim + In-Graph 512 Sampler + MTP VMM P2P** | `196.6` all / `209.1` warm | **`210.14` all / `236.10` warm (`287.7–388.3` peak) / `178.3` @117K (`197.0` @174K)** | **Warm `+12.9%` / `+112` Experts** | 21/21 SHA256 exact, `9,982` experts, `42.3M/GPU` Verifier, `51M` BAR1 |

---

## 2. Step-by-Step Reproduction Guide (Zero Guesswork + 1-Command Rollback)

> **Why this works safely on native 256 MiB BAR1 modded RTX 3080 20GB (without risky BAR1 resize):**
> 1. **Kernel Driver (`patches/nvidia-open-595.99.02-dynbar1-p2p.patch`)**: Maps only peer-shared allocations into 2 MiB-aligned dynamic BAR1 windows (`align3` law v3), requiring **zero PCI `0x610` BAR1 resize**.
> 2. **Strata `TpComm` (`src/kernels/cuda/tp_comm.cu`)**: Uses CUDA Driver VMM (`cuMemCreate` / `cuMemMap` / `cuMemSetAccess`) to expose **only its 36 MiB communication slab** to the peer GPU — never calling process-wide `cudaDeviceEnablePeerAccess()` (so 18 GB of weights & KV use **0 bytes** of BAR1). All transfers use 1-hop `st.global.wt` push-writes + same-window VRAM-flag barriers with Ampere `__nanosleep(64)` backoff.

### Step 0: Prerequisites & Mandatory Backups
```bash
uname -r && ls -d /lib/modules/$(uname -r)/build
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # must match open-gpu tag (e.g. 595.99.02)
ls -lh /lib/firmware/nvidia/595.99.02/gsp_ga10x.bin
mokutil --sb-state                                             # must be disabled (or sign modules)
lsinitramfs /boot/initrd.img-$(uname -r) | grep -E 'nvidia\.ko|gpu_unlock'  # must NOT contain nvidia.ko

sudo mkdir -p /root/backup-nvidia-595
sudo cp -a $(modinfo -n nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem) /root/backup-nvidia-595/
sudo cp -a /etc/default/grub /root/backup-nvidia-595/grub.orig
```

### Step 1: Enable `72 × 1 GiB` Static HugePages + `iommu=pt`
```bash
sudo rm -f /etc/sysctl.d/99-strata-hugepages.conf
# Add "hugepagesz=1G hugepages=72 iommu=pt" to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub
# (keep existing flags like gpu_unlock.targets=0000:01:00.0,0000:48:00.0)
sudo update-grub && sudo reboot
# After reboot verify:
dmesg | grep -E "HugeTLB|iommu: Default|gpu_unlock: unlocked"
grep -E "Hugetlb|MemAvailable" /proc/meminfo
```

### Step 2: Build Version-Matched `595.99.02` Dynamic BAR1 P2P Open Kernel Modules
```bash
git clone https://github.com/zongxinqian/strata-ampere.git
STRATA_DIR=$(pwd)/strata-ampere

sudo mkdir -p /root/nv-open-p2p && cd /root/nv-open-p2p
sudo git clone --depth 1 --branch 595.99.02 \
  https://github.com/NVIDIA/open-gpu-kernel-modules.git open-595.99.02
cd open-595.99.02
sudo git apply --verbose "$STRATA_DIR/patches/nvidia-open-595.99.02-dynbar1-p2p.patch"
sudo make modules -j$(nproc) SYSSRC=/lib/modules/$(uname -r)/build
```

### Step 3: Safe Live Trial (`insmod`, Stock Driver Untouched on Disk)
```bash
sudo systemctl stop strata@iq3_s gpu-arbiter-watchdog gpu-exporter nvidia-persistenced 2>/dev/null || true
sudo nvidia-smi -pm 0 2>/dev/null || true; sleep 2
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia 2>/dev/null || true
sudo modprobe ecc ecdh_generic drm drm_kms_helper video

KO=/root/nv-open-p2p/open-595.99.02/kernel-open
sudo insmod $KO/nvidia.ko NVreg_EnableResizableBar=0
sleep 1
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm 2>/dev/null || true
sudo insmod $KO/nvidia-uvm.ko uvm_disable_hmm=1
sudo insmod $KO/nvidia-modeset.ko
sudo insmod $KO/nvidia-drm.ko

nvidia-smi topo -p2p r && nvidia-smi topo -p2p w   # Expect OK both ways!
```

### Step 4: Persistent Install + 1-Command Rollback Script
```bash
KVER=$(uname -r)
KO=/root/nv-open-p2p/open-595.99.02/kernel-open
sudo mkdir -p /lib/modules/$KVER/updates/p2p
sudo install -m 0644 $KO/nvidia*.ko /lib/modules/$KVER/updates/p2p/

sudo tee /etc/depmod.d/p2p.conf >/dev/null <<EOF
override nvidia $KVER updates/p2p
override nvidia-uvm $KVER updates/p2p
override nvidia-modeset $KVER updates/p2p
override nvidia-drm $KVER updates/p2p
override nvidia-peermem $KVER updates/p2p
EOF

sudo tee /etc/modprobe.d/zz-nvidia-p2p.conf >/dev/null <<'EOF'
options nvidia NVreg_EnableResizableBar=0
options nvidia-uvm uvm_disable_hmm=1
EOF

sudo tee /usr/local/sbin/nvidia-p2p-rollback.sh >/dev/null <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -f /etc/depmod.d/p2p.conf /etc/modprobe.d/zz-nvidia-p2p.conf
depmod -a $KVER
echo "Rolled back to stock proprietary nvidia-595.99.02. Run 'sudo reboot' to activate."
EOF
sudo chmod +x /usr/local/sbin/nvidia-p2p-rollback.sh

sudo depmod -a $KVER
modinfo -n nvidia nvidia-uvm nvidia-modeset nvidia-drm
sudo reboot
```
> **1-Command Rollback Anytime**: `sudo /usr/local/sbin/nvidia-p2p-rollback.sh && sudo reboot`

### Step 5: Build & Run `strata-ampere` v1.3.0 (Including ACS Disable & `/dev/cpu_dma_latency=0`)
```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_WINDOW_MAX_T=16 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)

# Apply 240W Pareto OC + MRRS 4096B
sudo python3 scripts/ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240 --pcie-mrrs 4096 --thp-defrag defer+madvise

# Disable PCIe Root Port ACS on both GPU root ports (check your bus IDs with `lspci -tv`)
sudo setpci -s 00:01.1 2a6.w=0000
sudo setpci -s 40:03.1 2a6.w=0000

# Hold /dev/cpu_dma_latency=0 via a tiny companion systemd unit
sudo tee /etc/systemd/system/strata-dma-latency.service >/dev/null <<'EOF'
[Unit]
Description=Hold /dev/cpu_dma_latency=0 for low-jitter MoE CPU expert dispatch
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -c "import os,struct,time; fd=os.open('/dev/cpu_dma_latency',os.O_WRONLY); os.write(fd,struct.pack('i',0)); time.sleep(10**9)"
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now strata-dma-latency.service

# Reset stale pre-P2P draft-costs.json if upgrading
rm -f /path/to/draft-costs.json

# Launch (example config includes "user_args": ["--spec", "8", "--draft-conf", "0.50"] + STRATA_DF2_HEAD_Q4=1)
python3 serve/server.py --config examples/strata-2x3080-20g-dflash2.example.json
```

Verify success in logs:
- `strata generate: expert arena: cudaHostRegister PORTABLE ok; hugetlb 1 GB pages (72 x 1 GiB)`
- `strata tp_comm: NVLink/P2P enabled (vmm-push, decode=p2p, CUDA0 <-> CUDA1)`
- `strata verify: window up to 12 tokens, 42.3 MiB of device buffers [TP2 Vocab-Parallel lm_head]`
- `strata dflash2: P2P TP=2 draft head ON (GPU0 53152 rows + GPU1 53147 rows, 73.0 MiB/GPU)`
- `strata serve: STRATA_TP2=1 ready: CUDA0 (4873 experts) + CUDA1 (5109 experts) = 9982 total resident experts`
- Post-graph free VRAM **`~160–172 MiB`** on GPU0 and **`~124 MiB`** on GPU1 (verified `0` OOM through `116.8K` & `173.8K` context stress), `BAR1 Memory Usage: Used 51 MiB / Total 256 MiB` (`205 MiB` free) on both GPUs, `0 Xid / 0 AER`.
- Open `http://<server-ip>:8080/` for the built-in bilingual live telemetry dashboard (with all `/v1/*` OpenAI endpoints intact).

---

## 3. Attribution & License

- Engine fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) under the **MIT License** ([LICENSE](LICENSE)).
- Dynamic BAR1 P2P kernel patch (Dual MIT/GPLv2) builds on [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules), [tinygrad](https://github.com/tinygrad/open-gpu-kernel-modules), [aikitoria](https://github.com/aikitoria/open-gpu-kernel-modules), [Duanyll](https://github.com/Duanyll/open-gpu-kernel-modules) (Method 3), [dingiv](https://github.com/dingiv/open-gpu-kernel-modules) (`align3`), and [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p).
