# strata-ampere

<p align="center">
  <b>Strictly lossless, high-throughput TP=2 + EP=2 fork of <a href="https://github.com/Niko1221/Strata">Strata</a> engineered for dual consumer Ampere (<code>sm_86</code> RTX 3080 20G) & Turing (<code>sm_75</code> RTX 2080 Ti 22G + NVLink) MoE inference.</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/语言-简体中文-00b894.svg" alt="简体中文"></a>
</p>

<p align="center">
  <img src="docs/assets/dashboard-8080-live.jpg" alt="strata-ampere Port 8080 Live Telemetry Dashboard" width="100%">
  <br>
  <sub><b>Built-in Port 8080 Live Telemetry Dashboard</b>: 2× RTX 3080 20GB running 12H+ uptime, 180K context with 100% cache reuse, <code>193.5 tok/s</code> peak, <code>39–45 °C</code> load, <code>0 Xid · 0 AER · 0 OOM</code></sub>
</p>

`strata-ampere` is a systems-engineered fork of [Strata](https://github.com/Niko1221/Strata) (`0.1.60.1`) built to maximize **100% strictly lossless** MoE **Decode & Prefill** throughput on two popular high-value modded dual-GPU platforms:
- **2× Modded RTX 3080 20 GB** (Ampere `sm_86`, 40 GB total VRAM, non-NVLink / non-P2P)
- **2× Modded RTX 2080 Ti 22 GB** (Turing `sm_75`, 44 GB total VRAM, native NVLink P2P + hardware FP16 + `sm_75` INT8 Tensor Core)

> **Benchmark Honesty**: Every headline metric below is a **multi-run median or average** (never cherry-picked single-run peaks), verified under `temperature=0`, zero quantization downgrade, `STRATA_ROUTE_TAIL_SKIP=0` (zero dropped cold experts), and full `BF16` cross-GPU AllReduce.

---

## 1. Verified Decode & Prefill Benchmarks (Medians / Averages)

### Decode Speed (6-Workload Strictly Lossless Medians, `REPS=3`, Stock 320 W)

Tested on `Qwen3.8-Flash-Next-UD-Q4_K_XL` (`MTP-6` drafter, 3-run medians):

| Workload | Baseline (Layer-Split) | Early TP=2 Milestone | **`strata-ampere` Median (`srv_decode`)** | Speedup vs Baseline |
|---|---:|---:|---:|---:|
| `ctx512` | `91.6 tok/s` | `103.2 tok/s` | **`155.2 tok/s`** (`154.9`) | `+69.4%` |
| `ctx2048` | `97.2 tok/s` | `118.5 tok/s` | **`168.4 tok/s`** (`168.0`) | `+73.3%` |
| `ctx8192` | `91.9 tok/s` | `115.9 tok/s` | **`169.1 tok/s`** (`168.7`) | `+84.0%` |
| `ctx32768` | `101.1 tok/s` | `111.6 tok/s` | **`171.9 tok/s`** (`171.3`) | `+70.0%` |
| `creative` | `77.4 tok/s` | `93.1 tok/s` | **`131.5 tok/s`** (`131.3`) | `+69.9%` |
| `code` | `89.3 tok/s` | `123.8 tok/s` | **`162.0 tok/s`** (`161.5`) | `+81.4%` |
| **6-Workload Median Mean** | **`91.42 tok/s`** | **`111.02 tok/s`** | **`159.68 tok/s`** (`159.28`) | **`+74.7%`** |

### Drafter Switch, DFlash2 BF16 TC & Headless `240W` Pareto OC (Multi-Run Averages)

| Stage / Workload | Before | After (Live Verified) | Delta | Notes |
|---|---:|---:|---:|---|
| **6-Workload Mean** (`MTP-6` → `DFlash2 Q4_0`) | `153.16 tok/s` | **`157.45 tok/s`** | `+2.80%` | `code 171.42` / `math 166.87` / `reasoning 197.95` |
| **Reasoning Decode** (`DFlash2 BF16 TC`) | `148.30 tok/s` | **`154.61 tok/s`** | **`+4.25%`** | Cold + warm 2-pass avg; warm 16K ctx `+4.06%` |
| **Code Decode** (Headless OC + **`240W` Pareto**) | `189.42 tok/s` (stock 320W) | **`207.52 tok/s`** (2-run avg) | **`+9.6%`** | **-80 to -93 W draw, `41 °C` peak, `+19.3%` perf/W** |
| **Code Decode** (Headless OC + `320W` Peak) | `189.42 tok/s` (stock 320W) | **`209.80 tok/s`** (2-run avg) | **`+10.8%`** | Real GDDR6X `10,151 MHz` (below ECC retry cliff) |
| **Code Refactor / Agent JSON / Cross-Turn** (Hybrid `T=16` Suffix + DFlash2 Chaining + Rolling Cache) | `185.46 tok/s` (4-task warm mean) | **`229.98 tok/s`** (3-round all-in mean `206.08`) | **`+24.0%`** | Agent JSON `237.5 → 290.4 tok/s` (`+22.3%`), Python refactor `196.0 → 215.9 tok/s` (`+10.1%`), bit-exact greedy SHA |
| **Deep Kernel Suite P0–P4** (`W16 G=1` Fusion + Prefill Pre-Warm + Reg-MLP AR + `sm_86` INT8 TC) | `186.72 tok/s` (4-task warm baseline) | **`194.55 tok/s`** (`240W`, 84K delta **`203.5 tok/s`**) | **`+4.2%`** | 84K cold hit rate `70.0% → 79.2%` (`+9.2%`), decode `140.1 → 150.7 tok/s` (`+7.6%`), `cross_req +13.6%`, bit-exact SHA |

### Prefill Throughput (Strictly Lossless, Zero Decode Regression)

| Prompt Length / Metric | Before | After (Live Verified) | Gain | Notes |
|---|---:|---:|---:|---|
| **5K E2E HTTP TTFT** (`4,908 tok`) | `~2,086 tok/s` | **`2,231.5 tok/s`** | **`+7.0%`** | Async pre-pinned CPU share + parallel PLE |
| **10K E2E HTTP TTFT** (`9,758 tok`) | `~2,500 tok/s` | **`2,842.8 tok/s`** | **`+13.7%`** | 100% identical greedy hashes |
| **31K E2E HTTP TTFT** (`30,648 tok`) | `~3,483 tok/s` | **`3,696 ~ 3,788 tok/s`** | **`+8.7%`** | Single full chunk (`240W` `3,696` / `320W` `3,788`) |
| **40K E2E HTTP TTFT** (`39,551 tok`) | `3,142.9 tok/s` | **`3,690.9 tok/s`** | **`+17.4%`** | Parallel PLE (`+10.3%`) + `STRATA_PREFILL_EQUAL=1` (`+8.46%`) |
| **32K Chunk Internal** (`32,768 tok`) | `3,823 tok/s` (`8,570 ms`) | **`4,267 tok/s`** (`7,680 ms`) | **`+11.6%`** | PLE wait slashed `945 ms → 66 ms` (**14.3× faster**) |

### Dual RTX 2080 Ti 22 GB (Turing `sm_75` + NVLink) Verified Results

`strata-ampere` now natively supports **2× Modded RTX 2080 Ti 22 GB (44 GB total VRAM + NVLink `NV2`)** with Turing `sm_75` INT8 Tensor Core (`mma.sync.aligned.m8n8k16`), hardware FP16 NVLink P2P Reg-MLP AllReduce, and the full P0–P4 `W16` stack. Verified on a workstation with a 6-core `Xeon W-2135` + 4-channel DDR4-2666 + PCIe Gen3 x16 running `Qwen3.8-Flash-Next-UD-Q4_K_XL` (256K ctx):

| Metric (2× RTX 2080 Ti 22G + Xeon W-2135 6C) | Before (PCIe BF16 / `W8`) | **After (NVLink HW FP16 + `sm_75` INT8 TC + P0–P4 `W16`)** | Gain / Notes |
|---|---:|---:|---|
| **GPU Verify Window Latency** (`W2 / W3 / W4`) | `55 / 68 / 78 ms` | **`36 / 41 / 46 ms`** | **`-41.0%`** (`65.8 GB/s` NVLink P2P + FP16 Reg-MLP) |
| **Short Code Prompt** (`67 → 200 tok`) | Prefill `1,124 ms` / `68.7 tok/s` | **Prefill `695 ms` / `72.8 tok/s`** | TTFT **`+61.6%`**, P1 pre-warm hit rate `69.7% → 81.0%` |
| **16K–25K Cold Codebase** | `1,361 tok/s` / `74.6 tok/s` | **`1,474.9 tok/s` / `87.7 tok/s`** | Prefill **`+8.4%`**, 320 hot experts pre-warmed in `41.7 ms` (`85.7%` hit) |
| **16K–25K Multi-Turn Refactor (`W16` Suffix)** | Delta `~122 tok/s` / `101 tok/s` | **Delta `150.2 tok/s` / `107.7 ~ 109.2 tok/s`** | **`91.3%`** cache hit (`MTP-6` `107.7` / `DFlash2` `109.2 tok/s`) |
| **74K–154K Ultra-Long Context** | >32K `mtp.cpp` OOB crash / `63.7 tok/s` | **110K Prefill `1,570.3 tok/s`, 74K–154K Decode `91.0 ~ 103.6 tok/s`** | **Eliminates >32K OOB crash**, Decode **`+55.9%`** |

> **Drafter & CPU note for 2080 Ti 22G rigs**: In a strict 5-workload `T=0` A/B on the 6-core host, **`MTP-6` (`92.68 tok/s` mean)** outperformed **`DFlash2` (`87.32 tok/s`) by `+6.1%`** (especially creative writing `88.8 vs 70.5 tok/s` and cold 16K `80.7 vs 74.2 tok/s`) because Turing lacks `sm_80+` BF16 attention Tensor Cores and a 6-core CPU pool penalizes wider speculative windows, while `DFlash2` slightly led pure code refactoring (`109.2 vs 107.7 tok/s`). Pairing dual 2080 Ti 22G with a 14+-core CPU (e.g., `Xeon W-2170B / W-2275`) saturates quad-channel DDR4 and unlocks another `+15% ~ +20%` decode headroom.

---

## 2. Full Hardware & Software Testbed (Transparent & Reproducible)

| Component | Exact Specification |
|---|---|
| **GPUs** | **2× Modded RTX 3080 20 GB** (Ampere `sm_86`, 68 SMs / 20 GB GDDR6X each, 40 GB total VRAM) |
| **Interconnect** | **No NVLink, No PCIe P2P**; dual CPU-direct PCIe Gen4 x16 (`nvidia-smi topo -m` = `NODE`, across CPU Host Bridge, `BAR1 = 256 MiB`) |
| **GPU Clocks & Power** | Stock `320 W` (NVIDIA CUDA P2 state downclocks GDDR6X to `9,251 MHz`); live Pareto profile: **`240 W` per GPU + `10,151 MHz` VRAM (`+1800 eff`) + `2,100 MHz` core (`+120`)**, peak temp **`41 °C`** |
| **CPU** | **AMD Ryzen Threadripper 3960X** (24c/48t, 8 CCXs, Linux `performance` governor) |
| **System RAM** | **128 GB 4-channel DDR4-3200** (~77.8 GB/s effective CPU expert pool BW) |
| **OS & Stack** | Linux Kernel `6.8.0-138-generic`, NVIDIA Driver **`595.99.02`**, CUDA Toolkit **`13.1`** |
| **Model & Precision** | **`Qwen3.8-Flash-Next-UD-Q4_K_XL`** (48 layers, 512 routed experts/layer = 24,576 experts, 256K ctx; ~43% hot experts resident in 40 GB VRAM, cold experts streamed via PCIe/CPU) |
| **Key Flags** | `--kv int8 --kv-resident 32768 --ple-io ram --prompt-cache 6 --pool-workers 15`, `STRATA_TP2=1`, `STRATA_TP_AR_MODE=bf16`, `STRATA_ROUTE_TAIL_SKIP=0` |

---

## 3. Optimization Architecture & Strict Losslessness Guarantee

### Strict Losslessness Rules (Zero Shortcuts)
Every gain comes from systems architecture, CUDA microkernels, scheduling, and hardware clock/efficiency tuning — **never by degrading model quality**:
- **Zero quantization downgrade** (keeps original `UD-Q4_K_XL` weights).
- **Zero dropped experts** (`STRATA_ROUTE_TAIL_SKIP=0`).
- **Full `BF16` AllReduce** (`STRATA_TP_AR_MODE=q8` strictly rejected).
- **Verified Parity**:
  - **19/19** core CUDA/CPU parity & microbench targets pass with `0` failures.
  - **Long-Context Needle Test**: **3/3** exact hits at depth 0.5 (~150K chars / ~87K prompt tokens).
  - **Strict A/B Discipline**: Every regression was immediately reverted, including `STRATA_NV_CHUNK=16` (`-8.0%` 31K / `-7.2%` decode), `DEQUANT_SIDE + PLE_COMPACT` (`-7.8%` decode), `gr_write_norm_rs` single-block `bo` reuse (`+283 ms` register spill), upstream `iq_kernels.cu` Q4_K/Q5_K fused GU (`-7.0%` on `sm_86`), and `+2200 MHz` VRAM (`-7.4%` GDDR6X ECC retry cliff).

### Key Engineering Work
1. **Non-P2P `TP=2 + EP=2` & Zero-Sleep Ring AllReduce** (`src/kernels/cuda/tp_comm.cu`): 50/50 full-layer TP + disjoint 50/50 EP; decode uses Mapped Pinned Host Memory ring buffers with zero-sleep atomic GPU spin barriers (`~80–91 us` saved per 96× AR); prefill uses double-buffered 16 MB full-duplex CE DMA BF16 AllReduce.
2. **Ampere `sm_86` Microkernels**: Tuned `Q8_0`/`Q4_0` kernels (`q8_sm86_fast.cuh`, `q4_sm86.cuh`), re-benchmarked 68-SM `kIlRows_86` table, tree-aware 36-layer GDN Split (`STRATA_GDN_SPLIT=1`), `__ldg` in `S27`, and **`sm_80+` PTX BF16 Tensor Core DFlash2 Attention** (`src/kernels/cuda/dflash2.cu`, `+4.25%` reasoning / `+4.06%` warm 16K ctx).
3. **14.3× Parallel RAM-Backed PLE & Balanced Multi-Chunk Prefill**: 8-thread `PleTable::gather_batch` skipping redundant `madvise()` on locked/faulted RAM (`src/kernels/ngram.cpp`, `945 ms → 66 ms`, 32K chunk `3,823 → 4,267 tok/s`), async pre-pinned `cpu_x_stream` + `ExpertPool::run_range` (`+8.9%` on 4K–5K), and `STRATA_PREFILL_EQUAL=1` (`+8.46%` on 40K E2E).
4. **Headless Linux NVML OC, PCIe `MaxReadReq=4096B` & `THP defer+madvise`**: Built [scripts/ampere-oc-power.py](scripts/ampere-oc-power.py) to bypass NVIDIA's `-250 MHz` CUDA P2 trap (`10,151 MHz` @ `240 W`, `207.52 tok/s`, `41 °C`, `-93 W`, `+19.3%` perf/W), plus live PCIe `MaxReadRequestSize 512B → 4096B` (`+5.07%` code / `+3.38%` reasoning decode at zero extra watts) and async `THP defrag=defer+madvise`.
5. **Hybrid Dual-Stage Speculative Drafting (`T=16` Suffix Lookup + DFlash2 Chaining + Persistent Rolling Suffix Cache)** (`include/strata/spec/rolling_suffix.hpp`, `src/core/dflash2.cpp`, `src/core/verify.cpp`, `src/program/generate.cpp`): Lifted the `n_new <= 8` cap in `Dflash2Drafter::draft_launch`, made `Verifier::prepare(T)` TP=2-aware, unlocked `--lookup-max-t 16 --lookup-sizes 8,12,16 --lookup-chain 6` + `STRATA_DF2_LOOKUP_SKIP=6`, and added a bounded 256K-token cross-request `RollingSuffixSource` (`STRATA_SUFFIX_CACHE=1`, filtering multimodal `<|image_pad|>` tokens). Delivers **`+24.0%`** warm mean speedup across code refactoring, Agent JSON editing (**`290.40 tok/s`**), and multi-turn coding with bit-exact greedy hashes and zero regression on novel prompts.
6. **TP=2 Dual-GPU Multi-Conversation RAM Parking (`ConversationCache`)** (`src/program/generate.cpp`): Wired GPU 1 (`tp1->ss` + `tp1_part` checkpoints + incremental `reused_kv_bytes`) into `ConversationCache` instead of hard-disabling parking under `STRATA_TP2=1`. On a 2-tab Coding Agent switch (Tab A 12.6K ⇄ Tab B 6.8K), switching back to Tab A jumps from **`0% → 99.7%` KV reuse (`55.0 ms` restore)**, cutting TTFT **`4,869.8 ms → 499.3 ms` (9.75× faster)** with verified checkpoint rewind (`57.4 ms`).
7. **Deep Microkernel & Long-Context Suite (P0–P4, 100% Strictly Lossless)**:
   - **P0 (`-DSTRATA_WINDOW_MAX_T=16` `G=1` Full Fusion & 5-Bucket `DraftPolicy`)**: Lifted `n <= 8` caps in `verify.cpp` and `native_router.cu` (`can_mega_router`, `STRATA_LFUSE`, `hits_in_combine`) to `kVerifyMaxT=16` (halving `T=12/16` AllReduce barriers `192 → 96`) and expanded `DraftPolicy` to 5 context buckets (`<6K / <24K / <64K / <128K / >=128K`).
   - **P1 (Prefill-Guided Expert Pre-Warm & Dual-GPU VRAM Balance)**: Reuses `Prefill::on_routes` (`STRATA_PREFILL_PREWARM=1`) to promote top missing prompt experts across GPU 0/1 in `~13.5 ms` right before token 1, paired with GPU 0 reserve `1050 → 680 MiB` (`+368 MiB` resident experts) — boosting 84K cold hit rate `70.0% → 79.2%` and decode `140.1 → 150.7 tok/s`.
   - **P2 (`tp_comm.cu` Register-Cached + MLP-Pipelined BF16 AllReduce)**: `tp_allreduce_bf16_reg_kernel<K>` (`K=1,2,3,5`) keeps `local_buf` in registers across the barrier (zero redundant VRAM reload) and issues all `K` peer `load_cv_u4` PCIe reads concurrently.
   - **P3 (Multi-Hit PCIe 4096B vs CPU Split A/B)**: Implemented opt-in `STRATA_PCIE_HIT_MIN=N` in `expert_source.cpp`; live A/B proved `--pcie-frac 0.30` regresses `-4.6%` on 4-channel DDR4-3200 (`77.8 GB/s`) due to disabling `can_mega_router` + PCIe host-bridge contention with BF16 AllReduce, so default stays `--pcie-frac 0` (opt-in available for slower memory systems).
   - **P4 (Ampere `sm_86` + Turing `sm_75` INT8 Tensor Core)**: Added `q8_imma_m16n8k32_kernel` in `q8_sm86_fast.cuh` + single-launch `ncols=8..16` dispatch in `native_mmvq.cu` (`STRATA_Q8_TC=1`) — using `mma.sync.aligned.m16n8k32` on `sm_80+` and 4× `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` on Turing `sm_75` (`0/4096` exact int32 parity verified on RTX 2080 Ti). Hits `194.55 tok/s` warm mean (`203.50 tok/s` at 84K delta) on 2× 3080 20G @ 240W.
   - **Turing `sm_75` + NVLink P2P & >32K Stability Fix**: Added NVLink P2P direct VRAM path (`STRATA_TP_AR_P2P=1`, `65.8 GB/s` bidirectional), `tp_allreduce_fp16_reg_kernel<K>`, and hardware FP16 prefill pack/unpack kernels (`STRATA_TP_AR_MODE=fp16` / `STRATA_TP_AR_DECODE=fp16`) in `tp_comm.hpp`/`.cu`; fixed the >32K `mtp.cpp` `ident_` OOB crash (`window_ids` overwriting the iota array read by `iq_embed_rows`) with CUDA 12.x/13.x `cudaGraphAddNode` compatibility.

---

## 4. Build, Quick Start & Headless OC Tool

```bash
# Build for 2× RTX 3080 20G (Ampere sm_86, CUDA 12.x/13.1 + Ninja + C++20)
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_WINDOW_MAX_T=16 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)

# Build for 2× RTX 2080 Ti 22G (Turing sm_75, enables m8n8k16 INT8 Tensor Core + NVLink HW FP16)
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -B build75 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_WINDOW_MAX_T=16 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build75 --target strata -j $(nproc)

# Launch (3080 20G DFlash2 / MTP-6, or 2080 Ti 22G NVLink FP16)
python3 serve/server.py --config examples/strata-2x3080-20g-dflash2.example.json
python3 serve/server.py --config examples/strata-2x2080ti-22g.example.json

# Apply Pareto 240W + GDDR6X 10,151 MHz + PCIe MaxReadReq=4096B + THP defer+madvise live (no restart needed)
sudo python3 scripts/ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240 --pcie-mrrs 4096 --thp-defrag defer+madvise

# Generate a domain-specialized MoE VRAM residency heatmap (coding / writing / research / balanced, or --prompts-file)
sudo python3 scripts/profile-experts.py --config /path/to/strata-config.json --unit strata@iq3_s --domain coding --out /path/to/expert-profile-coding.bin --activate
```

### Built-in Port `8080` Live Telemetry Dashboard (Zero Dependencies)
After starting `serve/server.py`, open `http://<server-ip>:8080/` directly in your browser for the bilingual (`EN / 中文`) live dashboard shown above while keeping all OpenAI `/v1/*` endpoints 100% intact. The read-only numeric endpoint `GET /api/ampere/metrics` ([serve/ampere_metrics.py](serve/ampere_metrics.py)) surfaces dual-GPU thermals/power/clocks/MRRS, MoE VRAM residency, hybrid `T=16` draft acceptance, TP=2 `ConversationCache` parking, and `dmesg` Xid/AER/OOM health checks with zero prompt text or API key exposure.

---

## 5. Attribution & License

`strata-ampere` is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) under the **MIT License** — see [LICENSE](LICENSE).
