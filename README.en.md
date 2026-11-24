# strata-ampere

<p align="center">
  <b>Strictly lossless, high-throughput TP=2 + EP=2 fork of <a href="https://github.com/Niko1221/Strata">Strata</a> engineered for dual consumer Ampere (<code>sm_86</code>) MoE inference.</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/语言-简体中文-00b894.svg" alt="简体中文"></a>
</p>

`strata-ampere` is a systems-engineered fork of [Strata](https://github.com/Niko1221/Strata) (`0.1.60.1`) built to maximize **100% strictly lossless** MoE **Decode & Prefill** throughput on **non-NVLink, non-P2P** dual consumer Ampere GPUs (`2× modded RTX 3080 20 GB`) paired with a 4-channel DDR4 host CPU expert pool.

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

### Prefill Throughput (Strictly Lossless, Zero Decode Regression)

| Prompt Length / Metric | Before | After (Live Verified) | Gain | Notes |
|---|---:|---:|---:|---|
| **5K E2E HTTP TTFT** (`4,908 tok`) | `~2,086 tok/s` | **`2,231.5 tok/s`** | **`+7.0%`** | Async pre-pinned CPU share + parallel PLE |
| **10K E2E HTTP TTFT** (`9,758 tok`) | `~2,500 tok/s` | **`2,842.8 tok/s`** | **`+13.7%`** | 100% identical greedy hashes |
| **31K E2E HTTP TTFT** (`30,648 tok`) | `~3,483 tok/s` | **`3,696 ~ 3,788 tok/s`** | **`+8.7%`** | Single full chunk (`240W` `3,696` / `320W` `3,788`) |
| **40K E2E HTTP TTFT** (`39,551 tok`) | `3,142.9 tok/s` | **`3,690.9 tok/s`** | **`+17.4%`** | Parallel PLE (`+10.3%`) + `STRATA_PREFILL_EQUAL=1` (`+8.46%`) |
| **32K Chunk Internal** (`32,768 tok`) | `3,823 tok/s` (`8,570 ms`) | **`4,267 tok/s`** (`7,680 ms`) | **`+11.6%`** | PLE wait slashed `945 ms → 66 ms` (**14.3× faster**) |

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
5. **Hybrid Dual-Stage Speculative Drafting (`T=16` Suffix Lookup + DFlash2 Chaining + Persistent Rolling Suffix Cache)** (`include/strata/spec/rolling_suffix.hpp`, `src/core/dflash2.cpp`, `src/core/verify.cpp`, `src/program/generate.cpp`): Lifted the `n_new <= 8` cap in `Dflash2Drafter::draft_launch`, made `Verifier::prepare(T)` TP=2-aware, unlocked `--lookup-max-t 16 --lookup-sizes 8,12,16 --lookup-chain 6` + `STRATA_DF2_LOOKUP_SKIP=6`, and added a bounded 256K-token cross-request `RollingSuffixSource` (`STRATA_SUFFIX_CACHE=1`). Delivers **`+24.0%`** warm mean speedup across code refactoring, Agent JSON editing (**`290.40 tok/s`**), and multi-turn coding with bit-exact greedy hashes and zero regression on novel prompts.

---

## 4. Build, Quick Start & Headless OC Tool

```bash
# Build (sm_86, CUDA 12.x/13.1 + Ninja + C++20)
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)

# Launch (pick DFlash2 Q4_0 for code/math/reasoning or MTP-6 for balanced/creative)
python3 serve/server.py --config examples/strata-2x3080-20g-dflash2.example.json

# Apply Pareto 240W + GDDR6X 10,151 MHz + PCIe MaxReadReq=4096B + THP defer+madvise live (no restart needed)
sudo python3 scripts/ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240 --pcie-mrrs 4096 --thp-defrag defer+madvise

# Generate a domain-specialized MoE VRAM residency heatmap (coding / writing / research / balanced, or --prompts-file)
sudo python3 scripts/profile-experts.py --config /path/to/strata-config.json --unit strata@iq3_s --domain coding --out /path/to/expert-profile-coding.bin --activate
```

---

## 5. Attribution & License

`strata-ampere` is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) under the **MIT License** — see [LICENSE](LICENSE).
