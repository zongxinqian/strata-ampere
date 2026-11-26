# strata-ampere

<p align="center">
  <b>Strictly lossless, high-throughput TP=2 + EP=2 fork of <a href="https://github.com/Niko1221/Strata">Strata</a> for dual modded Ampere (<code>sm_86</code> RTX 3080 20G · 50.86 GB/s Direct PCIe 4.0 P2P on native 256 MiB BAR1).</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/语言-简体中文-00b894.svg" alt="简体中文"></a>
  <img src="https://img.shields.io/badge/v1.1.0-231.4%20tok%2Fs%20Warm%20Mean-00d4ff.svg" alt="231.4 tok/s">
  <img src="https://img.shields.io/badge/Peak-287.2%20tok%2Fs%20%40%20240W-7c4dff.svg" alt="287.2 tok/s Peak">
  <img src="https://img.shields.io/badge/PCIe%204.0%20P2P-50.86%20GB%2Fs%20Full--Duplex-00e676.svg" alt="50.86 GB/s P2P">
  <img src="https://img.shields.io/badge/Precision-100%25%20Bit--Exact%20Lossless-ffbe28.svg" alt="Lossless">
</p>

<p align="center">
  <img src="docs/assets/dashboard-8080-live.jpg" alt="strata-ampere Port 8080 Live Telemetry Dashboard" width="100%">
  <br>
  <sub><b>Built-in Port 8080 Live Telemetry Dashboard</b>: 2× Modded RTX 3080 20GB now running <b>v1.1.0 6-direction kernel & system optimizations + 50.86 GB/s direct PCIe 4.0 x16 P2P + 72×1 GiB Static HugePages</b> — hitting <code>231.35 tok/s</code> warm mean (<code>202.50 tok/s</code> fresh-start warm mean, <code>287.20 tok/s</code> peak), <code>3,870 tok/s</code> 84K prefill, <code>9,807</code> resident experts, <code>43 °C / 38 °C</code> load, <code>4.8 W</code> idle, <code>0 Xid · 0 AER · 0 DMAR</code></sub>
</p>

`strata-ampere` is a systems-engineered fork of [Strata](https://github.com/Niko1221/Strata) (`0.1.60.1`) built specifically for **2× Modded RTX 3080 20 GB (`sm_86`, 40 GB total VRAM)** to deliver **100% strictly lossless** MoE **Decode & Prefill** performance:
- **v1.1.0 6-Direction Deep Optimization Campaign (`240W` cap, 100% strictly lossless)**: Adds **GPU1 Sparse-Hit MoE Combine** (eliminating 48× `cudaMemsetAsync` + >50% zero-row reads per step), **Capacity-Weighted `48:52` Bresenham Expert Partitioning** (`9,764 → 9,807` resident experts, eliminating mid-rank eviction gaps), **Ampere `sm_86` Hardware `__nanosleep(64)` P2P Spin-Barrier + Warp-Rounded `K=1` Blocks**, **Zen 2 AVX2 Weight Prefetch + PCIe Root Port ACS Disable + `/dev/cpu_dma_latency=0`**, and **`STRATA_DF2_HEAD_Q4=1` + true `--spec 8 --draft-conf 0.50` (`user_args`) + recalibrated `draft-costs.json`** — pushing hot-warm 4-task mean to **`231.35 tok/s`** (`287.20 tok/s` peak) while restoring GPU0 free VRAM from `508 MiB` to **`528+ MiB`**.
- **50.86 GB/s full-duplex card-to-card PCIe 4.0 x16 P2P on the card's native `256 MiB BAR1` (ZERO BAR1 resize / ZERO PCI `0x610` hacking!)** via our version-matched dynamic BAR1 open-gpu kernel patch ([patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)), CUDA Driver VMM selective peer mapping (`39 / 256 MiB` BAR1 used), 1-hop Push-Write + VRAM-Flag AllReduce, and `72 × 1 GiB` Static HugePages + `iommu=pt`.

> **Strict Losslessness Guarantee**: Every headline number below is a multi-run average or median under `temperature=0`, original target `UD-Q4_K_XL` quantization, `STRATA_ROUTE_TAIL_SKIP=0` (zero dropped experts), full `BF16` AllReduce, and bit-exact greedy SHA256 verification (`STRATA_DF2_HEAD_Q4=1` applies only to the DFlash2 drafter; target model verification remains 100% lossless).

---

## 1. Verified Benchmarks (Strictly Lossless, `Qwen3.8-Flash-Next-UD-Q4_K_XL`)

### Latest Milestone — v1.1.0 6-Direction Deep Kernel & System Suite (`240W` Pareto, Strict A/B)

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

### Step 5: Build & Run `strata-ampere` v1.1.0 (Including ACS Disable & `/dev/cpu_dma_latency=0`)
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
- `strata serve: STRATA_TP2=1 ready: CUDA0 (4702 experts) + CUDA1 (5105 experts) = 9807 total resident experts`
- GPU0 free VRAM **`~528–532 MiB`**, `BAR1 Memory Usage: Used 39 MiB / Total 256 MiB` on both GPUs, `0 Xid / 0 AER`.
- Open `http://<server-ip>:8080/` for the built-in bilingual live telemetry dashboard (with all `/v1/*` OpenAI endpoints intact).

---

## 3. Attribution & License

- Engine fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) under the **MIT License** ([LICENSE](LICENSE)).
- Dynamic BAR1 P2P kernel patch (Dual MIT/GPLv2) builds on [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules), [tinygrad](https://github.com/tinygrad/open-gpu-kernel-modules), [aikitoria](https://github.com/aikitoria/open-gpu-kernel-modules), [Duanyll](https://github.com/Duanyll/open-gpu-kernel-modules) (Method 3), [dingiv](https://github.com/dingiv/open-gpu-kernel-modules) (`align3`), and [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p).
