# strata-ampere

<p align="center">
  <b>面向消费级双卡 Ampere（<code>sm_86</code> 魔改 RTX 3080 20G · 解锁 50.86 GB/s 原生 256 MiB BAR1 直接 PCIe 4.0 P2P）MoE 大模型推理的 <a href="https://github.com/Niko1221/Strata">Strata</a> 严格无损高吞吐 TP=2 + EP=2 优化分支</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.en.md"><img src="https://img.shields.io/badge/Language-English-00b894.svg" alt="English"></a>
  <img src="https://img.shields.io/badge/2x3080%2020G-201.4%20tok%2Fs%20Warm%20Mean-00d4ff.svg" alt="201.4 tok/s">
  <img src="https://img.shields.io/badge/PCIe%204.0%20P2P-50.86%20GB%2Fs%20Full--Duplex-00e676.svg" alt="50.86 GB/s P2P">
  <img src="https://img.shields.io/badge/Precision-100%25%20Bit--Exact%20Lossless-ffbe28.svg" alt="Lossless">
</p>

<p align="center">
  <img src="docs/assets/dashboard-8080-live.jpg" alt="strata-ampere Port 8080 Live Telemetry Dashboard" width="100%">
  <br>
  <sub><b>内置 8080 端口实时监控仪表盘实拍</b>：双魔改 RTX 3080 20GB 7×24 稳定运行，现已解锁 <b>50.86 GB/s 直接 PCIe 4.0 P2P + 72×1 GiB 静态大页</b>，热态均速首次突破 <code>201.40 tok/s</code>（单任务 <code>252.01 tok/s</code>）、84K Prefill <code>3,870 tok/s</code>、满载 <code>41℃ / 36℃</code>、<code>0 Xid · 0 AER · 0 DMAR</code></sub>
</p>

`strata-ampere` 基于 [Strata](https://github.com/Niko1221/Strata)（`0.1.60.1`）深度全栈工程重构，专为 **双魔改 RTX 3080 20 GB（Ampere `sm_86`，40 GB 总显存）** 打造，在 **100% 严格无损精度** 下榨干每一分硬件极限：
- **无需改写 PCIe 寄存器扩 BAR1（原生 `256 MiB BAR1`）**：通过自研 **VMM 选择性 BAR1 映射 + 1-Hop Push-Write + VRAM Flag 屏障** 配合版本精确匹配的开源动态 BAR1 P2P 内核驱动（[patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)）与 **`72 × 1 GiB` 静态大页 + `iommu=pt`**，实测打通 **`50.86 GB/s` 全双工卡对卡直连**，热态均速首次突破 **`201.40 tok/s`**！

> **口径承诺（100% 严格无损铁律）**：本文所有核心速度均采用**多轮均速或中位数**，所有实验均在 `temperature=0`、零降量化（原始 `UD-Q4_K_XL`）、`STRATA_ROUTE_TAIL_SKIP=0`（绝不丢弃冷门专家）、`BF16` 全精度跨卡 AllReduce 的条件下实测复现，并强制校验 Greedy SHA256（`code_refactor = 79e43b1e6dee`、`agent_json_edit = 6d0b72b248bc`）逐位一致。

---

## 一、核心优化实测速度（全历程可复现数据）

### 1. 最新全栈里程碑：1 GiB 静态大页 + `iommu=pt` + 50.86 GB/s 直接 PCIe 4.0 P2P（240W 帕累托档）

基于 `Qwen3.8-Flash-Next-UD-Q4_K_XL`（256K 上下文，`DFlash2 Q4_0` + `T=16` 持久化后缀树混合草稿，单卡 `240W` + 显存 `10,151 MHz` + `MRRS 4096B`）：

| 严格 A/B 阶段（同硬件、同 240W、同测试集） | 4 任务热态均速 (`cli / srv`) | 2 轮总均速 (`srv`) | `agent_json_edit` | `unseen_code` | 84K Cold Prefill (`83,656 tok`) | Greedy SHA256 |
|---|---:|---:|---:|---:|---:|:---:|
| P4 基线（THP + 主机路由 AllReduce） | `194.55 / — tok/s` | `187.28 tok/s` | `~240 tok/s` | `~196 tok/s` | `3,780.0 tok/s` (`22,131 ms`) | ✅ 逐位一致 |
| + 2 MiB 静态 HugePages（`36,727 × 2M`） | `194.21 / — tok/s` | `187.76 tok/s` | `~242 tok/s` | `~198 tok/s` | `3,780.0 tok/s` (`22,131 ms`) | ✅ 逐位一致 |
| + **`72 × 1 GiB` 静态 HugePages + `iommu=pt`** | `194.8 / — tok/s` | `188.1 tok/s` | `~243 tok/s` | `~200 tok/s` | `3,860.0 tok/s` (`21,673 ms`) | ✅ 逐位一致 |
| + **50.86 GB/s VMM Push-Write PCIe P2P（当前线上）** | **`201.40 / 200.53 tok/s`** | **`192.26 tok/s`** | **`252.01 tok/s`** | **`212.91 tok/s`** | **`3,870.0 tok/s` (`21,616 ms`)** | ✅ **100% 逐位一致** |

- **双卡 P2P 硬件实测微基准（原生 `256 MiB BAR1`，零 BAR1 扩容、零 ACS 修改）**：
  - `nvidia-smi topo -p2p r` / `w`：双向 **`OK`**
  - `cuMemcpyDtoDAsync` 单向 **`26.32 GB/s`** / 全双工双向 **`50.86 GB/s`**（跑满 PCIe 4.0 x16 线速）
  - SM Kernel 直接 Push-Pack 双向 **`50.75 GB/s`**（零中间缓冲、零主桥二次绕行）
  - 2-GPU 端到端 Push-Write + VRAM Flag AllReduce 延迟：`T=1`（8 KiB BF16）**`6.05 µs`** / `T=4`（32 KiB）**`8.97 µs`** / `T=16`（128 KiB）**`23.47 µs`**
  - BAR1 实际占用：双卡各仅 **`39 MiB / 256 MiB`**（剩余 **`217 MiB`**，模型权重与 KV 占 **0 字节** BAR1）
  - `gpu_unlock` 满血 BF16 算力完好确认：GPU0 **`124.2 TFLOPS`** / GPU1 **`126.2 TFLOPS`**（实测 `4096³` BF16 GEMM）
  - 满载温度 **`41℃ / 36℃`**，冷重启全自动拉起，内核 `dmesg` 全程 **`0 Xid / 0 AER / 0 DMAR`**

### 2. 早期各阶段演进汇总（从 `91.4 tok/s` 一路压榨到 `201.4 / 252.0 tok/s`）

| 调优阶段 | 优化前 | 优化后（线上实测） | 净收益 | 核心技术点 |
|---|---:|---:|---:|---|
| **切层流水线 → 真 `TP=2 + EP=2`**（6 场景中位数） | `91.42 tok/s` | **`159.68 tok/s`** | **`+74.7%`** | 全 48 层 TP + 24,576 专家不重叠 EP + `sm_86` 68-SM 微核 |
| **草稿升级 `DFlash2 Q4_0` + BF16 TC 注意力** | `148.30 tok/s` | **`154.61 tok/s`** | **`+4.25%`** | `mma.sync.m16n8k16` BF16 Tensor Core 注意力 |
| **无头显存超频 + `240W` 帕累托 + `MRRS 4096B`** | `189.42 tok/s` (320W) | **`207.52 tok/s`** (240W) | **`+9.6%` / `-93W`** | 破 CUDA P2 降频至 `10,151 MHz`，满载 `41℃`，能效 `+19.3%` |
| **双级混合草稿 `T=16` + 持久化后缀树** | `185.46 tok/s` | **`229.98 tok/s`** | **`+24.0%`** | 256K 跨请求滚动后缀树，Agent JSON 最高 `290.4 tok/s` |
| **TP=2 双卡多对话 RAM 泊车（双标签 Agent）** | `4,869.8 ms` TTFT (`0%`) | **`499.3 ms` (`99.7%`)** | **`9.75×` 提速** | GPU 1 接入 `ConversationCache`，`55 ms` 内存恢复 |
| **深度内核 P0–P4 套件（`W16 G=1` + P1 预热 + P2 MLP + P4 INT8 TC）** | `186.72 tok/s` | **`194.55 tok/s`** | **`+4.2%`** | AllReduce 次数砍半、84K 命中 `+9.2%`、`sm_86` `m16n8k32` INT8 TC |
| **1 GiB 静态大页 + `iommu=pt` + 50.86 GB/s VMM Push-P2P** | `187.28` / `3,780` | **`192.26` (`201.4` 热态) / `3,870`** | **破 200 均速大关** | 72×1 GiB 锁死 dTLB + 原生 256 MiB BAR1 直接 P2P |

### 3. Prefill 提示词吞吐实测（严格无损）

| 上下文长度 / 测试项 | 优化前 | 优化后（线上实测） | 提升幅度 | 备注 |
|---|---:|---:|---:|---|
| **5K 端到端 HTTP TTFT**（`4,908 tok`） | `~2,086 tok/s` | **`2,231.5 tok/s`** | **`+7.0%`** | 异步预锁页 CPU 专家池 + 并行 PLE |
| **10K 端到端 HTTP TTFT**（`9,758 tok`） | `~2,500 tok/s` | **`2,842.8 tok/s`** | **`+13.7%`** | Greedy Hash 100% 一致 |
| **32K 单块引擎内部吞吐**（`32,768 tok`） | `3,823 tok/s` (`8,570 ms`) | **`4,267 tok/s`** (`7,680 ms`) | **`+11.6%`** | PLE 耗时由 `945 ms → 66 ms`（**14.3× 提速**） |
| **40K 端到端 HTTP TTFT**（`39,551 tok`） | `3,142.9 tok/s` | **`3,690.9 tok/s`** | **`+17.4%`** | 并行 PLE + `STRATA_PREFILL_EQUAL=1` 均衡分块 |
| **84K 长文冷启动 Prefill**（`83,656 tok`） | `3,780.0 tok/s` (`22,131 ms`) | **`3,870.0 tok/s`** (`21,616 ms`) | **新纪录 (`-515 ms`)** | 1 GiB 静态大页 + `iommu=pt` + 50.75 GB/s P2P Push-Pack |

---

## 二、完整宿主机硬件与软件配置

| 组件 | 实测配置详情 |
|---|---|
| **GPU** | **2× 魔改 RTX 3080 20 GB**（Ampere `sm_86`，PCI ID `10de:2206`，vBIOS `94.02.27.00.14`，原生 `BAR1 = 256 MiB`，合计 40 GB 显存） |
| **多卡互联** | **双卡直连 CPU PCIe 4.0 x16**（`00:01.1 → 01:00.0`、`40:03.1 → 48:00.0`），通过动态 BAR1 P2P 解锁 **`50.86 GB/s` 全双工直连**（`nvidia-smi topo -p2p r/w` 均为 `OK`） |
| **GPU 频率与功耗** | 单卡 **`240W`** 帕累托档 + GDDR6X **`10,151 MHz`** (`+1800 eff`) + 核心 **`2,100 MHz`** (`+120`) + PCIe **`MRRS 4096B`**，满载 **`41℃ / 36℃`** |
| **CPU 与内存** | **AMD Ryzen Threadripper 3960X**（24C/48T，8 CCX）+ **128 GB 四通道 DDR4-3200**（`72 × 1 GiB` 静态 HugePages 永久锁入内核 `Hugetlb`） |
| **系统与驱动** | Ubuntu Linux Kernel `6.8.0-138-generic`（`iommu=pt`，SecureBoot 关闭）+ **`gpu_unlock`**（initramfs 解锁满血 BF16）+ 补丁版开源内核模块 **`595.99.02`** + CUDA Toolkit **`13.1`** |
| **模型与权重** | **`Qwen3.8-Flash-Next-UD-Q4_K_XL`**（48 层、24,576 路由专家、256K 上下文；40 GB 显存常驻 9,764 热专家，其余 71.73 GiB 锁在 1 GiB 静态大页中） |

---

## 三、手把手完整复现指南（零踩坑 + 一键安全回退）

> **为什么这套方案能在原生 256 MiB BAR1 的魔改 RTX 3080 20GB 上零风险成功？**
> 传统消费卡 P2P 补丁依赖 *Static BAR1*（要求 `BAR1 >= 显存`，必须冒险刷 BIOS 或改写 PCIe `0x610` 寄存器扩 BAR1 到 32 GiB）；而 `llama.cpp` 等单进程引擎调用 `cudaDeviceEnablePeerAccess()` 会急切把全部 18 GB 模型权重塞进 256 MiB BAR1 立即爆 OOM。
> **本方案的两层核心解耦彻底绕开上述全部雷区**：
> 1. **内核驱动层（Dynamic BAR1 + `align3` 2 MiB 对齐）**：仅按需为 peer 映射的分配动态开 2 MiB 对齐的 BAR1 窗口，**无需改写 PCIe 寄存器扩 BAR1**，保留原厂 `256 MiB BAR1`；
> 2. **推理引擎层（Strata `TpComm` VMM 选择性映射 + 1-Hop Push-Write）**：彻底不调用进程级 `cudaDeviceEnablePeerAccess()`，改用 CUDA Driver VMM API（`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`）**只把 `TpComm` 的 36 MiB 通信缓冲区开放给对端 GPU**（模型权重与 KV 占 **0 字节** BAR1），并通过 `st.global.wt` 单向推送 + 对端 VRAM Flag 严格保序屏障彻底消灭 3× 慢速的 PCIe P2P 远端读！

### Step 0：前置检查与强制安全备份（千万别省！）

请先确认以下 4 项前提全部满足（本指南以 Ubuntu + 驱动 **`595.99.02`** + 内核 `6.8.0-138-generic` 为例；**开源内核模块版本号必须与系统已安装的 `nvidia-driver` 用户态完全一致**）：

```bash
# 1. 确认内核版本、内核头文件、用户态驱动版本、GSP 固件与 Secure Boot 状态
uname -r
ls -d /lib/modules/$(uname -r)/build
nvidia-smi --query-gpu=driver_version --format=csv,noheader
ls -lh /lib/firmware/nvidia/595.99.02/gsp_ga10x.bin
mokutil --sb-state   # 必须为 SecureBoot disabled（或自行签名模块）

# 2. 确认 initramfs 里没有打包 nvidia.ko（确保任何异常重启都能干净进入 SSH）
lsinitramfs /boot/initrd.img-$(uname -r) | grep -E 'nvidia\.ko|gpu_unlock'
# 预期：看不到 nvidia.ko（如有 gpu_unlock.ko 属正常且完全兼容！）

# 3. 备份原版闭源驱动模块与 GRUB 配置（永久留底）
sudo mkdir -p /root/backup-nvidia-595
sudo cp -a /lib/modules/$(uname -r)/updates/dkms/nvidia*.ko /root/backup-nvidia-595/ 2>/dev/null || \
  sudo cp -a $(modinfo -n nvidia nvidia-uvm nvidia-modeset nvidia-drm nvidia-peermem) /root/backup-nvidia-595/
sudo cp -a /etc/default/grub /root/backup-nvidia-595/grub.orig
```

---

### Step 1：开启 `72 × 1 GiB` 静态大页 + `iommu=pt`（128 GB 内存适用）

> 如宿主机内存为 128 GB 且模型专家池为 `71.73 GiB`，预留 `72` 个 1 GiB 静态大页后系统仍剩 ~20+ GiB 可用内存（零 Swap）。若内存不足 96 GB，请改用 `2 MiB` 静态大页或跳过 `hugepagesz=1G`（保留 `iommu=pt` 即可）。

```bash
# 1. 若之前配置过 2M 静态大页 sysctl，先移除避免叠加爆内存
sudo rm -f /etc/sysctl.d/99-strata-hugepages.conf

# 2. 编辑 /etc/default/grub，在 GRUB_CMDLINE_LINUX_DEFAULT 中加入 hugepagesz=1G hugepages=72 iommu=pt
#    （保留你原有的 gpu_unlock.targets=... 等参数！）
# 示例：
# GRUB_CMDLINE_LINUX_DEFAULT="gpu_unlock.targets=0000:01:00.0,0000:48:00.0 hugepagesz=1G hugepages=72 iommu=pt"
sudo nano /etc/default/grub
sudo update-grub
sudo reboot
```

重启后验证：
```bash
dmesg | grep -E "HugeTLB|iommu: Default|gpu_unlock: unlocked"
# 预期看到：
#   iommu: Default domain type: Passthrough
#   HugeTLB: registered 1.00 GiB page size, pre-allocated 72 pages
#   pci 0000:01:00.0: gpu_unlock: unlocked
#   pci 0000:48:00.0: gpu_unlock: unlocked

grep -E "Hugetlb|MemAvailable" /proc/meminfo
# 预期：Hugetlb: 75497472 kB（72 GiB），MemAvailable > 18 GiB
```

---

### Step 2：编译版本精确匹配的 `595.99.02` 动态 BAR1 P2P 开源内核模块

本仓库已直接附带经过实机验证的干净补丁 [patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)（仅改动 9 个核心文件，包含 Method 3 动态 BAR1 窗口、GA102 HAL 接线、`align3` 2 MiB 对齐修复、`memdescAddRef` 生命周期加固及 32 MiB 保留区调优）：

```bash
# 1. 克隆本仓库（或直接拿 patches/nvidia-open-595.99.02-dynbar1-p2p.patch）
git clone https://github.com/zongxinqian/strata-ampere.git
STRATA_DIR=$(pwd)/strata-ampere

# 2. 克隆官方 NVIDIA open-gpu-kernel-modules 精确对应 tag（必须与 nvidia-smi 驱动版本一致！）
sudo mkdir -p /root/nv-open-p2p && cd /root/nv-open-p2p
sudo git clone --depth 1 --branch 595.99.02 \
  https://github.com/NVIDIA/open-gpu-kernel-modules.git open-595.99.02

# 3. 打上动态 BAR1 P2P + align3 补丁并编译全部 5 个内核模块（24 核约 50 秒）
cd /root/nv-open-p2p/open-595.99.02
sudo git apply --verbose "$STRATA_DIR/patches/nvidia-open-595.99.02-dynbar1-p2p.patch"
sudo make modules -j$(nproc) SYSSRC=/lib/modules/$(uname -r)/build

# 4. 校验生成的 5 个 .ko 文件版本与内核 vermagic 完全匹配
modinfo kernel-open/nvidia.ko | grep -E "version|vermagic|license"
# 预期：version: 595.99.02 | license: Dual MIT/GPL | vermagic 匹配当前 uname -r
```

---

### Step 3：免覆盖安全热加载试跑（验证通过前绝不修改系统默认驱动！）

这一步使用 `insmod` 临时加载刚编好的 `.ko`，**磁盘上的原版闭源驱动分毫未动**——即使出现任何异常，直接 `sudo reboot` 即可 100% 回到原状！

```bash
# 1. 停掉所有占用 GPU 的服务（注意包含监控 exporter、watchdog、persistenced、推理服务）
sudo systemctl stop strata@iq3_s gpu-arbiter-watchdog gpu-exporter nvidia-persistenced 2>/dev/null || true
sudo nvidia-smi -pm 0 2>/dev/null || true
sleep 2

# 2. 卸载当前闭源模块并确认已清空
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia 2>/dev/null || true
lsmod | grep -E '^nvidia'   # 必须无输出！如仍被占用，用 sudo lsof /dev/nvidia* 查进程

# 3. 加载开源 nvidia.ko 依赖的内核密码学模块（关键！漏了会报 Unknown symbol ecc_*）
sudo modprobe ecc ecdh_generic drm drm_kms_helper video

# 4. 临时 insmod 我们编译的 595.99.02 动态 BAR1 P2P 模块
KO=/root/nv-open-p2p/open-595.99.02/kernel-open
sudo insmod $KO/nvidia.ko NVreg_EnableResizableBar=0
sleep 1
# 注意：udev 可能会瞬间自动 modprobe 原版 nvidia_uvm，先卸载再显式加载配套开源版！
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm 2>/dev/null || true
sudo insmod $KO/nvidia-uvm.ko uvm_disable_hmm=1
sudo insmod $KO/nvidia-modeset.ko
sudo insmod $KO/nvidia-drm.ko

# 5. 检查 P2P 拓扑与内核健康！
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
# 预期 GPU0 <-> GPU1 读写全部显示 OK！
dmesg | tail -n 20   # 必须看到 Open Kernel Module 595.99.02，且 0 RmInitAdapter failed / 0 Xid / 0 AER
```

---

### Step 4：持久化安装 + 准备「一键回退原版闭源驱动」脚本

热加载验证 `nvidia-smi topo -p2p r/w == OK` 且零报错后，将其干净安装到独立目录 `updates/p2p/`（原版闭源模块保留在原路径完全不覆盖）：

```bash
KVER=$(uname -r)
KO=/root/nv-open-p2p/open-595.99.02/kernel-open

# 1. 复制 5 个补丁模块到独立目录
sudo mkdir -p /lib/modules/$KVER/updates/p2p
sudo install -m 0644 $KO/nvidia*.ko /lib/modules/$KVER/updates/p2p/

# 2. 仅针对当前内核版本覆盖解析顺序（其它内核仍默认启动原版）
sudo tee /etc/depmod.d/p2p.conf >/dev/null <<EOF
override nvidia $KVER updates/p2p
override nvidia-uvm $KVER updates/p2p
override nvidia-modeset $KVER updates/p2p
override nvidia-drm $KVER updates/p2p
override nvidia-peermem $KVER updates/p2p
EOF

# 3. 持久化模块参数（关闭静态 ReBAR 保持动态 BAR1，关闭 HMM）
sudo tee /etc/modprobe.d/zz-nvidia-p2p.conf >/dev/null <<'EOF'
options nvidia NVreg_EnableResizableBar=0
options nvidia-uvm uvm_disable_hmm=1
EOF

# 4. 生成一键安全回退脚本（任何时候执行即可秒回原版闭源 595.99.02）
sudo tee /usr/local/sbin/nvidia-p2p-rollback.sh >/dev/null <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -f /etc/depmod.d/p2p.conf /etc/modprobe.d/zz-nvidia-p2p.conf
depmod -a $KVER
echo "已回退至原版闭源 nvidia-595.99.02，请执行 sudo reboot 生效。"
EOF
sudo chmod +x /usr/local/sbin/nvidia-p2p-rollback.sh

# 5. 更新模块依赖并验证解析路径
sudo depmod -a $KVER
modinfo -n nvidia nvidia-uvm nvidia-modeset nvidia-drm
# 预期全部指向 /lib/modules/$KVER/updates/p2p/*.ko

# 6. 冷重启验证全链路开机自启！
sudo reboot
```

> **随时一键回退命令**：如果遇到任何问题或想恢复原版闭源驱动，只需执行：
> ```bash
> sudo /usr/local/sbin/nvidia-p2p-rollback.sh && sudo reboot
> ```

---

### Step 5：编译并启动 `strata-ampere`（验证 VMM Push-Write P2P + 1G 大页全上线）

```bash
# 1. 编译 Ampere sm_86（双 RTX 3080 20G，开启 W16 单组全融合）
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_WINDOW_MAX_T=16 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)

# 2. 应用 240W 帕累托超频 + PCIe MRRS=4096B + THP defer+madvise
sudo python3 scripts/ampere-oc-power.py \
  --mem-offset 1800 --gpc-offset 120 --power-limit 240 \
  --pcie-mrrs 4096 --thp-defrag defer+madvise

# 3. 启动服务（复制 examples/strata-2x3080-20g-dflash2.example.json 修改路径与 api_key）
python3 serve/server.py --config /path/to/your-strata-config.json
```

**如何确认复现 100% 成功？** 检查 Strata 启动日志与 `nvidia-smi`，看到以下 **4 条金标准日志** 即大功告成：
```text
strata generate: expert arena: cudaHostRegister PORTABLE ok; hugetlb 1 GB pages (72 x 1 GiB)
strata tp_comm: NVLink/P2P enabled (vmm-push, decode=p2p, CUDA0 <-> CUDA1)
strata serve: STRATA_TP2=1 ready: CUDA0 (4666 experts) + CUDA1 (5098 experts) = 9764 total resident experts
```
同时 `nvidia-smi -q -d MEMORY | grep -A 3 "BAR1 Memory Usage"` 显示双卡 BAR1 均仅占用 **`39 MiB / 256 MiB`**（剩余 `217 MiB`），`dmesg` 零 `Xid` / 零 `AER`！

### 常见排坑速查（FAQ）
- **`insmod nvidia.ko` 报 `Unknown symbol ecc_*` / `crypto_ecdh_*`**：执行 `sudo modprobe ecc ecdh_generic` 即可。
- **`rmmod nvidia` 报 `Module nvidia is in use`**：检查 `sudo lsof /dev/nvidia*`，并确认已关闭 `nvidia-smi -pm 0`、`nvidia-persistenced` 及任何监控 exporter / watchdog 服务。
- **热加载 `insmod` 时 `nvidia_uvm` 报 `File exists`**：`udev` 自动加载了原版闭源 `nvidia_uvm.ko`，执行 `sudo rmmod nvidia_uvm && sudo insmod $KO/nvidia-uvm.ko uvm_disable_hmm=1` 替换即可（Step 4 持久化 `depmod` 后开机自动加载配套开源版，不会再遇到）。
- **如何关闭 P2P 或仅关闭 Decode P2P 做严格 A/B 对照**：无需换驱动，直接设置环境变量 `STRATA_TP_AR_P2P=0`（完全回落主机路由）或 `STRATA_TP_AR_P2P_DECODE=0`（仅 Prefill 走 P2P）即可！
- **多卡内置 8080 实时监控面板**：启动后浏览器直访 `http://<server-ip>:8080/`（支持中英切换），`/v1/*` OpenAI API 100% 兼容不变；领域热力图采集脚本见 `scripts/profile-experts.py`。

---

## 四、核心优化技术原理全景（100% 严格无损）

1. **原生 256 MiB BAR1 直接 PCIe 4.0 P2P + VMM 选择性映射 + 1-Hop Push-Write**（`src/kernels/cuda/tp_comm.cu` + `patches/nvidia-open-595.99.02-dynbar1-p2p.patch`）：
   - 仅将 `TpComm` 所需的 **36 MiB 连续通信块**（2 MiB Decode 环 + 2×16 MiB Prefill 双缓冲 + 64B VRAM 同步 Flag）通过 CUDA Driver VMM API（`dlopen("libcuda.so.1")` 零链接依赖，内置 bad-VA-hole 检测重试与双向硬件写入自检）映射进对端 256 MiB BAR1；
   - **全部采用单向 Push-Write（`st.global.wt`）+ 本地 760 GB/s VRAM 读取**，彻底规避比写入慢 3× 的 PCIe P2P 远端读；同步 Flag 位于同一对端 BAR1 窗口，依赖 PCIe Base Spec Posted-Write 严格保序规则确保零数据竞态且自旋等待零 PCIe 流量。
2. **`72 × 1 GiB` 静态 HugePages 专家竞技场**（`src/core/pinned.cu`、`src/core/expert_source.cpp`）：优先分配 `MAP_HUGETLB | MAP_HUGE_1GB`（回退 `MAP_HUGE_2MB` / THP），71.73 GiB 专家池仅占 72 条 TLB 表项（100% 装入 Zen 2 2048 项 L2 dTLB），彻底消除 Swap 与 `khugepaged` 整理卡顿。
3. **深度微核与长上下文套件（P0–P4）**：`T=16` 单组 `G=1` 全融合（AllReduce 次数 `192 → 96`）+ 5 桶长上下文 `DraftPolicy`（P0）、Prefill 路由引导 `13.5 ms` 热专家抢跑预热（P1）、寄存器缓存去重读 + `K` 路并发 PCIe MLP AllReduce（P2）、Ampere `sm_86` (`m16n8k32`) INT8 Tensor Core（P4）。
4. **双级混合草稿（`T=16` 256K 持久化后缀树 + DFlash2 BF16 TC）与 TP=2 多对话 RAM 泊车**：双标签页 Coding Agent 切换 TTFT 提速 `9.75×`（`99.7%` 复用）。
5. **14.3× 并行内存 PLE（`945 ms → 66 ms`）+ `STRATA_PREFILL_EQUAL=1` 均衡切分 + 无头 `240W` 超频 / `MRRS 4096B`**：零掉精度，全面压榨每一瓦算力。

---

## 五、上游致谢与开源协议

- 引擎主体派生自 [Niko1221/Strata](https://github.com/Niko1221/Strata)，遵循 **MIT License**（见 [LICENSE](LICENSE)）。
- 动态 BAR1 P2P 内核驱动补丁（Dual MIT/GPLv2，见 [patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)）站在以下开源先驱的肩膀上构建并移植至 `595.99.02`：
  - [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules)
  - [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules)（消费卡 BAR1 P2P 开创者）
  - [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)（新驱动与跨代 P2P 移植）
  - [Duanyll/open-gpu-kernel-modules](https://github.com/Duanyll/open-gpu-kernel-modules)（Method 3 动态 BAR1 逐分配窗口）
  - [dingiv](https://github.com/dingiv/open-gpu-kernel-modules) & [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p)（RTX 3080 20GB `align3` 2 MiB 对齐规律 v3 与生命周期加固）
