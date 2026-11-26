# strata-ampere

<p align="center">
  <b>面向消费级双卡 Ampere（<code>sm_86</code> 魔改 RTX 3080 20G · 解锁 50.86 GB/s 原生 256 MiB BAR1 直接 PCIe 4.0 P2P）MoE 大模型推理的 <a href="https://github.com/Niko1221/Strata">Strata</a> 严格无损高吞吐 TP=2 + EP=2 优化分支</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.en.md"><img src="https://img.shields.io/badge/Language-English-00b894.svg" alt="English"></a>
  <img src="https://img.shields.io/badge/v1.3.0-217%20Warm%20%7C%20204.7%20All%20%7C%20173.6%20%40%20117K-00d4ff.svg" alt="v1.3.0 Decode">
  <img src="https://img.shields.io/badge/Peak-276.6%E2%80%93373%20tok%2Fs%20%40%20240W-7c4dff.svg" alt="Peak 276.6-373 tok/s">
  <img src="https://img.shields.io/badge/PCIe%204.0%20P2P-50.86%20GB%2Fs%20Full--Duplex-00e676.svg" alt="50.86 GB/s P2P">
  <img src="https://img.shields.io/badge/Precision-17%2F17%20Bit--Exact%20Lossless-ffbe28.svg" alt="17/17 Bit-Exact Lossless">
</p>

<p align="center">
  <img src="docs/assets/dashboard-8080-live.jpg" alt="strata-ampere Port 8080 Live Telemetry Dashboard" width="100%">
  <br>
  <sub><b>内置 8080 端口实时监控仪表盘实拍</b>：双魔改 RTX 3080 20GB 7×24 稳定运行，现已全面升级至 <b>v1.3.0 全仓后 P2P 原生架构深度重构 + DFlash2 推测树验证 + 50.86 GB/s 直接 PCIe 4.0 P2P + 72×1 GiB 静态大页</b> — 标准 3 轮全均值 <code>204.67 tok/s</code>（热态均速 <code>214–217 tok/s</code>，单任务峰值 <code>276.6 tok/s</code>，瞬时突发峰值 <code>373.0 tok/s</code>）、<b>117K 超长上下文冷启 Prefill <code>3,830.9–3,842.6 tok/s</code> + Decode 暴涨至 <code>173.6 tok/s</code>（<code>+45.0%</code>）</b>、<b>9,870 常驻热专家</b>、多轮 4.1K 会话跨卡恢复 <code>32.9 ms</code>、待机 CPU <code>0.00%</code>、满载 <code>43℃ / 38℃</code>、<code>0 Xid · 0 AER · 0 OOM</code></sub>
</p>

`strata-ampere` 基于 [Strata](https://github.com/Niko1221/Strata)（`0.1.60.1`）深度全栈工程重构，专为 **双魔改 RTX 3080 20 GB（Ampere `sm_86`，40 GB 总显存）** 打造，在 **100% 严格无损精度（17/17 SHA256 逐位一致）** 下榨干每一分硬件极限：
- **v1.3.0 全仓后 P2P 原生架构清查重构 + DFlash2 推测树验证（240W 严格无损，17/17 Bit-Exact SHA256 门禁全通）**：彻底扫除“无 P2P 时代遗留老路”——**① DFlash2 前沿分支推测树验证（`STRATA_DF2_TREE=1`）+ `sm_86` 快速 Q8/MMVQ/AVX2 微核**：将不确定前沿的双候选分支同时送入单次 Verify 图验证，显著抬高每窗接受 token 数；**② 图内统一 TP=2 BAR1 P2P Top-64 采样器（含真实对话惩罚项，`STRATA_TP_HEAD_FAST=1`）**：将 `presence_penalty=1.5` 等历史惩罚（`hist_len <= 128`）通过相干映射缓冲 `TpHeadSampleCfg` 直接纳入 Verify CUDA Graph 内部，消除默认流同步拷贝与图外二次采样，GPU1 仅经 BAR1 推送 `Top-64` 候选；**③ 解锁 DFlash2 原生 `Q4_0` 50/50 P2P 草稿头 + 原生 BF16 KV 环（`STRATA_DF2_KV_BF16=1`）**：打通 `STRATA_DF2_HEAD_Q4=1` 与 `STRATA_DF2_HEAD_TP2=1` 直连共存，消除 GPU0 `98.7 MiB` 启动瞬时峰值、将 DFlash2 单窗草稿耗时从 `1.32 ms` 压低至 **`0.87–1.17 ms`（最高 `-34%`）**，叠加原生 BF16 KV 环与 `22 MiB` 精简 VMM 块，常驻专家再增 **`+40` → `9,870`**；**④ 融合 BAR1 P2P `peer_bo_bf16` Prefill AllReduce（`STRATA_TP_FUSE_PF_AR=1`）+ 紧凑异步 PLE 上传**：`gr_write_norm_rs_kernel` 直接跨 BAR1 融合读取对端 BF16 部分和，**117K（`116,800` tok）冷启 Prefill 达 `3,830.9–3,842.6 tok/s`，长上下文 Decode 从 `119.70 → 173.60 tok/s`（暴涨 `+45.0%`）**；**⑤ 按卡异步会话驻留/恢复（`32.9 ms` 恢复 4.1K 会话）+ 零空转 `tp1` 工作线程（`0.00%` 待机 CPU）+ 遥测去抖 & SSE `TCP_NODELAY`**！
- **v1.2.0 P2P 原生架构深度优化（240W 严格无损，4/4 Bit-Exact SHA256 校验通过）**：在解锁 `50.86 GB/s` VMM Push-Write P2P 后进一步重构两大高频通信与草稿瓶颈——**① Direct P2P ArgHalf & 全词表 Logits 直传（`STRATA_TP_HEAD_P2P=1`）**：彻底消除真实采样（`temp > 0`）每步 `16 MiB` GPU1→Host→GPU0 绕行拷贝，4 任务真实聊天均速从 `164.57 → 185.50 tok/s`（**`+20.93 tok/s / +12.7%`**）；**② DFlash2 106K CJK 草稿头 P2P TP=2 词表并行（`STRATA_DF2_HEAD_TP2=1`）**：将 `146.0 MiB` Q4_0 草稿头按行 50/50 拆分到双卡（`53,152 + 53,147` 行，各 `73.0 MiB`），GPU0 通过 BAR1 推送 `20 KiB` 激活唤醒 GPU1 捕获图并行算后回推 `896 B` Top-16、单 Warp 32 线程 Bitonic 精确归并，**直接释放 `73.0 MiB` GPU0 显存**（`--vram-reserve-mib 608` 再多装 **`+24` 个常驻专家 → `9,831`**）；**③ Ampere `sm_86` 专属 `STRATA_DF2_GEMM4=26`**（unroll=1 降寄存器压力，自动按 `sm_86` 生效），标准 4 任务热态均速提升至 **`203.22–208.27 tok/s`**（`+8.4% ~ +11.1%`），冻结 Adapt 下 **4/4 任务 2,560 token 逐位一致**！
- **v1.1.0 六方向深度微核与系统调优**：**GPU1 稀疏命中 MoE Combine**、**48:52 容量加权 Bresenham 专家切分**、**Ampere `sm_86` 硬件 `__nanosleep(64)` P2P 自旋屏障 + `K=1` Warp 对齐**、**Zen 2 AVX2 权重预取 + PCIe Root Port ACS 关闭 + `/dev/cpu_dma_latency=0`**、**`STRATA_DF2_HEAD_Q4=1` + 真 `--spec 8 --draft-conf 0.50`（`user_args`）+ `draft-costs.json` 重校准**。
- **无需改写 PCIe 寄存器扩 BAR1（原生 `256 MiB BAR1`）**：通过自研 **VMM 选择性 BAR1 映射 + 1-Hop Push-Write + VRAM Flag 屏障** 配合版本精确匹配的开源动态 BAR1 P2P 内核驱动（[patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)）与 **`72 × 1 GiB` 静态大页 + `iommu=pt`**，实测打通 **`50.86 GB/s` 全双工卡对卡直连**。

> **口径承诺（100% 严格无损铁律）**：本文所有核心速度均采用**多轮均速或中位数**，所有实验均在 `temperature=0`、目标模型零降量化（原始 `UD-Q4_K_XL`）、`STRATA_ROUTE_TAIL_SKIP=0`（绝不丢弃冷门专家）、`BF16` 全精度跨卡 AllReduce 的条件下实测复现，并强制校验 Greedy SHA256 逐位一致（`STRATA_DF2_HEAD_Q4=1` 仅作用于草稿模型 DFlash2，目标模型验证 100% 精确无损）。

---

## 一、核心优化实测速度（全历程可复现数据）

### 1. v1.3.0 最新里程碑：全仓后 P2P 原生架构清查重构 + DFlash2 推测树验证（240W 严格无损）

基于 `Qwen3.8-Flash-Next-UD-Q4_K_XL`（256K 上下文，`DFlash2 Q4_0` P2P 50/50 双卡草稿头 + 原生 BF16 KV 环 + 前沿推测树 + `T=12` 持久化后缀树，单卡 `240W` + 显存 `10,151 MHz` + `MRRS 4096B` + `50.86 GB/s` PCIe 4.0 P2P，4 任务标准套件 `moe_arch / code_cuda / agent_json / kv_tech`，640 tok，`enable_thinking=False`）：

| 核心验收维度（240W 严格无损 `UD-Q4_K_XL`） | 无 P2P 基线 | 树推测前序基线 | **v1.3.0 全仓后 P2P 重构终版** | 净提升与关键亮点 |
|---|---:|---:|---:|---|
| **3 轮标准 `bench_6dir` 全均值 (`all_mean`)** | `165.30 tok/s` | `200.03 tok/s` | **`204.67 tok/s`** | 较无 P2P **`+23.8%`**，命中率 **`80.92%`** |
| **热态均值 (`r2 / r3`) · 单任务峰值** | `176.2` / `~210` | `215.30` / `272.80` | **`217.00 / 214.00` · `276.60 tok/s`** | 面板实测瞬时突发峰值 **`373.0 tok/s`** |
| **真实对话 `temp=0.6`（默认 `presence=1.5`）** | `~142 tok/s` | `172.15 / 172.75` | **`180.38 / 177.83 tok/s`（峰 `201.2`）** | 历史惩罚采样 **100% 移入 Verify 图内**（**`+3.9%`**） |
| **`117K` (`116,800` tok) Cold Prefill** | `~3,150 tok/s` | `3,798.8 tok/s` | **`3,830.9–3,842.6 tok/s`** | 融合 BAR1 P2P `peer_bo_bf16` AllReduce + 紧凑异步 PLE |
| **`117K` (`116,800` tok) 超长文 Decode** | `~92 tok/s` | `119.70 tok/s` | **`173.60 tok/s`** | **`+53.90 tok/s`（暴涨 `+45.0%`！）** |
| **双卡常驻热专家 / DFlash2 单窗耗时** | `9,656` / `1.74 ms` | `9,830` / `1.32 ms` | **`9,870 (4766+5104)` / `0.87–1.17 ms`** | **`+40` 常驻专家**，草稿延迟最高 **`-34%`** |
| **4.1K 多轮会话跨卡恢复 / 待机 CPU** | 同步阻塞 / 偏高 | `~115 ms` / 自旋耗核 | **`32.9 ms` (+`17.6 ms` 重建) / `0.00%`** | 按卡异步 `ParkAsync` + 无锁休眠唤醒，`0 Xid / 0 AER / 0 OOM` |

- **17/17 严格 Bit-Exact SHA256 对齐门禁（对称锁定 `4500 + 4500 = 9000` 常驻专家，`--adapt-every 999999`）**：
  - **Pure Greedy (`4/4 EXACT`)**、**无惩罚 `temp=0.6` (`4/4 EXACT`)**、**真实对话 `presence_penalty=1.5` 图内采样 (`4/4 EXACT`)**、**带惩罚 Greedy (`2/2 EXACT`)**、**`penalty_last_n=256` 图外 P2P 归并 (`2/2 EXACT`)**、**`9.7K` 稀疏块选择长文 Prefill+Decode (`1/1 EXACT`)** —— 全部 **17/17 组 SHA256 100.000% 逐位一致** 且同专家数下全项正提速！
- **v1.3.0 诚实消融披露（保留正向默认开启，实验特性保留为可选开关）**：
  - ✅ **默认开启**：`STRATA_DF2_TREE=1`（推测树验证）、`STRATA_TP_HEAD_FAST=1`（图内含惩罚统一 P2P 采样器）、`STRATA_DF2_KV_BF16=1`（原生 BF16 KV 环）、`STRATA_DF2_HEAD_TP2=1` + `STRATA_DF2_HEAD_Q4=1`（Q4_0 50/50 P2P 草稿头）、`STRATA_TP_FUSE_PF_AR=1`（融合 Prefill P2P AllReduce）、`STRATA_PLE_COMPACT=1` + `STRATA_PLE_COPY_STREAM=1`、`STRATA_TP_VMM_PIPE_MB=22`。
  - ⚙️ **保留为可选开关（默认 `=0`）**：`STRATA_TP_QSA_IDX_SPLIT=1`（6+6 跨卡 QSA Indexer 分片可将双卡常驻专家推高至 `9,998` 个，但长文下因 BF16 双卡 1-ulp 归约顺序差异出现 `16/17` 微差且 Prefill 慢 `~4.5%`，严守 100% 逐位无损红线默认关闭）；`STRATA_ADAPT_SMART=0`（实测提高换入阈值会把实时热专家换入从 `29.6` 压制到 `3.3 swaps/win`，保持无节流实时适应更快）。

### 2. v1.2.0 里程碑：P2P 原生架构深度重构（240W 严格无损，同二进制严格隔离 A/B）

基于 `Qwen3.8-Flash-Next-UD-Q4_K_XL`（256K 上下文，`DFlash2 Q4_0` 106,299-token CJK 草稿头 + `T=12` 持久化后缀树，单卡 `240W` + 显存 `10,151 MHz` + `MRRS 4096B` + `50.86 GB/s` PCIe 4.0 P2P，4 任务标准套件 `moe_arch / code_cuda / agent_json / kv_tech`，640 tok，`enable_thinking=False`）：

| 严格对照实验组（同 `build86_w16` 二进制、同 240W） | Greedy r1 均值 | Greedy 热态均值 | **真实聊天 `temp=0.6` 均值** | 常驻热专家 / 空闲显存 | 84K / 28K Cold Prefill | 无损校验 |
|---|---:|---:|---:|---:|---:|:---:|
| **v1.1.0 严格基线（全部 v1.2 特性关闭）** | `175.10 tok/s` | `187.45 tok/s` | **`164.57 tok/s`** | `9,807` (`548 / 230 MiB`) | `3,692.9 tok/s` (28K) | 基准 |
| **+ #1 `STRATA_TP_HEAD_P2P=1`（P2P Logits 直传）** | `177.80 tok/s` | `189.45 tok/s` | **`180.78 tok/s` (`+9.9%`)** | `9,807` (`548 / 230 MiB`) | `3,699.5 tok/s` (28K) | ✅ 逐位一致 |
| **+ #1 + #2 `STRATA_DF2_HEAD_TP2=1`（草稿头 P2P TP=2）** | `178.68 tok/s` | **`203.03 tok/s` (`+8.3%`)** | **`179.43 tok/s` (`+9.0%`)** | `9,807` (**`620 / 152 MiB`**, GPU0 省 73M) | `3,705.9 tok/s` (28K) | ✅ **4/4 SHA256** |
| **v1.2.0 Champion（+#1+#2+`GEMM4=26`+`reserve=608` 线上）** | **`179.20–185.82`** | **`203.22–208.27` (`+8.4% ~ +11.1%`)** | **`185.50 tok/s` (`+12.7%`!!)** | **`9,831 (+24)`** (`530 / 152 MiB`) | **`3,836.2 tok/s` (84K, 0 OOM)** | ✅ **4/4 100% 一致** |

- **4/4 严格 Bit-Exact Greedy SHA256 证明（冻结 Adapt `--adapt-every 999999`，4 任务共 2,560 tokens）**：
  - `moe_arch`: `113.9 → 118.0 tok/s` (`+3.6%`, `sha=a03df34b38d867cb` **EXACT**)
  - `code_cuda`: `106.6 → 110.2 tok/s` (`+3.4%`, `sha=c03f0df80b846de4` **EXACT**)
  - `agent_json`: `109.1 → 111.8 tok/s` (`+2.5%`, `sha=cb0d17a11e0089d8` **EXACT**)
  - `kv_tech`: `138.0 → 143.6 tok/s` (`+4.1%`, `sha=9ae96e090151696e` **EXACT**)
- **v1.2.0 六方向完整实测复盘（保留正向，诚实关闭负收益实验开关）**：
  - ✅ **#1 `STRATA_TP_HEAD_P2P=1`（默认开启）**：消除 `temp > 0` 每步 `16 MiB` Host 往返拷贝，真实聊天直接暴涨 `+16.2 ~ +20.9 tok/s`；Greedy 路径直传 `ArgHalf` 到 VMM 尾部零开销自复位。
  - ✅ **#2 `STRATA_DF2_HEAD_TP2=1`（默认开启）**：将 `106,299`-token CJK Q4_0 草稿头（`146.0 MiB`）50/50 拆分到双卡（`73.0 MiB/GPU`），BAR1 推 `20 KiB` `xq_` + 回推 `896 B` Top-16 + 单 Warp Bitonic 精确归并，既提速 `+13.6 tok/s` 又帮 GPU0 腾出 `73 MiB` 多装 `+24` 个常驻专家。
  - ✅ **`STRATA_DF2_GEMM4=26`（`sm_86` 自动默认）**：针对 68-SM Ampere 降低 Q4_0 草稿 GEMM 寄存器压力（替代面向 110-SM `sm_120` 的 variant `17`），无损再涨 `+5~11 tok/s`。
  - ❌ **#3 `STRATA_TP_AR_GR_NORM=0`（默认关闭）**：`n_tok <= 8` CTA 数过少导致 PCIe 并发度不如原有 `allreduce_add` + `gr_norm_split_fast`，实测 `-17 ~ -30 tok/s`，诚实关闭。
  - ❌ **#4 `STRATA_TP2_GPU1_CPU_MISS=0` / `STRATA_TP2_HEAT_W0=0.96`（保持关闭/原值）**：GPU1 接管 CPU miss 出现 MoE 归并同步问题且 `HEAT_W0=0.82` 在 `temp=0.6` 综合表现弱于 `0.96`，保持关闭。
  - ❌ **#5 `STRATA_TP_P2P_PIPE=0`（默认关闭）**：Prefill 4 MiB 微块流水线在 28K Cold Prefill 上增加 4× 跨卡 barrier/event 开销（`3,630` vs `3,713 tok/s`，慢 `~165 ms`），原有 16 MiB 双缓冲 P2P 胜出。
  - ❌ **#6 `STRATA_TP2_P2P_REBALANCE=0`（默认关闭，保留 opt-in）**：零 DDR4 跨卡 P2P 专家互换实测在短周期下收益中性略波动，默认关闭保留为可选实验开关。

### 2. v1.1.0 里程碑：六方向深度内核 + 系统调优（240W 帕累托档，严格 A/B）

基于 `Qwen3.8-Flash-Next-UD-Q4_K_XL`（256K 上下文，`DFlash2 Q4_0` + `T=12` 持久化后缀树混合草稿，单卡 `240W` + 显存 `10,151 MHz` + `MRRS 4096B` + `50.86 GB/s` PCIe 4.0 P2P，4 任务标准套件 `moe_arch / code_cuda / agent_json / kv_tech`，`temperature=0`）：

| 对比组（同硬件、同 240W、同 4 任务套件） | 4 任务热态均速 | `moe_arch` | `code_cuda` | `agent_json` | `kv_tech` | 常驻热专家 / GPU0 剩余显存 | Greedy SHA256 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **v1.0.0 线上基线（同冷启 r2 热态）** | `186.22 tok/s` | `204.2 tok/s` | `234.6 tok/s` | `177.0 tok/s` | `129.1 tok/s` | `9,764` / `508 MiB` | ✅ 逐位一致 |
| **v1.1.0 同冷启 r2 热态（严格同起跑线）** | **`202.50 tok/s` (`+8.7%`)** | `200.6 tok/s` | **`254.1 tok/s`** | **`222.6 tok/s` (`+25.8%`)** | `132.7 tok/s` | **`9,807 (+43)` / `528 MiB`** | ✅ 逐位一致 |
| **v1.0.0 充分预热 4 任务均速** | `199.68 tok/s` | — | — | — | — | `9,764` / `508 MiB` | ✅ 逐位一致 |
| **v1.1.0 充分预热 r3 冠军均速（线上）** | **`231.35 tok/s` (`+15.9%`)** | **`287.20 tok/s` (97.5% acc)** | **`256.70 tok/s` (92.1% acc)** | **`198.90 tok/s`** | **`182.60 tok/s`** | **`9,807 (+43)` / `528 MiB`** | ✅ **100% 逐位一致** |

- **关键硬件与延迟指标实测（v1.1.0 线上实机）**：
  - `draft-costs.json` 重校准后验证窗口延迟：`T=5` **`20.43 ms`**、`T=6` **`20.39 ms`**（热态 `T=6` 低至 **`18.75 ms`**，彻底清除 P2P 升级前被污染的 `42.19 ms` 倒挂表项）
  - DFlash2 草稿模型显存与耗时：`862 MiB → 733 MiB`（`-129 MiB`），单轮草稿 **`1.737 ms`**（Head 仅 `0.374 ms`），GPU0 空闲显存回到 **`528–532 MiB`**（解除 `<512 MiB` 告警），GPU1 `228 MiB`
  - 双卡 P2P 微基准（原生 `256 MiB BAR1`）：单向 **`26.32 GB/s`** / 全双工 **`50.86 GB/s`** / SM Push-Pack **`50.75 GB/s`** / BAR1 仅占 **`39 / 256 MiB`**（权重与 KV 占 **0 字节** BAR1）
  - 功耗、温度与稳定性：满载 **`238.6W / 238.3W`**（`43℃ / 38℃`），待机回落至 **`4.8W`**（`210 / 405 MHz`，`23–28℃`），`dmesg` 全程 **`0 Xid · 0 AER · 0 DMAR`**

### 2. v1.1.0 六方向深度探索明细（含如实披露的负收益实验）

本版本完整实现了全部 6 个工程方向，并经逐项严格 A/B 实测筛选——**所有正收益项默认开启上线，所有负收益/中性项保留为 Bit-Exact 可复现环境变量开关并默认关闭**：

| 方向 | 技术改动 | 线上状态 | 实测结论与原因 |
|---|---|:---:|---|
| **方向 1** | GPU1 零拷贝直读 72×1G 大页 (`STRATA_TP2_GPU1_MISS_256`) | 默认关闭 (`=0`) | 代码与 Bit-Exact 均跑通（0 Xid/AER，命中率 `79%→89%`），但 144B/176B 非合并 Q4_K/Q5_K SM 跨 PCIe 读仅跑 `4–6 GB/s` 且与 CPU 争抢 4 通道 DDR4-3200，吞吐反降至 `143.3 tok/s`，诚实关闭 |
| **方向 2** | 稀疏命中 GPU1 MoE Combine (`STRATA_TP2_GPU1_SPARSE_COMBINE`) + 融合 Push (`STRATA_TP_FUSE_MOE_PUSH`) | **稀疏 Combine ON (`=1`)** · 融合 Push OFF (`=0`) | 稀疏 Combine 干掉每步 48 次 `cudaMemsetAsync` + 跳过 >50% 零行读（稳定提速）；多 Block 融合 Push 因每 Block 需 `__threadfence_system()` 并重读 `local_buf`，反慢 `~0.4 ms`，保留为 `=0` 开关 |
| **方向 3** | **48:52 容量加权 Bresenham 专家切分** (`STRATA_TP2_EXPERT_SPLIT=48:52`) + **Ampere `__nanosleep(64)` P2P 屏障 & `K=1` Warp 对齐** | **默认开启** | 匹配 GPU0:GPU1 真实可用显存比（`4,702 : 5,105`），常驻热专家 **`+43` (`9,764 → 9,807`)** 且彻底消灭中段排名驱逐空洞；`__nanosleep(64)` 省下 GPU1 在 240W 功耗墙下的空转能耗 |
| **方向 4** | **`STRATA_DF2_HEAD_Q4=1` + `--lookup-max-t 12 --lookup-sizes 12` + `user_args` 真 `--spec 8 --draft-conf 0.50` + `draft-costs.json` 重校准** | **默认开启** | 绕开 `drafting.py` 自动覆写 `--spec 6` 的限制，完全释放 DFlash2 原生 7 步草稿（`--spec 8`）并用 `0.50` 剪枝低置信文本；`draft-costs` 重校准 + Q4 Head 释放 `129 MiB` 显存，是本次提速核心引擎之一 |
| **方向 5** | Ampere `sm_86` `STRATA_EXPDEC=1` 实测验证 | **否决不启用** | 在 `sm_86` 上反而拖慢 `~1.5 ms/window`（该路径针对其他架构设计），实测淘汰 |
| **方向 6** | **Zen 2 AVX2 权重软件预取 (`kq_avx2.cpp`) + PCIe Root Port ACS 关闭 + `/dev/cpu_dma_latency=0`**（对比 22 核 & `L3_PREWARM=2`） | **AVX2 预取 + ACS + DMA Latency 开启** · 维持 15+1 核 | 实测 15+1 核已完全打满 TR 3960X 四通道 DDR4-3200，扩到 22 核 `<0.2 ms` 无增益；保留最优雅 15+1 核配置，叠加 AVX2 预取 + ACS 关闭 + `cpu_dma_latency=0` 消除微秒级长尾抖动 |

### 3. 全历程演进汇总（从 `91.4 tok/s` 一路压榨到 `231.4 / 287.2 tok/s`）

| 调优阶段 | 优化前 | 优化后（线上实测） | 净收益 | 核心技术点 |
|---|---:|---:|---:|---|
| **切层流水线 → 真 `TP=2 + EP=2`**（6 场景中位数） | `91.42 tok/s` | **`159.68 tok/s`** | **`+74.7%`** | 全 48 层 TP + 24,576 专家不重叠 EP + `sm_86` 68-SM 微核 |
| **草稿升级 `DFlash2 Q4_0` + BF16 TC 注意力** | `148.30 tok/s` | **`154.61 tok/s`** | **`+4.25%`** | `mma.sync.m16n8k16` BF16 Tensor Core 注意力 |
| **无头显存超频 + `240W` 帕累托 + `MRRS 4096B`** | `189.42 tok/s` (320W) | **`207.52 tok/s`** (240W) | **`+9.6%` / `-93W`** | 破 CUDA P2 降频至 `10,151 MHz`，满载 `41℃`，能效 `+19.3%` |
| **双级混合草稿 + 256K 持久化后缀树** | `185.46 tok/s` | **`229.98 tok/s`** | **`+24.0%`** | 256K 跨请求滚动后缀树，严格无损逐位一致 |
| **TP=2 双卡多对话 RAM 泊车（双标签 Agent）** | `4,869.8 ms` TTFT (`0%`) | **`499.3 ms` (`99.7%`)** | **`9.75×` 提速** | GPU 1 接入 `ConversationCache`，`55 ms` 内存恢复 |
| **深度内核 P0–P4 套件（`W16 G=1` + P1 + P2 + P4 INT8 TC）** | `186.72 tok/s` | **`194.55 tok/s`** | **`+4.2%`** | AllReduce 次数砍半、84K 命中 `+9.2%`、`sm_86` `m16n8k32` INT8 TC |
| **v1.0.0：1 GiB 静态大页 + `iommu=pt` + 50.86 GB/s VMM Push-P2P** | `187.28` / `3,780` | **`201.40` 热态 / `3,870`** | **破 200 大关** | 72×1 GiB 锁死 dTLB + 原生 256 MiB BAR1 直接 P2P |
| **v1.1.0：六方向深度微核 + 48:52 EP + `user_args --spec 8` + ACS/DMA** | `186.22` / `199.68` | **`202.50` 冷启 / `231.35` 热态 (`287.2` 峰)** | **`+8.7% ~ +15.9%`** | 稀疏 Combine + 9,807 专家 + `__nanosleep` + `DF2_HEAD_Q4` + 重校准 |
| **v1.2.0：P2P Logits 直传 + DFlash2 106K 头 TP=2 + `sm_86` GEMM4=26** | `164.6` 聊天 / `187.5` Greedy | **`185.5` 真实聊天 / `203.2–208.3` 热态 (`273` 峰)** | **`+12.7%` 聊天 / `+8.4%~11.1%` Greedy** | 消除 16M Host 绕行 + GPU0 省 73M (`9,831` 专家) + 4/4 SHA256 逐位一致 |
| **v1.3.0：全仓后 P2P 重构 + 图内惩罚采样 + BF16 KV + 融合 Prefill AR + 推测树** | `200.0` 全均 / `119.7` @117K | **`204.67` 全均 (`217` 热 / `276.6` 峰) / `173.6` @117K** | **`+45.0%` @117K / `3,842.6` Prefill** | 17/17 SHA256 逐位一致、`9,870` 专家、草稿 `-34%`、`32.9 ms` 跨卡恢复、`0.00%` 待机 CPU |

### 4. Prefill 提示词吞吐实测（严格无损）

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
| **多卡互联** | **双卡直连 CPU PCIe 4.0 x16**（`00:01.1 → 01:00.0`、`40:03.1 → 48:00.0`，Root Port ACS 已关闭），通过动态 BAR1 P2P 解锁 **`50.86 GB/s` 全双工直连**（`nvidia-smi topo -p2p r/w` 均为 `OK`） |
| **GPU 频率与功耗** | 单卡 **`240W`** 帕累托档 + GDDR6X **`10,151 MHz`** (`+1800 eff`) + 核心 **`2,100 MHz`** (`+120`) + PCIe **`MRRS 4096B`**，满载 **`43℃ / 38℃`**，待机 **`4.8W`** |
| **CPU 与内存** | **AMD Ryzen Threadripper 3960X**（24C/48T，8 CCX，`/dev/cpu_dma_latency=0`）+ **128 GB 四通道 DDR4-3200**（`72 × 1 GiB` 静态 HugePages 永久锁入内核 `Hugetlb`） |
| **系统与驱动** | Ubuntu Linux Kernel `6.8.0-138-generic`（`iommu=pt`，SecureBoot 关闭）+ **`gpu_unlock`**（initramfs 解锁满血 BF16）+ 补丁版开源内核模块 **`595.99.02`** + CUDA Toolkit **`13.1`** |
| **模型与权重** | **`Qwen3.8-Flash-Next-UD-Q4_K_XL`**（48 层、24,576 路由专家、256K 上下文；40 GB 显存通过 `48:52` Bresenham + DFlash2 TP=2 草稿头腾出 73 MiB 后常驻 **9,831** 个最热专家，其余 71.73 GiB 锁在 1 GiB 静态大页中） |

---

## 三、手把手完整复现指南（零踩坑 + 一键安全回退）

> **为什么这套方案能在原生 256 MiB BAR1 的魔改 RTX 3080 20GB 上零风险成功？**
> 传统消费卡 P2P 补丁依赖 *Static BAR1*（要求 `BAR1 >= 显存`，必须冒险刷 BIOS 或改写 PCIe `0x610` 寄存器扩 BAR1 到 32 GiB）；而 `llama.cpp` 等单进程引擎调用 `cudaDeviceEnablePeerAccess()` 会急切把全部 18 GB 模型权重塞进 256 MiB BAR1 立即爆 OOM。
> **本方案的两层核心解耦彻底绕开上述全部雷区**：
> 1. **内核驱动层（Dynamic BAR1 + `align3` 2 MiB 对齐）**：仅按需为 peer 映射的分配动态开 2 MiB 对齐的 BAR1 窗口，**无需改写 PCIe 寄存器扩 BAR1**，保留原厂 `256 MiB BAR1`；
> 2. **推理引擎层（Strata `TpComm` VMM 选择性映射 + 1-Hop Push-Write）**：彻底不调用进程级 `cudaDeviceEnablePeerAccess()`，改用 CUDA Driver VMM API（`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`）**只把 `TpComm` 的 36 MiB 通信缓冲区开放给对端 GPU**（模型权重与 KV 占 **0 字节** BAR1），并通过 `st.global.wt` 单向推送 + 对端 VRAM Flag + `__nanosleep(64)` 严格保序屏障彻底消灭 3× 慢速的 PCIe P2P 远端读！

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
grep -E "Hugetlb|MemAvailable" /proc/meminfo
# 预期：Hugetlb: 75497472 kB（72 GiB），MemAvailable > 18 GiB
```

---

### Step 2：编译版本精确匹配的 `595.99.02` 动态 BAR1 P2P 开源内核模块

本仓库已直接附带经过实机验证的干净补丁 [patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)（仅改动 9 个核心文件，包含 Method 3 动态 BAR1 窗口、GA102 HAL 接线、`align3` 2 MiB 对齐修复、`memdescAddRef` 生命周期加固及 32 MiB 保留区调优）：

```bash
# 1. 克隆本仓库
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
```

---

### Step 3：免覆盖安全热加载试跑（验证通过前绝不修改系统默认驱动！）

这一步使用 `insmod` 临时加载刚编好的 `.ko`，**磁盘上的原版闭源驱动分毫未动**——即使出现任何异常，直接 `sudo reboot` 即可 100% 回到原状！

```bash
# 1. 停掉所有占用 GPU 的服务（包含监控 exporter、watchdog、persistenced、推理服务）
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
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm 2>/dev/null || true
sudo insmod $KO/nvidia-uvm.ko uvm_disable_hmm=1
sudo insmod $KO/nvidia-modeset.ko
sudo insmod $KO/nvidia-drm.ko

# 5. 检查 P2P 拓扑与内核健康！
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
# 预期 GPU0 <-> GPU1 读写全部显示 OK！
dmesg | tail -n 20   # 必须看到 Open Kernel Module 595.99.02，且 0 Xid / 0 AER
```

---

### Step 4：持久化安装 + 准备「一键回退原版闭源驱动」脚本

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
echo "已回退至原版闭源 nvidia-595.99.02，请执行 sudo reboot 生效。"
EOF
sudo chmod +x /usr/local/sbin/nvidia-p2p-rollback.sh

sudo depmod -a $KVER
modinfo -n nvidia nvidia-uvm nvidia-modeset nvidia-drm
sudo reboot
```

> **随时一键回退驱动**：`sudo /usr/local/sbin/nvidia-p2p-rollback.sh && sudo reboot`

---

### Step 5：编译并启动 `strata-ampere` v1.3.0（含 ACS 关闭 + `/dev/cpu_dma_latency=0` 持久化）

```bash
# 1. 编译 Ampere sm_86（已内置 v1.3.0 全部后 P2P 原生架构重构、图内惩罚采样器、BF16 KV 环与推测树验证）
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

# 3. （v1.1.0 新增，推荐）关闭两张 GPU 上游 PCIe Root Port 的 ACS，并常驻 /dev/cpu_dma_latency=0 消除 CPU C-State 唤醒抖动
# 先查你的 GPU 上游 Root Port（例：00:01.1 与 40:03.1）：
lspci -tv | grep -iE '3080|NVIDIA'
# 确认 ACS 控制位偏移（通常为 ECAP_ACS+6，如 2a6.w）后置零：
sudo setpci -s 00:01.1 2a6.w=0000
sudo setpci -s 40:03.1 2a6.w=0000

# 创建轻量守护服务把 /dev/cpu_dma_latency 锁为 0 µs（极低开销，不影响 GPU 待机 4.8W）
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

# 4. 重置旧的 draft-costs.json（若是从无 P2P 升级而来，务必删掉旧表让引擎按 P2P 实速重测！）
rm -f /path/to/draft-costs.json

# 5. 启动服务（复制 examples/strata-2x3080-20g-dflash2.example.json 修改路径与 api_key）
# 注意：v1.1.0 示例配置已包含 "user_args": ["--spec", "8", "--draft-conf", "0.50"] 与 STRATA_DF2_HEAD_Q4=1
python3 serve/server.py --config /path/to/your-strata-config.json
```

**如何确认 v1.2.0 复现 100% 成功？** 检查 Strata 启动日志与 `nvidia-smi`，看到以下 **4 条金标准日志** 即大功告成：
```text
strata generate: expert arena: cudaHostRegister PORTABLE ok; hugetlb 1 GB pages (72 x 1 GiB)
strata tp_comm: NVLink/P2P enabled (vmm-push, decode=p2p, CUDA0 <-> CUDA1)
strata dflash2: P2P TP=2 draft head ON (GPU0 53152 rows + GPU1 53147 rows, 73.0 MiB/GPU)
strata serve: STRATA_TP2=1 ready: CUDA0 (4726 experts) + CUDA1 (5105 experts) = 9831 total resident experts
```
同时 `nvidia-smi` 显示 GPU0 剩余显存 **`~530–550 MiB`**（GPU1 `~152 MiB`，84K 长文 Prefill 零 OOM）、双卡 BAR1 均仅占用 **`39 MiB / 256 MiB`**（剩余 `217 MiB`），`dmesg` 零 `Xid` / 零 `AER`！

### 常见排坑速查（FAQ）
- **为什么配置文件里既写了 `args` 又加了 `"user_args": ["--spec", "8", "--draft-conf", "0.50"]`？** `serve/strata_control/drafting.py` 会把 `args` 中的 `--spec` 自动重置为默认 `6`；只有把 `--spec 8 --draft-conf 0.50` 放进 `user_args`，才能真正生效并释放 DFlash2 全部 7 步草稿。
- **`insmod nvidia.ko` 报 `Unknown symbol ecc_*` / `crypto_ecdh_*`**：执行 `sudo modprobe ecc ecdh_generic` 即可。
- **`rmmod nvidia` 报 `Module nvidia is in use`**：检查 `sudo lsof /dev/nvidia*`，并确认已关闭 `nvidia-smi -pm 0`、`nvidia-persistenced` 及监控 exporter / watchdog 服务。
- **热加载 `insmod` 时 `nvidia_uvm` 报 `File exists`**：`udev` 自动加载了原版闭源 `nvidia_uvm.ko`，执行 `sudo rmmod nvidia_uvm && sudo insmod $KO/nvidia-uvm.ko uvm_disable_hmm=1` 替换即可（Step 4 持久化 `depmod` 后开机自动加载配套开源版）。
- **如何做严格 A/B 对照**：所有关键特性均支持环境变量一键开关，无需重新编译：
  - `STRATA_TP_AR_P2P=0`（关闭 P2P） / `STRATA_TP_AR_P2P_DECODE=0`（仅 Prefill 走 P2P）
  - `STRATA_TP2_GPU1_SPARSE_COMBINE=0`（关闭 GPU1 稀疏 Combine）
  - `STRATA_TP2_EXPERT_SPLIT=50:50`（切回对称切分 vs 默认 `48:52`）
  - `STRATA_DF2_HEAD_Q4=0`（切回 DFlash2 F16 Head）
  - `STRATA_TP_FUSE_MOE_PUSH=1` / `STRATA_TP2_GPU1_MISS_256=1`（实验性负收益开关，默认关闭）
- **内置 8080 实时监控面板**：启动后浏览器直访 `http://<server-ip>:8080/`（支持中英切换），`/v1/*` OpenAI API 100% 兼容不变；领域热力图采集脚本见 `scripts/profile-experts.py`。

---

## 四、核心优化技术原理全景（100% 严格无损）

1. **v1.1.0 六方向深度微核与系统优化**（[patches/strata-ampere-v1.1.0-tp2-opt.patch](patches/strata-ampere-v1.1.0-tp2-opt.patch)）：
   - **GPU1 稀疏命中 MoE Combine**（`src/kernels/cuda/native_moe.cu`、`src/core/verify.cpp`）：`combine_k10_vec4_sparse_hits` 仅对实际命中显存的槽位做向量化归约并直接清零未命中行，省去每步 48 次 `cudaMemsetAsync` 与 >50% 无效显存读；
   - **48:52 容量加权 Bresenham 专家切分**（`src/program/generate.cpp`）：按 GPU0/GPU1 真实剩余显存比例精准交错分配高热专家，把 Top-9,807 专家无缝装满双卡，消除 50:50 切分在 GPU0 满载后造成的中间热专家断层；
   - **Ampere `sm_86` 硬件 `__nanosleep(64)` P2P 屏障 + `K=1` Warp 对齐**（`src/kernels/cuda/tp_comm.cu`）：等待对端推送时用 SM 硬件休眠替代紧密空转，在 240W 功耗墙下把功耗预算留给有效算力；小窗口 `T=1..3` 用 `320/640/960` 线程替代 `1024`；
   - **Zen 2 AVX2 软件预取 + PCIe ACS 关闭 + `/dev/cpu_dma_latency=0` + `DF2_HEAD_Q4` + `user_args --spec 8 --draft-conf 0.50`**：端到端压低 CPU 专家访存、PCIe 路由与投机验证长尾延迟。
2. **原生 256 MiB BAR1 直接 PCIe 4.0 P2P + VMM 选择性映射 + 1-Hop Push-Write**（`src/kernels/cuda/tp_comm.cu` + `patches/nvidia-open-595.99.02-dynbar1-p2p.patch`）：
   - 仅将 `TpComm` 所需的 **36 MiB 连续通信块**通过 CUDA Driver VMM API（`dlopen("libcuda.so.1")` 零链接依赖）映射进对端 256 MiB BAR1；
   - 全部采用单向 Push-Write（`st.global.wt`）+ 本地 760 GB/s VRAM 读取，规避比写入慢 3× 的 PCIe P2P 远端读。
3. **`72 × 1 GiB` 静态 HugePages 专家竞技场**（`src/core/pinned.cu`、`src/core/expert_source.cpp`）：71.73 GiB 专家池仅占 72 条 TLB 表项（100% 装入 Zen 2 L2 dTLB），彻底消除 Swap 与 `khugepaged` 卡顿。
4. **深度微核与长上下文套件（P0–P4）**：`T=16` 单组 `G=1` 全融合 + 5 桶长上下文 `DraftPolicy`（P0）、Prefill 路由引导 `13.5 ms` 热专家抢跑预热（P1）、寄存器缓存去重读 + `K` 路并发 PCIe MLP AllReduce（P2）、Ampere `sm_86` (`m16n8k32`) INT8 Tensor Core（P4）。
5. **双级混合草稿（256K 持久化后缀树 + DFlash2 BF16 TC）、TP=2 多对话 RAM 泊车与 14.3× 并行内存 PLE**：全面兼顾高吞吐、低 TTFT 与多标签 Agent 复用。

---

## 五、上游致谢与开源协议

- 引擎主体派生自 [Niko1221/Strata](https://github.com/Niko1221/Strata)，遵循 **MIT License**（见 [LICENSE](LICENSE)）。
- 动态 BAR1 P2P 内核驱动补丁（Dual MIT/GPLv2，见 [patches/nvidia-open-595.99.02-dynbar1-p2p.patch](patches/nvidia-open-595.99.02-dynbar1-p2p.patch)）站在以下开源先驱的肩膀上构建并移植至 `595.99.02`：
  - [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules)
  - [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules)（消费卡 BAR1 P2P 开创者）
  - [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)（新驱动与跨代 P2P 移植）
  - [Duanyll/open-gpu-kernel-modules](https://github.com/Duanyll/open-gpu-kernel-modules)（Method 3 动态 BAR1 逐分配窗口）
  - [dingiv](https://github.com/dingiv/open-gpu-kernel-modules) & [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p)（RTX 3080 20GB `align3` 2 MiB 对齐规律 v3 与生命周期加固）
