# strata-ampere

<p align="center">
  <b>面向消费级双卡 Ampere（<code>sm_86</code>）MoE 大模型推理的 <a href="https://github.com/Niko1221/Strata">Strata</a> 严格无损高吞吐 TP=2 + EP=2 优化分支</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.en.md"><img src="https://img.shields.io/badge/Language-English-00b894.svg" alt="English"></a>
</p>

`strata-ampere` 基于 [Strata](https://github.com/Niko1221/Strata)（`0.1.60.1`）深度工程优化，目标是在**无 NVLink、无 PCIe P2P** 的消费级双卡 Ampere（`2× 魔改 RTX 3080 20 GB`）+ 四通道 DDR4 混合推理平台上，以 **100% 严格无损精度** 全面压榨 MoE 大模型的 **Decode 与 Prefill 吞吐**。

> **口径承诺**：本文所有核心速度均采用**多轮中位数或均速**（绝不拿单次偶发峰值做主指标），所有实验均在 `temperature=0`、零降量化、`STRATA_ROUTE_TAIL_SKIP=0`（不丢弃任何冷门专家）、`BF16` 跨卡 AllReduce 的严格无损条件下实测复现。

---

## 一、核心优化实测速度（中位数 / 均速）

### 1. Decode 生成速度（6 场景严格无损中位数，`REPS=3`）

在默认 `320W` 频率下，基于 `Qwen3.8-Flash-Next-UD-Q4_K_XL`（`MTP-6` 草稿模型，3 轮中位数）：

| 测试场景 | 原始基线（切层流水线） | 早期 TP=2 里程碑 | **`strata-ampere` 优化后中位数（`srv_decode`）** | 相对基线提升 |
|---|---:|---:|---:|---:|
| `ctx512` | `91.6 tok/s` | `103.2 tok/s` | **`155.2 tok/s`**（`154.9`） | `+69.4%` |
| `ctx2048` | `97.2 tok/s` | `118.5 tok/s` | **`168.4 tok/s`**（`168.0`） | `+73.3%` |
| `ctx8192` | `91.9 tok/s` | `115.9 tok/s` | **`169.1 tok/s`**（`168.7`） | `+84.0%` |
| `ctx32768` | `101.1 tok/s` | `111.6 tok/s` | **`171.9 tok/s`**（`171.3`） | `+70.0%` |
| `creative`（中文写作） | `77.4 tok/s` | `93.1 tok/s` | **`131.5 tok/s`**（`131.3`） | `+69.9%` |
| `code`（代码生成） | `89.3 tok/s` | `123.8 tok/s` | **`162.0 tok/s`**（`161.5`） | `+81.4%` |
| **6 场景综合中位均值** | **`91.42 tok/s`** | **`111.02 tok/s`** | **`159.68 tok/s`**（`159.28`） | **`+74.7%`** |

### 2. 草稿模型切换与最新 `DFlash2` + `240W` 无头显存超频实测

同一份二进制无需重编即可切换 `DFlash2 Q4_0`（并已集成 `sm_80+` BF16 Tensor Core 注意力）：

| 场景 / 调优阶段（均为多轮均速） | 优化前 | 优化后（线上实测） | 变化 | 说明 |
|---|---:|---:|---:|---|
| **6 场景均速**（`MTP-6` → `DFlash2 Q4_0`） | `153.16 tok/s` | **`157.45 tok/s`** | `+2.80%` | `code 171.42` / `math 166.87` / `reasoning 197.95` |
| **多步推理 Decode**（开启 DFlash2 BF16 TC） | `148.30 tok/s` | **`154.61 tok/s`** | **`+4.25%`** | 冷启动 + 热缓存 2 轮均值；16K 热缓存同步 `+4.06%` |
| **代码 Decode**（无头超频 + `240W` 帕累托档） | `189.42 tok/s`（默认 320W） | **`207.52 tok/s`**（2 轮均值） | **`+9.6%`** | **功耗直降 80～93 W，满载仅 `41℃`，能效比 `+19.3%`** |
| **代码 Decode**（无头超频 + `320W` 满血档） | `189.42 tok/s`（默认 320W） | **`209.80 tok/s`**（2 轮均值） | **`+10.8%`** | 显存实频 `10,151 MHz`（GDDR6X ECC 临界点内） |
| **代码重构 / Agent JSON / 跨请求**（双级混合草稿 `T=16` + 持久化后缀树） | `185.46 tok/s`（4 场景热态均值） | **`229.98 tok/s`**（3 轮含冷态均值 `206.08`） | **`+24.0%`** | Agent JSON `237.5 → 290.4 tok/s` (`+22.3%`)，Python 重构 `196.0 → 215.9 tok/s` (`+10.1%`)，贪婪 SHA 逐位一致 |

### 3. Prefill 提示词处理速度（严格无损，Decode 零退化）

| 上下文长度 / 测试项 | 优化前 | 优化后（线上实测） | 提升幅度 | 备注 |
|---|---:|---:|---:|---|
| **5K 端到端 HTTP TTFT**（`4,908 tok`） | `~2,086 tok/s` | **`2,231.5 tok/s`** | **`+7.0%`** | 异步预锁页 CPU 专家池 + 并行 PLE |
| **10K 端到端 HTTP TTFT**（`9,758 tok`） | `~2,500 tok/s` | **`2,842.8 tok/s`** | **`+13.7%`** | Greedy Hash 100% 一致 |
| **31K 端到端 HTTP TTFT**（`30,648 tok`） | `~3,483 tok/s` | **`3,696 ~ 3,788 tok/s`** | **`+8.7%`** | 单块满载（240W 档 `3,696` / 320W 档 `3,788`） |
| **40K 端到端 HTTP TTFT**（`39,551 tok`） | `3,142.9 tok/s` | **`3,690.9 tok/s`** | **`+17.4%`** | 并行 PLE (`+10.3%`) + `STRATA_PREFILL_EQUAL=1` 均衡分块 (`+8.46%`) |
| **32K 单块引擎内部吞吐**（`32,768 tok`） | `3,823 tok/s`（`8,570 ms`） | **`4,267 tok/s`**（`7,680 ms`） | **`+11.6%`** | PLE 耗时由 `945 ms → 66 ms`（**14.3× 提速**） |

---

## 二、完整硬件与软件环境（公开透明，欢迎复现）

为避免对测试结果的任何质疑，以下为本次所有数据对应的完整宿主机配置：

| 组件 | 实测配置详情 |
|---|---|
| **GPU** | **2× 魔改 RTX 3080 20 GB**（Ampere `sm_86`，每卡 68 SMs / 20 GB GDDR6X，合计 40 GB 显存） |
| **多卡互联** | **无 NVLink，无 PCIe P2P**；双卡直连 CPU PCIe Gen4 x16（`nvidia-smi topo -m` 拓扑为 `NODE`，跨 CPU Host Bridge，当前 `BAR1 = 256 MiB`） |
| **GPU 频率与功耗** | 默认 `320W`（CUDA P2 状态下显存会被降频至 `9,251 MHz`）；线上已启用无头 NVML 帕累托档：**单卡 `240W` + 显存 `10,151 MHz` (`+1800 eff`) + 核心 `2,100 MHz` (`+120`)**，满载温度 **`41℃`** |
| **CPU** | **AMD Ryzen Threadripper 3960X**（24 核 48 线程，8 CCX，Linux `performance` 调频） |
| **系统内存** | **128 GB 四通道 DDR4-3200**（CPU 专家池实测有效吞吐 ~77.8 GB/s） |
| **系统与驱动** | Linux Kernel `6.8.0-138-generic`，NVIDIA Driver **`595.99.02`**，CUDA Toolkit **`13.1`** |
| **测试模型与权重** | **`Qwen3.8-Flash-Next-UD-Q4_K_XL`**（48 层、每层 512 个路由专家、总计 24,576 专家、256K 上下文；40 GB 总显存常驻约 43% 热专家，其余冷专家按需经 PCIe/CPU 混合执行） |
| **关键运行参数** | `--kv int8 --kv-resident 32768 --ple-io ram --prompt-cache 6 --pool-workers 15`、`STRATA_TP2=1`、`STRATA_TP_AR_MODE=bf16`、`STRATA_ROUTE_TAIL_SKIP=0` |

---

## 三、核心优化原理与工作拆解（100% 严格无损精度）

### 严格无损铁律（我们绝不做的事）
所有速度提升**均来自底层架构、算子微核、通信调度与硬件频率/能效调优，绝不以牺牲模型精度换跑分**：
- **零降量化**：保持原始 `UD-Q4_K_XL` 权重精度，不降位宽；
- **零专家丢弃**：强制 `STRATA_ROUTE_TAIL_SKIP=0`，绝不跳过任何低权重路由专家；
- **拒绝有损通信**：跨卡 AllReduce 全程保持 **`BF16`**，明确禁用 `STRATA_TP_AR_MODE=q8`（8-bit 量化 AllReduce）；
- **完整精度与一致性校验**：
  - **19/19 项核心 CUDA/CPU Parity 与微基准回归全过（0 失败）**；
  - **超长上下文大海捞针测试**：~150K 字符 / ~87K token（深度 0.5）**3/3 精确命中**；
  - **严格 A/B 淘汰纪律**：所有实测导致精度偏差或负优化的方案均当场回滚，包括 `STRATA_NV_CHUNK=16`（31K `-8.0%` / Decode `-7.2%`）、`STRATA_DEQUANT_SIDE + PLE_COMPACT`（Decode `-7.8%`）、`gr_write_norm_rs` 单 Block 复用（寄存器溢出反慢 `+283 ms`）、上游 `iq_kernels.cu` Q4_K/Q5_K 融合核（覆盖原有 `sm_86` `s27` 核导致 Decode `-7.0%`）以及显存 `+2200 MHz`（触发 GDDR6X 静默 ECC 重传掉速 `-7.4%`）。

---

### 1. Decode 优化原理：打破非 P2P 双卡通信墙与 Ampere 微核瓶颈
- **非 P2P 真 `TP=2 + EP=2` 全链路重构**（`src/kernels/cuda/tp_comm.cu`）：
  - 全 48 层注意力 / GDN / QSA / 共享 MLP / LM Head 50/50 Tensor Parallel + 24,576 路由专家 50/50 互不重叠 Expert Parallel，双卡同时干活。
  - 针对消费级 3080 屏蔽 PCIe P2P 的限制，手写**锁页主机内存（Mapped Pinned Host Memory）环形缓冲 + 原子 `step` 零休眠自旋同步（Zero-Sleep Barrier）**，移除 `ld.acquire.sys` 轮询中的 `__nanosleep(40)`，彻底绕过 CPU 调度与同步延迟，96 次 AllReduce 实测节省 `~80–91 us`。
- **Ampere `sm_86`（68 SMs）手调 CUDA 微核**：
  - 定制 `Q8_0` / `Q4_0` 专用微核（`q8_sm86_fast.cuh`、`q4_sm86.cuh`），重调 `RPB/TPR` 分块并支持 `native_mmvq_pair`。
  - 针对 RTX 3080 的 68 SMs 重新实测标定 `kIlRows_86` 调度表（替代原 RTX A4000 参数，如 `Q4_K 6144×2560 T=2` 从 `24.8 us → 16.0 us`）。
  - 36 层 GDN Split 补齐树解码支持（`STRATA_GDN_SPLIT=1`）；`S27` 常驻专家核加入 `__ldg` 只读缓存加载。
- **DFlash2 `sm_80+` BF16 Tensor Core 注意力**（`src/kernels/cuda/dflash2.cu`）：
  - 移植基于 PTX `mma.sync.aligned.m16n8k16` 的 BF16 Tensor Core 注意力，保留精确 FP32 在线 Softmax，多步推理 Decode 再涨 `+4.25%`、16K 长上下文热缓存再涨 `+4.06%`。
- **代码/Agent 零算力持久化后缀树 + DFlash2 双级混合草稿（16-Token 长窗口）**（`include/strata/spec/rolling_suffix.hpp`、`src/core/dflash2.cpp`、`src/core/verify.cpp`、`src/program/generate.cpp`）：
  - 解除 `Dflash2Drafter::draft_launch` 对单轮提交行数 `<= 8` 的硬上限（`n_new > 8` 时自动分块写入 KV 环与 ring log 后直接触发块草稿），并将 `Verifier::prepare(T)` 升级为 TP=2 双卡协同校验，彻底打通双卡 + DFlash2 下的 **`--lookup-max-t 16 --lookup-sizes 8,12,16 --lookup-chain 6`** 与 `STRATA_DF2_LOOKUP_SKIP=6`（强匹配时跳过 DFlash2 神经网络草稿直接走零算力后缀查表）。
  - 新增跨请求持久化滚动后缀树 `RollingSuffixSource`（`STRATA_SUFFIX_CACHE=1`，默认 256K Token 有界自动压缩，~10 MiB 内存），同时服务主后缀查表与 `--lookup-chain` 接龙，并自动过滤多模态 `<|image_pad|>` 占位符防污染。实测 4 场景热态均速 **`185.46 → 229.98 tok/s（+24.0%）`**，Agent JSON 改写达 **`290.40 tok/s（+22.3%）`**、代码重构达 **`215.85 tok/s（+10.1%）`**，全新未见 Prompt 零退化、贪婪解码 SHA 100% 逐位一致。
- **TP=2 双卡多对话内存泊车（`ConversationCache`）**（`src/program/generate.cpp`）：
  - 解除原版 `STRATA_TP2=1` 下硬编码禁用多对话缓存（`o.conversation_cache_mib = 0` 与 `park_current` 跳过 `tp1`）的限制，将 GPU 1（`tp1->ss` 与 `c.tp1_part` 检查点）完整接入 `ConversationCache` 增量泊车与回滚恢复链路。
  - 在真实双标签页 Coding Agent（Tab A 12.6K ⇄ Tab B 6.8K）切换实测中，切回 Tab A 的缓存复用率从 **`0% → 99.7%`（12,648 tok，仅 `55 ms` 内存恢复）**，TTFT 从 **`4,869.8 ms → 499.3 ms`（9.75× 提速）**，且完整支持泊车对话内的编辑/重算检查点回滚（`57.4 ms` 精确恢复）。
- **CPU 专家池 CCX 绑核对齐**：
  - `--pool-workers 15` 跨 Threadripper 8 个 CCX 绑核（`15 worker + 1 host` = 16 活跃线程），吃满四通道 DDR4-3200 带宽且避免跨 CCX 争抢。

### 2. Prefill 优化原理：用 Profiler 揪出隐形软件堵点
- **14.3× 并行内存 PLE 查表**（`src/kernels/ngram.cpp`）：
  - 通过 `STRATA_PREFILL_TIMING=1` 发现原版 `--ple-io ram` 在单线程中对 524,288 行狂刷数十万次无意义的 `madvise()` 系统调用，在 GPU 0 吃掉 `945 ms` 并连累 GPU 1 在 AllReduce 栅栏空等 `+518 ms`。
  - 改为最多 8 线程并行 `PleTable::gather_batch` 并在内存已锁页/预热时跳过 `madvise()`，PLE 耗时从 **`945 ms → 66 ms`（占比仅 `0.9%`）**，32K 单块吞吐直冲 **`4,267 tok/s`**（已逼近 `96× BF16 PCIe Gen4 AllReduce` 16.1 GB 物理带宽墙）。
- **异步预锁页 CPU 专家池协作**（`src/prefill/prefill.cpp`、`src/kernels/cpu/pool.cpp`）：
  - 预锁页 `cpu_x / cpu_rows / cpu_ones` + 独立异步 `cpu_x_stream` D2H 拷贝流 + `ExpertPool::run_range` 多线程并行量化，并调优 `STRATA_PREFILL_CPU_SHARE_NT=32, MAX=6144`，4K～5K Prefill 提速 `+8.9%`。
- **多块均衡切分（`STRATA_PREFILL_EQUAL=1`）**：
  - 解决 40K 长提示词被切成 `32K + 7K` 导致小尾块重复扫 PCIe 冷专家掉速的问题，改为 `20K + 20K` 均衡双块，40K 端到端 Prefill 再涨 **`+8.46%`**（直接追平 31K 单块速度）。

### 3. Linux 底层无头调优：NVML 显存超频、240W 能效与 PCIe `MaxReadReq=4096B`
- **解除 CUDA P2 显存降频 + 240W 帕累托能效**：发现 NVIDIA 驱动默认 CUDA P2 状态会将 RTX 3080 GDDR6X 降频 `-250 MHz`（`9,501 → 9,251 MHz`）；通过开源脚本 [scripts/ampere-oc-power.py](scripts/ampere-oc-power.py) 锁定 Micron GDDR6X ECC 峰顶（`+1800` 为峰顶，`+2200` 触发静默 ECC 重传掉速），配合单卡 **`240W`** 功耗墙保留 **98.9% 极限 Decode 速度（`207.52 tok/s`）**，Prefill 功耗直降 **93 W**、满载温度仅 **`41℃`**、能效比提升 **`+19.3%`**。
- **PCIe `MaxReadReq 512B → 4096B` + `THP defrag=defer+madvise`**：绝大多数主板默认将消费级 GPU 的 PCIe `MaxReadRequestSize` 限制在 `512 bytes`，导致每次冷专家 DMA 读取被切碎成 8 倍事务包。热调至 **`4096 bytes`**（`setpci CAP_EXP+0x08.w`）并将内核透明大页整理设为异步 `defer+madvise` 后，**零额外功耗下 `decode_code` 均速再涨 `+5.07%`、`decode_reason` 再涨 `+3.38%`**，所有 Linux 多卡用户均可一键复现！

---

## 四、编译、快速启动与无头超频工具

### 1. 编译（`sm_86`）
依赖 CUDA Toolkit 13.1（或兼容 `sm_86` 的 12.x+）、CMake、Ninja 与 C++20 编译器：

```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)
```

### 2. 草稿模型选型与配置启动
同一份 `build86/strata` 二进制无需重编即可切换草稿模型：
- **偏代码 / 数学 / 逻辑推理（推荐）**：使用 `DFlash2 Q4_0` 模板 [examples/strata-2x3080-20g-dflash2.example.json](examples/strata-2x3080-20g-dflash2.example.json)（内置 `--draft-conf 0.35`、`--vram-reserve-mib 1050`、`STRATA_PREFILL_EQUAL=1`、`STRATA_PREFILL_CPU_SHARE_NT=32`）。
- **偏中文创意写作 / 全能均衡**：使用 `MTP-6` 模板 [examples/strata-2x3080-20g.example.json](examples/strata-2x3080-20g.example.json)（高随机性创作快 `~12.5 tok/s`）。

复制模板并替换模型路径与 `CHANGE_ME_API_KEY` 后启动：
```bash
python3 serve/server.py --config /path/to/strata-2x3080-20g-dflash2.example.json
```

### 3. 一键无头显存超频、PCIe `4096B` 与 240W 帕累托能效（无需重启引擎）
```bash
# 一键应用 240W + GDDR6X 10,151 MHz + PCIe MaxReadReq=4096B + THP defer+madvise
sudo python3 scripts/ampere-oc-power.py --mem-offset 1800 --gpc-offset 120 --power-limit 240 --pcie-mrrs 4096 --thp-defrag defer+madvise

# 一键恢复出厂默认频率与功耗墙
sudo python3 scripts/ampere-oc-power.py --reset
```

### 4. 一键采集领域专属 MoE 专家热力图（`scripts/profile-experts.py`）
针对半显存常驻架构，内置 `coding` / `writing` / `research` / `balanced` 四套语料（亦支持 `--prompts-file` 传入你自己的日志），结合 RRF 贝叶斯先验融合（`--prior-blend 0.45`）将高频领域专家精准晋升进显存（实测 Coding 冷启动显存命中率 `82.13% → 84.00%`，Python/C++ 未见任务冷启动最高 `+13.3%`）：
```bash
sudo python3 scripts/profile-experts.py \
  --config /path/to/strata-config.json \
  --unit strata@iq3_s \
  --domain coding \
  --out /path/to/expert-profile-coding.bin \
  --activate
```

---

## 五、上游致谢与开源协议

`strata-ampere` 派生自 [Niko1221/Strata](https://github.com/Niko1221/Strata)，遵循原仓库的 **MIT License**（见 [LICENSE](LICENSE)）。基础引擎架构、Pack 格式、KV Streaming 与验证框架均归功于 Niko1221 及 Strata 社区贡献者。
