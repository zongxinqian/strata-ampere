# strata-ampere

<p align="center">
  <b>面向 Ampere（<code>sm_86</code>）双卡 MoE 推理的 <a href="https://github.com/Niko1221/Strata">Strata</a> 无损高吞吐 TP=2 + EP=2 魔改分支。</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="README.md">English</a>
</p>

`strata-ampere` 基于 [Strata](https://github.com/Niko1221/Strata)（`0.1.60.1`）深度优化，目标是在消费级双卡 Ampere（`sm_86`，例如 2× 魔改 RTX 3080 20 GB，跨 CPU Host Bridge 非 P2P PCIe Gen4 x16）+ 多通道 DDR4 主机 CPU 混合推理架构上，以**严格无损精度**榨干 MoE 大模型的 Decode 吞吐。

在 `Qwen3.8-Flash-Next-UD-Q4_K_XL`（48 层、每层 512 路由专家、256K 上下文、`int8` KV 且 32K 常驻显存）上，`strata-ampere` 实测 6 场景中位均值达 **`159.68 tok/s`**（相比基线 `91.42 tok/s` **+74.7%**，相比早期 TP=2 Surgery 5 `111.02 tok/s` **+43.8%**），全程保持**零降量化**、**绝不跳过冷门专家**（`STRATA_ROUTE_TAIL_SKIP=0`）与 **BF16** TP2 AllReduce 精度。

---

## 核心优化亮点

1. **非 P2P 双卡 `TP=2 + EP=2` 全链路架构**
   - 全 48 层注意力 / GDN / QSA / 共享 MLP / LM Head 50/50 Tensor Parallel + 24,576 专家 50/50 互不重叠 Expert Parallel（`STRATA_TP2=1`），完整支持 TP=2 多轮前缀缓存 Checkpoint（`--prompt-cache 6`）。
   - Prefill 采用全双工 Copy-Engine DMA BF16 `TpComm`，Decode 采用低延迟环形 Mapped-Host BF16 AllReduce；移除 `src/kernels/cuda/tp_comm.cu` 中 `ld.acquire.sys` 轮询里的 `__nanosleep(40)`（`96x AR` 实测降低 `~80–91 us`）。
   - Prefill 锁页冷专家采用 50/50 奇偶 EP 并行加载与计算，均衡双卡负载。

2. **Ampere（`sm_86`）Dense 与 MoE 专家 CUDA 微核深度优化**
   - **`Q8_0` / `Q4_0` `sm_86` 专用微核**（`src/kernels/cuda/q8_sm86_fast.cuh`、`src/kernels/cuda/q4_sm86.cuh`、`src/kernels/cuda/native_mmvq.cu`）：重调 `RPB/TPR` 分块，新增 `subwarp_kernel`、`row1_direct_kernel`、显式 `__launch_bounds__` 及 `native_mmvq_pair`（支持 `Q8_0`/`Q4_0` `ncols >= 1`）。
   - **RTX 3080（68 SMs）实测 `kIlRows_86` 表格**（`src/kernels/cuda/native_mmvq.cu`）：基于 `mmvq_il_parity --bench --emit-table` 重标 `Q4_K`、`Q6_K`、`IQ4_XS`、`Q3_K`（替换原 RTX A4000 表项；如 `Q4_K 6144x2560 T=2` 从 `24.8 us` 降至 `16.0 us`）。
   - **36 层 GDN Split 兼容树解码**（`src/kernels/cuda/verify_kernels.cu`）：为 `gdn_step_split_kernel` 补齐 `bool TREE`（`tree_fork`/`tree_parent`）分支并在 `sm_86` 默认开启 `STRATA_GDN_SPLIT=1`。
   - **`S27` 常驻专家核优化**（`src/kernels/cuda/iq_kernels.cu`）：`S27<13>::load` 加入 `__ldg` 只读缓存加载，`sm_86` 默认 `gy_gu=16`、`gy_dn=16`（保持 `gth=128`）。

3. **MTP TP=2 草稿融合与树投机策略校准**
   - `--mtp-q4 all` 草稿头 Q4_0 缩容重分配，为 GPU0 腾出 ~130 MiB 换取更多常驻热专家。
   - `uint4` 向量化 TP2 发布/等待、融合 `bump_round`、`(double)` warp 并行 `sumexp` 归约、`mtp_tp2_merge_head` 内嵌 `dvocab` 映射，并用 `native_mmvq_pair` 融合 `k_proj + v_proj`（`src/core/mtp.cpp`、`src/kernels/cuda/verify_kernels.cu`）。
   - 基于 `STRATA_SPEC_TRACE` 实测重标 `STRATA_SPEC_FIXED_TABLE=10.5,12.2,14.0,16.1,20.2,23.6`、`STRATA_SPEC_R_OP=0.130`、`STRATA_SPEC_NO_HICONF=0` 与 `--spec-min-p 0.22`（配合 `STRATA_TREE_FORK=1`）。

4. **CPU 专家池与内存带宽对齐**
   - `--pool-workers 15` 跨 8 个 CCX 绑核（`15 worker + 1 host` = 16 活跃线程），在 Threadripper 3960X 四通道 DDR4-3200 上稳定吃满 ~77.8 GB/s 带宽上限且避免跨 CCX 争抢。

---

## 实测性能（`REPS=3` 中位数）

**测试平台**：2× RTX 3080 20 GB（`sm_86`，320 W，非 P2P PCIe Gen4 x16，CUDA 13.1），AMD Threadripper 3960X（24c/48t，`performance` 调频），128 GB 四通道 DDR4-3200。  
**测试模型与口径**：`Qwen3.8-Flash-Next-UD-Q4_K_XL`（`temperature=0`、`presence_penalty=0`、`enable_thinking=False`、`--kv int8 --kv-resident 32768`、`STRATA_ROUTE_TAIL_SKIP=0`、`STRATA_TP_AR_MODE=bf16`）。

| 场景 | 基线 (11-20) | TP=2 Surgery 5 (11-21) | **`strata-ampere` 客户端 (`srv_decode`)** | 3 轮峰值 | `e2e_out_tok_s` |
|---|---:|---:|---:|---:|---:|
| `ctx512` | 91.6 | 103.2 (`103.1`) | **155.2** (`154.9`) | **160.2** | 101.6 |
| `ctx2048` | 97.2 | 118.5 (`118.3`) | **168.4** (`168.0`) | **171.7** | 87.9 |
| `ctx8192` | 91.9 | 115.9 (`115.8`) | **169.1** (`168.7`) | **169.7** | 56.3 |
| `ctx32768` | 101.1 | 111.6 (`111.4`) | **171.9** (`171.3`) | **176.4** | 25.9（`3,407.1` srv prefill） |
| `creative` | 77.4 | 93.1 (`93.0`) | **131.5** (`131.3`) | 135.3 | 114.0 |
| `code` | 89.3 | 123.8 (`123.6`) | **162.0** (`161.5`) | 165.0* | 136.4 |
| **6 项综合均值** | **91.42** | **111.02** (`110.87`) | **`159.68 tok/s`** (`159.28`) | — | **`87.02 tok/s`** |

> \*同版二进制在 `STRATA_SPEC_R_OP=0.120` 下 `code` 曾录得单次 `189.1 tok/s` 峰值。`bench.py` 中 `creative` 紧接 `code` 交替执行，瞬时专家缓存命中率由 `84.1%` 降至 `78.4–79.1%`。

### 精度与稳定性验证

- **Parity 回归套件**：19/19 项核心 CUDA/CPU parity 与微基准 `0` 失败（含 `verify_parity`、`native_multi_parity`、`gdn_parity` 默认/`SPLIT=0`/`SPLIT=1`、`router_top10_parity`、`mmvq_il_parity`、`sampler_parity`、`gr_parity`、`native_expert_parity`、`q8_gemv_bench`、`router_chain_bench`、`mmvf_rows_parity`、`spec_proposal_parity`、`spec_verify_parity`、`qsa_parity`、`shared_expert_parity`）。`fused_gr_bench` 中实际生产路径 `Q8`/`Q8-PK` 对齐正常（`rel rms mixed 0.0012–0.0016`），仅预存历史 BF16 `apply 1` 存在 12 例已知差异（`apply 0` 逐比特一致）。
- **超长上下文大海捞针**：~150K 字符 / ~87K prompt tokens（深度 0.5）**3/3 精确命中**。
- **显存与告警**：全过程 `0` 条 `#2065` / `min_free` 告警，压测后 GPU0 余量 `536 MiB >= 512 MiB`。

---

## 编译指南（`sm_86`）

依赖 CUDA Toolkit 13.1（或兼容 `sm_86` 的 12.x+）、CMake、Ninja 与 C++20 编译器：

```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build86 --target strata -j $(nproc)
```

---

## 快速启动与草稿模型选型

同一份 `build86/strata` 二进制无需重新编译，即可通过配置文件在 **MTP-6** 与 **DFlash2 Q4_0** 两种草稿模型间自由切换：

### 草稿模型对比：`MTP-6` vs `DFlash2 Q4_0`

在 `2× RTX 3080 20 GB`（`TP=2 + EP=2`、`Qwen3.8-Flash-Next-UD-Q4_K_XL`、`temperature=0`、`STRATA_ROUTE_TAIL_SKIP=0`）上的同口径 6 场景端到端实测对比：

| 场景 | `MTP-6`（均衡通用） | `DFlash2 Q4_0`（`conf=0.35`，代码/推理向） | 变化 |
|---|---:|---:|---:|
| `code`（Python 算法与测试） | 156.76 tok/s | **171.42 tok/s** | **+9.4%** |
| `math`（数论与递推证明） | 158.14 tok/s | **166.87 tok/s** | **+5.5%** |
| `reasoning`（多步逻辑推理） | 183.94 tok/s | **197.95 tok/s** | **+7.6%** |
| `en_tech`（CUDA 架构解析） | 142.48 tok/s | **144.98 tok/s** | +1.8% |
| `zh_qa`（中文系统技术问答） | **142.54 tok/s** | 140.87 tok/s | -1.2% |
| `creative`（中文微科幻创作） | **135.11 tok/s** | 122.64 tok/s | -9.2% |
| **6 场景均值（decode / e2e）** | `153.16 / 137.48 tok/s` | **`157.45 / 141.01 tok/s`** | **+2.80%** |

- **偏代码 / 数学 / 逻辑推理推荐选 `DFlash2 Q4_0`**：使用 [examples/strata-2x3080-20g-dflash2.example.json](examples/strata-2x3080-20g-dflash2.example.json)（建议保持 `--draft-conf 0.35` 与 `--vram-reserve-mib 1050`；在半驻留显存架构下把 `--draft-conf` 调低会引入更多冷专家 DDR4 读取，反而变慢）。
- **偏中文写作 / 全场景均衡推荐选 `MTP-6`**：使用 [examples/strata-2x3080-20g.example.json](examples/strata-2x3080-20g.example.json)（`creative` 高随机性创作快 `~12.5 tok/s`）。

### 启动步骤

1. 按需复制对应的脱敏配置模板（`examples/strata-2x3080-20g-dflash2.example.json` 或 `examples/strata-2x3080-20g.example.json`），替换其中的 `/path/to/...` 模型/引擎路径以及 `CHANGE_ME_API_KEY`。
2. 按你的 CPU 物理核拓扑调整 `--pool-workers` 与 `--pool-cores`（模板默认针对 24 核 Threadripper 3960X 跨 8 CCX 绑定 15 个 worker）。
3. 使用 [examples/start-strata-ampere.sh](examples/start-strata-ampere.sh) 或直接运行：

```bash
python3 serve/server.py --config /path/to/strata-2x3080-20g.json
```

---

## 上游致谢与开源协议

`strata-ampere` 派生自 [Niko1221/Strata](https://github.com/Niko1221/Strata)，继续遵循原仓库的 **MIT License**（见 [LICENSE](LICENSE)）。基础引擎架构、Pack 格式、KV Streaming 与验证框架均归功于 Niko1221 及 Strata 社区贡献者。
