# Deployment record: TP=2 + DFlash2 on 2× RTX 3080 20 GB (PCIe Gen3 x16)

How this box was built and what it produces, end to end. The *differences* from the reference
platform and the dead ends are in [TP2-GEN3-NOTES.md](TP2-GEN3-NOTES.md); this file is the build
and run record.

## 1. Hardware and software

| Item | Value |
|---|---|
| GPU | 2× modded RTX 3080 **20 GB** (GA102, 320-bit), `sm_86`, ~760 GB/s, capped at 300 W |
| CPU | Intel i7-6950X (10C/20T), ~61 GB/s measured read bandwidth |
| RAM | 128 GB DDR4 |
| Link | PCIe **Gen3 x16** |
| P2P | `nvidia-smi topo -p2p r` OK both directions; DMA 10.30 GB/s one way, 19.40 both, NCCL all_reduce 9.54 |
| BAR1 | **32 GiB per card** (dynamic BAR1); peer BAR1 mapped by the engine through CUDA Driver VMM, so no kernel patch is required once BAR1 is large |
| Driver | 615.71.09, **open kernel module**, locally compiled (gcc 15.2) |
| CUDA | **13.3** (a 13.1 install also exists; the build must point at 13.3 explicitly) |
| Kernel cmdline | `pcie_aspm=off pcie_port_pm=off nvme_core.default_ps_max_latency_us=0 gpu_unlock.targets=all` |
| Storage | weights on NVMe — with a large page cache the cold path is disk-bound otherwise |

## 2. Prerequisites we actually check

```bash
journalctl -k -b | grep -c 'gpu_unlock: unlocked'      # must be 2, one per card
nvidia-smi --query-gpu=pcie.link.gen.current,width.current --format=csv,noheader
nvidia-smi topo -p2p r                                  # GPU0 <-> GPU1 must be OK
```

Idle bands down to Gen1 — that is power saving, not a fault. The engine prints its own verdict at
startup: `strata tp_comm: NVLink/P2P enabled (vmm-push, decode=p2p, CUDA0 <-> CUDA1)`.

## 3. Build

```bash
cmake -B build86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_WINDOW_MAX_T=16 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build86 --target strata -j 6
```

`-j` is a memory decision, not a CPU one: `ptxas` peaks at 2.4–2.5 GiB RSS per process and
oversubscribing fails as a zram deadlock (SSH completes the handshake but never sends a banner).
Rule: `-j ≈ available_GB / 3`. Full build on this box: **~1.5 h**. Touching
`src/program/generate.cpp` alone: one TU plus relink, **~100 s**.

## 4. Artifacts

| Path | Contents |
|---|---|
| `packs/ud-iq4_xs/` | routed-expert pack (`dense.bin`, `conversions.json`, `index.txt`, `tokenizer/`) |
| `flash-next/UD-IQ4_XS/*.gguf` | 3 shards, ~93.7 GB total; **the PLE table is in shard 2** |
| `dflash2/*.q4_0-selq4.pack` | DFlash2 drafter, ~340 MB |
| `mtp/rt/draft_vocab.bin` | vocabulary only; no MTP layer is loaded |

## 5. Config

Baseline is `examples/strata-2x3080-20g-dflash2.example.json`. Final `args` on this box:

| Flag | Value |
|---|---|
| `--pack` | `packs/ud-iq4_xs` |
| `--native` | `UD-IQ4_XS-00001-of-00003.gguf` |
| `--ple-gguf` | `UD-IQ4_XS-00002-of-00003.gguf` |
| `--expert-cache` / `--prefill` | `auto` / `auto` |
| `--max-context` | `524288` |
| `--kv` / `--kv-resident` | `int8` / `32768` |
| `--rope-scaling` / `--rope-scale` | `yarn` / `2` |
| `--vram-reserve-mib` | `500` |
| `--pcie-frac` | `0` |
| `--draft` / `--draft-fallback` | `dflash2` / `none` |
| `--spec` / `--draft-cap` | `4` / `4` |
| `--conversation-cache-mib` / `-slots` | `32768` / `8` |

Server side: `host 0.0.0.0`, `port` from the service, `model_name`, `api_key`, `tokenizer`,
`layer_split: auto`, and the environment variable **`STRATA_TP2=1`**.

Why those four values differ from the example, and what each one was measured against, is in
[TP2-GEN3-NOTES.md](TP2-GEN3-NOTES.md).

## 6. Launch

A systemd *user* unit, launching the server module by absolute path:

```ini
[Service]
Environment=STRATA_TP2=1
ExecStart=/path/to/venv/bin/python /path/to/serve/server.py \
          --engine strata --config /path/to/config-tp2-df2.json --host 0.0.0.0 --port 8081
Restart=on-failure
TimeoutStartSec=900
LimitMEMLOCK=8388608
```

Do not start it as `python -m serve.server`: that imports a different `serve/` package from the
working directory, which silently switches the drafter to MTP. The startup line must read
`drafting: DFlash2 alone`.

Cold start takes about **40 s** after the weights are in the page cache; the first start after a
reboot is a few minutes.

## 7. Verify

Two startup lines carry the state:

```
strata serve: STRATA_TP2=1 ready: CUDA0 (6282 experts) + CUDA1 (6502 experts) = 12784 total resident experts
strata serve: DFlash2 drafter on (...); windows up to 4 tokens; no MTP layer is loaded
```

Then, in order: no API key → 401; with key → 200 and normal output; `/v1/models` reports
`n_ctx = 524288`; a >130 K-token cold prompt (log must say `0 reused`) confirms the long-context
path.

## 8. Results on this box

| Workload | prefill | decode | draft acceptance |
|---|---|---|---|
| 117 K token cold context (mean of 3) | **3,321 tok/s** | **143.6 tok/s** | 44–67 % |
| 20 K token | 2,830–3,000 | 112–117 | 70–81 % |
| short chat | — | 65–93 | — |

| Resource | Value |
|---|---|
| VRAM | **19.98 / 20.0 GB** per card |
| Resident experts | **12,784** (6,282 CUDA0 + 6,502 CUDA1) |
| Engine RSS (120 K-token session) | RssAnon **58.5 GiB** = experts ~55.4 + PLE ~5.8 + buffers |
| cgroup `MemoryCurrent` | 105 GiB (includes reclaimable page cache) |
| Cold start | ~40 s |
| CPU power | 31 W idle, 66 W during prefill, **191 W during decode** (~9.5 cores) |

Decode is the CPU-heavy phase, not prefill: at 11.7 万 tokens prefill keeps one core busy feeding
chunks, while decode runs the expert kernel across ~9.5 cores.

## 9. Coexistence

TP=2 saturates both cards, so it is mutually exclusive with the layer-split service on the same
box — stop one before starting the other.
