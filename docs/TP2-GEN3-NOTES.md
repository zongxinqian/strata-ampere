# What we changed on a PCIe Gen3 x16 box (2× modded RTX 3080 20 GB)

Parameter baseline is `examples/strata-2x3080-20g-dflash2.example.json`; everything below is what
we changed on top of it and what we measured. Nothing here restates the example file or the guide.

## Our machine

| Item | Value |
|---|---|
| GPU | 2× modded RTX 3080 **20 GB**, `sm_86`, 320-bit, capped at 300 W |
| Link | PCIe **Gen3 x16** (idle bands down to Gen1 — normal) |
| P2P | DMA after `cudaDeviceEnablePeerAccess`: **10.30 GB/s** one way, **19.40** both |
| BAR1 | **32 GiB per card** (dynbar1); the engine maps peer BAR1 through CUDA Driver VMM, so no kernel patch is needed once BAR1 is large |
| Driver / CUDA | 615.71.09 (open) / **CUDA 13.3**, `-DCMAKE_CUDA_ARCHITECTURES=86` |
| CPU / RAM | i7-6950X, 128 GB DDR4 (~61 GB/s read) |

## Four deliberate deviations from the example config

**1. `--spec 4` / `--draft-cap 4`** — the DFlash2 drafter caps its window at 4 regardless
(`--draft dflash2: windows capped at 4 tokens`). Measured 4 / 6 / 8: identical. Anything above 4
is a no-op, so it is set to what actually runs.

**2. `--vram-reserve-mib 500`** — a hard floor on a 20 GB card:

| Value | Result |
|---|---|
| 400 | 28 MiB free; the driver starts evicting and decode collapses to single digits |
| **500** | ~470 MiB free, stable |
| 1000 | works, but costs expert slots for nothing |

At 500 the engine still warns `only XXX MiB of VRAM free (below 512 MiB)` — expected noise.

**3. `--conversation-cache-mib 32768 --conversation-cache-slots 8`** — the defaults (8192 MiB /
4 slots) evict a 120 K-token conversation:

```
conversation cache: parked=4 bytes=8.45 GB evictions=15
```

After eviction the next message reports `0 reused` and rebuilds 120 K tokens (~37 s) instead of
restoring in ~10.7 s. **Eviction fires on slot count**, so raising `mib` alone changes nothing.
Mean decode went 121.3 → **143.6 tok/s (+18 %)**.

**4. `UD-IQ4_XS` instead of a larger pack** — 93.7 GB leaves room for the PLE table and page cache
on a 128 GB box, at 89.55 % top-1 / KLD 0.0836. A 111.3 GB pack (92.26 % / 0.0469) does not fit
beside them. On a 96 GB box use `16384` / `6` slots and keep the weights on NVMe.

## What we measured

| Workload | prefill | decode | draft acceptance |
|---|---|---|---|
| 117 K token cold context (mean of 3) | **3,321 tok/s** | **143.6 tok/s** | 44–67 % |
| 20 K token | 2,830–3,000 | 112–117 | 70–81 % |
| short chat | — | 65–93 | — |

Both cards sit at **19.98 / 20.0 GB**, 12,784 resident experts (6,282 CUDA0 + 6,502 CUDA1), cold
start ~40 s.

These are **~15 % below the reference numbers**, consistently at both 19 K and 117 K. We attribute
that to the link generation rather than to config — TP=2 puts every layer's AllReduce on PCIe. If
you are on Gen3, budget the 15 % instead of hunting for a setting.

## TP=2 vs the layer split (same pack, same box)

| Context | TP=2 + DFlash2 | layer split + MTP | winner |
|---|---|---|---|
| **2 K prefill** | 670 tok/s | **1,501 tok/s** | layer split **2.2×** |
| 12 K prefill | ~2,700 | 2,390 | TP=2 slightly |
| 117 K prefill | **3,321** | <3,000 | TP=2 |
| decode ≤12 K | 112–164 | 100–166 | comparable |
| decode at 117 K | **143.6** | 96.3 | TP=2 **+49 %** |

Crossover sits around **5–8 K tokens**: below it the per-layer work cannot hide the AllReduce round
trip, so TP=2 on a short-chat workload is a net loss.

## Decode is not PCIe-bound here

At 128 K the intra-step budget is ~6.7 ms and inter-card traffic is ~0.0006 ms. Raising the link
from 6.5 to 9.5 GB/s changed decode by noise. The limits are VRAM bandwidth and the CPU expert
kernel, so link tuning will not buy decode on this box.

## Quantisation: the expert *type* matters, not the bit width

| Pack | `native_experts.txt` types | Result |
|---|---|---|
| Unsloth `UD-IQ4_XS` | 51 × `IQ4_XS` | works |
| a custom pack | `GSQ` + `IQ3_S` | `non-finite logits (#879)` after ~330 tokens under TP=2 |

The same custom pack runs fine **single-card**, which isolates it to the TP=2 expert kernel.
`IQ3_XXS` is different — it has a dequant kernel and a dispatch case — so the rule is "check
`native_experts.txt` for standard ggml types", not "avoid low bit widths".

## Three pitfalls we hit

**Launch `serve/server.py` by absolute path.** Running it as `python -m serve.server` imports a
different `serve/` package from the working directory, which does not know DFlash2, and the engine
silently switches drafter:

```
Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)
```

Everything still works — with the wrong drafter. The startup line must read `drafting: DFlash2 alone`.

**This fork's MTP path crashes.** `mtp draft (cond): an illegal memory access was encountered`,
across packs and with `STRATA_MTP_COND=0`, while the same MTP directory is stable on the upstream
engine. DFlash2 only.

**Do not oversubscribe the build.** `ptxas` peaks at 2.4–2.5 GiB RSS; going over fails as a zram
deadlock (SSH completes the handshake but never sends a banner). `-j ≈ available_GB / 3`. Rebuilding
after touching `src/program/generate.cpp` alone is one TU and ~100 s.

## Tuning attempts that produced nothing (measured, do not repeat)

| Attempt | Result |
|---|---|
| the 11-variable `PIN_BY_NEED` / `KV_PREFETCH` / `PF_*` set | prefill +2 %, **decode −16 %** (VRAM down to 126 MiB free) |
| `STRATA_PREFILL_CPU_SHARE*` variants | prefill unchanged, CPU power unchanged (66 W), decode −16 % |
| `--pipeline-windows 2` | prefill flat, **decode −34 %** (borrows ~4,800 expert slots per card) |
| `--draft-conf 0.50` | a wash (143.1 vs 143.6), more variance |
| PCIe MRRS = 4096 B (`setpci CAP_EXP+08`) | **no change**: 10.15 / 19.38 GB/s identical before and after, same binary, peer access on |

## Measuring without fooling yourself

- **P2P**: `cudaMemcpyPeerAsync` *without* `cudaDeviceEnablePeerAccess` silently bounces through
  host memory and reads **5.90 GB/s**, which looks like broken P2P. Enable peer access first.
- **Cold prefill**: change the text between runs and confirm `0 reused` in the log.
- **`prompt-lookup`** inflates acceptance on homogeneous prompts (320 tok/s reported; natural text
  gives 143).
- **Token density drifts** with length: the same Chinese paragraph is 25 tok/segment over 300
  segments but ~33 over 5,000. For an exact count, send once, read `prompt_tokens`, scale, resend.

## Vision under TP=2

Handled separately in #1 / PR #2: the peer stage never received its own image-position table or
`Prefill::embd_rows`, so image cells were decoded with identity positions.
