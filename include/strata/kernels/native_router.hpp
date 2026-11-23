#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// Fused top-10 softmax router + conditional doorbell publish (for 512 experts, 1..8 tokens, no bias/prior/rs).
void native_router_top10_bell(const float* logits, int32_t* ids, float* weights,
                              const float* x, const int32_t* d_res, int n_embd, int n_tok,
                              float* x_out, int32_t* ids_out, uint32_t* d_seq, uint32_t value,
                              bool per_tok, void* stream);
/// The same contract for a model family's (experts, top-k): 512 / 10 (Flash-Next: the two calls above, unchanged) and
/// 256 / 8 (Qwen3.6-35B-A3B).  logits [n_tok, n_expert], ids / weights [n_tok, k].  CUDA / HIP.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
/// STRATA_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream);
/// Fused BF16 router projection (n_in=2560 -> 512) + optional shared-expert gate (2560->1) +
/// top-10 routing + optional host doorbell (x->x_out, ids->ids_out, d_seq=ring) + TP2 device residency plan.
/// Zero allocations; safe for CUDA graph capture (`done_ctr` must be pre-zeroed once and self-resets).
void native_router_fused_bf16_bell_plan(
    const float* x, int64_t ldx, const uint16_t* w, float* logits, int64_t ldy, int n_tok,
    const uint16_t* w_aux, float* y_aux, int64_t ldy_aux,
    float* x_out, int32_t* ids, float* weights, int32_t* ids_out, uint32_t* d_seq, uint32_t ring,
    const int32_t* res_self, const int32_t* res_peer,
    const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
    int32_t* pl, long long capx, uint32_t* skip, uint32_t* done_ctr, bool bell, void* stream);
}
