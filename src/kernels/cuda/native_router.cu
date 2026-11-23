// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// The same contract for other (experts, top-k) shapes - Qwen3.6-35B-A3B's 256 / 8 - one warp per token, n_tok tokens
// (8 per block).  The 512 / 10 path keeps its own kernels above (the Flash-Next bits), this template is only used
// for the other shapes.
template <int NE, int K>
__launch_bounds__(256, 1)
__global__ void route_topk(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                           int n_tok) {
    static_assert(NE % 32 == 0 && K <= 32, "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE; ids += (size_t) tk * K; weights += (size_t) tk * K;
    const int lane = threadIdx.x;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

__launch_bounds__(256, 1)
__global__ void route_top10_bell_k(
        const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
        const float* __restrict__ x, const int32_t* __restrict__ d_res,
        int n_embd, int n_tok, float* x_out, int32_t* ids_out, uint32_t* seq, uint32_t value, bool per_tok) {
    __shared__ uint32_t tok_mask;
    if (threadIdx.y == 0 && threadIdx.x == 0) tok_mask = 0u;
    __syncthreads();
    const int tk = (int) threadIdx.y;
    const int lane = (int) threadIdx.x;
    if (tk < n_tok) {
        const float* l = logits + (size_t) tk * 512;
        int32_t* id = ids + (size_t) tk * 10;
        int32_t* id_out = ids_out + (size_t) tk * 10;
        float* w = weights + (size_t) tk * 10;
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = l[lane + i * 32];
        __syncwarp();
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
        maximum = warp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = expf(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        int sel_id = 0;
        int miss = 0;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                selected_sum += best;
            }
            if (rank == lane) { selected = best; sel_id = expert; }
        }
        if (lane < 10 && (d_res == nullptr || d_res[sel_id] < 0)) miss = 1;
        selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < 10) {
            id[lane] = sel_id;
            id_out[lane] = sel_id;
            w[lane] = selected * inverse_selected_sum;
        }
        if (__any_sync(0xffffffffu, miss) && lane == 0) atomicOr(&tok_mask, 1u << tk);
    }
    __syncthreads();
    const uint32_t raw_mask = tok_mask;
    const int tid = (int) threadIdx.y * 32 + (int) threadIdx.x;
    if (raw_mask != 0u && x != nullptr && x_out != nullptr) {
        const uint32_t full = n_tok == 32 ? 0xffffffffu : ((1u << n_tok) - 1u);
        const uint32_t mask = per_tok ? raw_mask : full;
        const bool aligned4 = ((((uintptr_t) x | (uintptr_t) x_out) & 15u) == 0) && ((n_embd & 3) == 0);
        if (mask == full) {
            const int tot = n_tok * n_embd;
            if (aligned4) {
                const int n4 = tot >> 2;
                const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
                float4* __restrict__ o4 = reinterpret_cast<float4*>(x_out);
                for (int i = tid; i < n4; i += 256) o4[i] = x4[i];
            } else {
                for (int i = tid; i < tot; i += 256) x_out[i] = x[i];
            }
        } else {
            for (int t = 0; t < n_tok; ++t) {
                if (((mask >> t) & 1u) == 0u) continue;
                const float* xt = x + (size_t) t * n_embd;
                float* ot = x_out + (size_t) t * n_embd;
                if (aligned4) {
                    const int p4 = n_embd >> 2;
                    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xt);
                    float4* __restrict__ o4 = reinterpret_cast<float4*>(ot);
                    for (int i = tid; i < p4; i += 256) o4[i] = x4[i];
                } else {
                    for (int i = tid; i < n_embd; i += 256) ot[i] = xt[i];
                }
            }
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    __syncthreads();
    if (tid == 0) {
        const uint32_t v = value != 0u ? value : (*(volatile uint32_t*) seq + 1u);
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(seq), "r"(v) : "memory");
    }
#else
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) seq = value != 0u ? value : (*(volatile uint32_t*) seq + 1u);
    }
#endif
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_multi<<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_router_topk_supported(int n_expert, int k) { return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8); }
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream) {
    if (n_expert == 512 && k == 10) {   // Flash-Next: the pinned kernels above, unchanged
        if (n_tok == 1) native_router_top10(logits, ids, weights, stream);
        else native_router_top10_multi(logits, ids, weights, n_tok, stream);
        return;
    }
    if (!stream || n_tok < 1 || !native_router_topk_supported(n_expert, k) ||
        !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("native_router_topk: a stream, a supported (experts, k) and aligned buffers");
    route_topk<256, 8><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// STRATA_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
//
// One warp per token.  This used to be one THREAD per token scanning all 512 experts (4 ranks x 512 x 12 scalar
// ops on up to 6 threads): 45 us per layer, 2.2 ms of a 19.4 ms window, which is exactly what the option saved in
// PCIe experts (~2.4 ms) - the swap cost as much as the misses it removed.  The ranks are still visited in order
// (a swap changes the set already picked), but the scan is coalesced (lane j reads f = j, j + 32, ...) and reduced
// within the warp.  The bits are unchanged: a strictly greater logit wins, the smallest expert index breaks ties,
// and the softmax is the same expression.
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                                 unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    const int lane = (int) threadIdx.x;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int32_t my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;                      // already resident: uniform across the warp
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
        my[r] = __shfl_sync(0xffffffffu, my[r], 0);     // all lanes keep the same picked set
    }
    int after = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) after += res[my[r]] < 0;
    const int sw = __shfl_sync(0xffffffffu, swaps, 0);
    if (sw) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            atomicAdd(stats + 0, (unsigned long long) tail);
            atomicAdd(stats + 1, (unsigned long long) sw);
            atomicAdd(stats + 2, (unsigned long long) before);
            atomicAdd(stats + 3, (unsigned long long) after);
        }
    }
}
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || !res_layer) throw std::invalid_argument("native_route_resident: bad arguments");
    route_resident_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, res_layer, n_tok, margin, rank_lo, rank_hi, stats);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_router_top10_bell(const float* logits, int32_t* ids, float* weights,
                              const float* x, const int32_t* d_res, int n_embd, int n_tok,
                              float* x_out, int32_t* ids_out, uint32_t* d_seq, uint32_t value,
                              bool per_tok, void* stream) {
    if (!stream || n_tok < 1 || n_tok > 8 || n_embd < 1 || ((x == nullptr) != (x_out == nullptr)) || !ids_out || !d_seq ||
        !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native_router_top10_bell: bad arguments");
    route_top10_bell_k<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, x, d_res, n_embd, n_tok, x_out, ids_out, d_seq, value, per_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

namespace {
__device__ __forceinline__ float mega_f32_bf16(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

template <int RPB, int NT, bool AUX, bool BELL>
__global__ void __launch_bounds__(256) bf16_2560_router_mega_k(
        const float* __restrict__ x, int64_t ldx,
        const uint16_t* __restrict__ w, float* __restrict__ y, int64_t ldy,
        const uint16_t* __restrict__ w_aux, float* __restrict__ y_aux, int64_t ldy_aux,
        float* __restrict__ x_out, uint32_t* __restrict__ done_ctr,
        int32_t* __restrict__ ids, float* __restrict__ weights,
        const int32_t* __restrict__ res0, const int32_t* __restrict__ res1,
        int32_t* __restrict__ ids_out, uint32_t* __restrict__ seq, uint32_t ring,
        const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
        int32_t* __restrict__ pl, long long capx, uint32_t* __restrict__ skip) {
    constexpr int n_out = 512;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int n_row_blks = n_out / RPB;
    const int o0 = (int) blockIdx.x * RPB;
    __shared__ float partials[RPB][NT][8];

    if constexpr (BELL) {
        if (x_out != nullptr) {
            const int gtid = (int) blockIdx.x * 256 + t;
            const int total_f4 = NT * 640;
            const int stride = (int) gridDim.x * 256;
            for (int idx = gtid; idx < total_f4; idx += stride) {
                const int tok = idx / 640;
                const int off4 = idx - tok * 640;
                reinterpret_cast<float4*>(x_out + (size_t) tok * 2560)[off4] =
                    reinterpret_cast<const float4*>(x + (size_t) tok * ldx)[off4];
            }
        }
    }

    if (AUX && (int) blockIdx.x == n_row_blks) {
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
        const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w_aux);
#pragma unroll
        for (int qq = 0; qq < 5; ++qq) {
            const int pair = t + qq * 256;
            const uint32_t wv = __ldg(w2 + pair);
            const float w0 = mega_f32_bf16((uint16_t) wv), w1 = mega_f32_bf16((uint16_t) (wv >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float2 xv = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
                acc[k] = __fmaf_rn(w0, xv.x, acc[k]);
                acc[k] = __fmaf_rn(w1, xv.y, acc[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const float ws = warp_sum(acc[k]);
            if (lane == 0) partials[0][k][warp] = ws;
        }
        __syncthreads();
        if (warp == 0 && lane < 8) {
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                float s = partials[0][k][lane];
                s += __shfl_xor_sync(0xffu, s, 4, 8);
                s += __shfl_xor_sync(0xffu, s, 2, 8);
                s += __shfl_xor_sync(0xffu, s, 1, 8);
                if (lane == 0) y_aux[(size_t) k * ldy_aux] = s;
            }
        }
    } else {
        float acc[RPB][NT];
#pragma unroll
        for (int r = 0; r < RPB; ++r)
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
        const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w + (size_t) o0 * 2560);
#pragma unroll
        for (int qq = 0; qq < 5; ++qq) {
            const int pair = t + qq * 256;
            float2 xv[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k)
                xv[k] = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
            uint32_t wv[RPB];
#pragma unroll
            for (int r = 0; r < RPB; ++r) wv[r] = __ldg(w2 + (size_t) r * 1280 + pair);
#pragma unroll
            for (int r = 0; r < RPB; ++r) {
                const float w0 = mega_f32_bf16((uint16_t) wv[r]), w1 = mega_f32_bf16((uint16_t) (wv[r] >> 16));
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    acc[r][k] = __fmaf_rn(w0, xv[k].x, acc[r][k]);
                    acc[r][k] = __fmaf_rn(w1, xv[k].y, acc[r][k]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPB; ++r)
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float ws = warp_sum(acc[r][k]);
                if (lane == 0) partials[r][k][warp] = ws;
            }
        __syncthreads();
        if (warp < RPB && lane < 8) {
            const int r = warp;
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                float s = partials[r][k][lane];
                s += __shfl_xor_sync(0xffu, s, 4, 8);
                s += __shfl_xor_sync(0xffu, s, 2, 8);
                s += __shfl_xor_sync(0xffu, s, 1, 8);
                if (lane == 0) y[(size_t) k * ldy + o0 + r] = s;
            }
        }
    }

    __shared__ bool is_last_blk;
    __threadfence();
    __syncthreads();
    if (t == 0) is_last_blk = (atomicAdd(done_ctr, 1u) == (gridDim.x - 1u));
    __syncthreads();
    if (!is_last_blk) return;
    __threadfence();

    __shared__ int32_t s_ids[128];
    if (warp < NT) {
        const float* l = y + (size_t) warp * ldy;
        int32_t* id = ids + (size_t) warp * 10;
        float* wt = weights + (size_t) warp * 10;
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = l[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
        maximum = warp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
        const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) { values[i] *= reciprocal; if (__isnanf(values[i])) values[i] = -FLT_MAX; }
        float selected = 0.0f, selected_sum = 0.0f;
        int sel_id = 0;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0]; int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i) { if (values[i] > best) { best = values[i]; expert = lane + i * 32; } }
#pragma unroll
            for (int m = 16; m; m >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, m, 32);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, m, 32);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) { values[expert / 32] = -INFINITY; selected_sum += best; }
            if (rank == lane) { selected = best; sel_id = expert; }
        }
        selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
        const float inv = 1.0f / selected_sum;
        if (lane < 10) { id[lane] = sel_id; s_ids[warp * 10 + lane] = sel_id; wt[lane] = selected * inv; }
    }
    __syncthreads();
    const int n_ent = NT * 10;
    if constexpr (BELL) {
        if (ids_out != nullptr && t < n_ent) ids_out[t] = s_ids[t];
        __syncthreads();
        if (t == 0) {
            *done_ctr = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
            const uint32_t v = ring != 0u ? ring : (*(volatile uint32_t*) seq + 1u);
            asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(seq), "r"(v) : "memory");
#else
            __threadfence_system();
            *(volatile uint32_t*) seq = ring != 0u ? ring : (*(volatile uint32_t*) seq + 1u);
#endif
        }
    } else {
        if (t == 0) *done_ctr = 0u;
    }
    if (pl != nullptr) {
        if (n_ent <= 32) {
            if (warp == 0) {
                int32_t eid = -1, slot = -1; bool hit1 = false;
                if (lane < n_ent) {
                    eid = s_ids[lane];
                    if (eid >= 0 && eid < 512) { slot = res0[eid]; if (slot < 0 && res1 != nullptr && res1[eid] >= 0) hit1 = true; }
                }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
                const int32_t key = (lane < n_ent && slot >= 0) ? eid : (-2 - lane);
                const uint32_t mask = __match_any_sync(0xffffffffu, key);
                const bool act = (lane < n_ent && slot >= 0);
                const int first_j = act ? (__ffs((int) mask) - 1) : lane;
                const int rank_in_group = __popc(mask & ((1u << lane) - 1u));
                const int count_same = __popc(mask);
#else
                __shared__ int32_t sm32[32];
                sm32[lane] = (lane < n_ent && slot >= 0) ? eid : -1;
                __syncwarp();
                int first_j = lane, rank_in_group = 0, count_same = 0;
                const bool act = (lane < n_ent && slot >= 0);
                if (act) for (int j = 0; j < n_ent; ++j) if (sm32[j] == eid) { if (j < first_j) first_j = j; if (j < lane) ++rank_in_group; ++count_same; }
#endif
                const bool is_first = act && (first_j == lane);
                const int my_pack = ((is_first ? count_same : 0) << 16) | (is_first ? 1 : 0);
                const int my_h1 = (lane < n_ent && hit1) ? 1 : 0;
                int pref = my_pack, pref1 = my_h1;
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const int up = __shfl_up_sync(0xffffffffu, pref, d);
                    const int up1 = __shfl_up_sync(0xffffffffu, pref1, d);
                    if (lane >= d) { pref += up; pref1 += up1; }
                }
                const int excl = pref - my_pack, excl1 = pref1 - my_h1;
                const int tot_pack = __shfl_sync(0xffffffffu, pref, 31);
                const int tot_hit1 = __shfl_sync(0xffffffffu, pref1, 31);
                const int fj_tot = __shfl_sync(0xffffffffu, excl, first_j);
                const int groups = tot_pack & 0xffff, nhits0 = tot_pack >> 16;
                int32_t* counts = pl; int32_t* start = pl + 4; int32_t* dst = start + capx + 1; int32_t* tok = dst + capx;
                const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
                unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
                int32_t* start2 = pl + ptr_off + 4 * capx;
                if (is_first) {
                    const int g = excl & 0xffff;
                    ptr[g] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
                    start[g] = excl >> 16;
                }
                if (act) { const int oi = (fj_tot >> 16) + rank_in_group; dst[oi] = lane; tok[oi] = lane / 10; }
                else if (lane < n_ent && hit1) { dst[nhits0 + excl1] = lane; }
                if (lane == 0) {
                    start[groups] = nhits0; start2[0] = nhits0;
                    counts[0] = groups; counts[1] = nhits0; counts[2] = 0; counts[3] = nhits0 + tot_hit1;
                    if (skip != nullptr) { __threadfence(); *skip = (nhits0 + tot_hit1 == n_ent) ? ring : 0; }
                }
            }
        } else {
            __shared__ int32_t p_ids[128], p_excl[128], p_wsum[8], p_excl1[128], p_wsum1[8];
            int32_t eid = -1, slot = -1; bool hit1 = false;
            if (t < n_ent) {
                eid = s_ids[t];
                if (eid >= 0 && eid < 512) { slot = res0[eid]; if (slot < 0 && res1 != nullptr && res1[eid] >= 0) hit1 = true; }
            }
            if (t < 128) p_ids[t] = (t < n_ent && slot >= 0) ? eid : -1;
            __syncthreads();
            int first_j = t, rank_in_group = 0, count_same = 0;
            if (t < n_ent && slot >= 0) {
                for (int j = 0; j < n_ent; ++j) {
                    if (p_ids[j] == eid) { if (j < first_j) first_j = j; if (j < t) ++rank_in_group; ++count_same; }
                }
            }
            const bool is_first = (t < n_ent && slot >= 0 && first_j == t);
            const int my_pack = ((is_first ? count_same : 0) << 16) | (is_first ? 1 : 0);
            const int my_h1 = (t < n_ent && hit1) ? 1 : 0;
            int pref = my_pack, pref1 = my_h1;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int up = __shfl_up_sync(0xffffffffu, pref, d);
                const int up1 = __shfl_up_sync(0xffffffffu, pref1, d);
                if (lane >= d) { pref += up; pref1 += up1; }
            }
            if (t < 128) { p_excl[t] = pref - my_pack; p_excl1[t] = pref1 - my_h1; }
            if (lane == 31) { p_wsum[warp] = pref; p_wsum1[warp] = pref1; }
            __syncthreads();
            int tot_pack = 0, tot_hit1 = 0;
            for (int w_i = 0; w_i < 8; ++w_i) { tot_pack += p_wsum[w_i]; tot_hit1 += p_wsum1[w_i]; }
            const int groups = tot_pack & 0xffff, nhits0 = tot_pack >> 16;
            int32_t* counts = pl; int32_t* start = pl + 4; int32_t* dst = start + capx + 1; int32_t* tok = dst + capx;
            const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
            unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
            int32_t* start2 = pl + ptr_off + 4 * capx;
            if (is_first) {
                int tot = p_excl[t]; for (int w_i = 0; w_i < warp; ++w_i) tot += p_wsum[w_i];
                const int g = tot & 0xffff;
                ptr[g] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
                start[g] = tot >> 16;
            }
            if (t < n_ent && slot >= 0) {
                const int fw = first_j >> 5;
                int ft = p_excl[first_j]; for (int w_i = 0; w_i < fw; ++w_i) ft += p_wsum[w_i];
                const int oi = (ft >> 16) + rank_in_group; dst[oi] = t; tok[oi] = t / 10;
            } else if (t < n_ent && hit1) {
                int p1 = p_excl1[t]; for (int w_i = 0; w_i < warp; ++w_i) p1 += p_wsum1[w_i];
                dst[nhits0 + p1] = t;
            }
            if (t == 0) {
                start[groups] = nhits0; start2[0] = nhits0;
                counts[0] = groups; counts[1] = nhits0; counts[2] = 0; counts[3] = nhits0 + tot_hit1;
                if (skip != nullptr) { __threadfence(); *skip = (nhits0 + tot_hit1 == n_ent) ? ring : 0; }
            }
        }
    }
}

template <int RPB, int NT, bool AUX, bool BELL>
inline void launch_router_mega(
    const float* x, int64_t ldx, const uint16_t* w, float* logits, int64_t ldy,
    const uint16_t* w_aux, float* y_aux, int64_t ldy_aux, float* x_out, uint32_t* done_ctr,
    int32_t* ids, float* weights, const int32_t* res0, const int32_t* res1,
    int32_t* ids_out, uint32_t* seq, uint32_t ring, const uint8_t* cache_base,
    const unsigned long long* slot_off, long long blob, int32_t* pl, long long capx,
    uint32_t* skip, cudaStream_t st) {
    const unsigned nb = (unsigned) (512 / RPB + (AUX ? 1 : 0));
    bf16_2560_router_mega_k<RPB, NT, AUX, BELL><<<nb, 256, 0, st>>>(
        x, ldx, w, logits, ldy, w_aux, y_aux, ldy_aux, x_out, done_ctr, ids, weights,
        res0, res1, ids_out, seq, ring, cache_base, slot_off, blob, pl, capx, skip);
}
}  // namespace

void native_router_fused_bf16_bell_plan(
    const float* x, int64_t ldx, const uint16_t* w, float* logits, int64_t ldy, int n_tok,
    const uint16_t* w_aux, float* y_aux, int64_t ldy_aux,
    float* x_out, int32_t* ids, float* weights, int32_t* ids_out, uint32_t* d_seq, uint32_t ring,
    const int32_t* res_self, const int32_t* res_peer,
    const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
    int32_t* pl, long long capx, uint32_t* skip, uint32_t* done_ctr, bool bell, void* stream) {
    if (!stream || n_tok < 1 || n_tok > 8 || (ldx & 1) != 0 || !x || !w || !done_ctr ||
        !valid(logits, (size_t) n_tok * ldy * 4) || ldy < 512 ||
        !valid(ids, (size_t) n_tok * 10 * 4) || !valid(weights, (size_t) n_tok * 10 * 4) ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (bell && (!ids_out || !d_seq || (x_out != nullptr && (((uintptr_t) x | (uintptr_t) x_out) & 15u) != 0))))
        throw std::invalid_argument("native_router_fused_bf16_bell_plan: bad arguments");
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const bool has_aux = (w_aux != nullptr && y_aux != nullptr);
#define STRATA_MEGA_CASE(NT) case NT: {         constexpr int RPB = (NT == 1) ? 2 : 4;         if (bell) {             if (has_aux) launch_router_mega<RPB, NT, true, true>(x, ldx, w, logits, ldy, w_aux, y_aux, ldy_aux, x_out, done_ctr, ids, weights, res_self, res_peer, ids_out, d_seq, ring, cache_base, slot_off, blob, pl, capx, skip, st);             else launch_router_mega<RPB, NT, false, true>(x, ldx, w, logits, ldy, w_aux, y_aux, ldy_aux, x_out, done_ctr, ids, weights, res_self, res_peer, ids_out, d_seq, ring, cache_base, slot_off, blob, pl, capx, skip, st);         } else {             if (has_aux) launch_router_mega<RPB, NT, true, false>(x, ldx, w, logits, ldy, w_aux, y_aux, ldy_aux, x_out, done_ctr, ids, weights, res_self, res_peer, ids_out, d_seq, ring, cache_base, slot_off, blob, pl, capx, skip, st);             else launch_router_mega<RPB, NT, false, false>(x, ldx, w, logits, ldy, w_aux, y_aux, ldy_aux, x_out, done_ctr, ids, weights, res_self, res_peer, ids_out, d_seq, ring, cache_base, slot_off, blob, pl, capx, skip, st);         }     } break
    switch (n_tok) {
        STRATA_MEGA_CASE(1);
        STRATA_MEGA_CASE(2);
        STRATA_MEGA_CASE(3);
        STRATA_MEGA_CASE(4);
        STRATA_MEGA_CASE(5);
        STRATA_MEGA_CASE(6);
        STRATA_MEGA_CASE(7);
        STRATA_MEGA_CASE(8);
    }
#undef STRATA_MEGA_CASE
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
