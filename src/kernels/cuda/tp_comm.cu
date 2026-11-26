#include "strata/core/tp_comm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
#include <cuda.h>
#include <dlfcn.h>
#endif
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace {

__device__ __forceinline__ float4 load_cv_f4(const float4* p) {
    float4 v;
    asm volatile("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void store_wt_f4(float4* p, float4 v) {
    asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

__device__ __forceinline__ uint4 load_cv_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void store_wt_u4(uint4* p, uint4 v) {
    asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__global__ void tp_allreduce_kernel(float* __restrict__ local_buf,
                                    float* __restrict__ my_map_base,
                                    const float* __restrict__ peer_map_base,
                                    volatile uint32_t* __restrict__ my_flag,
                                    volatile uint32_t* __restrict__ peer_flag,
                                    int n_elems,
                                    size_t slot_floats,
                                    uint32_t step) {
    const int tid = threadIdx.x;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    float* my_map = my_map_base + slot_off;
    const float* peer_map = peer_map_base + slot_off;

    const int n4 = n_elems >> 2;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    float4* m4 = reinterpret_cast<float4*>(my_map);
    const float4* p4 = reinterpret_cast<const float4*>(peer_map);
    for (int i = tid; i < n4; i += blockDim.x) {
        store_wt_f4(m4 + i, l4[i]);
    }
    for (int i = (n4 << 2) + tid; i < n_elems; i += blockDim.x) {
        *(volatile float*) (my_map + i) = local_buf[i];
    }
    __syncthreads();
    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#elif defined(__CUDA_ARCH__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        __threadfence_system();
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
        } while (pf < step);
#else
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {
        }
        __threadfence_system();
#endif
    }
    __syncthreads();
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 a = l4[i];
        const float4 b = load_cv_f4(p4 + i);
        l4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    for (int i = (n4 << 2) + tid; i < n_elems; i += blockDim.x) {
        local_buf[i] += *(const volatile float*) (peer_map + i);
    }
}

// P2: Register-cached + MLP-pipelined BF16 AllReduce (100% bitwise identical to tp_allreduce_bf16_kernel).
// Eliminates the redundant post-barrier VRAM reload of local_buf AND issues all K peer PCIe reads
// back-to-back in flight concurrently (hiding PCIe round-trip latency for T=4..16 windows).
template <int K>
__global__ void __launch_bounds__(1024, 1)
tp_allreduce_bf16_reg_kernel(float* __restrict__ local_buf,
                             float* __restrict__ my_map_base,
                             const float* __restrict__ peer_map_base,
                             volatile uint32_t* __restrict__ my_flag,
                             volatile uint32_t* __restrict__ peer_flag,
                             int n_elems,
                             size_t slot_floats,
                             uint32_t step) {
    const int tid = threadIdx.x;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    float* my_map = my_map_base + slot_off;
    const float* peer_map = peer_map_base + slot_off;

    const int n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    uint4* m4 = reinterpret_cast<uint4*>(my_map);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_map);

    float4 ra[K], rb[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const float4 a = l4[2 * i];
            const float4 b = l4[2 * i + 1];
            ra[k] = a;
            rb[k] = b;
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(a.x, a.y);
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(a.z, a.w);
            const __nv_bfloat162 p2 = __floats2bfloat162_rn(b.x, b.y);
            const __nv_bfloat162 p3 = __floats2bfloat162_rn(b.z, b.w);
            uint4 v;
            std::memcpy(&v.x, &p0, 4);
            std::memcpy(&v.y, &p1, 4);
            std::memcpy(&v.z, &p2, 4);
            std::memcpy(&v.w, &p3, 4);
            store_wt_u4(m4 + i, v);
        }
    }
    __syncthreads();
    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#elif defined(__CUDA_ARCH__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        __threadfence_system();
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
        } while (pf < step);
#else
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {
        }
        __threadfence_system();
#endif
    }
    __syncthreads();
    uint4 pv[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) pv[k] = load_cv_u4(p4 + i);
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const uint4 v = pv[k];
            __nv_bfloat162 p0, p1, p2, p3;
            std::memcpy(&p0, &v.x, 4);
            std::memcpy(&p1, &v.y, 4);
            std::memcpy(&p2, &v.z, 4);
            std::memcpy(&p3, &v.w, 4);
            const float2 f0 = __bfloat1622float2(p0);
            const float2 f1 = __bfloat1622float2(p1);
            const float2 f2 = __bfloat1622float2(p2);
            const float2 f3 = __bfloat1622float2(p3);
            float4 a = ra[k];
            float4 b = rb[k];
            a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
            b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
            l4[2 * i] = a;
            l4[2 * i + 1] = b;
        }
    }
}

// Wave 2A (#3): Fused P2P BF16 AllReduce + Hyper-Connection residual write (R += gw * bo) + RMSNorm (rs, xn).
// Eliminates bo_ VRAM writeback/2x-reloads, eliminates gr_up's R_out writeback, and deletes gr_norm_multi_kernel
// across ~89 of 96 AllReduces per verify window. 100% deterministic warp-tree reduction (no atomicAdd).
template <int K>
__global__ void __launch_bounds__(1024, 1)
tp_allreduce_bf16_gr_norm_kernel(float* __restrict__ local_buf,
                                 float* __restrict__ R_base,
                                 const float* __restrict__ inj_base,
                                 const float* __restrict__ w_norm,
                                 float* __restrict__ xn_out,
                                 float* __restrict__ rs_out,
                                 float eps,
                                 int n_tok,
                                 float* __restrict__ my_map_base,
                                 const float* __restrict__ peer_map_base,
                                 volatile uint32_t* __restrict__ my_flag,
                                 volatile uint32_t* __restrict__ peer_flag,
                                 size_t slot_floats,
                                 uint32_t step,
                                 bool also_write_bo) {
    constexpr int N = 2560;
    constexpr int HC = 4;
    constexpr int N8_TOK = N >> 3;  // 320 (exact multiple of 32: 10 warps per token!)
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n8 = n_tok * N8_TOK;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    float* my_map = my_map_base + slot_off;
    const float* peer_map = peer_map_base + slot_off;

    __shared__ float s_gw[8][HC];
    __shared__ float s_part[32][8][HC];
    __shared__ float s_rs[8][HC];

    if (tid < n_tok * HC) {
        s_gw[tid >> 2][tid & 3] = 2.0f / (1.0f + __expf(-inj_base[tid] * 0.25f));
    }
    // Zero this warp's 8*4 partial slots (lane 0..31 -> 1 float each!)
    s_part[warp][lane >> 2][lane & 3] = 0.0f;

    float4* l4 = reinterpret_cast<float4*>(local_buf);
    uint4* m4 = reinterpret_cast<uint4*>(my_map);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_map);

    float4 ra[K], rb[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const float4 a = l4[2 * i];
            const float4 b = l4[2 * i + 1];
            ra[k] = a;
            rb[k] = b;
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(a.x, a.y);
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(a.z, a.w);
            const __nv_bfloat162 p2 = __floats2bfloat162_rn(b.x, b.y);
            const __nv_bfloat162 p3 = __floats2bfloat162_rn(b.z, b.w);
            uint4 v;
            std::memcpy(&v.x, &p0, 4);
            std::memcpy(&v.y, &p1, 4);
            std::memcpy(&v.z, &p2, 4);
            std::memcpy(&v.w, &p3, 4);
            store_wt_u4(m4 + i, v);
        }
    }
    __syncthreads();
    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#else
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {}
        __threadfence_system();
#endif
    }
    __syncthreads();

    uint4 pv[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) pv[k] = load_cv_u4(p4 + i);
    }

#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const int tok = i / N8_TOK;
            const int d8 = i - tok * N8_TOK;
            const uint4 v = pv[k];
            __nv_bfloat162 p0, p1, p2, p3;
            std::memcpy(&p0, &v.x, 4);
            std::memcpy(&p1, &v.y, 4);
            std::memcpy(&p2, &v.z, 4);
            std::memcpy(&p3, &v.w, 4);
            const float2 f0 = __bfloat1622float2(p0);
            const float2 f1 = __bfloat1622float2(p1);
            const float2 f2 = __bfloat1622float2(p2);
            const float2 f3 = __bfloat1622float2(p3);
            float4 a = ra[k];
            float4 b = rb[k];
            a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
            b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
            if (also_write_bo) {
                l4[2 * i] = a;
                l4[2 * i + 1] = b;
            }
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float gw = s_gw[tok][c];
                const size_t roff = ((size_t) tok * HC + (size_t) c) * (size_t) N + (size_t) d8 * 8u;
                const size_t woff = (size_t) c * (size_t) N + (size_t) d8 * 8u;
                float4* r4 = reinterpret_cast<float4*>(R_base + roff);
                const float4* wn4 = reinterpret_cast<const float4*>(w_norm + woff);
                float4* x4 = reinterpret_cast<float4*>(xn_out + roff);
                float4 r0 = r4[0];
                float4 r1 = r4[1];
                r0.x = fmaf(a.x, gw, r0.x); r0.y = fmaf(a.y, gw, r0.y);
                r0.z = fmaf(a.z, gw, r0.z); r0.w = fmaf(a.w, gw, r0.w);
                r1.x = fmaf(b.x, gw, r1.x); r1.y = fmaf(b.y, gw, r1.y);
                r1.z = fmaf(b.z, gw, r1.z); r1.w = fmaf(b.w, gw, r1.w);
                r4[0] = r0;
                r4[1] = r1;
                const float4 wn0 = wn4[0];
                const float4 wn1 = wn4[1];
                x4[0] = make_float4(r0.x * wn0.x, r0.y * wn0.y, r0.z * wn0.z, r0.w * wn0.w);
                x4[1] = make_float4(r1.x * wn1.x, r1.y * wn1.y, r1.z * wn1.z, r1.w * wn1.w);
                float sq = r0.x * r0.x + r0.y * r0.y + r0.z * r0.z + r0.w * r0.w +
                           r1.x * r1.x + r1.y * r1.y + r1.z * r1.z + r1.w * r1.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
                if (lane == 0) s_part[warp][tok][c] += sq;
            }
        }
    }
    __syncthreads();
    if (tid < n_tok * HC) {
        const int tok = tid >> 2;
        const int c = tid & 3;
        const int nwarps = blockDim.x >> 5;
        float s = 0.0f;
        for (int w = 0; w < nwarps; ++w) s += s_part[w][tok][c];
        const float rsv = rsqrtf(s * (1.0f / (float) N) + eps);
        s_rs[tok][c] = rsv;
        rs_out[tid] = rsv;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const int tok = i / N8_TOK;
            const int d8 = i - tok * N8_TOK;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float rsv = s_rs[tok][c];
                const size_t roff = ((size_t) tok * HC + (size_t) c) * (size_t) N + (size_t) d8 * 8u;
                float4* x4 = reinterpret_cast<float4*>(xn_out + roff);
                float4 x0 = x4[0];
                float4 x1 = x4[1];
                x0.x *= rsv; x0.y *= rsv; x0.z *= rsv; x0.w *= rsv;
                x1.x *= rsv; x1.y *= rsv; x1.z *= rsv; x1.w *= rsv;
                x4[0] = x0;
                x4[1] = x1;
            }
        }
    }
}

// Fused-epilogue companion: local_buf was already pushed to peer BAR1 + flagged by combine epilogue.
// Waits on local VRAM flag (with Ampere nanosleep), then reads peer BF16 from local VRAM and adds into local_buf.
template <int K>
__global__ void __launch_bounds__(1024, 1)
tp_allreduce_wait_add_bf16_kernel(float* __restrict__ local_buf,
                                  const float* __restrict__ peer_map_base,
                                  volatile uint32_t* __restrict__ peer_flag,
                                  int n_elems,
                                  size_t slot_floats,
                                  uint32_t step) {
    const int tid = threadIdx.x;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    const float* peer_map = peer_map_base + slot_off;
    const int n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_map);

    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#else
        while (*peer_flag < step) {}
        __threadfence_system();
#endif
    }
    __syncthreads();

    uint4 pv[K];
    float4 ra[K], rb[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * blockDim.x;
        if (i < n8) {
            pv[k] = load_cv_u4(p4 + i);
            ra[k] = l4[2 * i];
            rb[k] = l4[2 * i + 1];
        }
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * blockDim.x;
        if (i < n8) {
            const uint4 v = pv[k];
            __nv_bfloat162 p0, p1, p2, p3;
            std::memcpy(&p0, &v.x, 4);
            std::memcpy(&p1, &v.y, 4);
            std::memcpy(&p2, &v.z, 4);
            std::memcpy(&p3, &v.w, 4);
            const float2 f0 = __bfloat1622float2(p0);
            const float2 f1 = __bfloat1622float2(p1);
            const float2 f2 = __bfloat1622float2(p2);
            const float2 f3 = __bfloat1622float2(p3);
            float4 a = ra[k];
            float4 b = rb[k];
            a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
            b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
            l4[2 * i] = a;
            l4[2 * i + 1] = b;
        }
    }
}

__global__ void tp_p2p_selftest_write_kernel(uint32_t* __restrict__ remote_dst, uint32_t magic, int n) {
    const int tid = threadIdx.x;
    if (tid < n) {
        const uint32_t v = magic ^ (uint32_t) (tid * 2654435761u);
        asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(remote_dst + tid), "r"(v) : "memory");
    }
    __syncthreads();
    if (tid == 0) __threadfence_system();
}

__global__ void tp_push_bytes_kernel(const uint8_t* __restrict__ src,
                                     uint8_t* __restrict__ peer_dst,
                                     size_t n_bytes) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n16 = n_bytes >> 4;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(peer_dst);
    for (size_t i = tid; i < n16; i += stride) {
        store_wt_u4(d4 + i, s4[i]);
    }
    for (size_t i = (n16 << 4) + tid; i < n_bytes; i += stride) {
        *(volatile uint8_t*) (peer_dst + i) = src[i];
    }
    if (threadIdx.x == 0) __threadfence_system();
}

__global__ void tp_push_logits_f32_kernel(const float* __restrict__ src,
                                          float* __restrict__ peer_dst,
                                          int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n4 = n_elems >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    float4* d4 = reinterpret_cast<float4*>(peer_dst);
    for (int64_t i = tid; i < n4; i += stride) {
        store_wt_f4(d4 + i, s4[i]);
    }
    for (int64_t i = (n4 << 2) + tid; i < n_elems; i += stride) {
        *(volatile float*) (peer_dst + i) = src[i];
    }
    if (threadIdx.x == 0) __threadfence_system();
}

__global__ void tp_signal_flag_kernel(volatile uint32_t* __restrict__ flag, uint32_t val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag), "r"(val) : "memory");
#else
    __threadfence_system();
    *flag = val;
    __threadfence_system();
#endif
}

__global__ void tp_wait_flag_kernel(volatile uint32_t* __restrict__ flag, uint32_t val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    uint32_t pf = 0;
    for (;;) {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(flag) : "memory");
        if (pf >= val) break;
        __nanosleep(32);
    }
#else
    while (*flag < val) {}
    __threadfence_system();
#endif
}

__global__ void tp_scatter_logits_2d_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            int T,
                                            int64_t src_cols,
                                            int64_t dst_stride,
                                            int64_t dst_col_off) {
    const int row = blockIdx.y;
    if (row >= T) return;
    const float* s = src + (size_t) row * (size_t) src_cols;
    float* d = dst + (size_t) row * (size_t) dst_stride + (size_t) dst_col_off;
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n4 = src_cols >> 2;
    if ((((uintptr_t) s | (uintptr_t) d) & 15u) == 0) {
        const float4* s4 = reinterpret_cast<const float4*>(s);
        float4* d4 = reinterpret_cast<float4*>(d);
        for (int64_t i = tid; i < n4; i += stride) d4[i] = s4[i];
        for (int64_t i = (n4 << 2) + tid; i < src_cols; i += stride) d[i] = s[i];
    } else {
        for (int64_t i = tid; i < src_cols; i += stride) d[i] = s[i];
    }
}

__global__ void tp_allreduce_bf16_kernel(float* __restrict__ local_buf,
                                         float* __restrict__ my_map_base,
                                         const float* __restrict__ peer_map_base,
                                         volatile uint32_t* __restrict__ my_flag,
                                         volatile uint32_t* __restrict__ peer_flag,
                                         int n_elems,
                                         size_t slot_floats,
                                         uint32_t step) {
    const int tid = threadIdx.x;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    float* my_map = my_map_base + slot_off;
    const float* peer_map = peer_map_base + slot_off;

    const int n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    uint4* m4 = reinterpret_cast<uint4*>(my_map);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_map);
    for (int i = tid; i < n8; i += blockDim.x) {
        const float4 a = l4[2 * i];
        const float4 b = l4[2 * i + 1];
        const __nv_bfloat162 p0 = __floats2bfloat162_rn(a.x, a.y);
        const __nv_bfloat162 p1 = __floats2bfloat162_rn(a.z, a.w);
        const __nv_bfloat162 p2 = __floats2bfloat162_rn(b.x, b.y);
        const __nv_bfloat162 p3 = __floats2bfloat162_rn(b.z, b.w);
        uint4 v;
        std::memcpy(&v.x, &p0, 4);
        std::memcpy(&v.y, &p1, 4);
        std::memcpy(&v.z, &p2, 4);
        std::memcpy(&v.w, &p3, 4);
        store_wt_u4(m4 + i, v);
    }
    __syncthreads();
    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#elif defined(__CUDA_ARCH__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        __threadfence_system();
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
        } while (pf < step);
#else
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {
        }
        __threadfence_system();
#endif
    }
    __syncthreads();
    for (int i = tid; i < n8; i += blockDim.x) {
        float4 a = l4[2 * i];
        float4 b = l4[2 * i + 1];
        const uint4 v = load_cv_u4(p4 + i);
        __nv_bfloat162 p0, p1, p2, p3;
        std::memcpy(&p0, &v.x, 4);
        std::memcpy(&p1, &v.y, 4);
        std::memcpy(&p2, &v.z, 4);
        std::memcpy(&p3, &v.w, 4);
        const float2 f0 = __bfloat1622float2(p0);
        const float2 f1 = __bfloat1622float2(p1);
        const float2 f2 = __bfloat1622float2(p2);
        const float2 f3 = __bfloat1622float2(p3);
        a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
        b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
        l4[2 * i] = a;
        l4[2 * i + 1] = b;
    }
}

// P2-FP16: Register-cached + MLP-pipelined FP16 AllReduce (hardware __half2 on Turing sm_75 & Ampere+ sm_80+).
// Eliminates redundant post-barrier VRAM reload of local_buf AND pipelines K concurrent peer reads (NVLink or PCIe).
template <int K>
__global__ void __launch_bounds__(1024, 1)
tp_allreduce_fp16_reg_kernel(float* __restrict__ local_buf,
                             float* __restrict__ my_map_base,
                             const float* __restrict__ peer_map_base,
                             volatile uint32_t* __restrict__ my_flag,
                             volatile uint32_t* __restrict__ peer_flag,
                             int n_elems,
                             size_t slot_floats,
                             uint32_t step) {
    const int tid = threadIdx.x;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    float* my_map = my_map_base + slot_off;
    const float* peer_map = peer_map_base + slot_off;

    const int n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    uint4* m4 = reinterpret_cast<uint4*>(my_map);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_map);

    float4 ra[K], rb[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const float4 a = l4[2 * i];
            const float4 b = l4[2 * i + 1];
            ra[k] = a;
            rb[k] = b;
            const __half2 p0 = __floats2half2_rn(a.x, a.y);
            const __half2 p1 = __floats2half2_rn(a.z, a.w);
            const __half2 p2 = __floats2half2_rn(b.x, b.y);
            const __half2 p3 = __floats2half2_rn(b.z, b.w);
            uint4 v;
            std::memcpy(&v.x, &p0, 4);
            std::memcpy(&v.y, &p1, 4);
            std::memcpy(&v.z, &p2, 4);
            std::memcpy(&v.w, &p3, 4);
            store_wt_u4(m4 + i, v);
        }
    }
    __syncthreads();
    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#else
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {}
        __threadfence_system();
#endif
    }
    __syncthreads();
    uint4 pv[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) pv[k] = load_cv_u4(p4 + i);
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k * 1024;
        if (i < n8) {
            const uint4 v = pv[k];
            __half2 p0, p1, p2, p3;
            std::memcpy(&p0, &v.x, 4);
            std::memcpy(&p1, &v.y, 4);
            std::memcpy(&p2, &v.z, 4);
            std::memcpy(&p3, &v.w, 4);
            const float2 f0 = __half22float2(p0);
            const float2 f1 = __half22float2(p1);
            const float2 f2 = __half22float2(p2);
            const float2 f3 = __half22float2(p3);
            float4 a = ra[k];
            float4 b = rb[k];
            a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
            b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
            l4[2 * i] = a;
            l4[2 * i + 1] = b;
        }
    }
}

__global__ void tp_allreduce_fp16_kernel(float* __restrict__ local_buf,
                                         float* __restrict__ my_map_base,
                                         const float* __restrict__ peer_map_base,
                                         volatile uint32_t* __restrict__ my_flag,
                                         volatile uint32_t* __restrict__ peer_flag,
                                         int n_elems,
                                         size_t slot_floats,
                                         uint32_t step) {
    const int tid = threadIdx.x;
    const size_t slot_off = (size_t) (step & (TpComm::kRingSlots - 1u)) * slot_floats;
    float* my_map = my_map_base + slot_off;
    const float* peer_map = peer_map_base + slot_off;

    const int n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    uint4* m4 = reinterpret_cast<uint4*>(my_map);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_map);
    for (int i = tid; i < n8; i += blockDim.x) {
        const float4 a = l4[2 * i];
        const float4 b = l4[2 * i + 1];
        const __half2 p0 = __floats2half2_rn(a.x, a.y);
        const __half2 p1 = __floats2half2_rn(a.z, a.w);
        const __half2 p2 = __floats2half2_rn(b.x, b.y);
        const __half2 p3 = __floats2half2_rn(b.z, b.w);
        uint4 v;
        std::memcpy(&v.x, &p0, 4);
        std::memcpy(&v.y, &p1, 4);
        std::memcpy(&v.z, &p2, 4);
        std::memcpy(&v.w, &p3, 4);
        store_wt_u4(m4 + i, v);
    }
    __syncthreads();
    if (tid == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
            if (pf >= step) break;
            __nanosleep(64);
        }
#elif defined(__CUDA_ARCH__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
        __threadfence_system();
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
        uint32_t pf = 0;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
        } while (pf < step);
#else
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {
        }
        __threadfence_system();
#endif
    }
    __syncthreads();
    for (int i = tid; i < n8; i += blockDim.x) {
        float4 a = l4[2 * i];
        float4 b = l4[2 * i + 1];
        const uint4 v = load_cv_u4(p4 + i);
        __half2 p0, p1, p2, p3;
        std::memcpy(&p0, &v.x, 4);
        std::memcpy(&p1, &v.y, 4);
        std::memcpy(&p2, &v.z, 4);
        std::memcpy(&p3, &v.w, 4);
        const float2 f0 = __half22float2(p0);
        const float2 f1 = __half22float2(p1);
        const float2 f2 = __half22float2(p2);
        const float2 f3 = __half22float2(p3);
        a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
        b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
        l4[2 * i] = a;
        l4[2 * i + 1] = b;
    }
}

__global__ void tp_push_kernel(const float* __restrict__ local_buf,
                               float* __restrict__ my_map,
                               int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n4 = n_elems >> 2;
    const float4* l4 = reinterpret_cast<const float4*>(local_buf);
    float4* m4 = reinterpret_cast<float4*>(my_map);
    for (int64_t i = tid; i < n4; i += stride) {
        store_wt_f4(m4 + i, l4[i]);
    }
    for (int64_t i = (n4 << 2) + tid; i < n_elems; i += stride) {
        *(volatile float*) (my_map + i) = local_buf[i];
    }
    if (threadIdx.x == 0) __threadfence_system();
}

__global__ void tp_barrier_kernel(volatile uint32_t* __restrict__ my_flag,
                                  volatile uint32_t* __restrict__ peer_flag,
                                  uint32_t step) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(my_flag), "r"(step) : "memory");
    uint32_t pf = 0;
    for (;;) {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(pf) : "l"(peer_flag) : "memory");
        if (pf >= step) break;
        __nanosleep(20);
    }
#else
    __threadfence_system();
    *my_flag = step;
    __threadfence_system();
    while (*peer_flag < step) {}
    __threadfence_system();
#endif
}

__global__ void tp_pull_kernel(float* __restrict__ local_buf,
                               const float* __restrict__ peer_map,
                               int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n4 = n_elems >> 2;
    float4* l4 = reinterpret_cast<float4*>(local_buf);
    const float4* p4 = reinterpret_cast<const float4*>(peer_map);
    for (int64_t i = tid; i < n4; i += stride) {
        const float4 a = l4[i];
        const float4 b = load_cv_f4(p4 + i);
        l4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    for (int64_t i = (n4 << 2) + tid; i < n_elems; i += stride) {
        local_buf[i] += *(const volatile float*) (peer_map + i);
    }
}

__global__ void tp_pack_bf16_kernel(const float* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst,
                                    int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n_elems >> 3;
    const float4* l4 = reinterpret_cast<const float4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int64_t i = tid; i < n8; i += stride) {
        const float4 a = l4[2 * i];
        const float4 b = l4[2 * i + 1];
        const __nv_bfloat162 p0 = __floats2bfloat162_rn(a.x, a.y);
        const __nv_bfloat162 p1 = __floats2bfloat162_rn(a.z, a.w);
        const __nv_bfloat162 p2 = __floats2bfloat162_rn(b.x, b.y);
        const __nv_bfloat162 p3 = __floats2bfloat162_rn(b.z, b.w);
        uint4 v;
        std::memcpy(&v.x, &p0, 4);
        std::memcpy(&v.y, &p1, 4);
        std::memcpy(&v.z, &p2, 4);
        std::memcpy(&v.w, &p3, 4);
        d4[i] = v;
    }
    for (int64_t i = (n8 << 3) + tid; i < n_elems; i += stride) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

__global__ void tp_unpack_add_bf16_kernel(float* __restrict__ dst,
                                          const __nv_bfloat16* __restrict__ peer_dev,
                                          int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(dst);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_dev);
    for (int64_t i = tid; i < n8; i += stride) {
        float4 a = l4[2 * i];
        float4 b = l4[2 * i + 1];
        const uint4 v = load_cv_u4(p4 + i);
        __nv_bfloat162 p0, p1, p2, p3;
        std::memcpy(&p0, &v.x, 4);
        std::memcpy(&p1, &v.y, 4);
        std::memcpy(&p2, &v.z, 4);
        std::memcpy(&p3, &v.w, 4);
        const float2 f0 = __bfloat1622float2(p0);
        const float2 f1 = __bfloat1622float2(p1);
        const float2 f2 = __bfloat1622float2(p2);
        const float2 f3 = __bfloat1622float2(p3);
        a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
        b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
        l4[2 * i] = a;
        l4[2 * i + 1] = b;
    }
    for (int64_t i = (n8 << 3) + tid; i < n_elems; i += stride) {
        uint16_t raw = *(const volatile uint16_t*) (reinterpret_cast<const uint16_t*>(peer_dev) + i);
        __nv_bfloat16 bf;
        std::memcpy(&bf, &raw, 2);
        dst[i] += __bfloat162float(bf);
    }
}

__global__ void tp_pack_fp16_kernel(const float* __restrict__ src,
                                    __half* __restrict__ dst,
                                    int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n_elems >> 3;
    const float4* l4 = reinterpret_cast<const float4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int64_t i = tid; i < n8; i += stride) {
        const float4 a = l4[2 * i];
        const float4 b = l4[2 * i + 1];
        const __half2 p0 = __floats2half2_rn(a.x, a.y);
        const __half2 p1 = __floats2half2_rn(a.z, a.w);
        const __half2 p2 = __floats2half2_rn(b.x, b.y);
        const __half2 p3 = __floats2half2_rn(b.z, b.w);
        uint4 v;
        std::memcpy(&v.x, &p0, 4);
        std::memcpy(&v.y, &p1, 4);
        std::memcpy(&v.z, &p2, 4);
        std::memcpy(&v.w, &p3, 4);
        d4[i] = v;
    }
    for (int64_t i = (n8 << 3) + tid; i < n_elems; i += stride) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void tp_unpack_add_fp16_kernel(float* __restrict__ dst,
                                          const __half* __restrict__ peer_dev,
                                          int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n_elems >> 3;
    float4* l4 = reinterpret_cast<float4*>(dst);
    const uint4* p4 = reinterpret_cast<const uint4*>(peer_dev);
    for (int64_t i = tid; i < n8; i += stride) {
        float4 a = l4[2 * i];
        float4 b = l4[2 * i + 1];
        const uint4 v = load_cv_u4(p4 + i);
        __half2 p0, p1, p2, p3;
        std::memcpy(&p0, &v.x, 4);
        std::memcpy(&p1, &v.y, 4);
        std::memcpy(&p2, &v.z, 4);
        std::memcpy(&p3, &v.w, 4);
        const float2 f0 = __half22float2(p0);
        const float2 f1 = __half22float2(p1);
        const float2 f2 = __half22float2(p2);
        const float2 f3 = __half22float2(p3);
        a.x += f0.x; a.y += f0.y; a.z += f1.x; a.w += f1.y;
        b.x += f2.x; b.y += f2.y; b.z += f3.x; b.w += f3.y;
        l4[2 * i] = a;
        l4[2 * i + 1] = b;
    }
    for (int64_t i = (n8 << 3) + tid; i < n_elems; i += stride) {
        uint16_t raw = *(const volatile uint16_t*) (reinterpret_cast<const uint16_t*>(peer_dev) + i);
        __half hf;
        std::memcpy(&hf, &raw, 2);
        dst[i] += __half2float(hf);
    }
}

__global__ void tp_pack_q8_kernel(const float* __restrict__ src,
                                  int8_t* __restrict__ qd,
                                  half* __restrict__ qs,
                                  int64_t n_elems) {
    const int64_t blk = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t nblk = n_elems >> 5;
    const int64_t bstride = (int64_t) gridDim.x * (blockDim.x >> 5);
    const int lane = threadIdx.x & 31;
    for (int64_t b = blk; b < nblk; b += bstride) {
        const float v = src[b * 32 + lane];
        float am = fabsf(v);
        for (int mask = 16; mask > 0; mask >>= 1) {
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, mask));
        }
        const float d = am * (1.0f / 127.0f);
        const float id = am > 0.0f ? 127.0f / am : 0.0f;
        int q = __float2int_rn(v * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qd[b * 32 + lane] = (int8_t) q;
        if (lane == 0) qs[b] = __float2half(d);
    }
}

__global__ void tp_unpack_add_q8_kernel(float* __restrict__ dst,
                                        const int8_t* __restrict__ qd,
                                        const half* __restrict__ qs,
                                        int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n4 = n_elems >> 2;
    float4* l4 = reinterpret_cast<float4*>(dst);
    const char4* q4 = reinterpret_cast<const char4*>(qd);
    for (int64_t i = tid; i < n4; i += stride) {
        const float d = __half2float(qs[i >> 3]);
        const char4 c = q4[i];
        float4 a = l4[i];
        a.x += d * (float) c.x;
        a.y += d * (float) c.y;
        a.z += d * (float) c.z;
        a.w += d * (float) c.w;
        l4[i] = a;
    }
}

__global__ void tp_pack_bf16_push_kernel(const float* __restrict__ src,
                                         __nv_bfloat16* __restrict__ peer_dst,
                                         int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n_elems >> 3;
    const float4* l4 = reinterpret_cast<const float4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(peer_dst);
    for (int64_t i = tid; i < n8; i += stride) {
        const float4 a = l4[2 * i];
        const float4 b = l4[2 * i + 1];
        const __nv_bfloat162 p0 = __floats2bfloat162_rn(a.x, a.y);
        const __nv_bfloat162 p1 = __floats2bfloat162_rn(a.z, a.w);
        const __nv_bfloat162 p2 = __floats2bfloat162_rn(b.x, b.y);
        const __nv_bfloat162 p3 = __floats2bfloat162_rn(b.z, b.w);
        uint4 v;
        std::memcpy(&v.x, &p0, 4);
        std::memcpy(&v.y, &p1, 4);
        std::memcpy(&v.z, &p2, 4);
        std::memcpy(&v.w, &p3, 4);
        store_wt_u4(d4 + i, v);
    }
    for (int64_t i = (n8 << 3) + tid; i < n_elems; i += stride) {
        __nv_bfloat16 bf = __float2bfloat16(src[i]);
        uint16_t raw;
        std::memcpy(&raw, &bf, 2);
        *(volatile uint16_t*) (reinterpret_cast<uint16_t*>(peer_dst) + i) = raw;
    }
    if (threadIdx.x == 0) __threadfence_system();
}

__global__ void tp_pack_fp16_push_kernel(const float* __restrict__ src,
                                         __half* __restrict__ peer_dst,
                                         int64_t n_elems) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n_elems >> 3;
    const float4* l4 = reinterpret_cast<const float4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(peer_dst);
    for (int64_t i = tid; i < n8; i += stride) {
        const float4 a = l4[2 * i];
        const float4 b = l4[2 * i + 1];
        const __half2 p0 = __floats2half2_rn(a.x, a.y);
        const __half2 p1 = __floats2half2_rn(a.z, a.w);
        const __half2 p2 = __floats2half2_rn(b.x, b.y);
        const __half2 p3 = __floats2half2_rn(b.z, b.w);
        uint4 v;
        std::memcpy(&v.x, &p0, 4);
        std::memcpy(&v.y, &p1, 4);
        std::memcpy(&v.z, &p2, 4);
        std::memcpy(&v.w, &p3, 4);
        store_wt_u4(d4 + i, v);
    }
    for (int64_t i = (n8 << 3) + tid; i < n_elems; i += stride) {
        __half hf = __float2half(src[i]);
        uint16_t raw;
        std::memcpy(&raw, &hf, 2);
        *(volatile uint16_t*) (reinterpret_cast<uint16_t*>(peer_dst) + i) = raw;
    }
    if (threadIdx.x == 0) __threadfence_system();
}

#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
struct CudaVmmApi {
    using FnGran = CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
    using FnCreate = CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    using FnReserve = CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    using FnMap = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    using FnSetAccess = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    using FnUnmap = CUresult (*)(CUdeviceptr, size_t);
    using FnAddrFree = CUresult (*)(CUdeviceptr, size_t);
    using FnRelease = CUresult (*)(CUmemGenericAllocationHandle);
    using FnPtrAttr = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);
    FnGran gran = nullptr;
    FnCreate create = nullptr;
    FnReserve reserve = nullptr;
    FnMap map = nullptr;
    FnSetAccess set_access = nullptr;
    FnUnmap unmap = nullptr;
    FnAddrFree addr_free = nullptr;
    FnRelease release = nullptr;
    FnPtrAttr ptr_attr = nullptr;
    bool ok = false;

    static const CudaVmmApi& get() {
        static const CudaVmmApi api = [] {
            CudaVmmApi a;
            void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
            if (!lib) lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
            if (!lib) return a;
            a.gran = reinterpret_cast<FnGran>(dlsym(lib, "cuMemGetAllocationGranularity"));
            a.create = reinterpret_cast<FnCreate>(dlsym(lib, "cuMemCreate"));
            a.reserve = reinterpret_cast<FnReserve>(dlsym(lib, "cuMemAddressReserve"));
            a.map = reinterpret_cast<FnMap>(dlsym(lib, "cuMemMap"));
            a.set_access = reinterpret_cast<FnSetAccess>(dlsym(lib, "cuMemSetAccess"));
            a.unmap = reinterpret_cast<FnUnmap>(dlsym(lib, "cuMemUnmap"));
            a.addr_free = reinterpret_cast<FnAddrFree>(dlsym(lib, "cuMemAddressFree"));
            a.release = reinterpret_cast<FnRelease>(dlsym(lib, "cuMemRelease"));
            a.ptr_attr = reinterpret_cast<FnPtrAttr>(dlsym(lib, "cuPointerGetAttribute"));
            a.ok = a.gran && a.create && a.reserve && a.map && a.set_access &&
                   a.unmap && a.addr_free && a.release && a.ptr_attr;
            return a;
        }();
        return api;
    }
};

static bool vmm_alloc_peer_slab(int owner, int peer, size_t req_bytes,
                                void*& out_ptr, uint64_t& out_handle, size_t& out_bytes) {
    const auto& api = CudaVmmApi::get();
    if (!api.ok) return false;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = owner;
    size_t gran = 0;
    CUresult cr = api.gran(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (cr != CUDA_SUCCESS || gran == 0) {
        std::fprintf(stderr, "strata tp_comm: vmm gran failed (owner=%d cr=%d)\n", owner, (int) cr);
        return false;
    }
    const size_t bytes = (req_bytes + gran - 1u) & ~(gran - 1u);
    CUmemGenericAllocationHandle handle = 0;
    cr = api.create(&handle, bytes, &prop, 0);
    if (cr != CUDA_SUCCESS) {
        std::fprintf(stderr, "strata tp_comm: vmm create failed (owner=%d bytes=%zu cr=%d)\n", owner, bytes, (int) cr);
        return false;
    }

    // Retry VA placement if driver returns a bad VA hole (geforce-dynbar1-p2p D-1 guard)
    CUdeviceptr bad_vas[4] = {};
    int n_bad = 0;
    CUdeviceptr ptr = 0;
    bool mapped = false;
    for (int attempt = 0; attempt < 4; ++attempt) {
        cr = api.reserve(&ptr, bytes, gran, 0, 0);
        if (cr != CUDA_SUCCESS) {
            std::fprintf(stderr, "strata tp_comm: vmm reserve failed (owner=%d cr=%d)\n", owner, (int) cr);
            break;
        }
        cr = api.map(ptr, bytes, 0, handle, 0);
        if (cr != CUDA_SUCCESS) {
            std::fprintf(stderr, "strata tp_comm: vmm map failed (owner=%d cr=%d)\n", owner, (int) cr);
            api.addr_free(ptr, bytes);
            ptr = 0;
            break;
        }
        CUmemAccessDesc descs[2]{};
        descs[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        descs[0].location.id = owner;
        descs[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        descs[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        descs[1].location.id = peer;
        descs[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cr = api.set_access(ptr, bytes, descs, 2);
        if (cr != CUDA_SUCCESS) {
            std::fprintf(stderr, "strata tp_comm: vmm set_access failed (owner=%d peer=%d cr=%d)\n", owner, peer, (int) cr);
            api.unmap(ptr, bytes);
            api.addr_free(ptr, bytes);
            ptr = 0;
            break;
        }
        unsigned int mtype = 0;
        cudaSetDevice(owner);
        CUresult cr_attr = api.ptr_attr(&mtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
        cudaError_t ce_ms = cudaMemset(reinterpret_cast<void*>(ptr), 0, bytes);
        if (ce_ms == cudaSuccess) ce_ms = cudaDeviceSynchronize();
        if (cr_attr == CUDA_SUCCESS && mtype == CU_MEMORYTYPE_DEVICE && ce_ms == cudaSuccess) {
            mapped = true;
            break;
        }
        std::fprintf(stderr, "strata tp_comm: vmm attempt %d rejected (ptr=0x%llx cr_attr=%d mtype=%u ce_ms=%d)\n",
                     attempt, (unsigned long long) ptr, (int) cr_attr, mtype, (int) ce_ms);
        (void) cudaGetLastError();
        api.unmap(ptr, bytes);
        bad_vas[n_bad++] = ptr;
        ptr = 0;
    }
    for (int i = 0; i < n_bad; ++i) api.addr_free(bad_vas[i], bytes);
    if (!mapped) {
        api.release(handle);
        return false;
    }
    out_ptr = reinterpret_cast<void*>(ptr);
    out_handle = static_cast<uint64_t>(handle);
    out_bytes = bytes;
    return true;
}

static void vmm_free_peer_slab(void*& ptr, uint64_t& handle, size_t& bytes) {
    const auto& api = CudaVmmApi::get();
    if (api.ok && ptr != nullptr && bytes > 0) {
        api.unmap(reinterpret_cast<CUdeviceptr>(ptr), bytes);
        api.addr_free(reinterpret_cast<CUdeviceptr>(ptr), bytes);
    }
    if (api.ok && handle != 0) {
        api.release(static_cast<CUmemGenericAllocationHandle>(handle));
    }
    ptr = nullptr;
    handle = 0;
    bytes = 0;
}
#endif

}  // namespace

TpComm::~TpComm() {
    release();
}

void TpComm::release_pipe() {
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    for (int r = 0; r < 2; ++r) {
        if (!pipe_init_[r] && !p2p_pipe_init_[r] && !s_rx_[r] && !d_tx_[r][0] && !d_rx_[r][0] &&
            !d_dec_buf_[r] && !d_p2p_tx_[r][0] && !d_p2p_flag_[r] && !vmm_base_[r]) continue;
        cudaSetDevice(dev_[r]);
        if (s_rx_[r] != nullptr) {
            cudaStreamDestroy((cudaStream_t) s_rx_[r]);
            s_rx_[r] = nullptr;
        }
        if (ev_bar_[r] != nullptr) {
            cudaEventDestroy((cudaEvent_t) ev_bar_[r]);
            ev_bar_[r] = nullptr;
        }
        if (ev_done_[r] != nullptr) {
            cudaEventDestroy((cudaEvent_t) ev_done_[r]);
            ev_done_[r] = nullptr;
        }
        for (int s = 0; s < kP2pPipeSlots; ++s) {
            if (ev_p2p_rx_[r][s] != nullptr) {
                cudaEventDestroy((cudaEvent_t) ev_p2p_rx_[r][s]);
                ev_p2p_rx_[r][s] = nullptr;
            }
        }
        p2p_pipe_init_[r] = false;
        if (p2p_vmm_) {
#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
            vmm_free_peer_slab(vmm_base_[r], vmm_handle_[r], vmm_bytes_[r]);
#endif
            d_dec_buf_[r] = nullptr;
            d_p2p_flag_[r] = nullptr;
            d_p2p_head_arg_[r] = nullptr;
            d_p2p_head_flag_[r] = nullptr;
            d_p2p_logits_flag_[r] = nullptr;
            d_p2p_swap_flag_[r] = nullptr;
            d_p2p_df2_flag_[r] = nullptr;
            d_p2p_tx_[r][0] = nullptr;
            d_p2p_tx_[r][1] = nullptr;
            if (d_done_ctr_[r]) { cudaFree(d_done_ctr_[r]); d_done_ctr_[r] = nullptr; }
        } else {
            if (d_dec_buf_[r] != nullptr) {
                cudaFree(d_dec_buf_[r]);
                d_dec_buf_[r] = nullptr;
            }
            if (d_p2p_flag_[r] != nullptr) {
                cudaFree(d_p2p_flag_[r]);
                d_p2p_flag_[r] = nullptr;
            }
            d_p2p_head_arg_[r] = nullptr;
            d_p2p_head_flag_[r] = nullptr;
            d_p2p_logits_flag_[r] = nullptr;
            for (int b = 0; b < 2; ++b) {
                if (d_p2p_tx_[r][b] != nullptr) { cudaFree(d_p2p_tx_[r][b]); d_p2p_tx_[r][b] = nullptr; }
            }
        }
        for (int b = 0; b < 2; ++b) {
            if (d_tx_[r][b] != nullptr) { cudaFree(d_tx_[r][b]); d_tx_[r][b] = nullptr; }
            if (d_rx_[r][b] != nullptr) { cudaFree(d_rx_[r][b]); d_rx_[r][b] = nullptr; }
            if (ev_rx_[r][b] != nullptr) { cudaEventDestroy((cudaEvent_t) ev_rx_[r][b]); ev_rx_[r][b] = nullptr; }
        }
        pipe_init_[r] = false;
    }
    p2p_vmm_ = false;
    pipe_bytes_ = 0;
    cudaSetDevice(prev_dev);
}

bool TpComm::ensure_pipe(int rank) {
    if (pipe_init_[rank]) return true;
    const size_t slot_bytes = slot_floats_ * sizeof(float);
    if (slot_bytes == 0) return false;
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(dev_[rank]);
    pipe_bytes_ = slot_bytes;
    int low_p = 0, high_p = 0;
    cudaDeviceGetStreamPriorityRange(&low_p, &high_p);
    cudaStream_t srx = nullptr;
    cudaEvent_t eb = nullptr, ed = nullptr;
    if (cudaStreamCreateWithPriority(&srx, cudaStreamNonBlocking, high_p) != cudaSuccess ||
        cudaEventCreateWithFlags(&eb, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ed, cudaEventDisableTiming) != cudaSuccess) {
        if (srx) cudaStreamDestroy(srx);
        if (eb) cudaEventDestroy(eb);
        if (ed) cudaEventDestroy(ed);
        cudaSetDevice(prev_dev);
        return false;
    }
    s_rx_[rank] = srx;
    ev_bar_[rank] = eb;
    ev_done_[rank] = ed;
    for (int b = 0; b < 2; ++b) {
        cudaEvent_t er = nullptr;
        if (cudaMalloc(&d_tx_[rank][b], slot_bytes) != cudaSuccess ||
            cudaMalloc(&d_rx_[rank][b], slot_bytes) != cudaSuccess ||
            cudaEventCreateWithFlags(&er, cudaEventDisableTiming) != cudaSuccess) {
            if (er) cudaEventDestroy(er);
            cudaSetDevice(prev_dev);
            return false;
        }
        ev_rx_[rank][b] = er;
    }
    pipe_init_[rank] = true;
    cudaSetDevice(prev_dev);
    return true;
}

bool TpComm::ensure_p2p_pipe(int rank) {
    if (p2p_pipe_init_[rank]) return true;
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(dev_[rank]);
    int low_p = 0, high_p = 0;
    cudaDeviceGetStreamPriorityRange(&low_p, &high_p);
    if (s_rx_[rank] == nullptr) {
        cudaStream_t srx = nullptr;
        if (cudaStreamCreateWithPriority(&srx, cudaStreamNonBlocking, high_p) != cudaSuccess) {
            cudaSetDevice(prev_dev);
            return false;
        }
        s_rx_[rank] = srx;
    }
    if (ev_bar_[rank] == nullptr) {
        cudaEvent_t eb = nullptr;
        if (cudaEventCreateWithFlags(&eb, cudaEventDisableTiming) != cudaSuccess) {
            cudaSetDevice(prev_dev);
            return false;
        }
        ev_bar_[rank] = eb;
    }
    if (ev_done_[rank] == nullptr) {
        cudaEvent_t ed = nullptr;
        if (cudaEventCreateWithFlags(&ed, cudaEventDisableTiming) != cudaSuccess) {
            cudaSetDevice(prev_dev);
            return false;
        }
        ev_done_[rank] = ed;
    }
    for (int s = 0; s < kP2pPipeSlots; ++s) {
        if (ev_p2p_rx_[rank][s] == nullptr) {
            cudaEvent_t er = nullptr;
            if (cudaEventCreateWithFlags(&er, cudaEventDisableTiming) != cudaSuccess) {
                cudaSetDevice(prev_dev);
                return false;
            }
            ev_p2p_rx_[rank][s] = er;
        }
    }
    p2p_pipe_init_[rank] = true;
    cudaSetDevice(prev_dev);
    return true;
}

void TpComm::release() {
    release_pipe();
    for (int r = 0; r < 2; ++r) {
        if (h_buf_[r] != nullptr) {
            cudaFreeHost(h_buf_[r]);
            h_buf_[r] = nullptr;
        }
        if (h_flag_[r] != nullptr) {
            cudaFreeHost(h_flag_[r]);
            h_flag_[r] = nullptr;
        }
        for (int j = 0; j < 2; ++j) {
            m_buf_[r][j] = nullptr;
            m_flag_[r][j] = nullptr;
        }
    }
    p2p_ = false;
    p2p_dec_ = false;
    max_floats_ = 0;
    slot_floats_ = 0;
}

bool TpComm::init(int dev0, int dev1, size_t max_floats, std::string& err) {
    release();
    dev_[0] = dev0;
    dev_[1] = dev1;
    slot_floats_ = (max_floats + 63u) & ~size_t(63u);
    max_floats_ = slot_floats_ * kRingSlots;
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    {
        const char* pe = std::getenv("STRATA_TP_AR_P2P");
        const bool want = pe == nullptr || std::atoi(pe) != 0;
        int can01 = 0, can10 = 0;
        if (want) {
            cudaDeviceCanAccessPeer(&can01, dev_[0], dev_[1]);
            cudaDeviceCanAccessPeer(&can10, dev_[1], dev_[0]);
        }
        if (want && can01 != 0 && can10 != 0) {
            const size_t dec_bytes = kRingSlots * kP2pDecSlotFloats * sizeof(float);
            const size_t pipe_tx_bytes = (size_t) kP2pPipeElems * sizeof(uint16_t);
            const size_t slab_bytes = dec_bytes + 2u * pipe_tx_bytes + 4096u;
            bool alloc_ok = false;

#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
            // Preferred path: CUDA Driver VMM selective peer BAR1 mapping (~36 MiB per GPU).
            // Never calls process-wide cudaDeviceEnablePeerAccess, so model weights/KV consume 0 bytes of BAR1
            // (works natively on 256 MiB BAR1 dynamic-P2P cards as well as ReBAR / NVLink).
            const char* vmm_env = std::getenv("STRATA_TP_AR_P2P_VMM");
            const bool want_vmm = vmm_env == nullptr || std::atoi(vmm_env) != 0;
            if (want_vmm && CudaVmmApi::get().ok) {
                cudaSetDevice(dev_[0]); (void) cudaFree(0);
                cudaSetDevice(dev_[1]); (void) cudaFree(0);
                if (vmm_alloc_peer_slab(dev_[0], dev_[1], slab_bytes, vmm_base_[0], vmm_handle_[0], vmm_bytes_[0]) &&
                    vmm_alloc_peer_slab(dev_[1], dev_[0], slab_bytes, vmm_base_[1], vmm_handle_[1], vmm_bytes_[1])) {
                    for (int r = 0; r < 2; ++r) {
                        uint8_t* base = reinterpret_cast<uint8_t*>(vmm_base_[r]);
                        d_dec_buf_[r] = reinterpret_cast<float*>(base);
                        d_p2p_tx_[r][0] = base + dec_bytes;
                        d_p2p_tx_[r][1] = base + dec_bytes + pipe_tx_bytes;
                        uint8_t* tail = base + dec_bytes + 2u * pipe_tx_bytes;
                        d_p2p_flag_[r] = reinterpret_cast<uint32_t*>(tail);
                        d_p2p_head_arg_[r] = reinterpret_cast<void*>(tail + 256u);
                        d_p2p_head_flag_[r] = reinterpret_cast<uint32_t*>(tail + 512u);
                        d_p2p_logits_flag_[r] = reinterpret_cast<uint32_t*>(tail + 576u);
                        d_p2p_swap_flag_[r] = reinterpret_cast<uint32_t*>(tail + 640u);
                        d_p2p_df2_flag_[r] = reinterpret_cast<uint32_t*>(tail + 704u);
                    }
                    p2p_vmm_ = true;
                    alloc_ok = true;
                } else {
                    vmm_free_peer_slab(vmm_base_[0], vmm_handle_[0], vmm_bytes_[0]);
                    vmm_free_peer_slab(vmm_base_[1], vmm_handle_[1], vmm_bytes_[1]);
                }
            }
#endif
            if (!alloc_ok) {
                cudaSetDevice(dev_[0]);
                cudaError_t e0 = cudaDeviceEnablePeerAccess(dev_[1], 0);
                if (e0 == cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); e0 = cudaSuccess; }
                cudaSetDevice(dev_[1]);
                cudaError_t e1 = cudaDeviceEnablePeerAccess(dev_[0], 0);
                if (e1 == cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); e1 = cudaSuccess; }
                if (e0 == cudaSuccess && e1 == cudaSuccess) {
                    alloc_ok = true;
                    for (int r = 0; r < 2 && alloc_ok; ++r) {
                        cudaSetDevice(dev_[r]);
                        if (cudaMalloc((void**) &d_dec_buf_[r], dec_bytes) != cudaSuccess ||
                            cudaMalloc((void**) &d_p2p_flag_[r], 64) != cudaSuccess) {
                            alloc_ok = false;
                            break;
                        }
                        cudaMemset(d_dec_buf_[r], 0, dec_bytes);
                        cudaMemset(d_p2p_flag_[r], 0, 64);
                        for (int b = 0; b < 2; ++b) {
                            if (cudaMalloc(&d_p2p_tx_[r][b], pipe_tx_bytes) != cudaSuccess) {
                                alloc_ok = false;
                                break;
                            }
                        }
                    }
                }
            }
            if (alloc_ok) {
                // Self-verify bidirectional direct BAR1/NVLink kernel push-stores (#2666 guard against broken host bridges/ACS)
                constexpr int kTestN = 64;
                uint32_t verify[kTestN];
                for (int r = 0; r < 2 && alloc_ok; ++r) {
                    const uint32_t magic = r == 0 ? 0x5A3C96F1u : 0xA5C3690Eu;
                    cudaSetDevice(dev_[r]);
                    tp_p2p_selftest_write_kernel<<<1, kTestN>>>(reinterpret_cast<uint32_t*>(d_dec_buf_[1 - r]), magic, kTestN);
                    tp_p2p_selftest_write_kernel<<<1, 1>>>(d_p2p_flag_[1 - r], magic, 1);
                    cudaError_t ce_sync = cudaDeviceSynchronize();
                    if (ce_sync != cudaSuccess) {
                        std::fprintf(stderr, "strata tp_comm: selftest r=%d vmm=%d sync err=%d (%s)\n",
                                     r, (int) p2p_vmm_, (int) ce_sync, cudaGetErrorString(ce_sync));
                        alloc_ok = false;
                        break;
                    }
                    cudaSetDevice(dev_[1 - r]);
                    std::memset(verify, 0, sizeof verify);
                    uint32_t fver = 0;
                    cudaError_t ce_c1 = cudaMemcpy(verify, d_dec_buf_[1 - r], sizeof verify, cudaMemcpyDeviceToHost);
                    cudaError_t ce_c2 = cudaMemcpy(&fver, d_p2p_flag_[1 - r], sizeof fver, cudaMemcpyDeviceToHost);
                    if (ce_c1 != cudaSuccess || ce_c2 != cudaSuccess) {
                        std::fprintf(stderr, "strata tp_comm: selftest r=%d vmm=%d cpy err=%d/%d\n",
                                     r, (int) p2p_vmm_, (int) ce_c1, (int) ce_c2);
                        alloc_ok = false;
                        break;
                    }
                    if (fver != magic) {
                        std::fprintf(stderr, "strata tp_comm: selftest r=%d vmm=%d flag mismatch got=0x%08x exp=0x%08x\n",
                                     r, (int) p2p_vmm_, fver, magic);
                        alloc_ok = false;
                        break;
                    }
                    for (int i = 0; i < kTestN; ++i) {
                        if (verify[i] != (magic ^ (uint32_t) (i * 2654435761u))) {
                            std::fprintf(stderr, "strata tp_comm: selftest r=%d vmm=%d data[%d] mismatch got=0x%08x exp=0x%08x\n",
                                         r, (int) p2p_vmm_, i, verify[i], magic ^ (uint32_t) (i * 2654435761u));
                            alloc_ok = false;
                            break;
                        }
                    }
                    if (alloc_ok) {
                        (void) cudaMemset(d_dec_buf_[1 - r], 0, sizeof verify);
                        (void) cudaMemset(d_p2p_flag_[1 - r], 0, 64);
                    }
                }
                if (!alloc_ok) {
                    std::fprintf(stderr, "strata tp_comm: P2P reported available but direct BAR1 store self-test failed; falling back to host route\n");
                }
            }
            if (alloc_ok) {
                p2p_ = true;
                const char* pdec = std::getenv("STRATA_TP_AR_P2P_DECODE");
                p2p_dec_ = (pdec == nullptr || std::atoi(pdec) != 0);
                for (int r = 0; r < 2; ++r) {
                    cudaSetDevice(dev_[r]);
                    if (cudaMalloc((void**) &d_done_ctr_[r], 64) == cudaSuccess) {
                        (void) cudaMemset(d_done_ctr_[r], 0, 64);
                    } else {
                        (void) cudaGetLastError();
                        d_done_ctr_[r] = nullptr;
                    }
                }
            } else {
                (void) cudaGetLastError();
                if (p2p_vmm_) {
#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
                    vmm_free_peer_slab(vmm_base_[0], vmm_handle_[0], vmm_bytes_[0]);
                    vmm_free_peer_slab(vmm_base_[1], vmm_handle_[1], vmm_bytes_[1]);
#endif
                    p2p_vmm_ = false;
                    for (int r = 0; r < 2; ++r) {
                        d_dec_buf_[r] = nullptr;
                        d_p2p_flag_[r] = nullptr;
                        d_p2p_tx_[r][0] = nullptr;
                        d_p2p_tx_[r][1] = nullptr;
                    }
                } else {
                    for (int r = 0; r < 2; ++r) {
                        cudaSetDevice(dev_[r]);
                        if (d_dec_buf_[r]) { cudaFree(d_dec_buf_[r]); d_dec_buf_[r] = nullptr; }
                        if (d_p2p_flag_[r]) { cudaFree(d_p2p_flag_[r]); d_p2p_flag_[r] = nullptr; }
            if (d_done_ctr_[r]) { cudaFree(d_done_ctr_[r]); d_done_ctr_[r] = nullptr; }
                        for (int b = 0; b < 2; ++b) {
                            if (d_p2p_tx_[r][b]) { cudaFree(d_p2p_tx_[r][b]); d_p2p_tx_[r][b] = nullptr; }
                        }
                    }
                }
            }
            std::fprintf(stderr, "strata tp_comm: NVLink/P2P %s (%s, decode=%s, CUDA%d <-> CUDA%d)\n",
                         p2p_ ? "enabled" : "off (host route)",
                         p2p_ ? (p2p_vmm_ ? "vmm-push" : "legacy-push") : "host",
                         p2p_dec_ ? "p2p" : "host",
                         dev_[0], dev_[1]);
        }
        cudaSetDevice(prev_dev);
    }
    const unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable;
    const size_t host_floats = p2p_ ? (kRingSlots * kP2pDecSlotFloats) : max_floats_;
    for (int r = 0; r < 2; ++r) {
        cudaSetDevice(dev_[r]);
        if (cudaHostAlloc((void**) &h_buf_[r], host_floats * sizeof(float), flags) != cudaSuccess ||
            cudaHostAlloc((void**) &h_flag_[r], 64, flags) != cudaSuccess) {
            err = "TpComm: cudaHostAlloc failed";
            cudaSetDevice(prev_dev);
            release();
            return false;
        }
        std::memset(h_buf_[r], 0, host_floats * sizeof(float));
        std::memset(h_flag_[r], 0, 64);
    }
    for (int r = 0; r < 2; ++r) {
        cudaSetDevice(dev_[r]);
        for (int j = 0; j < 2; ++j) {
            if (cudaHostGetDevicePointer((void**) &m_buf_[r][j], h_buf_[j], 0) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &m_flag_[r][j], h_flag_[j], 0) != cudaSuccess) {
                err = "TpComm: cudaHostGetDevicePointer failed";
                cudaSetDevice(prev_dev);
                release();
                return false;
            }
        }
    }
    cudaSetDevice(prev_dev);
    reset(0);
    return true;
}

void TpComm::reset(uint32_t base) {
    if (h_flag_[0] != nullptr) *(volatile uint32_t*) h_flag_[0] = base;
    if (h_flag_[1] != nullptr) *(volatile uint32_t*) h_flag_[1] = base;
    if (d_p2p_flag_[0] != nullptr || d_p2p_flag_[1] != nullptr) {
        int prev_dev = 0;
        cudaGetDevice(&prev_dev);
        for (int r = 0; r < 2; ++r) {
            if (d_p2p_flag_[r] != nullptr) {
                cudaSetDevice(dev_[r]);
                (void) cudaMemcpy(d_p2p_flag_[r], &base, sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
            if (d_p2p_head_flag_[r] != nullptr) {
                const uint32_t z = 0u;
                cudaSetDevice(dev_[r]);
                (void) cudaMemcpy(d_p2p_head_flag_[r], &z, sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
            if (d_p2p_logits_flag_[r] != nullptr) {
                const uint32_t z = 0u;
                cudaSetDevice(dev_[r]);
                (void) cudaMemcpy(d_p2p_logits_flag_[r], &z, sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
            if (d_p2p_swap_flag_[r] != nullptr) {
                const uint32_t z = 0u;
                cudaSetDevice(dev_[r]);
                (void) cudaMemcpy(d_p2p_swap_flag_[r], &z, sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
            if (d_p2p_df2_flag_[r] != nullptr) {
                const uint32_t z = 0u;
                cudaSetDevice(dev_[r]);
                (void) cudaMemcpy(d_p2p_df2_flag_[r], &z, sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
        }
        logits_step_ = 0;
        swap_step_ = 0;
        cudaSetDevice(prev_dev);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void TpComm::allreduce_add(float* local_buf, int n_elems, uint32_t step, int rank, void* stream) {
    static const int dec_mode = [] {
        const char* d = std::getenv("STRATA_TP_AR_DECODE");
        const char* v = (d != nullptr && d[0] != '\0') ? d : std::getenv("STRATA_TP_AR_MODE");
        if (v == nullptr) return 1;
        if (std::strcmp(v, "fp32") == 0 || std::strcmp(v, "0") == 0) return 0;
        if (std::strcmp(v, "fp16") == 0 || std::strcmp(v, "half") == 0) return 2;
        return 1;
    }();
    const bool use_p2p = p2p_dec_ && (size_t) n_elems <= kP2pDecSlotFloats && d_p2p_flag_[0] != nullptr && d_p2p_flag_[1] != nullptr;
    // Push-to-peer BAR1 write + VRAM flag (strict PCIe posted-write ordering) + local-VRAM read (0 slow P2P reads!)
    float* my_base = use_p2p ? d_dec_buf_[1 - rank] : m_buf_[rank][rank];
    const float* peer_base = use_p2p ? d_dec_buf_[rank] : m_buf_[rank][1 - rank];
    volatile uint32_t* my_flag = use_p2p ? d_p2p_flag_[1 - rank] : m_flag_[rank][rank];
    volatile uint32_t* peer_flag = use_p2p ? d_p2p_flag_[rank] : m_flag_[rank][1 - rank];
    const size_t s_floats = use_p2p ? kP2pDecSlotFloats : (p2p_ ? kP2pDecSlotFloats : slot_floats_);

    if (dec_mode >= 1 && (n_elems & 7) == 0 && (size_t) n_elems * 2u <= s_floats * sizeof(float)) {
        const int n8 = n_elems >> 3;
        if (dec_mode == 2) {
            if (n8 <= 1024) {
                tp_allreduce_fp16_reg_kernel<1><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else if (n8 <= 2048) {
                tp_allreduce_fp16_reg_kernel<2><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else if (n8 <= 3072) {
                tp_allreduce_fp16_reg_kernel<3><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else if (n8 <= 5120) {
                tp_allreduce_fp16_reg_kernel<5><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else {
                tp_allreduce_fp16_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            }
        } else {
            if (n8 <= 1024) {
                const int nth = ((n8 + 31) & ~31);
                tp_allreduce_bf16_reg_kernel<1><<<1, nth > 0 ? nth : 32, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else if (n8 <= 2048) {
                tp_allreduce_bf16_reg_kernel<2><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else if (n8 <= 3072) {
                tp_allreduce_bf16_reg_kernel<3><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else if (n8 <= 5120) {
                tp_allreduce_bf16_reg_kernel<5><<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            } else {
                tp_allreduce_bf16_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(
                    local_buf, my_base, peer_base, my_flag, peer_flag, n_elems, s_floats, step);
            }
        }
        return;
    }
    const int threads = n_elems > 32768 ? 1024 : 512;
    tp_allreduce_kernel<<<1, threads, 0, (cudaStream_t) stream>>>(
        local_buf, my_base, peer_base,
        my_flag, peer_flag, n_elems, s_floats, step);
}

bool TpComm::allreduce_add_gr_norm(float* local_buf, float* R_base, const float* inj_base,
                                   const float* w_norm, float* xn_out, float* rs_out,
                                   float eps, int n_tok, int N, int HC,
                                   uint32_t step, int rank, void* stream, bool also_write_bo) {
    static const int dec_mode = [] {
        const char* d = std::getenv("STRATA_TP_AR_DECODE");
        const char* v = (d != nullptr && d[0] != '\0') ? d : std::getenv("STRATA_TP_AR_MODE");
        if (v == nullptr) return 1;
        if (std::strcmp(v, "fp32") == 0 || std::strcmp(v, "0") == 0) return 0;
        if (std::strcmp(v, "fp16") == 0 || std::strcmp(v, "half") == 0) return 2;
        return 1;
    }();
    if (!p2p_dec_ || dec_mode != 1 || N != 2560 || HC != 4 || n_tok < 1 || n_tok > 8 ||
        d_p2p_flag_[0] == nullptr || d_p2p_flag_[1] == nullptr ||
        local_buf == nullptr || R_base == nullptr || inj_base == nullptr ||
        w_norm == nullptr || xn_out == nullptr || rs_out == nullptr)
        return false;
    if ((((uintptr_t) local_buf | (uintptr_t) R_base | (uintptr_t) w_norm | (uintptr_t) xn_out) & 15u) != 0)
        return false;
    const int n_elems = n_tok * N;
    if ((size_t) n_elems > kP2pDecSlotFloats) return false;
    float* my_base = d_dec_buf_[1 - rank];
    const float* peer_base = d_dec_buf_[rank];
    volatile uint32_t* my_flag = d_p2p_flag_[1 - rank];
    volatile uint32_t* peer_flag = d_p2p_flag_[rank];
    const int n8 = n_elems >> 3;
    if (n8 <= 1024) {
        const int nth = ((n8 + 31) & ~31);
        tp_allreduce_bf16_gr_norm_kernel<1><<<1, nth, 0, (cudaStream_t) stream>>>(
            local_buf, R_base, inj_base, w_norm, xn_out, rs_out, eps, n_tok,
            my_base, peer_base, my_flag, peer_flag, kP2pDecSlotFloats, step, also_write_bo);
    } else if (n8 <= 2048) {
        tp_allreduce_bf16_gr_norm_kernel<2><<<1, 1024, 0, (cudaStream_t) stream>>>(
            local_buf, R_base, inj_base, w_norm, xn_out, rs_out, eps, n_tok,
            my_base, peer_base, my_flag, peer_flag, kP2pDecSlotFloats, step, also_write_bo);
    } else {
        tp_allreduce_bf16_gr_norm_kernel<3><<<1, 1024, 0, (cudaStream_t) stream>>>(
            local_buf, R_base, inj_base, w_norm, xn_out, rs_out, eps, n_tok,
            my_base, peer_base, my_flag, peer_flag, kP2pDecSlotFloats, step, also_write_bo);
    }
    return true;
}

bool TpComm::p2p_push_target(int n_elems, uint32_t step, int rank, P2pPushTarget& out) const {
    static const int dec_mode = [] {
        const char* d = std::getenv("STRATA_TP_AR_DECODE");
        const char* v = (d != nullptr && d[0] != '\0') ? d : std::getenv("STRATA_TP_AR_MODE");
        if (v == nullptr) return 1;
        if (std::strcmp(v, "fp32") == 0 || std::strcmp(v, "0") == 0) return 0;
        if (std::strcmp(v, "fp16") == 0 || std::strcmp(v, "half") == 0) return 2;
        return 1;
    }();
    if (!p2p_dec_ || dec_mode != 1 || (n_elems & 7) != 0 ||
        (size_t) n_elems > kP2pDecSlotFloats ||
        d_dec_buf_[1 - rank] == nullptr || d_p2p_flag_[0] == nullptr ||
        d_p2p_flag_[1] == nullptr || d_done_ctr_[rank] == nullptr)
        return false;
    const size_t slot_off = (size_t) (step & (kRingSlots - 1u)) * kP2pDecSlotFloats;
    out.peer_slot_buf = d_dec_buf_[1 - rank] + slot_off;
    out.peer_flag = d_p2p_flag_[1 - rank];
    out.done_ctr = d_done_ctr_[rank];
    out.step = step;
    return true;
}

void TpComm::allreduce_wait_add_bf16(float* local_buf, int n_elems, uint32_t step, int rank, void* stream) {
    const int n8 = n_elems >> 3;
    const float* peer_base = d_dec_buf_[rank];
    volatile uint32_t* peer_flag = d_p2p_flag_[rank];
    const size_t s_floats = kP2pDecSlotFloats;
    if (n8 <= 1024) {
        const int nth = ((n8 + 31) & ~31);
        tp_allreduce_wait_add_bf16_kernel<1><<<1, nth > 0 ? nth : 32, 0, (cudaStream_t) stream>>>(
            local_buf, peer_base, peer_flag, n_elems, s_floats, step);
    } else if (n8 <= 2048) {
        tp_allreduce_wait_add_bf16_kernel<2><<<1, 1024, 0, (cudaStream_t) stream>>>(
            local_buf, peer_base, peer_flag, n_elems, s_floats, step);
    } else if (n8 <= 3072) {
        tp_allreduce_wait_add_bf16_kernel<3><<<1, 1024, 0, (cudaStream_t) stream>>>(
            local_buf, peer_base, peer_flag, n_elems, s_floats, step);
    } else if (n8 <= 5120) {
        tp_allreduce_wait_add_bf16_kernel<5><<<1, 1024, 0, (cudaStream_t) stream>>>(
            local_buf, peer_base, peer_flag, n_elems, s_floats, step);
    } else {
        tp_allreduce_wait_add_bf16_kernel<8><<<1, 1024, 0, (cudaStream_t) stream>>>(
            local_buf, peer_base, peer_flag, n_elems, s_floats, step);
    }
}

bool TpComm::p2p_gather_logits_rank0(const float* logits0_dev, const float* logits1_dev,
                                     int T, int64_t half0, int64_t half1, int64_t full_vocab,
                                     float* full_logits_dev0, void* stream0, void* stream1) {
    if (!p2p_ || !p2p_vmm_ || T <= 0 || half0 <= 0 || half1 <= 0 ||
        d_p2p_tx_[0][0] == nullptr || d_p2p_logits_flag_[0] == nullptr ||
        logits0_dev == nullptr || logits1_dev == nullptr || full_logits_dev0 == nullptr)
        return false;
    const size_t peer_bytes = (size_t) T * (size_t) half1 * sizeof(float);
    const size_t vmm_tx_cap = 2u * (size_t) kP2pPipeElems * sizeof(uint16_t);
    if (peer_bytes > vmm_tx_cap) return false;

    const uint32_t cur_step = ++logits_step_;
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);

    // 1. Launch GPU1 direct P2P push of its half-vocab logits into GPU0's VMM BAR1 slab (idle during decode)
    {
        cudaSetDevice(dev_[1]);
        const cudaStream_t cs1 = (cudaStream_t) stream1;
        const int64_t n_elems = (int64_t) T * half1;
        const int64_t n4 = (n_elems + 3) >> 2;
        const int blocks = (int) ((n4 + 255) / 256 < 256 ? (n4 + 255) / 256 : 256);
        tp_push_logits_f32_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs1>>>(
            logits1_dev, reinterpret_cast<float*>(d_p2p_tx_[0][0]), n_elems);
        tp_signal_flag_kernel<<<1, 1, 0, cs1>>>(d_p2p_logits_flag_[0], cur_step);
        (void) cudaStreamQuery(cs1);
    }

    // 2. Overlap GPU0's own local HBM 2D scatter with GPU1's P2P push, then wait flag and scatter GPU1's half
    {
        cudaSetDevice(dev_[0]);
        const cudaStream_t cs0 = (cudaStream_t) stream0;
        const int bx0 = (int) (((half0 >> 2) + 255) / 256 < 128 ? ((half0 >> 2) + 255) / 256 : 128);
        tp_scatter_logits_2d_kernel<<<dim3((unsigned) (bx0 > 0 ? bx0 : 1), (unsigned) T), 256, 0, cs0>>>(
            logits0_dev, full_logits_dev0, T, half0, full_vocab, 0);
        tp_wait_flag_kernel<<<1, 1, 0, cs0>>>(d_p2p_logits_flag_[0], cur_step);
        const int bx1 = (int) (((half1 >> 2) + 255) / 256 < 128 ? ((half1 >> 2) + 255) / 256 : 128);
        tp_scatter_logits_2d_kernel<<<dim3((unsigned) (bx1 > 0 ? bx1 : 1), (unsigned) T), 256, 0, cs0>>>(
            reinterpret_cast<const float*>(d_p2p_tx_[0][0]), full_logits_dev0, T, half1, full_vocab, half0);
    }

    cudaSetDevice(prev_dev);
    return true;
}

bool TpComm::p2p_swap_expert_blobs(void* slot0_dev, void* slot1_dev, size_t bytes,
                                   void* stream0, void* stream1) {
    if (!p2p_ || !p2p_vmm_ || bytes == 0 ||
        slot0_dev == nullptr || slot1_dev == nullptr ||
        d_p2p_tx_[0][0] == nullptr || d_p2p_tx_[1][0] == nullptr ||
        d_p2p_swap_flag_[0] == nullptr || d_p2p_swap_flag_[1] == nullptr)
        return false;
    const size_t max_slot = (size_t) kP2pPipeElems * sizeof(uint16_t);
    if (bytes > max_slot) return false;
    if ((((uintptr_t) slot0_dev | (uintptr_t) slot1_dev) & 15u) != 0) return false;

    const uint32_t cur_step = ++swap_step_;
    const size_t n16 = (bytes + 15u) >> 4;
    const int blocks = (int) ((n16 + 255u) / 256u < 256u ? (n16 + 255u) / 256u : 256u);
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);

    // Full-duplex 50.86 GB/s bidirectional push into peer's idle VMM BAR1 slab (0 DDR4 traffic!)
    {
        cudaSetDevice(dev_[0]);
        const cudaStream_t cs0 = (cudaStream_t) stream0;
        tp_push_bytes_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs0>>>(
            reinterpret_cast<const uint8_t*>(slot0_dev),
            reinterpret_cast<uint8_t*>(d_p2p_tx_[1][0]), bytes);
        tp_signal_flag_kernel<<<1, 1, 0, cs0>>>(d_p2p_swap_flag_[1], cur_step);
        (void) cudaStreamQuery(cs0);
    }
    {
        cudaSetDevice(dev_[1]);
        const cudaStream_t cs1 = (cudaStream_t) stream1;
        tp_push_bytes_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs1>>>(
            reinterpret_cast<const uint8_t*>(slot1_dev),
            reinterpret_cast<uint8_t*>(d_p2p_tx_[0][0]), bytes);
        tp_signal_flag_kernel<<<1, 1, 0, cs1>>>(d_p2p_swap_flag_[0], cur_step);
        (void) cudaStreamQuery(cs1);
    }
    // Wait peer arrival, then copy locally at 760 GB/s HBM (~4 us) into the swapped slot
    {
        cudaSetDevice(dev_[0]);
        const cudaStream_t cs0 = (cudaStream_t) stream0;
        tp_wait_flag_kernel<<<1, 1, 0, cs0>>>(d_p2p_swap_flag_[0], cur_step);
        if (cudaMemcpyAsync(slot0_dev, d_p2p_tx_[0][0], bytes, cudaMemcpyDeviceToDevice, cs0) != cudaSuccess) {
            cudaSetDevice(prev_dev);
            return false;
        }
    }
    {
        cudaSetDevice(dev_[1]);
        const cudaStream_t cs1 = (cudaStream_t) stream1;
        tp_wait_flag_kernel<<<1, 1, 0, cs1>>>(d_p2p_swap_flag_[1], cur_step);
        if (cudaMemcpyAsync(slot1_dev, d_p2p_tx_[1][0], bytes, cudaMemcpyDeviceToDevice, cs1) != cudaSuccess) {
            cudaSetDevice(prev_dev);
            return false;
        }
    }
    cudaSetDevice(prev_dev);
    return true;
}

void TpComm::allreduce_add_large(float* local_buf, int64_t n_elems, uint32_t& step, int rank, void* stream) {
    if (n_elems <= 0 || slot_floats_ == 0) return;
    // Mode: "bf16" (1), "q8" (2), "fp16"/"half" (3, hardware on sm_75+), "fp32" (0)
    static const int ar_mode = [] {
        const char* v = std::getenv("STRATA_TP_AR_MODE");
        if (v == nullptr) return 1;
        if (std::strcmp(v, "fp32") == 0 || std::strcmp(v, "0") == 0) return 0;
        if (std::strcmp(v, "q8") == 0 || std::strcmp(v, "2") == 0) return 2;
        if (std::strcmp(v, "fp16") == 0 || std::strcmp(v, "half") == 0 || std::strcmp(v, "3") == 0) return 3;
        return 1;
    }();
    const cudaStream_t cs = (cudaStream_t) stream;
    int eff_mode = ar_mode;
    if (eff_mode == 2 && (n_elems & 31) != 0) eff_mode = 1;

    if (p2p_ && (eff_mode == 1 || eff_mode == 3) &&
        d_p2p_tx_[0][0] != nullptr && d_p2p_tx_[1][0] != nullptr &&
        d_p2p_flag_[0] != nullptr && d_p2p_flag_[1] != nullptr) {
        static const bool p2p_pipe_on = [] {
            const char* v = std::getenv("STRATA_TP_P2P_PIPE");
            return v != nullptr && std::atoi(v) != 0;   // opt-in (A/B showed 16 MiB double-buffered P2P beats 4 MiB micro-chunks)
        }();
        static const int64_t p2p_pipe_elems = [] {
            const char* v = std::getenv("STRATA_TP_P2P_CHUNK_MB");
            int mb = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 4;
            if (mb < 1) mb = 1;
            if (mb > 4) mb = 4;  // 8 slots * 4 MiB = 32 MiB (exact fit in 2 * 16 MiB VMM slab)
            return ((int64_t) mb << 20) / 2;  // 2 bytes per bf16/fp16 elem
        }();
        if (p2p_pipe_on && p2p_vmm_ && n_elems > p2p_pipe_elems && ensure_p2p_pipe(rank)) {
            const int64_t cap = p2p_pipe_elems;
            const size_t slot_bytes = (size_t) cap * sizeof(uint16_t);
            uint8_t* const peer_ring = reinterpret_cast<uint8_t*>(d_p2p_tx_[1 - rank][0]);
            uint8_t* const my_ring = reinterpret_cast<uint8_t*>(d_p2p_tx_[rank][0]);
            const cudaStream_t srx = (cudaStream_t) s_rx_[rank];
            const cudaEvent_t eb = (cudaEvent_t) ev_bar_[rank];
            const cudaEvent_t ed = (cudaEvent_t) ev_done_[rank];
            int64_t k = 0;
            for (int64_t off = 0; off < n_elems; off += cap, ++k) {
                const int64_t chunk = (n_elems - off) < cap ? (n_elems - off) : cap;
                const uint32_t cur_step = ++step;
                const int s = (int) (cur_step & (kP2pPipeSlots - 1));
                const int64_t n8 = (chunk + 7) >> 3;
                const int blocks = (int) ((n8 + 255) / 256 < 256 ? (n8 + 255) / 256 : 256);
                void* const peer_dst = peer_ring + (size_t) s * slot_bytes;
                const void* const my_src = my_ring + (size_t) s * slot_bytes;
                if (eff_mode == 3) {
                    tp_pack_fp16_push_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                        local_buf + off, reinterpret_cast<__half*>(peer_dst), chunk);
                } else {
                    tp_pack_bf16_push_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                        local_buf + off, reinterpret_cast<__nv_bfloat16*>(peer_dst), chunk);
                }
                // Wait for my unpack of (cur_step - 2) BEFORE barrier(cur_step) so both peers are guaranteed done
                // with slot (cur_step - 2) before it can ever be reused 6 steps later (8-slot ring).
                if (k >= 2) {
                    const int s_old = (int) ((cur_step - 2u) & (kP2pPipeSlots - 1));
                    cudaStreamWaitEvent(cs, (cudaEvent_t) ev_p2p_rx_[rank][s_old], 0);
                }
                tp_barrier_kernel<<<1, 1, 0, cs>>>(d_p2p_flag_[1 - rank], d_p2p_flag_[rank], cur_step);
                cudaEventRecord(eb, cs);
                cudaStreamWaitEvent(srx, eb, 0);
                if (eff_mode == 3) {
                    tp_unpack_add_fp16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, srx>>>(
                        local_buf + off, reinterpret_cast<const __half*>(my_src), chunk);
                } else {
                    tp_unpack_add_bf16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, srx>>>(
                        local_buf + off, reinterpret_cast<const __nv_bfloat16*>(my_src), chunk);
                }
                cudaEventRecord((cudaEvent_t) ev_p2p_rx_[rank][s], srx);
            }
            cudaEventRecord(ed, srx);
            cudaStreamWaitEvent(cs, ed, 0);
            return;
        }
        const int64_t cap = kP2pPipeElems;
        for (int64_t off = 0; off < n_elems; off += cap) {
            const int64_t chunk = (n_elems - off) < cap ? (n_elems - off) : cap;
            const uint32_t cur_step = ++step;
            const int b = (int) (cur_step & 1u);
            const int64_t n8 = (chunk + 7) >> 3;
            const int blocks = (int) ((n8 + 255) / 256 < 256 ? (n8 + 255) / 256 : 256);
            if (eff_mode == 3) {
                tp_pack_fp16_push_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                    local_buf + off, reinterpret_cast<__half*>(d_p2p_tx_[1 - rank][b]), chunk);
                tp_barrier_kernel<<<1, 1, 0, cs>>>(d_p2p_flag_[1 - rank], d_p2p_flag_[rank], cur_step);
                tp_unpack_add_fp16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                    local_buf + off, reinterpret_cast<const __half*>(d_p2p_tx_[rank][b]), chunk);
            } else {
                tp_pack_bf16_push_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                    local_buf + off, reinterpret_cast<__nv_bfloat16*>(d_p2p_tx_[1 - rank][b]), chunk);
                tp_barrier_kernel<<<1, 1, 0, cs>>>(d_p2p_flag_[1 - rank], d_p2p_flag_[rank], cur_step);
                tp_unpack_add_bf16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                    local_buf + off, reinterpret_cast<const __nv_bfloat16*>(d_p2p_tx_[rank][b]), chunk);
            }
        }
        return;
    }

    if (!p2p_ && eff_mode >= 1 && ensure_pipe(rank)) {
        const size_t slot_bytes = slot_floats_ * sizeof(float);
        const int64_t max_q8 = ((int64_t) (slot_bytes * 32u / 34u)) & ~int64_t(255);
        const int64_t max_bf16 = ((int64_t) (slot_floats_ * 2u)) & ~int64_t(255);
        // Pipelined chunks keep GPU-host bridge fed full-duplex: cap chunk at 8M elements (~16 MB bf16/fp16 / ~8.5 MB q8)
        const int64_t raw_cap = eff_mode == 2 ? max_q8 : max_bf16;
        const int64_t cap = raw_cap < (8LL << 20) ? raw_cap : (8LL << 20);
        const cudaStream_t srx = (cudaStream_t) s_rx_[rank];
        const cudaEvent_t eb = (cudaEvent_t) ev_bar_[rank];
        const cudaEvent_t ed = (cudaEvent_t) ev_done_[rank];
        int b = 0;
        int64_t k = 0;
        for (int64_t off = 0; off < n_elems; off += cap, ++k) {
            const int64_t chunk = (n_elems - off) < cap ? (n_elems - off) : cap;
            const uint32_t cur_step = ++step;
            const size_t slot_off = (size_t) (cur_step & (TpComm::kRingSlots - 1u)) * slot_floats_;
            uint8_t* h_mine = reinterpret_cast<uint8_t*>(h_buf_[rank] + slot_off);
            const uint8_t* h_peer = reinterpret_cast<const uint8_t*>(h_buf_[1 - rank] + slot_off);
            if (k >= 2) cudaStreamWaitEvent(cs, (cudaEvent_t) ev_rx_[rank][b], 0);
            size_t tx_bytes = 0;
            if (eff_mode == 2) {
                const int64_t nblk = chunk >> 5;
                const int blocks = (int) ((nblk + 7) / 8 < 256 ? (nblk + 7) / 8 : 256);
                int8_t* qd = reinterpret_cast<int8_t*>(d_tx_[rank][b]);
                half* qs = reinterpret_cast<half*>(reinterpret_cast<uint8_t*>(d_tx_[rank][b]) + (size_t) chunk);
                tp_pack_q8_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(local_buf + off, qd, qs, chunk);
                tx_bytes = (size_t) chunk + (size_t) nblk * sizeof(half);
            } else if (eff_mode == 3) {
                const int64_t n8 = (chunk + 7) >> 3;
                const int blocks = (int) ((n8 + 255) / 256 < 256 ? (n8 + 255) / 256 : 256);
                tp_pack_fp16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                    local_buf + off, reinterpret_cast<__half*>(d_tx_[rank][b]), chunk);
                tx_bytes = (size_t) chunk * 2u;
            } else {
                const int64_t n8 = (chunk + 7) >> 3;
                const int blocks = (int) ((n8 + 255) / 256 < 256 ? (n8 + 255) / 256 : 256);
                tp_pack_bf16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                    local_buf + off, reinterpret_cast<__nv_bfloat16*>(d_tx_[rank][b]), chunk);
                tx_bytes = (size_t) chunk * 2u;
            }
            cudaMemcpyAsync(h_mine, d_tx_[rank][b], tx_bytes, cudaMemcpyDeviceToHost, cs);
            tp_barrier_kernel<<<1, 1, 0, cs>>>(m_flag_[rank][rank], m_flag_[rank][1 - rank], cur_step);
            cudaEventRecord(eb, cs);

            cudaStreamWaitEvent(srx, eb, 0);
            cudaMemcpyAsync(d_rx_[rank][b], h_peer, tx_bytes, cudaMemcpyHostToDevice, srx);
            if (eff_mode == 2) {
                const int64_t n4 = chunk >> 2;
                const int blocks = (int) ((n4 + 255) / 256 < 256 ? (n4 + 255) / 256 : 256);
                const int8_t* qd = reinterpret_cast<const int8_t*>(d_rx_[rank][b]);
                const half* qs = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(d_rx_[rank][b]) + (size_t) chunk);
                tp_unpack_add_q8_kernel<<<blocks > 0 ? blocks : 1, 256, 0, srx>>>(local_buf + off, qd, qs, chunk);
            } else if (eff_mode == 3) {
                const int64_t n8 = (chunk + 7) >> 3;
                const int blocks = (int) ((n8 + 255) / 256 < 256 ? (n8 + 255) / 256 : 256);
                tp_unpack_add_fp16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, srx>>>(
                    local_buf + off, reinterpret_cast<const __half*>(d_rx_[rank][b]), chunk);
            } else {
                const int64_t n8 = (chunk + 7) >> 3;
                const int blocks = (int) ((n8 + 255) / 256 < 256 ? (n8 + 255) / 256 : 256);
                tp_unpack_add_bf16_kernel<<<blocks > 0 ? blocks : 1, 256, 0, srx>>>(
                    local_buf + off, reinterpret_cast<const __nv_bfloat16*>(d_rx_[rank][b]), chunk);
            }
            cudaEventRecord((cudaEvent_t) ev_rx_[rank][b], srx);
            b ^= 1;
        }
        cudaEventRecord(ed, srx);
        cudaStreamWaitEvent(cs, ed, 0);
        return;
    }

    const size_t host_slot_floats = p2p_ ? kP2pDecSlotFloats : slot_floats_;
    for (int64_t off = 0; off < n_elems; ) {
        const int64_t rem = n_elems - off;
        const int64_t chunk = rem < (int64_t) host_slot_floats ? rem : (int64_t) host_slot_floats;
        const uint32_t cur_step = ++step;
        if (chunk <= 32768) {
            tp_allreduce_kernel<<<1, 256, 0, cs>>>(
                local_buf + off, m_buf_[rank][rank], m_buf_[rank][1 - rank],
                m_flag_[rank][rank], m_flag_[rank][1 - rank], (int) chunk, host_slot_floats, cur_step);
        } else {
            const size_t slot_off = (size_t) (cur_step & (TpComm::kRingSlots - 1u)) * host_slot_floats;
            const int64_t n4 = (chunk + 3) >> 2;
            const int blocks = (int) ((n4 + 255) / 256 < 256 ? (n4 + 255) / 256 : 256);
            tp_push_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                local_buf + off, m_buf_[rank][rank] + slot_off, chunk);
            tp_barrier_kernel<<<1, 1, 0, cs>>>(
                m_flag_[rank][rank], m_flag_[rank][1 - rank], cur_step);
            tp_pull_kernel<<<blocks > 0 ? blocks : 1, 256, 0, cs>>>(
                local_buf + off, m_buf_[rank][1 - rank] + slot_off, chunk);
        }
        off += chunk;
    }
}

}  // namespace strata::core
