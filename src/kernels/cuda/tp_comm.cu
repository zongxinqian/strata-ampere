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

__global__ void tp_p2p_selftest_write_kernel(uint32_t* __restrict__ remote_dst, uint32_t magic, int n) {
    const int tid = threadIdx.x;
    if (tid < n) {
        const uint32_t v = magic ^ (uint32_t) (tid * 2654435761u);
        asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(remote_dst + tid), "r"(v) : "memory");
    }
    __syncthreads();
    if (tid == 0) __threadfence_system();
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
        if (!pipe_init_[r] && !s_rx_[r] && !d_tx_[r][0] && !d_rx_[r][0] &&
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
        if (p2p_vmm_) {
#if defined(__linux__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
            vmm_free_peer_slab(vmm_base_[r], vmm_handle_[r], vmm_bytes_[r]);
#endif
            d_dec_buf_[r] = nullptr;
            d_p2p_flag_[r] = nullptr;
            d_p2p_tx_[r][0] = nullptr;
            d_p2p_tx_[r][1] = nullptr;
        } else {
            if (d_dec_buf_[r] != nullptr) {
                cudaFree(d_dec_buf_[r]);
                d_dec_buf_[r] = nullptr;
            }
            if (d_p2p_flag_[r] != nullptr) {
                cudaFree(d_p2p_flag_[r]);
                d_p2p_flag_[r] = nullptr;
            }
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
                        d_p2p_flag_[r] = reinterpret_cast<uint32_t*>(base + dec_bytes + 2u * pipe_tx_bytes);
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
        }
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
                tp_allreduce_bf16_reg_kernel<1><<<1, 1024, 0, (cudaStream_t) stream>>>(
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
