#include "strata/core/tp_comm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
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
    __threadfence_system();
    *my_flag = step;
    __threadfence_system();
    while (*peer_flag < step) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(20);
#endif
    }
    __threadfence_system();
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
        const uint4 v = p4[i];
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
        dst[i] += __bfloat162float(peer_dev[i]);
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

}  // namespace

TpComm::~TpComm() {
    release();
}

void TpComm::release_pipe() {
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    for (int r = 0; r < 2; ++r) {
        if (!pipe_init_[r] && !s_rx_[r] && !d_tx_[r][0] && !d_rx_[r][0]) continue;
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
        for (int b = 0; b < 2; ++b) {
            if (d_tx_[r][b] != nullptr) { cudaFree(d_tx_[r][b]); d_tx_[r][b] = nullptr; }
            if (d_rx_[r][b] != nullptr) { cudaFree(d_rx_[r][b]); d_rx_[r][b] = nullptr; }
            if (ev_rx_[r][b] != nullptr) { cudaEventDestroy((cudaEvent_t) ev_rx_[r][b]); ev_rx_[r][b] = nullptr; }
        }
        pipe_init_[r] = false;
    }
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
    const unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable;
    for (int r = 0; r < 2; ++r) {
        cudaSetDevice(dev_[r]);
        if (cudaHostAlloc((void**) &h_buf_[r], max_floats_ * sizeof(float), flags) != cudaSuccess ||
            cudaHostAlloc((void**) &h_flag_[r], 64, flags) != cudaSuccess) {
            err = "TpComm: cudaHostAlloc failed";
            cudaSetDevice(prev_dev);
            release();
            return false;
        }
        std::memset(h_buf_[r], 0, max_floats_ * sizeof(float));
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
    if (dec_mode >= 1 && (n_elems & 7) == 0 && (size_t) n_elems * 2u <= slot_floats_ * sizeof(float)) {
        if (dec_mode == 2) {
            tp_allreduce_fp16_kernel<<<1, 512, 0, (cudaStream_t) stream>>>(
                local_buf, m_buf_[rank][rank], m_buf_[rank][1 - rank],
                m_flag_[rank][rank], m_flag_[rank][1 - rank], n_elems, slot_floats_, step);
        } else {
            tp_allreduce_bf16_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(
                local_buf, m_buf_[rank][rank], m_buf_[rank][1 - rank],
                m_flag_[rank][rank], m_flag_[rank][1 - rank], n_elems, slot_floats_, step);
        }
        return;
    }
    const int threads = n_elems > 32768 ? 1024 : 512;
    tp_allreduce_kernel<<<1, threads, 0, (cudaStream_t) stream>>>(
        local_buf, m_buf_[rank][rank], m_buf_[rank][1 - rank],
        m_flag_[rank][rank], m_flag_[rank][1 - rank], n_elems, slot_floats_, step);
}

void TpComm::allreduce_add_large(float* local_buf, int64_t n_elems, uint32_t& step, int rank, void* stream) {
    if (n_elems <= 0 || slot_floats_ == 0) return;
    // Mode: "bf16" (default, duplex CE DMA), "q8" (block-32 int8 + duplex CE DMA), "fp32" (exact FP32)
    static const int ar_mode = [] {
        const char* v = std::getenv("STRATA_TP_AR_MODE");
        if (v == nullptr) return 1; // 1 = bf16 duplex DMA
        if (std::strcmp(v, "fp32") == 0 || std::strcmp(v, "0") == 0) return 0;
        if (std::strcmp(v, "q8") == 0 || std::strcmp(v, "2") == 0) return 2;
        return 1;
    }();
    const cudaStream_t cs = (cudaStream_t) stream;
    int eff_mode = ar_mode;
    if (eff_mode == 2 && (n_elems & 31) != 0) eff_mode = 1;

    if (eff_mode >= 1 && ensure_pipe(rank)) {
        const size_t slot_bytes = slot_floats_ * sizeof(float);
        const int64_t max_q8 = ((int64_t) (slot_bytes * 32u / 34u)) & ~int64_t(255);
        const int64_t max_bf16 = ((int64_t) (slot_floats_ * 2u)) & ~int64_t(255);
        // Pipelined chunks keep GPU-host bridge fed full-duplex: cap chunk at 8M elements (~16 MB bf16 / ~8.5 MB q8)
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

    for (int64_t off = 0; off < n_elems; ) {
        const int64_t rem = n_elems - off;
        const int64_t chunk = rem < (int64_t) slot_floats_ ? rem : (int64_t) slot_floats_;
        const uint32_t cur_step = ++step;
        if (chunk <= 32768) {
            tp_allreduce_kernel<<<1, 256, 0, cs>>>(
                local_buf + off, m_buf_[rank][rank], m_buf_[rank][1 - rank],
                m_flag_[rank][rank], m_flag_[rank][1 - rank], (int) chunk, slot_floats_, cur_step);
        } else {
            const size_t slot_off = (size_t) (cur_step & (TpComm::kRingSlots - 1u)) * slot_floats_;
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
