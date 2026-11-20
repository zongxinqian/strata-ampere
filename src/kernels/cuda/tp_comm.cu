#include "strata/core/tp_comm.hpp"

#include <cuda_runtime.h>
#include <atomic>
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
        __threadfence_system();
        *my_flag = step;
        __threadfence_system();
        while (*peer_flag < step) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(40);
#endif
        }
        __threadfence_system();
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

}  // namespace

TpComm::~TpComm() {
    release();
}

void TpComm::release() {
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
    const int threads = n_elems > 32768 ? 1024 : 256;
    tp_allreduce_kernel<<<1, threads, 0, (cudaStream_t) stream>>>(
        local_buf, m_buf_[rank][rank], m_buf_[rank][1 - rank],
        m_flag_[rank][rank], m_flag_[rank][1 - rank], n_elems, slot_floats_, step);
}

}  // namespace strata::core
