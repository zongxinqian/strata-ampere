// src/kernels/cuda/q8_sm86_fast.cuh - sm_86 coalesced Q8_0 packed GEMV kernels.
// Included inside native_mmvq.cu after Q81Block is defined.

namespace q8sm86_fast {

__device__ __forceinline__ int4 load_x_half_q81(const Q81Block* xb, int half_b) {
    const int* xp = reinterpret_cast<const int*>(xb->qs) + (half_b << 2);
    return make_int4(__ldg(xp), __ldg(xp + 1), __ldg(xp + 2), __ldg(xp + 3));
}

template<int NCOLS, int RPB, int TPR, bool UNROLL2>
__launch_bounds__(RPB * TPR, (RPB * TPR <= 64 ? 16 : (NCOLS <= 3 ? 12 : 8)))
__global__ void q8_subwarp_rows_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                       const Q81Block* __restrict__ x, float* __restrict__ y,
                                       int n_in, int n_out) {
    const int nb = n_in >> 5;
    const int nb2 = nb << 1;
    const int tid = int(threadIdx.x);
    const int rloc = tid / TPR;
    const int lane = tid % TPR;
    const int row = int(blockIdx.x) * RPB + rloc;
    if (row >= n_out) return;

    const int4* w4 = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in);
    const half* dr = dpl + std::size_t(row) * nb;

    auto dot_step = [&](int hb0, float (&acc)[NCOLS]) {
        const int4 wv = __ldg(w4 + hb0);
        const float dw = __half2float(__ldg(dr + (hb0 >> 1)));
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * nb + (hb0 >> 1);
            const int4 xv = load_x_half_q81(xb, hb0 & 1);
            const float dx = __low2float(xb->ds);
            int s = STRATA_DP4A(wv.x, xv.x, 0);
            s = STRATA_DP4A(wv.y, xv.y, s);
            s = STRATA_DP4A(wv.z, xv.z, s);
            s = STRATA_DP4A(wv.w, xv.w, s);
            acc[j] += dw * dx * float(s);
        }
    };

    float acc[NCOLS] = {};
    int hb = lane;
    if constexpr (UNROLL2) {
        for (; hb + TPR < nb2; hb += 2 * TPR) {
            const int4 wv0 = __ldg(w4 + hb);
            const float dw0 = __half2float(__ldg(dr + (hb >> 1)));
            const int4 wv1 = __ldg(w4 + hb + TPR);
            const float dw1 = __half2float(__ldg(dr + ((hb + TPR) >> 1)));
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb0 = x + std::size_t(j) * nb + (hb >> 1);
                const int4 xv0 = load_x_half_q81(xb0, hb & 1);
                const float dx0 = __low2float(xb0->ds);
                const Q81Block* xb1 = x + std::size_t(j) * nb + ((hb + TPR) >> 1);
                const int4 xv1 = load_x_half_q81(xb1, (hb + TPR) & 1);
                const float dx1 = __low2float(xb1->ds);
                int s0 = STRATA_DP4A(wv0.x, xv0.x, 0);
                s0 = STRATA_DP4A(wv0.y, xv0.y, s0);
                s0 = STRATA_DP4A(wv0.z, xv0.z, s0);
                s0 = STRATA_DP4A(wv0.w, xv0.w, s0);
                acc[j] += dw0 * dx0 * float(s0);
                int s1 = STRATA_DP4A(wv1.x, xv1.x, 0);
                s1 = STRATA_DP4A(wv1.y, xv1.y, s1);
                s1 = STRATA_DP4A(wv1.z, xv1.z, s1);
                s1 = STRATA_DP4A(wv1.w, xv1.w, s1);
                acc[j] += dw1 * dx1 * float(s1);
            }
        }
    }
    for (; hb < nb2; hb += TPR) dot_step(hb, acc);

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template<int NCOLS>
bool launch_col(const int8_t* qs, const half* d, const Q81Block* x, float* y,
                int n_in, int n_out, cudaStream_t s) {
    if (n_out < 2048) return false;
    if (n_in <= 640) {
        constexpr int RPB = 4, TPR = 8;
        q8_subwarp_rows_kernel<NCOLS, RPB, TPR, true><<<unsigned((n_out + RPB - 1) / RPB), RPB * TPR, 0, s>>>(qs, d, x, y, n_in, n_out);
        return true;
    }
    if (n_out <= 2560) {
        constexpr int RPB = 1, TPR = 32;
        q8_subwarp_rows_kernel<NCOLS, RPB, TPR, true><<<unsigned(n_out), RPB * TPR, 0, s>>>(qs, d, x, y, n_in, n_out);
        return true;
    }
    if constexpr (NCOLS >= 5) {
        // 1-warp blocks with unroll-2 avoid register spill while keeping higher wave occupancy on sm_86.
        constexpr int RPB = 1, TPR = 32;
        q8_subwarp_rows_kernel<NCOLS, RPB, TPR, true><<<unsigned(n_out), RPB * TPR, 0, s>>>(qs, d, x, y, n_in, n_out);
        return true;
    }
    constexpr int RPB = 4, TPR = 32;
    q8_subwarp_rows_kernel<NCOLS, RPB, TPR, true><<<unsigned((n_out + RPB - 1) / RPB), RPB * TPR, 0, s>>>(qs, d, x, y, n_in, n_out);
    return true;
}

#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
// P4: Ampere sm_80+ INT8 Tensor Core (mma.sync.aligned.m16n8k32) kernel for packed Q8_0 x Q8_1 at NCOLS = 8..16.
// Exact int32 inner product per 32-element block (bitwise identical integer dots to __dp4a), loading each 16x32
// weight tile ONCE and sharing it across up to 16 columns (238 TOPS INT8 Tensor Core on RTX 3080 sm_86).
template<int NCOLS, int WARPS>
__launch_bounds__(WARPS * 32, 4)
__global__ void q8_imma_m16n8k32_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                        const Q81Block* __restrict__ x, float* __restrict__ y,
                                        int n_in, int n_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int nb = n_in >> 5;
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int gid = lane >> 2;        // 0..7
    const int tig = lane & 3;         // 0..3
    const int row0 = (int(blockIdx.x) * WARPS + warp) * 16;
    if (row0 >= n_out) return;

    const int r0 = row0 + gid;
    const int r1 = row0 + gid + 8;
    const bool v0 = (r0 < n_out);
    const bool v1 = (r1 < n_out);
    const int* w0 = v0 ? reinterpret_cast<const int*>(qs + std::size_t(r0) * n_in) : nullptr;
    const int* w1 = v1 ? reinterpret_cast<const int*>(qs + std::size_t(r1) * n_in) : nullptr;
    const half* dr0 = v0 ? (dpl + std::size_t(r0) * nb) : nullptr;
    const half* dr1 = v1 ? (dpl + std::size_t(r1) * nb) : nullptr;

    constexpr bool TWO_TILES = (NCOLS > 8);
    float f0_0 = 0.0f, f0_1 = 0.0f, f0_2 = 0.0f, f0_3 = 0.0f;
    float f1_0 = 0.0f, f1_1 = 0.0f, f1_2 = 0.0f, f1_3 = 0.0f;

    const int c0_0 = tig * 2;
    const int c0_1 = tig * 2 + 1;
    const int c1_0 = 8 + tig * 2;
    const int c1_1 = 8 + tig * 2 + 1;

    for (int b = 0; b < nb; ++b) {
        const int kb = b * 8 + tig;
        const uint32_t a0 = v0 ? static_cast<uint32_t>(__ldg(w0 + kb)) : 0u;
        const uint32_t a1 = v1 ? static_cast<uint32_t>(__ldg(w1 + kb)) : 0u;
        const uint32_t a2 = v0 ? static_cast<uint32_t>(__ldg(w0 + kb + 4)) : 0u;
        const uint32_t a3 = v1 ? static_cast<uint32_t>(__ldg(w1 + kb + 4)) : 0u;
        const float dw0 = v0 ? __half2float(__ldg(dr0 + b)) : 0.0f;
        const float dw1 = v1 ? __half2float(__ldg(dr1 + b)) : 0.0f;

        {
            const Q81Block* xb = x + std::size_t(gid) * nb + b;
            const int* xp = reinterpret_cast<const int*>(xb->qs);
            const uint32_t b0 = static_cast<uint32_t>(__ldg(xp + tig));
            const uint32_t b1 = static_cast<uint32_t>(__ldg(xp + tig + 4));
            int d0, d1, d2, d3;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(0), "r"(0), "r"(0), "r"(0));
            const float dx0 = __low2float(x[std::size_t(c0_0) * nb + b].ds);
            const float dx1 = __low2float(x[std::size_t(c0_1) * nb + b].ds);
            f0_0 += dw0 * dx0 * float(d0);
            f0_1 += dw0 * dx1 * float(d1);
            f0_2 += dw1 * dx0 * float(d2);
            f0_3 += dw1 * dx1 * float(d3);
        }
        if constexpr (TWO_TILES) {
            const int cg = 8 + gid;
            uint32_t b0 = 0u, b1 = 0u;
            if (cg < NCOLS) {
                const Q81Block* xb = x + std::size_t(cg) * nb + b;
                const int* xp = reinterpret_cast<const int*>(xb->qs);
                b0 = static_cast<uint32_t>(__ldg(xp + tig));
                b1 = static_cast<uint32_t>(__ldg(xp + tig + 4));
            }
            int d0, d1, d2, d3;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(0), "r"(0), "r"(0), "r"(0));
            const float dx0 = (c1_0 < NCOLS) ? __low2float(x[std::size_t(c1_0) * nb + b].ds) : 0.0f;
            const float dx1 = (c1_1 < NCOLS) ? __low2float(x[std::size_t(c1_1) * nb + b].ds) : 0.0f;
            f1_0 += dw0 * dx0 * float(d0);
            f1_1 += dw0 * dx1 * float(d1);
            f1_2 += dw1 * dx0 * float(d2);
            f1_3 += dw1 * dx1 * float(d3);
        }
    }

    if (v0) {
        y[std::size_t(c0_0) * n_out + r0] = f0_0;
        y[std::size_t(c0_1) * n_out + r0] = f0_1;
        if constexpr (TWO_TILES) {
            if (c1_0 < NCOLS) y[std::size_t(c1_0) * n_out + r0] = f1_0;
            if (c1_1 < NCOLS) y[std::size_t(c1_1) * n_out + r0] = f1_1;
        }
    }
    if (v1) {
        y[std::size_t(c0_0) * n_out + r1] = f0_2;
        y[std::size_t(c0_1) * n_out + r1] = f0_3;
        if constexpr (TWO_TILES) {
            if (c1_0 < NCOLS) y[std::size_t(c1_0) * n_out + r1] = f1_2;
            if (c1_1 < NCOLS) y[std::size_t(c1_1) * n_out + r1] = f1_3;
        }
    }
#endif
}

template<int NCOLS>
bool launch_tc(const int8_t* qs, const half* d, const Q81Block* x, float* y,
               int n_in, int n_out, cudaStream_t s) {
    constexpr int WARPS = 2;  // 32 output rows/block -> >= 80 blocks for n_out=2560 (saturates 68 SMs on sm_86)
    const unsigned grid = static_cast<unsigned>((n_out + WARPS * 16 - 1) / (WARPS * 16));
    q8_imma_m16n8k32_kernel<NCOLS, WARPS><<<grid, WARPS * 32, 0, s>>>(qs, d, x, y, n_in, n_out);
    return true;
}
#endif

inline bool launch(const int8_t* qs, const half* d, const Q81Block* x, float* y,
                   int n_in, int n_out, int ncols, cudaStream_t s) {
    static const int on = [] {
        const char* v = std::getenv("STRATA_Q8_FAST");
        return v ? std::atoi(v) : 1;
    }();
    static const int allow_head = [] {
        const char* v = std::getenv("STRATA_Q8_FAST_HEAD");
        return v ? std::atoi(v) : 0;
    }();
    static const int tc_on = [] {
        const char* v = std::getenv("STRATA_Q8_TC");
        return v ? std::atoi(v) : 1;
    }();
    if (on <= 0 || n_in % 32 != 0 || ncols < 1 || ncols > 16) return false;
    if (n_out < 2048) return false;
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    if (tc_on > 0 && ncols >= 8) {
        switch (ncols) {
            case 8:  return launch_tc<8>(qs, d, x, y, n_in, n_out, s);
            case 9:  return launch_tc<9>(qs, d, x, y, n_in, n_out, s);
            case 10: return launch_tc<10>(qs, d, x, y, n_in, n_out, s);
            case 11: return launch_tc<11>(qs, d, x, y, n_in, n_out, s);
            case 12: return launch_tc<12>(qs, d, x, y, n_in, n_out, s);
            case 13: return launch_tc<13>(qs, d, x, y, n_in, n_out, s);
            case 14: return launch_tc<14>(qs, d, x, y, n_in, n_out, s);
            case 15: return launch_tc<15>(qs, d, x, y, n_in, n_out, s);
            default: return launch_tc<16>(qs, d, x, y, n_in, n_out, s);
        }
    }
#endif
    if (ncols > 8 || (n_out >= 32768 && (allow_head ? 0 : ncols <= 4))) return false;
    switch (ncols) {
        case 1: return launch_col<1>(qs, d, x, y, n_in, n_out, s);
        case 2: return launch_col<2>(qs, d, x, y, n_in, n_out, s);
        case 3: return launch_col<3>(qs, d, x, y, n_in, n_out, s);
        case 4: return launch_col<4>(qs, d, x, y, n_in, n_out, s);
        case 5: return launch_col<5>(qs, d, x, y, n_in, n_out, s);
        case 6: return launch_col<6>(qs, d, x, y, n_in, n_out, s);
        case 7: return launch_col<7>(qs, d, x, y, n_in, n_out, s);
        default: return launch_col<8>(qs, d, x, y, n_in, n_out, s);
    }
}

} // namespace q8sm86_fast
