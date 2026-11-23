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
    if (on <= 0 || n_in % 32 != 0 || ncols < 1 || ncols > 8 || (n_out >= 32768 && (allow_head ? 0 : ncols <= 4))) return false;
    if (n_out < 2048) return false;
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
