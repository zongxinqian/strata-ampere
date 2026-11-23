// src/kernels/cuda/q4_sm86.cuh - Ampere sm_86 optimized Q4_0 GEMV kernels for in-place packed (qs, half d).
#pragma once

namespace q4sm86 {

constexpr int MAX_COLS = 8;

inline int sm_count() {
    static int cached[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 68;
    if (cached[dev] == 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0) cached[dev] = n;
        else cached[dev] = 68;
    }
    return cached[dev];
}

template<int NCOLS, int TPR, int THREADS>
__launch_bounds__(THREADS, (THREADS == 128 ? 8 : 4))
__global__ void smem_blk_kernel(const uint8_t* __restrict__ qs, const half* __restrict__ dpl,
                                const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int4 q4_smem[];
    const int nb = n_in / 32;
    int4* sx0 = q4_smem;
    int4* sx1 = sx0 + std::size_t(NCOLS) * nb;
    half2* sds = reinterpret_cast<half2*>(sx1 + std::size_t(NCOLS) * nb);

    const int tid = int(threadIdx.x);
    for (int j = 0; j < NCOLS; ++j) {
        const Q81Block* xcol = x + std::size_t(j) * nb;
        int4* d0 = sx0 + std::size_t(j) * nb;
        int4* d1 = sx1 + std::size_t(j) * nb;
        half2* dd = sds + std::size_t(j) * nb;
        for (int b = tid; b < nb; b += THREADS) {
            const int* xp = reinterpret_cast<const int*>(xcol[b].qs);
            d0[b] = make_int4(xp[0], xp[1], xp[2], xp[3]);
            d1[b] = make_int4(xp[4], xp[5], xp[6], xp[7]);
            dd[b] = xcol[b].ds;
        }
    }
    __syncthreads();

    constexpr int RPB = THREADS / TPR;
    const int lane = tid % TPR;
    const int rloc = tid / TPR;
    const int groups = (n_out + RPB - 1) / RPB;

    for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
        const int row = g * RPB + rloc;
        float acc[NCOLS] = {};
        if (row < n_out) {
            const int4* w4 = reinterpret_cast<const int4*>(qs + std::size_t(row) * (n_in / 2));
            const half* d = dpl + std::size_t(row) * nb;
            for (int b = lane; b < nb; b += TPR) {
                const int4 wv = __ldg(w4 + b);
                const float dw = __half2float(__ldg(d + b));
                const int lo0 = wv.x & 0x0f0f0f0f;
                const int lo1 = wv.y & 0x0f0f0f0f;
                const int lo2 = wv.z & 0x0f0f0f0f;
                const int lo3 = wv.w & 0x0f0f0f0f;
                const int hi0 = (wv.x >> 4) & 0x0f0f0f0f;
                const int hi1 = (wv.y >> 4) & 0x0f0f0f0f;
                const int hi2 = (wv.z >> 4) & 0x0f0f0f0f;
                const int hi3 = (wv.w >> 4) & 0x0f0f0f0f;
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int off = j * nb + b;
                    const int4 x0 = sx0[off];
                    const int4 x1 = sx1[off];
                    int s = STRATA_DP4A(lo0, x0.x, 0);
                    s = STRATA_DP4A(lo1, x0.y, s);
                    s = STRATA_DP4A(lo2, x0.z, s);
                    s = STRATA_DP4A(lo3, x0.w, s);
                    s = STRATA_DP4A(hi0, x1.x, s);
                    s = STRATA_DP4A(hi1, x1.y, s);
                    s = STRATA_DP4A(hi2, x1.z, s);
                    s = STRATA_DP4A(hi3, x1.w, s);
                    const float2 ds = __half22float2(sds[off]);
                    acc[j] += dw * (float(s) * ds.x - 8.0f * ds.y);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
            if (lane == 0 && row < n_out) y[std::size_t(j) * n_out + row] = v;
        }
    }
}

template<int NCOLS, int NW>
__launch_bounds__(NW * 32, 4)
__global__ void row1_direct_kernel(const uint8_t* __restrict__ qs, const half* __restrict__ dpl,
                                   const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int row = int(blockIdx.x);
    if (row >= n_out) return;
    const int tid = int(threadIdx.x);
    constexpr int NT = NW * 32;
    const int nb = n_in / 32;
    const int4* w4 = reinterpret_cast<const int4*>(qs + std::size_t(row) * (n_in / 2));
    const half* d = dpl + std::size_t(row) * nb;
    float acc[NCOLS] = {};
    for (int b = tid; b < nb; b += NT) {
        const int4 wv = __ldg(w4 + b);
        const float dw = __half2float(__ldg(d + b));
        const int lo0 = wv.x & 0x0f0f0f0f;
        const int lo1 = wv.y & 0x0f0f0f0f;
        const int lo2 = wv.z & 0x0f0f0f0f;
        const int lo3 = wv.w & 0x0f0f0f0f;
        const int hi0 = (wv.x >> 4) & 0x0f0f0f0f;
        const int hi1 = (wv.y >> 4) & 0x0f0f0f0f;
        const int hi2 = (wv.z >> 4) & 0x0f0f0f0f;
        const int hi3 = (wv.w >> 4) & 0x0f0f0f0f;
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * nb + b;
            const int* xp = reinterpret_cast<const int*>(xb->qs);
            int s = STRATA_DP4A(lo0, __ldg(xp), 0);
            s = STRATA_DP4A(lo1, __ldg(xp + 1), s);
            s = STRATA_DP4A(lo2, __ldg(xp + 2), s);
            s = STRATA_DP4A(lo3, __ldg(xp + 3), s);
            s = STRATA_DP4A(hi0, __ldg(xp + 4), s);
            s = STRATA_DP4A(hi1, __ldg(xp + 5), s);
            s = STRATA_DP4A(hi2, __ldg(xp + 6), s);
            s = STRATA_DP4A(hi3, __ldg(xp + 7), s);
            const float2 ds = __half22float2(xb->ds);
            acc[j] += dw * (float(s) * ds.x - 8.0f * ds.y);
        }
    }
    __shared__ float red[NCOLS][NW];
    const int warp = tid >> 5, lane = tid & 31;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[j][warp] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < NCOLS) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) tot += red[lane][w];
        y[std::size_t(lane) * n_out + row] = tot;
    }
}

template<int NCOLS, int RPB, int TPR>
__launch_bounds__(RPB * TPR, (RPB * TPR <= 64 ? 16 : 12))
__global__ void subwarp_kernel(const uint8_t* __restrict__ qs, const half* __restrict__ dpl,
                               const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int nb = n_in / 32;
    const int tid = int(threadIdx.x);
    const int rloc = tid / TPR;
    const int lane = tid % TPR;
    const int row = int(blockIdx.x) * RPB + rloc;
    if (row >= n_out) return;
    const int4* w4 = reinterpret_cast<const int4*>(qs + std::size_t(row) * (n_in / 2));
    const half* d = dpl + std::size_t(row) * nb;
    float acc[NCOLS] = {};
    for (int b = lane; b < nb; b += TPR) {
        const int4 wv = __ldg(w4 + b);
        const float dw = __half2float(__ldg(d + b));
        const int lo0 = wv.x & 0x0f0f0f0f;
        const int lo1 = wv.y & 0x0f0f0f0f;
        const int lo2 = wv.z & 0x0f0f0f0f;
        const int lo3 = wv.w & 0x0f0f0f0f;
        const int hi0 = (wv.x >> 4) & 0x0f0f0f0f;
        const int hi1 = (wv.y >> 4) & 0x0f0f0f0f;
        const int hi2 = (wv.z >> 4) & 0x0f0f0f0f;
        const int hi3 = (wv.w >> 4) & 0x0f0f0f0f;
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * nb + b;
            const int* xp = reinterpret_cast<const int*>(xb->qs);
            int s = STRATA_DP4A(lo0, __ldg(xp), 0);
            s = STRATA_DP4A(lo1, __ldg(xp + 1), s);
            s = STRATA_DP4A(lo2, __ldg(xp + 2), s);
            s = STRATA_DP4A(lo3, __ldg(xp + 3), s);
            s = STRATA_DP4A(hi0, __ldg(xp + 4), s);
            s = STRATA_DP4A(hi1, __ldg(xp + 5), s);
            s = STRATA_DP4A(hi2, __ldg(xp + 6), s);
            s = STRATA_DP4A(hi3, __ldg(xp + 7), s);
            const float2 ds = __half22float2(xb->ds);
            acc[j] += dw * (float(s) * ds.x - 8.0f * ds.y);
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template<int NCOLS, int NW>
__launch_bounds__(NW * 32)
__global__ void row1_pair_kernel(const uint8_t* __restrict__ qs1, const half* __restrict__ dpl1,
                                 const uint8_t* __restrict__ qs2, const half* __restrict__ dpl2,
                                 const Q81Block* __restrict__ x, float* __restrict__ y1, float* __restrict__ y2,
                                 int n_in, int n_out) {
    const int row = int(blockIdx.x);
    if (row >= n_out) return;
    const int tid = int(threadIdx.x);
    constexpr int NT = NW * 32;
    const int nb = n_in / 32;
    const int4* w4a = reinterpret_cast<const int4*>(qs1 + std::size_t(row) * (n_in / 2));
    const int4* w4b = reinterpret_cast<const int4*>(qs2 + std::size_t(row) * (n_in / 2));
    const half* d1 = dpl1 + std::size_t(row) * nb;
    const half* d2 = dpl2 + std::size_t(row) * nb;
    float acc1[NCOLS] = {}, acc2[NCOLS] = {};
    for (int b = tid; b < nb; b += NT) {
        const int4 wa = __ldg(w4a + b), wb = __ldg(w4b + b);
        const float dw1 = __half2float(__ldg(d1 + b)), dw2 = __half2float(__ldg(d2 + b));
        const int a0 = wa.x & 0x0f0f0f0f, a1 = wa.y & 0x0f0f0f0f, a2 = wa.z & 0x0f0f0f0f, a3 = wa.w & 0x0f0f0f0f;
        const int a4 = (wa.x >> 4) & 0x0f0f0f0f, a5 = (wa.y >> 4) & 0x0f0f0f0f, a6 = (wa.z >> 4) & 0x0f0f0f0f, a7 = (wa.w >> 4) & 0x0f0f0f0f;
        const int b0 = wb.x & 0x0f0f0f0f, b1 = wb.y & 0x0f0f0f0f, b2 = wb.z & 0x0f0f0f0f, b3 = wb.w & 0x0f0f0f0f;
        const int b4 = (wb.x >> 4) & 0x0f0f0f0f, b5 = (wb.y >> 4) & 0x0f0f0f0f, b6 = (wb.z >> 4) & 0x0f0f0f0f, b7 = (wb.w >> 4) & 0x0f0f0f0f;
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * nb + b;
            const int* xp = reinterpret_cast<const int*>(xb->qs);
            const int x0 = __ldg(xp), x1 = __ldg(xp + 1), x2 = __ldg(xp + 2), x3 = __ldg(xp + 3);
            const int x4 = __ldg(xp + 4), x5 = __ldg(xp + 5), x6 = __ldg(xp + 6), x7 = __ldg(xp + 7);
            int sa = STRATA_DP4A(a0, x0, 0); sa = STRATA_DP4A(a1, x1, sa); sa = STRATA_DP4A(a2, x2, sa); sa = STRATA_DP4A(a3, x3, sa);
            sa = STRATA_DP4A(a4, x4, sa); sa = STRATA_DP4A(a5, x5, sa); sa = STRATA_DP4A(a6, x6, sa); sa = STRATA_DP4A(a7, x7, sa);
            int sb = STRATA_DP4A(b0, x0, 0); sb = STRATA_DP4A(b1, x1, sb); sb = STRATA_DP4A(b2, x2, sb); sb = STRATA_DP4A(b3, x3, sb);
            sb = STRATA_DP4A(b4, x4, sb); sb = STRATA_DP4A(b5, x5, sb); sb = STRATA_DP4A(b6, x6, sb); sb = STRATA_DP4A(b7, x7, sb);
            const float2 ds = __half22float2(xb->ds);
            acc1[j] += dw1 * (float(sa) * ds.x - 8.0f * ds.y);
            acc2[j] += dw2 * (float(sb) * ds.x - 8.0f * ds.y);
        }
    }
    __shared__ float red1[NCOLS][NW], red2[NCOLS][NW];
    const int warp = tid >> 5, lane = tid & 31;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v1 = acc1[j], v2 = acc2[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { v1 += __shfl_xor_sync(0xffffffffu, v1, o); v2 += __shfl_xor_sync(0xffffffffu, v2, o); }
        if (lane == 0) { red1[j][warp] = v1; red2[j][warp] = v2; }
    }
    __syncthreads();
    if (warp == 0 && lane < NCOLS) {
        float t1 = 0.0f, t2 = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) { t1 += red1[lane][w]; t2 += red2[lane][w]; }
        y1[std::size_t(lane) * n_out + row] = t1;
        y2[std::size_t(lane) * n_out + row] = t2;
    }
}

template<int NCOLS>
void launch_cols(const uint8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const int nb = n_in / 32;
    const int sm = sm_count();
    static const int bps = [] {
        const char* v = std::getenv("STRATA_Q4_SM86_BPS");
        return v ? std::atoi(v) : 0;
    }();
    static const int row1_on = [] {
        const char* v = std::getenv("STRATA_Q4_SM86_ROW1");
        return v ? std::atoi(v) : 1;
    }();
    if (row1_on && n_out <= 1024) {
        row1_direct_kernel<NCOLS, 2><<<unsigned(n_out), 64, 0, s>>>(qs, d, x, y, n_in, n_out);
        return;
    }
    if (n_in >= 1024 && n_out <= 4096) {
        constexpr int RPB = 4, TPR = 16;
        subwarp_kernel<NCOLS, RPB, TPR><<<unsigned((n_out + RPB - 1) / RPB), RPB * TPR, 0, s>>>(qs, d, x, y, n_in, n_out);
        return;
    }
    if (nb >= 48) {
        if constexpr (NCOLS <= 2) {
            if (bps <= 0) {
                constexpr int RPB = 4, TPR = 32;
                subwarp_kernel<NCOLS, RPB, TPR><<<unsigned((n_out + RPB - 1) / RPB), RPB * TPR, 0, s>>>(qs, d, x, y, n_in, n_out);
                return;
            }
        }
        if constexpr (NCOLS == 3) {
            constexpr int TPR = 32, TH = 256, RPB = TH / TPR;
            const int groups = (n_out + RPB - 1) / RPB;
            const int grid = bps > 0 ? (std::min)(groups, sm * bps) : groups;
            const std::size_t smem = std::size_t(NCOLS) * std::size_t(nb) * 36;
            smem_blk_kernel<NCOLS, TPR, TH><<<unsigned(grid), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
        } else {
            constexpr int TPR = 32, TH = 128, RPB = TH / TPR;
            const int groups = (n_out + RPB - 1) / RPB;
            const int grid = bps > 0 ? (std::min)(groups, sm * bps) : groups;
            const std::size_t smem = std::size_t(NCOLS) * std::size_t(nb) * 36;
            smem_blk_kernel<NCOLS, TPR, TH><<<unsigned(grid), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
        }
    } else if (nb >= 16) {
        constexpr int TPR = 16, TH = 256, RPB = TH / TPR;
        const int groups = (n_out + RPB - 1) / RPB;
        const int grid = bps > 0 ? (std::min)(groups, sm * bps) : groups;
        const std::size_t smem = std::size_t(NCOLS) * std::size_t(nb) * 36;
        smem_blk_kernel<NCOLS, TPR, TH><<<unsigned(grid), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
    } else {
        constexpr int TPR = 8, TH = 128, RPB = TH / TPR;
        const int groups = (n_out + RPB - 1) / RPB;
        const int grid = bps > 0 ? (std::min)(groups, sm * bps) : groups;
        const std::size_t smem = std::size_t(NCOLS) * std::size_t(nb) * 36;
        smem_blk_kernel<NCOLS, TPR, TH><<<unsigned(grid), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
    }
}

inline bool launch(const uint8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                   cudaStream_t s) {
    if (n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > MAX_COLS) return false;
    if (reinterpret_cast<std::uintptr_t>(qs) % 16 != 0) return false;
    const int nb = n_in / 32;
    if (std::size_t(ncols) * std::size_t(nb) * 36 > 48 * 1024) return false;
    switch (ncols) {
        case 1: launch_cols<1>(qs, d, x, y, n_in, n_out, s); break;
        case 2: launch_cols<2>(qs, d, x, y, n_in, n_out, s); break;
        case 3: launch_cols<3>(qs, d, x, y, n_in, n_out, s); break;
        case 4: launch_cols<4>(qs, d, x, y, n_in, n_out, s); break;
        case 5: launch_cols<5>(qs, d, x, y, n_in, n_out, s); break;
        case 6: launch_cols<6>(qs, d, x, y, n_in, n_out, s); break;
        case 7: launch_cols<7>(qs, d, x, y, n_in, n_out, s); break;
        default: launch_cols<8>(qs, d, x, y, n_in, n_out, s); break;
    }
    return true;
}

inline bool launch_pair(const uint8_t* qs1, const half* d1, const uint8_t* qs2, const half* d2,
                        const Q81Block* x, float* y1, float* y2, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (n_out > 1024 || n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > MAX_COLS) return false;
    if ((reinterpret_cast<std::uintptr_t>(qs1) | reinterpret_cast<std::uintptr_t>(qs2)) % 16 != 0) return false;
    switch (ncols) {
        case 1: row1_pair_kernel<1, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        case 2: row1_pair_kernel<2, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        case 3: row1_pair_kernel<3, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        case 4: row1_pair_kernel<4, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        case 5: row1_pair_kernel<5, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        case 6: row1_pair_kernel<6, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        case 7: row1_pair_kernel<7, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
        default: row1_pair_kernel<8, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out); break;
    }
    return true;
}

} // namespace q4sm86
