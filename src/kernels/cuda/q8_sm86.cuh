// src/kernels/cuda/q8_sm86.cuh - Ampere sm_86 optimized Q8_0 GEMV kernels for in-place packed (qs, half d).
#pragma once

namespace q8sm86 {

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

// Bank-conflict-free staging with zero integer division per int load.
template<int THREADS>
__device__ __forceinline__ void stage_x_tile(const Q81Block* __restrict__ x, int x_stride_blk, int blk0, int nblk,
                                             int ncols, int8_t* __restrict__ sx, float* __restrict__ sd) {
    int* sxi = reinterpret_cast<int*>(sx);
    const int nints = nblk * 8;
    for (int col = 0; col < ncols; ++col) {
        const Q81Block* xcol = x + std::size_t(col) * x_stride_blk + blk0;
        int* dst = sxi + std::size_t(col) * nints;
        for (int t = int(threadIdx.x); t < nints; t += THREADS) {
            const int blk = t >> 3, lane8 = t & 7;
            dst[t] = reinterpret_cast<const int*>(xcol[blk].qs)[lane8];
        }
        float* ds_dst = sd + std::size_t(col) * nblk;
        for (int blk = int(threadIdx.x); blk < nblk; blk += THREADS) {
            ds_dst[blk] = __low2float(xcol[blk].ds);
        }
    }
}

// Whole-block (32-element, two int4 + one half d) GEMV with shared-memory x tiling (K_TILE_BLK = 64 -> <= 18 KB smem for 8 cols).
// Keeps TPR, THREADS, and K_TILE_BLK invariant across ncols so col 0 is bitwise identical to ncols=1!
template<int NCOLS, int TPR, int THREADS, int K_TILE_BLK>
__launch_bounds__(THREADS)
__global__ void smem_blk_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int4 sm86_smem[];
    const int nb = n_in / 32;
    const int tile_blk = K_TILE_BLK > 0 ? K_TILE_BLK : nb;
    int8_t* sx = reinterpret_cast<int8_t*>(sm86_smem);
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * tile_blk * 32);

    constexpr int RPB = THREADS / TPR;
    const int lane = int(threadIdx.x) % TPR;
    const int rloc = int(threadIdx.x) / TPR;
    const int groups = (n_out + RPB - 1) / RPB;

    if constexpr (K_TILE_BLK == 0) {
        stage_x_tile<THREADS>(x, nb, 0, nb, NCOLS, sx, sd);
        __syncthreads();
        for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
            const int row = g * RPB + rloc;
            float acc[NCOLS] = {};
            if (row < n_out) {
                const int4* w4 = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in);
                const half* d = dpl + std::size_t(row) * nb;
                for (int b = lane; b < nb; b += TPR) {
                    const int4 w0 = __ldg(w4 + 2 * b);
                    const int4 w1 = __ldg(w4 + 2 * b + 1);
                    const float dw = __half2float(d[b]);
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) {
                        const int4* x4 = reinterpret_cast<const int4*>(sx + std::size_t(j) * n_in);
                        const int4 x0 = x4[2 * b];
                        const int4 x1 = x4[2 * b + 1];
                        int s = STRATA_DP4A(w0.x, x0.x, 0);
                        s = STRATA_DP4A(w0.y, x0.y, s);
                        s = STRATA_DP4A(w0.z, x0.z, s);
                        s = STRATA_DP4A(w0.w, x0.w, s);
                        s = STRATA_DP4A(w1.x, x1.x, s);
                        s = STRATA_DP4A(w1.y, x1.y, s);
                        s = STRATA_DP4A(w1.z, x1.z, s);
                        s = STRATA_DP4A(w1.w, x1.w, s);
                        acc[j] += dw * sd[j * nb + b] * float(s);
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
    } else {
        for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
            const int row = g * RPB + rloc;
            float acc[NCOLS] = {};
            const int4* w4 = row < n_out ? reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in) : nullptr;
            const half* d = row < n_out ? (dpl + std::size_t(row) * nb) : nullptr;
            for (int b0 = 0; b0 < nb; b0 += K_TILE_BLK) {
                const int nblk = (nb - b0) < K_TILE_BLK ? (nb - b0) : K_TILE_BLK;
                stage_x_tile<THREADS>(x, nb, b0, nblk, NCOLS, sx, sd);
                __syncthreads();
                if (row < n_out) {
                    for (int bi = lane; bi < nblk; bi += TPR) {
                        const int b = b0 + bi;
                        const int4 w0 = __ldg(w4 + 2 * b);
                        const int4 w1 = __ldg(w4 + 2 * b + 1);
                        const float dw = __half2float(d[b]);
#pragma unroll
                        for (int j = 0; j < NCOLS; ++j) {
                            const int4* x4 = reinterpret_cast<const int4*>(sx + std::size_t(j) * nblk * 32);
                            const int4 x0 = x4[2 * bi];
                            const int4 x1 = x4[2 * bi + 1];
                            int s = STRATA_DP4A(w0.x, x0.x, 0);
                            s = STRATA_DP4A(w0.y, x0.y, s);
                            s = STRATA_DP4A(w0.z, x0.z, s);
                            s = STRATA_DP4A(w0.w, x0.w, s);
                            s = STRATA_DP4A(w1.x, x1.x, s);
                            s = STRATA_DP4A(w1.y, x1.y, s);
                            s = STRATA_DP4A(w1.z, x1.z, s);
                            s = STRATA_DP4A(w1.w, x1.w, s);
                            acc[j] += dw * sd[j * nblk + bi] * float(s);
                        }
                    }
                }
                __syncthreads();
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
}

// 16-byte chunk kernel (for small K like n_in=320 where n_in/16=20 fits TPR=4/8/16 evenly).
template<int NCOLS, int TPR, int THREADS>
__launch_bounds__(THREADS)
__global__ void smem_half_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                 const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int4 sm86_smem[];
    const int nb = n_in / 32;
    int8_t* sx = reinterpret_cast<int8_t*>(sm86_smem);
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * n_in);
    stage_x_tile<THREADS>(x, nb, 0, nb, NCOLS, sx, sd);
    __syncthreads();

    constexpr int RPB = THREADS / TPR;
    const int lane = int(threadIdx.x) % TPR;
    const int rloc = int(threadIdx.x) / TPR;
    const int nch = n_in / 16;
    const int groups = (n_out + RPB - 1) / RPB;
    for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
        const int row = g * RPB + rloc;
        float acc[NCOLS] = {};
        if (row < n_out) {
            const int4* w = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in);
            const half* d = dpl + std::size_t(row) * nb;
            for (int c = lane; c < nch; c += TPR) {
                const int4 wv = __ldg(w + c);
                const float dw = __half2float(d[c >> 1]);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int4 xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * n_in)[c];
                    int s = STRATA_DP4A(wv.x, xv.x, 0);
                    s = STRATA_DP4A(wv.y, xv.y, s);
                    s = STRATA_DP4A(wv.z, xv.z, s);
                    s = STRATA_DP4A(wv.w, xv.w, s);
                    acc[j] += dw * sd[j * nb + (c >> 1)] * float(s);
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

// 1-row-per-block kernel for small n_out (<= 1024, e.g. HC down 320, QSA k/v 512, shexp gate/up 640):
// launches n_out blocks (>= 4.7 blocks/SM on 68 SMs!), reading aligned 16B weight vectors and L1-cached x directly.
template<int NCOLS, int NW>
__launch_bounds__(NW * 32)
__global__ void row1_direct_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                   const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int row = int(blockIdx.x);
    if (row >= n_out) return;
    const int tid = int(threadIdx.x);
    constexpr int NT = NW * 32;
    const int nb = n_in / 32;
    const int nch = n_in / 16;
    const int4* w4 = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in);
    const half* d = dpl + std::size_t(row) * nb;
    float acc[NCOLS] = {};
    for (int c = tid; c < nch; c += NT) {
        const int b = c >> 1;
        const int half_b = c & 1;
        const int4 wv = __ldg(w4 + c);
        const float dw = __half2float(d[b]);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * nb + b;
            const int* xqs = reinterpret_cast<const int*>(xb->qs) + half_b * 4;
            int s = STRATA_DP4A(wv.x, __ldg(xqs), 0);
            s = STRATA_DP4A(wv.y, __ldg(xqs + 1), s);
            s = STRATA_DP4A(wv.z, __ldg(xqs + 2), s);
            s = STRATA_DP4A(wv.w, __ldg(xqs + 3), s);
            acc[j] += dw * __low2float(xb->ds) * float(s);
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

template<int NCOLS>
void launch_cols(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const int sm = sm_count();
    static const int bps = [] {
        const char* v = std::getenv("STRATA_Q8_SM86_BPS");
        return v ? std::atoi(v) : 4;
    }();
    static const int row1_max_cols = [] {
        const char* v = std::getenv("STRATA_Q8_SM86_ROW1_MAX");
        return v ? std::atoi(v) : 4;
    }();
    if (n_out <= 1024) {
        static const int r1_nw = [] { const char* v = std::getenv("STRATA_Q8_R1_NW"); return v ? std::atoi(v) : 2; }();
        if (r1_nw == 1) row1_direct_kernel<NCOLS, 1><<<unsigned(n_out), 32, 0, s>>>(qs, d, x, y, n_in, n_out);
        else if (r1_nw == 4) row1_direct_kernel<NCOLS, 4><<<unsigned(n_out), 128, 0, s>>>(qs, d, x, y, n_in, n_out);
        else if (r1_nw == 5) row1_direct_kernel<NCOLS, 5><<<unsigned(n_out), 160, 0, s>>>(qs, d, x, y, n_in, n_out);
        else row1_direct_kernel<NCOLS, 2><<<unsigned(n_out), 64, 0, s>>>(qs, d, x, y, n_in, n_out);
        return;
    }
    if (n_in <= 384) {
        if (n_out <= 4096) {
            static const int dn_cfg = [] { const char* v = std::getenv("STRATA_Q8_SHDN_CFG"); return v ? std::atoi(v) : 0; }();
            const std::size_t smem = std::size_t(NCOLS) * (std::size_t(n_in) + std::size_t(n_in / 32) * sizeof(float));
            if (dn_cfg == 1) {
                constexpr int TPR = 4, TH = 32, RPB = TH / TPR;
                smem_half_kernel<NCOLS, TPR, TH><<<unsigned((n_out + RPB - 1) / RPB), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
            } else if (dn_cfg == 2) {
                constexpr int TPR = 4, TH = 64, RPB = TH / TPR;
                smem_half_kernel<NCOLS, TPR, TH><<<unsigned((n_out + RPB - 1) / RPB), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
            } else {
                constexpr int TPR = 8, TH = 64, RPB = TH / TPR;
                smem_half_kernel<NCOLS, TPR, TH><<<unsigned((n_out + RPB - 1) / RPB), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
            }
            return;
        }
        // HC up 320->10240: TPR=4, 128t => RPB=32, bps*8
        constexpr int TPR = 4, TH = 128, RPB = TH / TPR;
        const int groups = (n_out + RPB - 1) / RPB;
        const int grid = bps > 0 ? (std::min)(groups, sm * 8) : groups;
        const std::size_t smem = std::size_t(NCOLS) * (std::size_t(n_in) + std::size_t(n_in / 32) * sizeof(float));
        smem_half_kernel<NCOLS, TPR, TH><<<unsigned(grid), TH, smem, s>>>(qs, d, x, y, n_in, n_out);
        return;
    }
    if (n_in <= 1024) {
        // shared down 640->2560: q8sm60::kernel<NCOLS, 8> achieves 5.4..6.2 us (all cols!)
        q8sm60::launch_one<NCOLS, 8>(qs, d, x, y, n_in, n_out, s);
        return;
    }
    if (n_in > 3072) {
        // GDN/QSA out 6144->2560: row1_direct NW=4 fastest at 1..3 cols; q8sm60 faster at 4..6 cols.
        if (NCOLS <= 3 || q8sm60::smem_bytes(NCOLS, n_in) > q8sm60::MAX_SMEM) {
            row1_direct_kernel<NCOLS, 4><<<unsigned(n_out), 128, 0, s>>>(qs, d, x, y, n_in, n_out);
        } else {
            q8sm60::launch_one<NCOLS, 32>(qs, d, x, y, n_in, n_out, s);
        }
        return;
    }
    // 1024 < n_in <= 3072 (n_in=2560: qkv, gate, q): row1 NW=4 dominates 1..4 cols (85-88% bw),
    // q8sm60::kernel<NCOLS, 32> dominates 5..8 cols!
    if (NCOLS <= row1_max_cols) {
        row1_direct_kernel<NCOLS, 4><<<unsigned(n_out), 128, 0, s>>>(qs, d, x, y, n_in, n_out);
    } else {
        q8sm60::launch_one<NCOLS, 32>(qs, d, x, y, n_in, n_out, s);
    }
}

inline bool launch(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                   cudaStream_t s) {
    if (n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > MAX_COLS) return false;
    if (n_in > 1024 && n_in <= 3072 && ncols > 4 && (std::size_t(ncols) * (std::size_t(n_in) + std::size_t(n_in / 32) * sizeof(float)) > 48 * 1024))
        return false;
    if (n_in <= 1024 && (std::size_t(ncols) * (std::size_t(n_in) + std::size_t(n_in / 32) * sizeof(float)) > 48 * 1024))
        return false;
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


template<int NCOLS, int NW>
__launch_bounds__(NW * 32)
__global__ void row1_pair_kernel(const int8_t* __restrict__ qs1, const half* __restrict__ dpl1,
                                 const int8_t* __restrict__ qs2, const half* __restrict__ dpl2,
                                 const Q81Block* __restrict__ x, float* __restrict__ y1, float* __restrict__ y2,
                                 int n_in, int n_out) {
    const int row = int(blockIdx.x);
    if (row >= n_out) return;
    const int tid = int(threadIdx.x);
    constexpr int NT = NW * 32;
    const int nb = n_in / 32;
    const int nch = n_in / 16;
    const int4* w4a = reinterpret_cast<const int4*>(qs1 + std::size_t(row) * n_in);
    const int4* w4b = reinterpret_cast<const int4*>(qs2 + std::size_t(row) * n_in);
    const half* d1 = dpl1 + std::size_t(row) * nb;
    const half* d2 = dpl2 + std::size_t(row) * nb;
    float acc1[NCOLS] = {}, acc2[NCOLS] = {};
    for (int c = tid; c < nch; c += NT) {
        const int b = c >> 1;
        const int half_b = c & 1;
        const int4 wva = __ldg(w4a + c);
        const int4 wvb = __ldg(w4b + c);
        const float dw1 = __half2float(d1[b]);
        const float dw2 = __half2float(d2[b]);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * nb + b;
            const int* xqs = reinterpret_cast<const int*>(xb->qs) + half_b * 4;
            const int x0 = __ldg(xqs), x1 = __ldg(xqs + 1), x2 = __ldg(xqs + 2), x3 = __ldg(xqs + 3);
            const float dx = __low2float(xb->ds);
            int sa = STRATA_DP4A(wva.x, x0, 0);
            sa = STRATA_DP4A(wva.y, x1, sa);
            sa = STRATA_DP4A(wva.z, x2, sa);
            sa = STRATA_DP4A(wva.w, x3, sa);
            int sb = STRATA_DP4A(wvb.x, x0, 0);
            sb = STRATA_DP4A(wvb.y, x1, sb);
            sb = STRATA_DP4A(wvb.z, x2, sb);
            sb = STRATA_DP4A(wvb.w, x3, sb);
            acc1[j] += dw1 * dx * float(sa);
            acc2[j] += dw2 * dx * float(sb);
        }
    }
    __shared__ float red1[NCOLS][NW], red2[NCOLS][NW];
    const int warp = tid >> 5, lane = tid & 31;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v1 = acc1[j], v2 = acc2[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_xor_sync(0xffffffffu, v1, o);
            v2 += __shfl_xor_sync(0xffffffffu, v2, o);
        }
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
void launch_pair_cols(const int8_t* qs1, const half* d1, const int8_t* qs2, const half* d2,
                      const Q81Block* x, float* y1, float* y2, int n_in, int n_out, cudaStream_t s) {
    static const int pr_nw = [] { const char* v = std::getenv("STRATA_Q8_PR_NW"); return v ? std::atoi(v) : 2; }();
    if (pr_nw == 1) row1_pair_kernel<NCOLS, 1><<<unsigned(n_out), 32, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out);
    else if (pr_nw == 4) row1_pair_kernel<NCOLS, 4><<<unsigned(n_out), 128, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out);
    else if (pr_nw == 5) row1_pair_kernel<NCOLS, 5><<<unsigned(n_out), 160, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out);
    else row1_pair_kernel<NCOLS, 2><<<unsigned(n_out), 64, 0, s>>>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out);
}

inline bool launch_pair(const int8_t* qs1, const half* d1, const int8_t* qs2, const half* d2,
                        const Q81Block* x, float* y1, float* y2, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (n_out > 1024 || n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > MAX_COLS) return false;
    switch (ncols) {
        case 1: launch_pair_cols<1>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        case 2: launch_pair_cols<2>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        case 3: launch_pair_cols<3>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        case 4: launch_pair_cols<4>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        case 5: launch_pair_cols<5>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        case 6: launch_pair_cols<6>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        case 7: launch_pair_cols<7>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
        default: launch_pair_cols<8>(qs1, d1, qs2, d2, x, y1, y2, n_in, n_out, s); break;
    }
    return true;
}

} // namespace q8sm86
