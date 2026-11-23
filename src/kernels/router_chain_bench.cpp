// src/kernels/router_chain_bench.cpp - what the FFN-side router chain of one layer costs in a captured graph:
// the router GEMV (bf16 [512][2560], T rows), the top-10, the doorbell publish; 48 layers' worth with distinct
// weights (cold), as one graph, per-layer microseconds.  Bench only (no parity: the kernels are the engine's own).
//
//     build/router_chain_bench [T=2] [reps=20]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)


int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int T = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int L = 48, N = 2560, NE = 512, KK = 10;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> w((size_t) NE * N);
    for (auto& x : w) {
        const float f = 0.05f * nd(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    std::vector<float> xh((size_t) T * N);
    for (auto& x : xh) x = nd(rng);
    std::vector<uint16_t*> dw(L);
    for (int l = 0; l < L; ++l) {
        CK(cudaMalloc((void**) &dw[l], w.size() * 2));
        CK(cudaMemcpy(dw[l], w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    }
    float *dx, *dlog, *dwt, *dxo;
    int32_t *dids, *dids_o, *dres;
    CK(cudaMalloc((void**) &dx, xh.size() * 4));
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dlog, (size_t) T * NE * 4));
    CK(cudaMalloc((void**) &dwt, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dids, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dids_o, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dxo, xh.size() * 4));
    CK(cudaMalloc((void**) &dres, NE * 4));
    std::vector<int32_t> res(NE);
    for (int i = 0; i < NE; ++i) res[i] = i;
    CK(cudaMemcpy(dres, res.data(), NE * 4, cudaMemcpyHostToDevice));
    uint32_t* hseq;
    CK(cudaHostAlloc((void**) &hseq, 64, cudaHostAllocMapped | cudaHostAllocPortable));
    *hseq = 0;
    uint32_t* dseq;
    CK(cudaHostGetDevicePointer((void**) &dseq, hseq, 0));
    int* dz;
    CK(cudaMalloc((void**) &dz, 16));
    CK(cudaMemset(dz, 0, 16));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    {   // Bitwise parity check: native_router_top10_bell vs native_router_top10_multi + doorbell_publish_res_value
        std::vector<float> lref(8 * NE), wt_ref(8 * KK), wt_new(8 * KK), x_ref(8 * N), x_new(8 * N);
        std::vector<int32_t> id_ref(8 * KK), id_new(8 * KK), ido_ref(8 * KK), ido_new(8 * KK), res_test(NE);
        for (auto& v : lref) v = nd(rng);
        float *dl, *dw1, *dw2, *dx1, *dx2;
        int32_t *di1, *di2, *do1, *do2, *dr;
        uint32_t *hs1, *hs2, *ds1, *ds2;
        CK(cudaMalloc(&dl, 8 * NE * 4)); CK(cudaMemcpy(dl, lref.data(), 8 * NE * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dw1, 8 * KK * 4)); CK(cudaMalloc(&dw2, 8 * KK * 4));
        CK(cudaMalloc(&di1, 8 * KK * 4)); CK(cudaMalloc(&di2, 8 * KK * 4));
        CK(cudaMalloc(&do1, 8 * KK * 4)); CK(cudaMalloc(&do2, 8 * KK * 4));
        CK(cudaMalloc(&dx1, 8 * N * 4)); CK(cudaMalloc(&dx2, 8 * N * 4));
        CK(cudaMalloc(&dr, NE * 4));
        CK(cudaHostAlloc(&hs1, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc(&hs2, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostGetDevicePointer(&ds1, hs1, 0)); CK(cudaHostGetDevicePointer(&ds2, hs2, 0));
        std::vector<float> xbig(8 * N);
        for (auto& v : xbig) v = nd(rng);
        float* dxin; CK(cudaMalloc(&dxin, 8 * N * 4)); CK(cudaMemcpy(dxin, xbig.data(), 8 * N * 4, cudaMemcpyHostToDevice));
        for (int n_test : {1, 2, 3, 6, 8}) {
            for (int per_tok : {0, 1}) {
                for (int mc = 0; mc < 4; ++mc) {
                    if (mc == 0) std::fill(res_test.begin(), res_test.end(), 0);         // all hit
                    else if (mc == 1) std::fill(res_test.begin(), res_test.end(), -1);   // all miss
                    else {                                                                // partial miss
                        for (int i = 0; i < NE; ++i) res_test[i] = (i & 1) ? 0 : -1;
                    }
                    CK(cudaMemcpy(dr, res_test.data(), NE * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemsetAsync(dx1, 0x5a, n_test * N * 4, s));
                    CK(cudaMemsetAsync(dx2, 0x5a, n_test * N * 4, s));
                    *hs1 = 17; *hs2 = 17;
                    const int32_t* dres_ptr = (mc == 3) ? nullptr : dr;
                    K::native_router_top10_multi(dl, di1, dw1, n_test, s);
                    K::doorbell_publish_res_value(dxin, di1, dres_ptr, NE, (int64_t) n_test * N, (int64_t) n_test * KK,
                                                  dx1, do1, ds1, 42u, s, per_tok ? KK : 0);
                    K::native_router_top10_bell(dl, di2, dw2, dxin, dres_ptr, N, n_test,
                                                dx2, do2, ds2, 42u, per_tok != 0, s);
                    CK(cudaStreamSynchronize(s));
                    CK(cudaMemcpy(id_ref.data(), di1, n_test * KK * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(id_new.data(), di2, n_test * KK * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(ido_ref.data(), do1, n_test * KK * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(ido_new.data(), do2, n_test * KK * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(wt_ref.data(), dw1, n_test * KK * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(wt_new.data(), dw2, n_test * KK * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(x_ref.data(), dx1, n_test * N * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(x_new.data(), dx2, n_test * N * 4, cudaMemcpyDeviceToHost));
                    if (*hs1 != *hs2 || *hs1 != 42u ||
                        std::memcmp(id_ref.data(), id_new.data(), n_test * KK * 4) != 0 ||
                        std::memcmp(ido_ref.data(), ido_new.data(), n_test * KK * 4) != 0 ||
                        std::memcmp(wt_ref.data(), wt_new.data(), n_test * KK * 4) != 0 ||
                        std::memcmp(x_ref.data(), x_new.data(), n_test * N * 4) != 0) {
                        std::fprintf(stderr, "PARITY FAIL: n=%d per_tok=%d mc=%d seq=(%u,%u)\n",
                                     n_test, per_tok, mc, *hs1, *hs2);
                        return 1;
                    }
                }
            }
        }
        std::printf("router_top10_bell parity: ALL 40 cases bitwise IDENTICAL\n");
        cudaFree(dl); cudaFree(dw1); cudaFree(dw2); cudaFree(di1); cudaFree(di2);
        cudaFree(do1); cudaFree(do2); cudaFree(dx1); cudaFree(dx2); cudaFree(dr); cudaFree(dxin);
        cudaFreeHost(hs1); cudaFreeHost(hs2);
    }
    {   // outputs of the router GEMV for several shapes and token counts as FNV hashes: compare across STRATA_MMVF_PF=0/1
        const int shapes[][2] = {{2560, 512}, {2560, 256}, {2560, 64}, {1280, 513}, {640, 100}, {4096, 50}, {2048, 7}, {256, 33}, {8192, 90}};
        for (const auto& sh : shapes) {
            const int ni = sh[0], no = sh[1];
            std::vector<uint16_t> wh((size_t) no * ni);
            for (auto& v : wh) {
                const float f = 0.05f * nd(rng);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                v = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
            }
            std::vector<float> xs((size_t) 8 * ni);
            for (auto& v : xs) v = nd(rng);
            uint16_t* dwp;
            float *dxp, *dyp;
            CK(cudaMalloc((void**) &dwp, wh.size() * 2));
            CK(cudaMemcpy(dwp, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc((void**) &dxp, xs.size() * 4));
            CK(cudaMemcpy(dxp, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc((void**) &dyp, (size_t) 8 * no * 4));
            for (int tk = 1; tk <= 8; ++tk) {
                CK(cudaMemset(dyp, 0xff, (size_t) 8 * no * 4));
                K::bf16_gemv_fp32_mmvf_multi(dxp, ni, dwp, dyp, no, ni, no, tk, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> yh((size_t) 8 * no);
                CK(cudaMemcpy(yh.data(), dyp, yh.size() * 4, cudaMemcpyDeviceToHost));
                unsigned long long h = 1469598103934665603ull;
                for (size_t b = 0; b < yh.size() * 4; ++b) h = (h ^ ((const unsigned char*) yh.data())[b]) * 1099511628211ull;
                std::printf("PARITY n_in %d n_out %d T %d hash %016llx\n", ni, no, tk, h);
            }
            cudaFree(dwp); cudaFree(dxp); cudaFree(dyp);
        }
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    {   // the BF16 GEMV by shape and token count, 48 distinct weight copies in one graph: us per call (STRATA_MMVF_PF=0 / 1 to compare)
        const int outs[] = {512, 256, 128, 64, 48, 8, 1};
        const int ins[] = {2560, 2560, 2560, 2560, 2560, 2560, 2560};
        for (int si = 0; si < 7; ++si) {
            const int no = outs[si], ni = ins[si];
            std::vector<uint16_t*> dws(L);
            for (int l = 0; l < L; ++l) {
                CK(cudaMalloc((void**) &dws[l], (size_t) no * ni * 2));
                CK(cudaMemcpy(dws[l], w.data(), (size_t) no * ni * 2 < w.size() * 2 ? (size_t) no * ni * 2 : w.size() * 2, cudaMemcpyHostToDevice));
            }
            for (int tk : {1, 2, 3, 4, 6}) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < L; ++l) K::bf16_gemv_fp32_mmvf_multi(dx, N, dws[l], dlog, NE, ni, no, tk, s);
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                std::vector<float> ms;
                for (int r = 0; r < reps; ++r) {
                    CK(cudaEventRecord(e0, s));
                    CK(cudaGraphLaunch(ge, s));
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    float m = 0;
                    CK(cudaEventElapsedTime(&m, e0, e1));
                    ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("SHAPE n_in %d n_out %3d T %d : %6.2f us per call\n", ni, no, tk, 1e3 * ms[ms.size() / 2] / L);
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
            }
            for (int l = 0; l < L; ++l) cudaFree(dws[l]);
        }
    }
    // mode bits: 1 mmvf, 2 route, 4 doorbell, 8 an empty kernel per layer (the node floor)
    const char* names[] = {"empty node", "mmvf", "route", "mmvf+route", "doorbell", "mmvf+doorbell", "route+doorbell", "mmvf+route+doorbell", "mmvf+route_bell"};
    for (int mode = 0; mode < 9; ++mode) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            if (mode == 0) K::gpu_stamp((unsigned long long*) dz, 0, s);
            if (mode & 1) K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
            if (mode == 8) {
                K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
                K::native_router_top10_bell(dlog, dids, dwt, dx, dres, N, T, dxo, dids_o, dseq, 0u, true, s);
            } else {
                if (mode & 2) K::native_router_top10_multi(dlog, dids, dwt, T, s);
                if (mode & 4) K::doorbell_publish_res(dx, dids, dres, NE, (int64_t) T * N, (int64_t) T * KK, dxo, dids_o, dseq, s);
            }
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float m = 0;
            CK(cudaEventElapsedTime(&m, e0, e1));
            ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T %d %-20s : %6.2f us per layer (median of %d graphs of %d layers)\n", T, names[mode], 1e3 * ms[ms.size() / 2] / L, reps, L);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    return 0;
}
