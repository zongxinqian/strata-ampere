// src/kernels/q8_gemv_bench.cpp - MODEL-UPDATE: the dense Q8_0 GEMVs of Unsloth's files (attention, SSM, shared expert,
// hyper-connections, head: all Q8_0 there) at the engine's shapes, 1..6 columns (a verify window's tokens), through
// native_mmvq - the call the engine makes - against the card's memory bandwidth.  Enough copies of each matrix are
// cycled to miss the GPU's last-level cache, as a layer's call does.
//
//     build/q8_gemv_bench [--cols 1,2,3,4,6] [--iters 50] [--bw GB/s]
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
struct Shape { const char* what; int n_in, n_out; };
const Shape SHAPES[] = {
    {"GDN qkv 2560->10240", 2560, 10240},  {"GDN gate 2560->6144", 2560, 6144}, {"GDN/QSA out 6144->2560", 6144, 2560},
    {"QSA q 2560->12288", 2560, 12288},    {"QSA k/v 2560->512", 2560, 512},    {"HC down 10240->320", 10240, 320},
    {"HC up 320->10240", 320, 10240},      {"shared gate/up 2560->640", 2560, 640}, {"shared down 640->2560", 640, 2560},
    {"head 2560->248320", 2560, 248320},
    {"head_tp2 2560->124160", 2560, 124160},
};
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> cols = {1, 2, 3, 4, 6};
    int iters = 50;
    double bw = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cols" && i + 1 < argc) cols = ints(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--bw" && i + 1 < argc) bw = std::atof(argv[++i]);
    }
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("q8_gemv_bench: %s\n", prop.name);
    std::mt19937 rng(7);
    for (const Shape& s : SHAPES) {
        const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 34;
        const size_t cap_mb = std::getenv("Q8_BENCH_MIB") ? (size_t) std::atoi(std::getenv("Q8_BENCH_MIB")) : 512;
        const int copies = (int) std::max<size_t>(1, std::min<size_t>(16, (cap_mb << 20) / bytes));
        std::vector<uint8_t> host(bytes);
        for (size_t b = 0; b < bytes; b += 34) {   // fp16 scale 0.01 (0x211f), then random int8 weights
            host[b] = 0x1f;
            host[b + 1] = 0x21;
            for (int k = 0; k < 32; ++k) host[b + 2 + k] = (uint8_t) (rng() & 0xff);
        }
        if (s.n_out > 20000 && !(std::getenv("Q8_BENCH_HEAD") && s.n_out == 124160)) continue;
        const bool do_pack = std::getenv("Q8_BENCH_PACK") && std::atoi(std::getenv("Q8_BENCH_PACK")) != 0;
        std::vector<uint8_t> packed_host;
        if (do_pack) {
            packed_host.resize(bytes);
            K::native_q8_0_pack_host(host.data(), packed_host.data(), s.n_in, s.n_out);
        }
        if (do_pack && std::getenv("Q8_BENCH_PARITY")) {
            void *wu = nullptr, *wp = nullptr, *xq_t = nullptr;
            float *xt = nullptr, *yu = nullptr, *yp = nullptr;
            const int mc = *std::max_element(cols.begin(), cols.end());
            ck(cudaMalloc(&wu, bytes), "pu"); ck(cudaMemcpy(wu, host.data(), bytes, cudaMemcpyHostToDevice), "cpu");
            ck(cudaMalloc(&wp, bytes), "pp"); ck(cudaMemcpy(wp, packed_host.data(), bytes, cudaMemcpyHostToDevice), "cpp");
            K::native_q8_0_packed_register(wp, wp, s.n_in, s.n_out);
            ck(cudaMalloc(&xt, (size_t) s.n_in * mc * 4), "px");
            std::vector<float> hx((size_t) s.n_in * mc);
            for (auto& v : hx) v = std::normal_distribution<float>(0.f, 1.f)(rng);
            ck(cudaMemcpy(xt, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "cpx");
            ck(cudaMalloc(&xq_t, K::native_q8_1_bytes(s.n_in, mc)), "pxq");
            ck(cudaMalloc(&yu, (size_t) s.n_out * mc * 4), "pyu");
            ck(cudaMalloc(&yp, (size_t) s.n_out * mc * 4), "pyp");
            cudaStream_t sp; ck(cudaStreamCreate(&sp), "sp");
            std::vector<float> y1_ref(s.n_out);
            for (int c : cols) {
                K::native_quantize_q8_1(xt, xq_t, s.n_in, c, sp);
                K::native_mmvq(8, wu, xq_t, yu, s.n_in, s.n_out, c, sp);
                K::native_mmvq(8, wp, xq_t, yp, s.n_in, s.n_out, c, sp);
                ck(cudaStreamSynchronize(sp), "psync");
                std::vector<float> hu((size_t) c * s.n_out), hp((size_t) c * s.n_out);
                cudaMemcpy(hu.data(), yu, hu.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hp.data(), yp, hp.size() * 4, cudaMemcpyDeviceToHost);
                double err = 0, ref = 0;
                for (size_t i = 0; i < hu.size(); ++i) { err += std::fabs(hp[i] - hu[i]); ref += std::fabs(hu[i]); }
                if (c == cols.front()) y1_ref.assign(hp.begin(), hp.begin() + s.n_out);
                const bool exact0 = (c > 4) || (std::memcmp(hp.data(), y1_ref.data(), y1_ref.size() * 4) == 0);
                const double rel = ref > 0 ? err / ref : 0.0;
                if (!(rel < 1e-5) || !exact0) {
                    std::fprintf(stderr, "PARITY FAIL %s cols=%d rel=%.3e exact0=%d\n", s.what, c, rel, (int) exact0);
                    return 2;
                }
            }

            // Also check prefill dequant_f32 / dequant_f16 / dequant_f16_ld / dequant_bf16 on packed vs unpacked!
            const int r0 = s.n_out > 64 ? 32 : 0, nr = std::min(64, s.n_out - r0), ldp = s.n_in + 64;
            void *dfu = nullptr, *dfp = nullptr;
            ck(cudaMalloc(&dfu, (size_t) nr * ldp * 4), "dfu"); ck(cudaMalloc(&dfp, (size_t) nr * ldp * 4), "dfp");
            auto chk_dq = [&](const char* tag, size_t elems) {
                ck(cudaStreamSynchronize(sp), tag);
                std::vector<uint8_t> hu(elems * 4), hp(elems * 4);
                cudaMemcpy(hu.data(), dfu, elems * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hp.data(), dfp, elems * 4, cudaMemcpyDeviceToHost);
                if (std::memcmp(hu.data(), hp.data(), elems * 4) != 0) {
                    std::fprintf(stderr, "DEQUANT PARITY FAIL %s %s\n", s.what, tag); std::exit(3);
                }
            };
            K::dequant_f32(8, wu, r0, nr, s.n_in, (float*) dfu, sp);
            K::dequant_f32(8, wp, r0, nr, s.n_in, (float*) dfp, sp);
            chk_dq("f32", (size_t) nr * s.n_in);
            K::dequant_f16(8, wu, r0, nr, s.n_in, (uint16_t*) dfu, sp);
            K::dequant_f16(8, wp, r0, nr, s.n_in, (uint16_t*) dfp, sp);
            chk_dq("f16", ((size_t) nr * s.n_in) / 2);
            cudaMemsetAsync(dfu, 0, (size_t) nr * ldp * 2, sp); cudaMemsetAsync(dfp, 0, (size_t) nr * ldp * 2, sp);
            K::dequant_f16_ld(8, wu, r0, nr, s.n_in, ldp, (uint16_t*) dfu, sp);
            K::dequant_f16_ld(8, wp, r0, nr, s.n_in, ldp, (uint16_t*) dfp, sp);
            chk_dq("f16_ld", ((size_t) nr * ldp) / 2);
            K::dequant_bf16(8, wu, r0, nr, s.n_in, (uint16_t*) dfu, sp);
            K::dequant_bf16(8, wp, r0, nr, s.n_in, (uint16_t*) dfp, sp);
            chk_dq("bf16", ((size_t) nr * s.n_in) / 2);
            cudaFree(dfu); cudaFree(dfp);
            K::native_q8_0_packed_unregister(wp);
            cudaStreamDestroy(sp); cudaFree(wu); cudaFree(wp); cudaFree(xt); cudaFree(xq_t); cudaFree(yu); cudaFree(yp);
            std::printf("[PARITY + DEQUANT OK: %s]\n", s.what);
        }
        std::vector<void*> w(copies);
        for (auto& p : w) {
            ck(cudaMalloc(&p, bytes), "malloc w");
            ck(cudaMemcpy(p, do_pack ? packed_host.data() : host.data(), bytes, cudaMemcpyHostToDevice), "copy w");
            if (do_pack && K::native_q8_0_packed_eligible(s.n_in, s.n_out))
                K::native_q8_0_packed_register(p, p, s.n_in, s.n_out);
        }
        const int maxc = *std::max_element(cols.begin(), cols.end());
        std::vector<float> xh((size_t) s.n_in * maxc);
        for (auto& v : xh) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        float *x = nullptr, *y = nullptr;
        void* xq = nullptr;
        ck(cudaMalloc(&x, xh.size() * 4), "x");
        ck(cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x copy");
        ck(cudaMalloc(&xq, K::native_q8_1_bytes(s.n_in, maxc)), "xq");
        ck(cudaMalloc(&y, (size_t) s.n_out * maxc * 4), "y");
        cudaStream_t st;
        ck(cudaStreamCreate(&st), "stream");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::printf("\n%s: %.2f MB, %d copies\n", s.what, bytes / 1e6, copies);
        for (int c : cols) {
            K::native_quantize_q8_1(x, xq, s.n_in, c, st);
            for (int i = 0; i < 3; ++i) K::native_mmvq(8, w[i % copies], xq, y, s.n_in, s.n_out, c, st);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) K::native_mmvq(8, w[i % copies], xq, y, s.n_in, s.n_out, c, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1000.0 * ms / iters, gbs = bytes / (us * 1e3);
            std::printf("  %d col%s: %8.1f us  %7.1f GB/s%s\n", c, c == 1 ? " " : "s", us, gbs,
                        bw > 0 ? (" = " + std::to_string((int) (100 * gbs / bw + 0.5)) + "% of bw").c_str() : "");
        }
        for (auto p : w) {
            if (do_pack && K::native_q8_0_packed_eligible(s.n_in, s.n_out)) K::native_q8_0_packed_unregister(p);
            cudaFree(p);
        }
        cudaFree(x);
        cudaFree(xq);
        cudaFree(y);
        cudaStreamDestroy(st);
    }
    if (std::getenv("TP2_MINI_BENCH")) {
        struct Sh { const char* what; int n_in, n_out; };
        const Sh shs[] = {{"TP2 sh_dn 320->2560", 320, 2560}, {"TP2 sh_gu 2560->320", 2560, 320}, {"TP2 QSA_kv 2560->256", 2560, 256}};
        std::printf("\n=== TP2 Q8 mini shapes ===\n");
        for (const auto& s : shs) {
            const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 34;
            std::vector<uint8_t> h(bytes), pk(bytes);
            for (size_t b = 0; b < bytes; b += 34) { h[b] = 0x1f; h[b+1] = 0x21; for (int k = 0; k < 32; ++k) h[b+2+k] = (uint8_t)(rng()&0xff); }
            K::native_q8_0_pack_host(h.data(), pk.data(), s.n_in, s.n_out);
            std::vector<void*> w(16);
            for (auto& p : w) { ck(cudaMalloc(&p, bytes), "w"); ck(cudaMemcpy(p, pk.data(), bytes, cudaMemcpyHostToDevice), "cw"); K::native_q8_0_packed_register(p, p, s.n_in, s.n_out); }
            void* xq = nullptr; float *x = nullptr, *y = nullptr;
            ck(cudaMalloc(&x, (size_t) s.n_in * 4 * 4), "x"); ck(cudaMalloc(&xq, K::native_q8_1_bytes(s.n_in, 4)), "xq"); ck(cudaMalloc(&y, (size_t) s.n_out * 4 * 4), "y");
            cudaStream_t st; ck(cudaStreamCreate(&st), "st"); cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::printf("%s: %.2f MB\n", s.what, bytes / 1e6);
            for (int c : {1, 2, 3, 4}) {
                K::native_quantize_q8_1(x, xq, s.n_in, c, st);
                for (int i = 0; i < 3; ++i) K::native_mmvq(8, w[i % 16], xq, y, s.n_in, s.n_out, c, st);
                cudaEventRecord(e0, st);
                for (int i = 0; i < iters; ++i) K::native_mmvq(8, w[i % 16], xq, y, s.n_in, s.n_out, c, st);
                cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "s"); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                std::printf("  %d cols: %.2f us (%.1f GB/s)\n", c, 1000.0 * ms / iters, bytes / ((1000.0 * ms / iters) * 1e3));
            }
            if (s.n_out == 320) {
                float* y2 = nullptr; ck(cudaMalloc(&y2, (size_t) s.n_out * 4 * 4), "y2");
                for (int c : {2, 3, 4}) {
                    K::native_quantize_q8_1(x, xq, s.n_in, c, st);
                    for (int i = 0; i < 3; ++i) K::native_mmvq_pair(8, w[i%16], w[(i+1)%16], xq, y, y2, s.n_in, s.n_out, c, st);
                    cudaEventRecord(e0, st);
                    for (int i = 0; i < iters; ++i) K::native_mmvq_pair(8, w[i%16], w[(i+1)%16], xq, y, y2, s.n_in, s.n_out, c, st);
                    cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "sp"); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    std::printf("  PAIR %d cols: %.2f us (%.1f GB/s for 2x)\n", c, 1000.0 * ms / iters, 2.0 * bytes / ((1000.0 * ms / iters) * 1e3));
                }
                cudaFree(y2);
            }
            for (auto p : w) { K::native_q8_0_packed_unregister(p); cudaFree(p); }
            cudaFree(x); cudaFree(xq); cudaFree(y); cudaStreamDestroy(st);
        }
    }
    if (std::getenv("Q4_BENCH")) {
        struct Q4Shape { const char* what; int n_in, n_out; };
        const Q4Shape q4s[] = {
            {"MTP dhead 2560->106299", 2560, 106299},
            {"MTP fc 2560->2560", 2560, 2560},
            {"MTP sh_gu 2560->640", 2560, 640},
            {"MTP sh_dn 640->2560", 640, 2560},
        };
        std::printf("\n=== Q4_0 Unpacked vs Packed (sm_86) ===\n");
        for (const auto& s : q4s) {
            const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 18;
            std::vector<uint8_t> host(bytes), pk(bytes);
            for (size_t b = 0; b < bytes; b += 18) {
                host[b] = 0x1f; host[b + 1] = 0x21;
                for (int k = 0; k < 16; ++k) host[b + 2 + k] = (uint8_t) (rng() & 0xff);
            }
            K::native_q4_0_pack_host(host.data(), pk.data(), s.n_in, s.n_out);
            void *wu = nullptr, *wp = nullptr, *xq = nullptr;
            float *x = nullptr, *yu = nullptr, *yp = nullptr;
            const int maxc = 4;
            ck(cudaMalloc(&wu, bytes), "q4u"); ck(cudaMemcpy(wu, host.data(), bytes, cudaMemcpyHostToDevice), "cq4u");
            ck(cudaMalloc(&wp, bytes), "q4p"); ck(cudaMemcpy(wp, pk.data(), bytes, cudaMemcpyHostToDevice), "cq4p");
            K::native_q4_0_packed_register(wp, wp, s.n_in, s.n_out);
            std::vector<float> hx((size_t) s.n_in * maxc);
            for (auto& v : hx) v = std::normal_distribution<float>(0.f, 1.f)(rng);
            ck(cudaMalloc(&x, hx.size() * 4), "q4x"); ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "cq4x");
            ck(cudaMalloc(&xq, K::native_q8_1_bytes(s.n_in, maxc)), "q4xq");
            ck(cudaMalloc(&yu, (size_t) s.n_out * maxc * 4), "q4yu");
            ck(cudaMalloc(&yp, (size_t) s.n_out * maxc * 4), "q4yp");
            cudaStream_t st; ck(cudaStreamCreate(&st), "q4st");
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::printf("%s: %.2f MB\n", s.what, bytes / 1e6);
            for (int c : {1, 2, 4}) {
                K::native_quantize_q8_1(x, xq, s.n_in, c, st);
                K::native_mmvq(2, wu, xq, yu, s.n_in, s.n_out, c, st);
                K::native_mmvq(2, wp, xq, yp, s.n_in, s.n_out, c, st);
                ck(cudaStreamSynchronize(st), "q4par");
                std::vector<float> hu((size_t) c * s.n_out), hp((size_t) c * s.n_out);
                cudaMemcpy(hu.data(), yu, hu.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hp.data(), yp, hp.size() * 4, cudaMemcpyDeviceToHost);
                double err = 0, ref = 0;
                for (size_t i = 0; i < hu.size(); ++i) { err += std::fabs(hp[i] - hu[i]); ref += std::fabs(hu[i]); }
                const double rel = ref > 0 ? err / ref : 0.0;
                if (!(rel < 1e-5)) { std::fprintf(stderr, "Q4 PARITY FAIL %s c=%d rel=%.3e\n", s.what, c, rel); return 4; }
                for (int i = 0; i < 3; ++i) K::native_mmvq(2, wu, xq, yu, s.n_in, s.n_out, c, st);
                cudaEventRecord(e0, st);
                for (int i = 0; i < iters; ++i) K::native_mmvq(2, wu, xq, yu, s.n_in, s.n_out, c, st);
                cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "u");
                float msu = 0; cudaEventElapsedTime(&msu, e0, e1);
                for (int i = 0; i < 3; ++i) K::native_mmvq(2, wp, xq, yp, s.n_in, s.n_out, c, st);
                cudaEventRecord(e0, st);
                for (int i = 0; i < iters; ++i) K::native_mmvq(2, wp, xq, yp, s.n_in, s.n_out, c, st);
                cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "p");
                float msp = 0; cudaEventElapsedTime(&msp, e0, e1);
                const double usu = 1000.0 * msu / iters, usp = 1000.0 * msp / iters;
                std::printf("  %d col%s: unpacked %6.1f us (%5.1f GB/s) -> packed %6.1f us (%5.1f GB/s) [rel %.2e]\n",
                            c, c == 1 ? " " : "s", usu, bytes / (usu * 1e3), usp, bytes / (usp * 1e3), rel);
            }
            K::native_q4_0_packed_unregister(wp);
            cudaFree(wu); cudaFree(wp); cudaFree(x); cudaFree(xq); cudaFree(yu); cudaFree(yp); cudaStreamDestroy(st);
        }
    }
    return 0;
}
