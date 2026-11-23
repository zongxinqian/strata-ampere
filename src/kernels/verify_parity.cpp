// src/kernels/verify_parity.cpp - batched / split verify-window kernels against the kernels they replace, BITWISE.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every kernel here is checked against the
// one it stands in for, on random inputs that include -0.0, denormals, NaN, -inf and large values.  The tests
// follow eddoursul's fork (MIT; verify_parity.cpp), ported to this tree's kernels.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// argmax_rows against sample_tokens' greedy pick and row_top_prob_split against row_top_prob, bitwise: rows of both
// vocabularies with ties at the maximum (the lowest index wins), NaN, -inf, a row with no value above -inf, and
// launches repeated on one scratch (each must leave its counters at zero)
int test_argmax(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8));   // one scratch for every row count, as in the engine
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8));
    check(cudaMemset(sa, 0, argmax_rows_scratch_bytes(8)), "memset");
    check(cudaMemset(st, 0, row_top_prob_scratch_bytes(8)), "memset");
    for (const int n : {248320, 40525, 4097, 33}) {
        for (const int rows : {4, 8, 1, 3, 8, 2}) {
            std::vector<float> h((size_t) rows * n);
            for (float& v : h) v = nd(rng);
            for (int r = 0; r < rows; ++r) {
                float* row = h.data() + (size_t) r * n;
                const int a = (int) (rng() % n), b = (int) (rng() % n);
                row[a] = row[b] = 40.0f;                       // a tie at the maximum
                row[(int) (rng() % n)] = std::nanf("");        // never picked
                row[(int) (rng() % n)] = -INFINITY;
                if (r == 2)                                    // no value above -inf: 0
                    for (int i = 0; i < n; ++i) row[i] = i % 7 ? -INFINITY : std::nanf("");
            }
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            int* d_ref = dev<int>(rows);
            int32_t* d_out = dev<int32_t>(rows);
            float* d_p1 = dev<float>(rows);
            float* d_p2 = dev<float>(rows);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(d_l, rows, n, nullptr, 0, sp, d_ref, s);
            row_top_prob(d_l, rows, n, d_ref, d_p1, s);
            for (int rep = 0; rep < 3; ++rep) {
                argmax_rows(d_l, rows, n, sa, d_out, s);
                row_top_prob_split(d_l, rows, n, d_out, d_p2, st, s);
                check(cudaStreamSynchronize(s), "argmax");
                const std::vector<int> ref = down(d_ref, rows);
                const std::vector<int32_t> out = down(d_out, rows);
                const std::vector<float> p1 = down(d_p1, rows), p2 = down(d_p2, rows);
                for (int r = 0; r < rows; ++r)
                    if (ref[r] != out[r] || std::memcmp(&p1[r], &p2[r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  argmax n %d rows %d row %d: %d / %d, p %.9g / %.9g\n", n, rows, r, ref[r],
                                        out[r], p1[r], p2[r]);
                        ++bad;
                    }
            }
            (void) cudaFree(d_l); (void) cudaFree(d_ref); (void) cudaFree(d_out); (void) cudaFree(d_p1); (void) cudaFree(d_p2);
        }
    }
    (void) cudaFree(sa);
    (void) cudaFree(st);
    std::printf("argmax_rows, row_top_prob_split: %s\n",
                bad ? "MISMATCH" : "bitwise equal to the one-block kernels (ties, NaN, -inf, 1-8 rows on one scratch)");
    return bad;
}

int test_mapped_multi(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    // 7-copy round layout + non-multiple-of-4 large copy (1025 words) + 4B-aligned non-16B-aligned large copy (1028 words)
    const int64_t words[7] = {5, 40, 1025, 1028, 51200, 4, 16};
    const int64_t off[7]   = {0,  0,    0,    1,     0, 0,  0};   // entry 3 is +1 word (+4 B) offset -> 4B-aligned, not 16B
    int32_t* h_src[7] = {};
    int32_t* d_src[7] = {};
    int32_t* d_ref[7] = {};
    int32_t* d_out[7] = {};
    MappedCopy mc[7] = {};
    for (int i = 0; i < 7; ++i) {
        const size_t alloc_w = (size_t) (words[i] + off[i]);
        check(cudaHostAlloc((void**) &h_src[i], alloc_w * 4, cudaHostAllocMapped), "cudaHostAlloc");
        check(cudaHostGetDevicePointer((void**) &d_src[i], h_src[i], 0), "cudaHostGetDevicePointer");
        for (size_t j = 0; j < alloc_w; ++j) h_src[i][j] = (int32_t) rng();
        d_ref[i] = dev<int32_t>(alloc_w);
        d_out[i] = dev<int32_t>(alloc_w);
        check(cudaMemset(d_ref[i], 0, alloc_w * 4), "memset");
        if (words[i] > 1024 && (words[i] & 3) == 0 && off[i] == 0)
            copy_from_mapped((float*) (d_ref[i] + off[i]), (const float*) (d_src[i] + off[i]), words[i], s);
        else
            copy_i32_from_mapped(d_ref[i] + off[i], d_src[i] + off[i], words[i], s);
        mc[i] = {d_out[i] + off[i], d_src[i] + off[i], words[i]};
    }
    const char* prev = std::getenv("STRATA_MTP_MULTI_COPY");
    const std::string saved = prev ? prev : "";
    for (const char* mode : {"1", "0"}) {
#if defined(_WIN32)
        _putenv_s("STRATA_MTP_MULTI_COPY", mode);
#else
        setenv("STRATA_MTP_MULTI_COPY", mode, 1);
#endif
        for (int i = 0; i < 7; ++i) check(cudaMemsetAsync(d_out[i], 0x5a, (size_t) (words[i] + off[i]) * 4, s), "poison");
        copy_from_mapped_multi(mc, 7, s);
        check(cudaStreamSynchronize(s), "mapped_multi");
        for (int i = 0; i < 7; ++i) {
            const std::vector<int32_t> r = down(d_ref[i] + off[i], (size_t) words[i]);
            const std::vector<int32_t> o = down(d_out[i] + off[i], (size_t) words[i]);
            if (r != o) ++bad;
        }
    }
#if defined(_WIN32)
    _putenv_s("STRATA_MTP_MULTI_COPY", prev ? saved.c_str() : "");
#else
    if (prev) setenv("STRATA_MTP_MULTI_COPY", saved.c_str(), 1);
    else unsetenv("STRATA_MTP_MULTI_COPY");
#endif
    for (int i = 0; i < 7; ++i) {
        (void) cudaFreeHost(h_src[i]);
        (void) cudaFree(d_ref[i]);
        (void) cudaFree(d_out[i]);
    }
    std::printf("copy_from_mapped_multi: %s\n",
                bad ? "MISMATCH" : "bitwise equal to individual mapped copies (multi=1 and fallback=0, aligned + 1025w + 4B-aligned)");
    return bad;
}

int test_quantize_q8_0_scaled(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (const int64_t n : {2560, 10240, 25600}) {
        const int64_t nb = n / 32;
        std::vector<float> hx((size_t) n);
        for (int64_t i = 0; i < n; ++i) hx[(size_t) i] = nd(rng);
        for (int64_t j = 0; j < 32 && j < n; ++j) hx[(size_t) j] = 0.0f;   // all-zero block
        if (nb > 2) {
            hx[64] = 127.0f;
            for (int j = 1; j < 32; ++j) hx[64 + j] = (float) j - 0.5f;    // exact .5 half-away ties
        }
        float* d_x = dev<float>((size_t) n);
        uint8_t* d_b = dev<uint8_t>((size_t) nb * 34);
        float* d_s = dev<float>((size_t) nb);
        up(d_x, hx);
        quantize_q8_0_scaled(d_x, d_b, d_s, n, s);
        check(cudaStreamSynchronize(s), "q8_0_scaled");
        const std::vector<uint8_t> gb = down(d_b, (size_t) nb * 34);
        const std::vector<float> gs = down(d_s, (size_t) nb);
        for (int64_t b = 0; b < nb; ++b) {
            const float* xb = hx.data() + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
            const float sc = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = sc > 0.f ? 1.f / sc : 0.f;
            if (std::memcmp(&gs[(size_t) b], &sc, 4) != 0) ++bad;
            const uint16_t d16 = f16_from_f32(sc);
            const uint8_t* ob = gb.data() + b * 34;
            if (ob[0] != (uint8_t) (d16 & 0xFF) || ob[1] != (uint8_t) (d16 >> 8)) ++bad;
            for (int i = 0; i < 32; ++i) {
                const float t = xb[i] * inv;
                const float r = t + (t >= 0.f ? 0.5f : -0.5f);
                int v = (int) r;
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                if (ob[2 + i] != (uint8_t) (int8_t) v) ++bad;
            }
        }
        (void) cudaFree(d_x);
        (void) cudaFree(d_b);
        (void) cudaFree(d_s);
    }
    std::printf("quantize_q8_0_scaled: %s\n", bad ? "MISMATCH" : "bitwise equal to CPU act_quant_q8_1 reference (zero, .5 ties, normal)");
    return bad;
}

void bench_head_ops(cudaStream_t s) {
    using namespace strata::kernels;
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8));
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8));
    check(cudaMemset(sa, 0, argmax_rows_scratch_bytes(8)), "memset");
    check(cudaMemset(st, 0, row_top_prob_scratch_bytes(8)), "memset");
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "ev");
    check(cudaEventCreate(&e1), "ev");
    constexpr int ITERS = 200;
    auto time_fn = [&](auto&& fn) {
        for (int i = 0; i < 20; ++i) fn();
        check(cudaStreamSynchronize(s), "warm");
        check(cudaEventRecord(e0, s), "e0");
        for (int i = 0; i < ITERS; ++i) fn();
        check(cudaEventRecord(e1, s), "e1");
        check(cudaEventSynchronize(e1), "sync");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms * 1000.0 / ITERS;
    };
    for (const int n : {106299, 248320}) {
        for (const int rows : {1, 4, 5}) {
            if (n == 106299 && rows > 1) continue;
            std::vector<float> h((size_t) rows * n, 0.1f);
            for (int r = 0; r < rows; ++r) h[(size_t) r * n + (r * 1337 + 42) % n] = 10.0f;
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            int32_t* d_out = dev<int32_t>(rows);
            float* d_p = dev<float>(rows);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            const double us_g1 = time_fn([&] { sample_tokens(d_l, rows, n, nullptr, 0, sp, d_out, s); });
            const double us_gm = time_fn([&] { argmax_rows(d_l, rows, n, sa, d_out, s); });
            const double us_p1 = time_fn([&] { row_top_prob(d_l, rows, n, d_out, d_p, s); });
            const double us_pm = time_fn([&] { row_top_prob_split(d_l, rows, n, d_out, d_p, st, s); });
            std::printf("  V=%6d rows=%d: greedy 1-blk %6.2f us -> argmax_rows %6.2f us (x%.2f) | top_prob 1-blk %6.2f us -> split %6.2f us (x%.2f)\n",
                        n, rows, us_g1, us_gm, us_g1 / us_gm, us_p1, us_pm, us_p1 / us_pm);
            (void) cudaFree(d_l); (void) cudaFree(d_out); (void) cudaFree(d_p);
        }
    }
    (void) cudaEventDestroy(e0);
    (void) cudaEventDestroy(e1);
    (void) cudaFree(sa);
    (void) cudaFree(st);
}


int test_mtp_tp2_head(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(1));
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(1));
    check(cudaMemset(sa, 0, argmax_rows_scratch_bytes(1)), "memset sa");
    check(cudaMemset(st, 0, row_top_prob_scratch_bytes(1)), "memset st");

    int32_t *blk0_id = dev<int32_t>(128), *blk1_id = dev<int32_t>(128), *out0_id = dev<int32_t>(1), *out_id = dev<int32_t>(1);
    float *blk0_max = dev<float>(128), *blk1_max = dev<float>(128), *part0 = dev<float>(64), *part1 = dev<float>(64);
    float *out0_max = dev<float>(1), *out0_sum = dev<float>(1), *out_prob = dev<float>(1);
    unsigned int *ctr0 = dev<unsigned int>(1), *ctr1 = dev<unsigned int>(1);
    check(cudaMemset(ctr0, 0, 4), "ctr0");
    check(cudaMemset(ctr1, 0, 4), "ctr1");
    uint64_t* d_round0 = dev<uint64_t>(1);
    check(cudaMemset(d_round0, 0, 8), "round0");
    MtpHeadPartial* hp_h = nullptr;
    uint64_t* done_h = nullptr;
    check(cudaHostAlloc((void**) &hp_h, sizeof(MtpHeadPartial), cudaHostAllocMapped | cudaHostAllocPortable), "hp_h");
    check(cudaHostAlloc((void**) &done_h, 8, cudaHostAllocMapped | cudaHostAllocPortable), "done_h");
    *done_h = 0ull;
    MtpHeadPartial* hp_m0 = nullptr;
    uint64_t* done_m0 = nullptr;
    check(cudaHostGetDevicePointer((void**) &hp_m0, hp_h, 0), "hp_m0");
    check(cudaHostGetDevicePointer((void**) &done_m0, done_h, 0), "done_m0");

    // 1. Exact edge cases + ties + NaN + -inf + all -inf across split boundary on single GPU
    struct Case { int n; int n0; int mode; };
    const Case cases[] = {
        {106299, 53152, 0}, {106299, 53152, 1}, {106299, 53152, 2}, {106299, 53152, 3},
        {53152,  26576, 0}, {4097,   2048,  1}, {4097,   2048,  2}, {53152,  26576, 3}
    };
    int32_t* ref_id = dev<int32_t>(1);
    float* ref_prob = dev<float>(1);
    for (int ci = 0; ci < (int) (sizeof(cases) / sizeof(cases[0])); ++ci) {
        const Case& c = cases[ci];
        std::vector<float> h((size_t) c.n);
        if (c.mode == 3) {
            for (int i = 0; i < c.n; ++i) h[i] = (i % 7) ? -INFINITY : std::nanf("");
        } else {
            for (float& v : h) v = nd(rng);
            h[(size_t) (rng() % (size_t) c.n)] = std::nanf("");
            h[(size_t) (rng() % (size_t) c.n)] = -INFINITY;
            if (c.mode == 0) {
                h[123] = 40.0f;
                h[(size_t) c.n0 + 456] = 40.0f;   // cross-shard tie: lower index wins
            } else if (c.mode == 1) {
                h[(size_t) c.n0 + 777] = 41.0f;
                h[(size_t) c.n0 + 999] = 41.0f;   // shard-1 internal tie
            } else {
                for (int i = 0; i < c.n0; ++i) h[i] = -INFINITY;
                h[(size_t) c.n0 + 17] = 12.5f;   // shard 0 all -inf
            }
        }
        float* d_l = dev<float>(h.size());
        up(d_l, h);
        argmax_rows(d_l, 1, c.n, sa, ref_id, s);
        row_top_prob_split(d_l, 1, c.n, ref_id, ref_prob, st, s);
        mtp_tp2_bump_round(d_round0, s);
        mtp_tp2_head_argmax_sumexp(d_l + c.n0, c.n - c.n0, c.n0, blk1_id, blk1_max, part1, ctr1,
                                   nullptr, nullptr, nullptr, hp_m0, done_m0, d_round0, 0, s);
        mtp_tp2_head_argmax_sumexp(d_l, c.n0, 0, blk0_id, blk0_max, part0, ctr0,
                                   out0_id, out0_max, out0_sum, nullptr, nullptr, d_round0, 0, s);
        mtp_tp2_merge_head(out0_id, out0_max, out0_sum, hp_m0, done_m0, d_round0, 0, out_id, out_prob, s);
        check(cudaStreamSynchronize(s), "tp2 head sync");
        const int32_t r_id = down(ref_id, 1)[0], t_id = down(out_id, 1)[0];
        const float r_pr = down(ref_prob, 1)[0], t_pr = down(out_prob, 1)[0];
        // row_top_prob_split propagates any NaN logit into p=nan, whereas mtp_tp2 filters !isnan(x)
        const bool pr_ok = std::isnan(r_pr)
            ? (c.mode == 3 ? t_pr == 0.0f : (c.mode == 2 ? t_pr > 0.9999f : std::fabs(t_pr - 0.5f) < 1e-5f))
            : std::fabs(r_pr - t_pr) <= 1e-6f * std::fmax(1.0f, r_pr);
        if (r_id != t_id || !pr_ok) {
            std::printf("  tp2_head case %d (n=%d mode=%d): id %d/%d p %.9g/%.9g\n",
                        ci, c.n, c.mode, r_id, t_id, r_pr, t_pr);
            ++bad;
        }
        (void) cudaFree(d_l);
    }

    for (int ci = 0; ci < 5; ++ci) {
        const int n = 106299, n0 = 53152;
        std::vector<float> h(n);
        for (float& v : h) v = nd(rng);
        if (ci & 1) h[n0 + (rng() % (n - n0))] = 14.0f;
        else h[rng() % n0] = 14.0f;
        h[rng() % n] = -INFINITY;
        float* d_l = dev<float>(n);
        up(d_l, h);
        argmax_rows(d_l, 1, n, sa, ref_id, s);
        row_top_prob_split(d_l, 1, n, ref_id, ref_prob, st, s);
        mtp_tp2_bump_round(d_round0, s);
        mtp_tp2_head_argmax_sumexp(d_l + n0, n - n0, n0, blk1_id, blk1_max, part1, ctr1,
                                   nullptr, nullptr, nullptr, hp_m0, done_m0, d_round0, 0, s);
        mtp_tp2_head_argmax_sumexp(d_l, n0, 0, blk0_id, blk0_max, part0, ctr0,
                                   out0_id, out0_max, out0_sum, nullptr, nullptr, d_round0, 0, s);
        mtp_tp2_merge_head(out0_id, out0_max, out0_sum, hp_m0, done_m0, d_round0, 0, out_id, out_prob, s);
        check(cudaStreamSynchronize(s), "tp2 clean sync");
        const int32_t r_id = down(ref_id, 1)[0], t_id = down(out_id, 1)[0];
        const float r_pr = down(ref_prob, 1)[0], t_pr = down(out_prob, 1)[0];
        if (r_id != t_id || std::fabs(r_pr - t_pr) > 1e-6f * std::fmax(1.0f, r_pr)) {
            std::printf("  tp2_head clean %d: id %d/%d p %.9g/%.9g\n", ci, r_id, t_id, r_pr, t_pr);
            ++bad;
        }
        (void) cudaFree(d_l);
    }

    // 2. Real 2-GPU sync + abort_peer + multi-round parity when >= 2 CUDA devices exist
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev >= 2) {
        const int n = 106299, n0 = 53152, n1 = n - n0, xbytes = 2880;
        uint8_t *xq_h = nullptr;
        uint64_t *go_h = nullptr, *done2_h = nullptr;
        check(cudaHostAlloc((void**) &xq_h, xbytes, cudaHostAllocMapped | cudaHostAllocPortable), "xq_h");
        check(cudaHostAlloc((void**) &go_h, 8, cudaHostAllocMapped | cudaHostAllocPortable), "go_h");
        check(cudaHostAlloc((void**) &done2_h, 8, cudaHostAllocMapped | cudaHostAllocPortable), "done2_h");
        *go_h = 0ull; *done2_h = 0ull;
        uint8_t *xq_m0 = nullptr, *xq_m1 = nullptr, *xq0 = nullptr, *xq1 = nullptr;
        uint64_t *go_m0 = nullptr, *go_m1 = nullptr, *done2_m0 = nullptr, *done2_m1 = nullptr;
        MtpHeadPartial* hp2_m1 = nullptr;
        check(cudaHostGetDevicePointer((void**) &xq_m0, xq_h, 0), "xq_m0");
        check(cudaHostGetDevicePointer((void**) &go_m0, go_h, 0), "go_m0");
        check(cudaHostGetDevicePointer((void**) &done2_m0, done2_h, 0), "done2_m0");
        xq0 = dev<uint8_t>(xbytes);
        float *l0 = dev<float>(n), *l1 = nullptr;
        int32_t *b1_id = nullptr;
        float *b1_max = nullptr, *p1 = nullptr;
        unsigned int *c1 = nullptr;
        uint64_t *r1 = nullptr;
        cudaStream_t s1 = nullptr;
        check(cudaSetDevice(1), "dev1");
        check(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "s1");
        check(cudaHostGetDevicePointer((void**) &xq_m1, xq_h, 0), "xq_m1");
        check(cudaHostGetDevicePointer((void**) &go_m1, go_h, 0), "go_m1");
        check(cudaHostGetDevicePointer((void**) &done2_m1, done2_h, 0), "done2_m1");
        check(cudaHostGetDevicePointer((void**) &hp2_m1, hp_h, 0), "hp2_m1");
        xq1 = dev<uint8_t>(xbytes);
        l1 = dev<float>(n1);
        b1_id = dev<int32_t>(128); b1_max = dev<float>(128); p1 = dev<float>(64);
        c1 = dev<unsigned int>(1); check(cudaMemset(c1, 0, 4), "c1");
        r1 = dev<uint64_t>(1); check(cudaMemset(r1, 0, 8), "r1");
        check(cudaSetDevice(0), "dev0");
        check(cudaMemset(d_round0, 0, 8), "reset r0");
        const int run_steps[4] = {2, 6, 1, 3};
        for (int rnd = 0; rnd < 4; ++rnd) {
            std::vector<float> h(n);
            for (float& v : h) v = nd(rng);
            if (rnd & 1) h[(size_t) n0 + (rng() % (size_t) n1)] = 35.0f;
            else h[(size_t) (rng() % (size_t) n0)] = 35.0f;
            std::vector<uint8_t> hx(xbytes);
            for (int i = 0; i < xbytes; ++i) hx[i] = (uint8_t) (rng() & 255);
            up(xq0, hx);
            up(l0, h);
            check(cudaSetDevice(1), "dev1 up");
            check(cudaMemcpy(l1, h.data() + n0, (size_t) n1 * 4, cudaMemcpyHostToDevice), "up l1");
            check(cudaSetDevice(0), "dev0 run");
            argmax_rows(l0, 1, n, sa, ref_id, s);
            row_top_prob_split(l0, 1, n, ref_id, ref_prob, st, s);
            for (int j = 0; j < run_steps[rnd]; ++j) {
                check(cudaSetDevice(1), "dev1 step");
                mtp_tp2_peer_wait(go_m1, xq_m1, xq1, r1, (uint32_t) j, (size_t) xbytes, 0ull, s1);
                mtp_tp2_head_argmax_sumexp(l1, n1, n0, b1_id, b1_max, p1, c1,
                                           nullptr, nullptr, nullptr, hp2_m1, done2_m1, r1, (uint32_t) j, s1);
                check(cudaSetDevice(0), "dev0 step");
                if (j == 0) mtp_tp2_bump_round(d_round0, s);
                mtp_tp2_publish_x(xq0, xq_m0, go_m0, d_round0, (uint32_t) j, (size_t) xbytes, s);
                mtp_tp2_head_argmax_sumexp(l0, n0, 0, blk0_id, blk0_max, part0, ctr0,
                                           out0_id, out0_max, out0_sum, nullptr, nullptr, d_round0, (uint32_t) j, s);
                mtp_tp2_merge_head(out0_id, out0_max, out0_sum, hp_m0, done2_m0, d_round0, (uint32_t) j, out_id, out_prob, s);
                check(cudaStreamSynchronize(s), "sync s0");
                check(cudaSetDevice(1), "dev1 sync");
                check(cudaStreamSynchronize(s1), "sync s1");
                std::vector<uint8_t> gx(xbytes);
                check(cudaMemcpy(gx.data(), xq1, xbytes, cudaMemcpyDeviceToHost), "dl xq1");
                check(cudaSetDevice(0), "dev0 cmp");
                if (std::memcmp(hx.data(), gx.data(), xbytes) != 0) ++bad;
                const int32_t r_id = down(ref_id, 1)[0], t_id = down(out_id, 1)[0];
                const float r_pr = down(ref_prob, 1)[0], t_pr = down(out_prob, 1)[0];
                if (r_id != t_id || std::fabs(r_pr - t_pr) > 1e-6f * std::fmax(1.0f, r_pr)) ++bad;
            }
            if (run_steps[rnd] < 6) mtp_tp2_abort_peer(d_round0, go_m0, s);
            check(cudaStreamSynchronize(s), "abort sync");
        }
        check(cudaSetDevice(1), "dev1 free");
        (void) cudaStreamDestroy(s1);
        (void) cudaFree(xq1); (void) cudaFree(l1); (void) cudaFree(b1_id);
        (void) cudaFree(b1_max); (void) cudaFree(p1); (void) cudaFree(c1); (void) cudaFree(r1);
        check(cudaSetDevice(0), "dev0 free");
        (void) cudaFree(xq0); (void) cudaFree(l0);
        (void) cudaFreeHost(xq_h); (void) cudaFreeHost(go_h); (void) cudaFreeHost(done2_h);
    }
    (void) cudaFree(sa); (void) cudaFree(st);
    (void) cudaFree(blk0_id); (void) cudaFree(blk1_id); (void) cudaFree(out0_id); (void) cudaFree(out_id);
    (void) cudaFree(blk0_max); (void) cudaFree(blk1_max); (void) cudaFree(part0); (void) cudaFree(part1);
    (void) cudaFree(out0_max); (void) cudaFree(out0_sum); (void) cudaFree(out_prob);
    (void) cudaFree(ctr0); (void) cudaFree(ctr1); (void) cudaFree(d_round0);
    (void) cudaFree(ref_id); (void) cudaFree(ref_prob);
    (void) cudaFreeHost(hp_h); (void) cudaFreeHost(done_h);
    return bad;
}

int test_tree_ops(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;

    // 1. argmax_top2_prob (single & row_dev!=nullptr across sizes, ties, NaN, -inf)
    {
        int a2_bad = 0;
        std::normal_distribution<float> nd(0.0f, 3.5f);
        uint8_t* scr = dev<uint8_t>(argmax_top2_prob_scratch_bytes());
        check(cudaMemset(scr, 0, argmax_top2_prob_scratch_bytes()), "top2 memset");
        int32_t* d_o1 = dev<int32_t>(1);
        int32_t* d_o2 = dev<int32_t>(1);
        int32_t* d_row = dev<int32_t>(1);
        float* d_p1 = dev<float>(1);
        float* d_p2 = dev<float>(1);

        for (int n : {248320, 40525, 4097, 33}) {
            const int R = 4;
            std::vector<float> h((size_t) R * n);
            for (float& v : h) v = nd(rng);
            // row 0: tie at top-1 -> lower index wins top1, second index wins top2
            {
                float* r0 = h.data();
                r0[5] = 42.0f;
                r0[n - 3] = 42.0f;
                r0[1] = std::nanf("");
                r0[2] = -INFINITY;
            }
            // row 1: unique max, tie at top-2 -> lowest tied index wins top2
            {
                float* r1 = h.data() + n;
                for (int i = 0; i < n; ++i) r1[i] = -10.0f + 0.0001f * (float) (i % 17);
                r1[11] = 25.0f;
                r1[20] = 21.0f;
                r1[3] = 21.0f;
            }
            // row 2: random normal with NaNs / -inf
            {
                float* r2 = h.data() + (size_t) 2 * n;
                r2[(int) (rng() % n)] = std::nanf("");
                r2[(int) (rng() % n)] = -INFINITY;
            }
            // row 3: all -inf/NaN -> id1=0, prob1=0, id2=-1, prob2=0
            {
                float* r3 = h.data() + (size_t) 3 * n;
                for (int i = 0; i < n; ++i) r3[i] = (i % 5 == 0) ? std::nanf("") : -INFINITY;
            }
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            for (int row = 0; row < R; ++row) {
                const bool use_rdev = (row > 0);
                up(d_row, std::vector<int32_t>{row});
                argmax_top2_prob(d_l, n, scr, d_o1, d_p1, d_o2, d_p2, s, use_rdev ? d_row : nullptr);
                check(cudaStreamSynchronize(s), "top2 sync");
                int32_t g1 = down(d_o1, 1)[0], g2 = down(d_o2, 1)[0];
                float gp1 = down(d_p1, 1)[0], gp2 = down(d_p2, 1)[0];

                const float* rr = h.data() + (size_t) row * n;
                int e1 = -1, e2 = -1;
                float m1 = -INFINITY, m2 = -INFINITY;
                for (int i = 0; i < n; ++i) {
                    const float v = rr[i];
                    if (std::isnan(v)) continue;
                    if (e1 < 0 || v > m1) { e2 = e1; m2 = m1; e1 = i; m1 = v; }
                    else if (e2 < 0 || v > m2) { e2 = i; m2 = v; }
                }
                double sum = 0.0;
                if (std::isfinite(m1)) {
                    for (int i = 0; i < n; ++i) if (!std::isnan(rr[i])) sum += std::exp((double) (rr[i] - m1));
                } else {
                    e1 = 0;
                    e2 = -1;
                }
                const float ep1 = (sum > 0.0) ? (float) (1.0 / sum) : 0.0f;
                const float ep2 = (e2 >= 0 && std::isfinite(m2) && std::isfinite(m1) && sum > 0.0)
                                      ? (float) (std::exp((double) (m2 - m1)) / sum) : 0.0f;
                if (g1 != e1 || g2 != e2 || std::fabs(gp1 - ep1) > 1e-5f || std::fabs(gp2 - ep2) > 1e-5f) {
                    if (a2_bad < 5)
                        std::printf("  argmax_top2 n=%d row=%d: got (%d %.6g, %d %.6g) exp (%d %.6g, %d %.6g)\n",
                                    n, row, g1, gp1, g2, gp2, e1, ep1, e2, ep2);
                    ++a2_bad;
                }
            }
            argmax_top2_prob(d_l, n, scr, nullptr, nullptr, d_o2, d_p2, s, nullptr);
            check(cudaStreamSynchronize(s), "top2 null sync");
            if (down(d_o2, 1)[0] != n - 3) ++a2_bad;
            (void) cudaFree(d_l);
        }
        (void) cudaFree(scr); (void) cudaFree(d_o1); (void) cudaFree(d_o2); (void) cudaFree(d_row);
        (void) cudaFree(d_p1); (void) cudaFree(d_p2);
        std::printf("argmax_top2_prob vs CPU oracle: %s (16 cases)\n", a2_bad ? "FAIL" : "exact");
        bad += a2_bad;
    }

    // 2. mtp_tp2_head_argmax_sumexp + mtp_tp2_merge_head top2 across 2 GPUs
    {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev >= 2) {
            int tp_bad = 0;
            std::normal_distribution<float> nd(0.0f, 3.0f);
            cudaStream_t s1 = nullptr;
            check(cudaSetDevice(1), "set dev1");
            check(cudaStreamCreate(&s1), "s1 create");
            check(cudaSetDevice(0), "set dev0");

            MtpHeadPartial* h_hp = nullptr; MtpHeadPartial* d_hp0 = nullptr; MtpHeadPartial* d_hp1 = nullptr;
            uint64_t* h_done = nullptr; uint64_t* d_done0 = nullptr; uint64_t* d_done1 = nullptr;
            check(cudaHostAlloc(&h_hp, sizeof(MtpHeadPartial) + 64, cudaHostAllocMapped | cudaHostAllocPortable), "alloc hp");
            check(cudaHostAlloc(&h_done, 64, cudaHostAllocMapped | cudaHostAllocPortable), "alloc done");
            std::memset(h_hp, 0, sizeof(MtpHeadPartial));
            *h_done = 0;
            check(cudaHostGetDevicePointer(&d_hp0, h_hp, 0), "hp0");
            check(cudaHostGetDevicePointer(&d_done0, h_done, 0), "done0");
            check(cudaSetDevice(1), "dev1 ptr");
            check(cudaHostGetDevicePointer(&d_hp1, h_hp, 0), "hp1");
            check(cudaHostGetDevicePointer(&d_done1, h_done, 0), "done1");
            uint64_t* d_rnd1 = dev<uint64_t>(1);
            check(cudaMemset(d_rnd1, 0, 8), "rnd1");
            int32_t* d1_bid = dev<int32_t>(128); float* d1_bmax = dev<float>(128);
            float* d1_part = dev<float>(64); unsigned int* d1_ctr = dev<unsigned int>(1);
            check(cudaMemset(d1_ctr, 0, 4), "ctr1");
            int32_t* d1_oid = dev<int32_t>(2); float* d1_omax = dev<float>(2); float* d1_osum = dev<float>(1);

            check(cudaSetDevice(0), "dev0 ptr");
            uint64_t* d_rnd0 = dev<uint64_t>(1);
            check(cudaMemset(d_rnd0, 0, 8), "rnd0");
            int32_t* d0_bid = dev<int32_t>(128); float* d0_bmax = dev<float>(128);
            float* d0_part = dev<float>(64); unsigned int* d0_ctr = dev<unsigned int>(1);
            check(cudaMemset(d0_ctr, 0, 4), "ctr0");
            int32_t* d0_oid = dev<int32_t>(2); float* d0_omax = dev<float>(2); float* d0_osum = dev<float>(1);
            int32_t* d_out1 = dev<int32_t>(1); float* d_prob1 = dev<float>(1);
            int32_t* d_out2 = dev<int32_t>(1); float* d_prob2 = dev<float>(1);

            const int n0 = 124160, n1 = 124160;
            for (int mode = 0; mode < 4; ++mode) {
                std::vector<float> v0(n0), v1(n1);
                for (float& x : v0) x = nd(rng);
                for (float& x : v1) x = nd(rng);
                if (mode == 0) { v0[10] = 30.0f; v0[500] = 28.0f; }
                if (mode == 1) { v0[100] = 30.0f; v1[200] = 28.5f; }
                if (mode == 2) { v1[77] = 31.0f; v1[99] = 29.0f; }
                if (mode == 3) { v0[300] = 30.0f; v1[400] = 30.0f; }

                check(cudaSetDevice(1), "m dev1");
                float* dl1 = dev<float>(n1);
                up(dl1, v1);
                const uint32_t step_j = (uint32_t) (mode + 1);
                mtp_tp2_head_argmax_sumexp(dl1, n1, n0, d1_bid, d1_bmax, d1_part, d1_ctr, d1_oid, d1_omax, d1_osum,
                                           d_hp1, d_done1, d_rnd1, step_j, s1, d1_oid + 1, d1_omax + 1);
                check(cudaSetDevice(0), "m dev0");
                float* dl0 = dev<float>(n0);
                up(dl0, v0);
                mtp_tp2_head_argmax_sumexp(dl0, n0, 0, d0_bid, d0_bmax, d0_part, d0_ctr, d0_oid, d0_omax, d0_osum,
                                           nullptr, nullptr, d_rnd0, step_j, s, d0_oid + 1, d0_omax + 1);
                mtp_tp2_merge_head(d0_oid, d0_omax, d0_osum, d_hp0, d_done0, d_rnd0, step_j,
                                   d_out1, d_prob1, s, d_out2, d_prob2);
                check(cudaStreamSynchronize(s), "merge s");
                check(cudaSetDevice(1), "sync s1");
                check(cudaStreamSynchronize(s1), "sync s1");
                check(cudaSetDevice(0), "back dev0");

                const int32_t g1 = down(d_out1, 1)[0], g2 = down(d_out2, 1)[0];
                const float gp1 = down(d_prob1, 1)[0], gp2 = down(d_prob2, 1)[0];

                int e1 = -1, e2 = -1; float m1 = -INFINITY, m2 = -INFINITY;
                for (int i = 0; i < n0 + n1; ++i) {
                    const float val = i < n0 ? v0[i] : v1[i - n0];
                    if (e1 < 0 || val > m1) { e2 = e1; m2 = m1; e1 = i; m1 = val; }
                    else if (e2 < 0 || val > m2) { e2 = i; m2 = val; }
                }
                double sum = 0.0;
                for (int i = 0; i < n0 + n1; ++i) sum += std::exp((double) ((i < n0 ? v0[i] : v1[i - n0]) - m1));
                const float ep1 = (float) (1.0 / sum), ep2 = (float) (std::exp((double) (m2 - m1)) / sum);
                if (g1 != e1 || g2 != e2 || std::fabs(gp1 - ep1) > 1e-5f || std::fabs(gp2 - ep2) > 1e-5f) {
                    if (tp_bad < 5)
                        std::printf("  tp2 top2 mode=%d: got (%d %.6g, %d %.6g) exp (%d %.6g, %d %.6g)\n",
                                    mode, g1, gp1, g2, gp2, e1, ep1, e2, ep2);
                    ++tp_bad;
                }
                (void) cudaFree(dl0);
                check(cudaSetDevice(1), "free dl1");
                (void) cudaFree(dl1);
                check(cudaSetDevice(0), "free dl0 done");
            }
            check(cudaSetDevice(1), "cleanup dev1");
            (void) cudaFree(d_rnd1); (void) cudaFree(d1_bid); (void) cudaFree(d1_bmax);
            (void) cudaFree(d1_part); (void) cudaFree(d1_ctr); (void) cudaFree(d1_oid);
            (void) cudaFree(d1_omax); (void) cudaFree(d1_osum);
            (void) cudaStreamDestroy(s1);
            check(cudaSetDevice(0), "cleanup dev0");
            (void) cudaFree(d_rnd0); (void) cudaFree(d0_bid); (void) cudaFree(d0_bmax);
            (void) cudaFree(d0_part); (void) cudaFree(d0_ctr); (void) cudaFree(d0_oid);
            (void) cudaFree(d0_omax); (void) cudaFree(d0_osum); (void) cudaFree(d_out1);
            (void) cudaFree(d_prob1); (void) cudaFree(d_out2); (void) cudaFree(d_prob2);
            (void) cudaFreeHost(h_hp); (void) cudaFreeHost(h_done);
            std::printf("mtp_tp2_head + merge top2 (2 GPUs): %s (4 cases)\n", tp_bad ? "FAIL" : "exact");
            bad += tp_bad;
        }
    }

    // 3. native_ple_postops_batch_snap(..., tree_fork, tree_parent) vs sequential native_ple_postops
    {
        int ple_bad = 0;
        constexpr int N = NG_N_EMBD, D = NG_HC_DIM, HS = NG_HIST * NG_HC_DIM;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> nk(D), nq(D), nc(D), hist0(HS);
        for (auto* v : {&nk, &nq, &nc}) for (float& x : *v) x = 1.0f + 0.1f * nd(rng);
        for (float& x : hist0) x = nd(rng);
        std::vector<uint16_t> conv((size_t) 4 * D);
        for (uint16_t& x : conv) x = f16_from_f32(0.3f * nd(rng));
        PleWeights w;
        float *dnk = dev<float>(D), *dnq = dev<float>(D), *dnc = dev<float>(D);
        uint16_t* dconv = dev<uint16_t>(conv.size());
        up(dnk, nk); up(dnq, nq); up(dnc, nc); up(dconv, conv);
        w.norm_key = dnk; w.norm_query = dnq; w.norm_conv = dnc; w.conv1d_f16 = dconv;

        struct TC { int T, fork, parent; };
        for (TC tc : {TC{5, 3, 0}, TC{5, 3, 1}, TC{6, 4, 1}}) {
            std::vector<float> key((size_t) tc.T * D), hid((size_t) tc.T * D), val((size_t) tc.T * N);
            for (float& x : key) x = 4.0f * nd(rng);
            for (float& x : hid) x = nd(rng);
            for (float& x : val) x = nd(rng);

            float *k0 = dev<float>(key.size()), *h0 = dev<float>(hid.size()), *v0 = dev<float>(val.size());
            up(k0, key); up(h0, hid); up(v0, val);
            float *ha = dev<float>(HS), *sa = dev<float>((size_t) tc.T * HS);
            up(ha, hist0);
            float *q = dev<float>(D), *nrm = dev<float>(D), *g = dev<float>(4), *gd = dev<float>(D), *cv = dev<float>(D);
            for (int t = 0; t < tc.T; ++t) {
                if (t == tc.fork) {
                    check(cudaMemcpyAsync(ha, sa + (size_t) tc.parent * HS, (size_t) HS * 4,
                                          cudaMemcpyDeviceToDevice, s), "ref restore");
                }
                NativePlePostopsBuffers b{q, nrm, g, gd, nrm, cv, h0 + (size_t) t * D};
                native_ple_postops(k0 + (size_t) t * D, h0 + (size_t) t * D, v0 + (size_t) t * N, ha, w, b, s);
                ple_history_advance(ha, nrm, s);
                check(cudaMemcpyAsync(sa + (size_t) t * HS, ha, (size_t) HS * 4, cudaMemcpyDeviceToDevice, s), "ref snap");
            }

            float *k1 = dev<float>(key.size()), *h1 = dev<float>(hid.size()), *v1 = dev<float>(val.size());
            up(k1, key); up(h1, hid); up(v1, val);
            float *hb = dev<float>(HS), *sb = dev<float>((size_t) tc.T * HS);
            up(hb, hist0);
            float *qn = dev<float>((size_t) tc.T * D), *gated = dev<float>((size_t) tc.T * D), *gate = dev<float>((size_t) tc.T * 4);
            native_ple_postops_batch_snap(k1, h1, v1, hb, w, qn, gated, gate, tc.T, sb, s, tc.fork, tc.parent);
            check(cudaStreamSynchronize(s), "ple tree sync");
            if (down(h0, hid.size()) != down(h1, hid.size()) ||
                down(sa, (size_t) tc.T * HS) != down(sb, (size_t) tc.T * HS) ||
                down(ha, HS) != down(hb, HS)) {
                ++ple_bad;
                std::printf("  PLE tree mismatch T=%d fork=%d parent=%d\n", tc.T, tc.fork, tc.parent);
            }
            for (float* p : {k0, h0, v0, ha, sa, q, nrm, g, gd, cv, k1, h1, v1, hb, sb, qn, gated, gate}) (void) cudaFree(p);
        }
        (void) cudaFree(dnk); (void) cudaFree(dnq); (void) cudaFree(dnc); (void) cudaFree(dconv);
        std::printf("native_ple_postops_batch_snap tree: %s (3 configs)\n", ple_bad ? "FAIL" : "bitwise");
        bad += ple_bad;
    }

    // 4. QSA tree indexer snap/restore + append_steps(rows_dev) + remap_sel
    {
        int qsa_bad = 0;
        const QsaShapes qs = qsa_real_shapes();
        const int ID = (int) qs.idx_dim, IB = (int) qs.idx_block;
        const int max_cells = 4096, max_blocks = max_cells / IB + 1;
        const int ISN = (IB + 5) * ID + 4;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> gamma(ID);
        for (float& x : gamma) x = 1.0f + 0.1f * nd(rng);
        float* d_gam = dev<float>(ID); up(d_gam, gamma);
        RopeScaling rsc{};

        const int T = 5, fk = 3;
        for (int pos0 : {2, 3, 2002, 2003}) {
            for (int par : {0, 1}) {
                std::vector<float> raw((size_t) T * ID), init_tail((size_t) (IB - 1) * ID), init_dead(ID),
                                   init_pool((size_t) max_blocks * ID);
                for (float& x : raw) x = nd(rng);
                for (float& x : init_tail) x = nd(rng);
                for (float& x : init_dead) x = nd(rng);
                for (float& x : init_pool) x = nd(rng);
                int32_t init_bp = (pos0 / IB) * IB;

                float *d_raw = dev<float>(raw.size()), *d_pre = dev<float>(ISN), *d_br = dev<float>(ISN);
                up(d_raw, raw);
                auto alloc_st = [&](float*& t, float*& d, float*& p, int32_t*& bp) {
                    t = dev<float>((IB - 1) * ID); d = dev<float>(ID); p = dev<float>(max_blocks * ID); bp = dev<int32_t>(1);
                    up(t, init_tail); up(d, init_dead); up(p, init_pool); up(bp, std::vector<int32_t>{init_bp});
                };
                float *t1, *d1, *p1, *t2, *d2, *p2; int32_t *bp1, *bp2;
                alloc_st(t1, d1, p1, bp1);
                alloc_st(t2, d2, p2, bp2);

                const int32_t wrows[4] = {0, par == 1 ? 1 : 3, par == 1 ? 3 : 4, 4};
                const int n_b = T - fk;
                const int nk = par + 1 + n_b;
                std::vector<int32_t> hst(T * kStepCount), hpos_b(nk * kStepCount), hrows(wrows, wrows + nk);
                for (int i = 0; i < par + 1; ++i) qsa_step_fill(hst.data() + i * kStepCount, pos0 + i, qs);
                for (int j = 0; j < n_b; ++j) qsa_step_fill(hst.data() + (fk + j) * kStepCount, pos0 + par + 1 + j, qs);
                for (int s_i = 0; s_i < nk; ++s_i) qsa_step_fill(hpos_b.data() + s_i * kStepCount, pos0 + s_i, qs);
                int32_t* d_st = dev<int32_t>(hst.size()); up(d_st, hst);
                int32_t* d_pb = dev<int32_t>(hpos_b.size()); up(d_pb, hpos_b);
                int32_t* d_rows = dev<int32_t>(hrows.size()); up(d_rows, hrows);

                QsaIndexerBuffers ib1{t1, d1, p1, bp1}, ib2{t2, d2, p2, bp2};
                std::vector<float> pack_raw((size_t) nk * ID);
                for (int s_i = 0; s_i < nk; ++s_i)
                    std::memcpy(pack_raw.data() + (size_t) s_i * ID, raw.data() + (size_t) wrows[s_i] * ID, (size_t) ID * 4);
                float* d_praw = dev<float>(pack_raw.size()); up(d_praw, pack_raw);
                native_qsa_indexer_append_steps(d_praw, d_pb + kStepPos, kStepCount, nk, 0, d_gam, 1e-6f, ib2, qs, max_cells, rsc, s);

                qsa_tree_idx_snap(d_st, t1, d1, p1, bp1, max_blocks, ID, IB, d_pre, s);
                native_qsa_indexer_append_steps(d_raw, d_st + kStepPos, kStepCount, par + 1, 0, d_gam, 1e-6f, ib1, qs, max_cells, rsc, s);
                qsa_tree_idx_snap(d_st, t1, d1, p1, bp1, max_blocks, ID, IB, d_br, s);
                if (fk - 1 > par) {
                    std::vector<int32_t> hstA(kStepCount * (fk - 1 - par));
                    for (int u = par + 1; u < fk; ++u) qsa_step_fill(hstA.data() + (u - par - 1) * kStepCount, pos0 + u, qs);
                    int32_t* d_stA = dev<int32_t>(hstA.size()); up(d_stA, hstA);
                    native_qsa_indexer_append_steps(d_raw + (size_t) (par + 1) * ID, d_stA + kStepPos, kStepCount,
                                                    fk - 1 - par, 0, d_gam, 1e-6f, ib1, qs, max_cells, rsc, s);
                    check(cudaStreamSynchronize(s), "stA");
                    (void) cudaFree(d_stA);
                }
                qsa_tree_idx_restore(d_st, t1, d1, p1, bp1, max_blocks, ID, IB, d_br, s);
                native_qsa_indexer_append_steps(d_raw + (size_t) fk * ID, d_st + fk * kStepCount + kStepPos, kStepCount,
                                                n_b, 0, d_gam, 1e-6f, ib1, qs, max_cells, rsc, s);
                check(cudaStreamSynchronize(s), "tree branch");
                if (down(t1, (IB - 1) * ID) != down(t2, (IB - 1) * ID) ||
                    down(d1, ID) != down(d2, ID) ||
                    down(p1, max_blocks * ID) != down(p2, max_blocks * ID) ||
                    down(bp1, 1)[0] != down(bp2, 1)[0]) {
                    ++qsa_bad;
                    std::printf("  QSA tree branch mismatch pos0=%d par=%d\n", pos0, par);
                }
                qsa_tree_idx_restore(d_st, t1, d1, p1, bp1, max_blocks, ID, IB, d_pre, s);
                check(cudaStreamSynchronize(s), "restore pre");
                if (down(t1, (IB - 1) * ID) != init_tail || down(d1, ID) != init_dead ||
                    down(p1, max_blocks * ID) != init_pool || down(bp1, 1)[0] != init_bp) {
                    ++qsa_bad;
                    std::printf("  QSA restore pre mismatch pos0=%d par=%d\n", pos0, par);
                }
                native_qsa_indexer_append_steps(d_raw, d_pb + kStepPos, kStepCount, nk, 0, d_gam, 1e-6f,
                                                ib1, qs, max_cells, rsc, s, d_rows);
                check(cudaStreamSynchronize(s), "append rows_dev");
                if (down(t1, (IB - 1) * ID) != down(t2, (IB - 1) * ID) ||
                    down(d1, ID) != down(d2, ID) ||
                    down(p1, max_blocks * ID) != down(p2, max_blocks * ID) ||
                    down(bp1, 1)[0] != down(bp2, 1)[0]) {
                    ++qsa_bad;
                    std::printf("  QSA rows_dev commit mismatch pos0=%d par=%d\n", pos0, par);
                }
                std::vector<int32_t> sel_h = {-1, 0, pos0, pos0 + par, pos0 + par + 1, pos0 + par + n_b, pos0 + par + n_b + 1};
                int32_t* d_sel = dev<int32_t>(sel_h.size()); up(d_sel, sel_h);
                qsa_tree_remap_sel(d_sel, (int) sel_h.size(), d_st, fk, par, n_b, s);
                check(cudaStreamSynchronize(s), "remap sel");
                auto sel_got = down(d_sel, sel_h.size());
                const int shift = fk - par - 1;
                std::vector<int32_t> sel_exp = {-1, 0, pos0, pos0 + par, pos0 + par + 1 + shift, pos0 + par + n_b + shift, pos0 + par + n_b + 1};
                if (sel_got != sel_exp) {
                    ++qsa_bad;
                    std::printf("  QSA remap_sel mismatch pos0=%d par=%d\n", pos0, par);
                }
                (void) cudaFree(d_sel); (void) cudaFree(d_raw); (void) cudaFree(d_praw);
                (void) cudaFree(d_pre); (void) cudaFree(d_br); (void) cudaFree(d_st); (void) cudaFree(d_pb); (void) cudaFree(d_rows);
                (void) cudaFree(t1); (void) cudaFree(d1); (void) cudaFree(p1); (void) cudaFree(bp1);
                (void) cudaFree(t2); (void) cudaFree(d2); (void) cudaFree(p2); (void) cudaFree(bp2);
            }
        }
        (void) cudaFree(d_gam);
        std::printf("QSA tree snap/restore + rows_dev + remap_sel: %s (8 cases)\n", qsa_bad ? "FAIL" : "bitwise");
        bad += qsa_bad;
    }

    // 5. kv_copy_tree_layers + copy_tree_R_rows
    {
        int kv_bad = 0;
        QsaShapes qs = qsa_real_shapes();
        qs.page_size = 4;
        const int H = (int) qs.n_head_kv, D = (int) qs.head_dim, P = (int) qs.page_size, G = D / KV_Q8_GROUP;
        const int L = 2, n_cells = 64, pages = n_cells / P;
        const int64_t pool_e = (int64_t) pages * H * P;
        std::normal_distribution<float> nd(0.0f, 1.0f);

        for (int pos0 : {2, 11}) {
            const int T = 5;
            std::vector<float> k_in((size_t) L * T * H * D), v_in((size_t) L * T * H * D);
            for (float& x : k_in) x = nd(rng);
            for (float& x : v_in) x = nd(rng);

            std::vector<int32_t> pt_h(pages);
            for (int i = 0; i < pages; ++i) pt_h[i] = i;
            std::shuffle(pt_h.begin(), pt_h.end(), rng);
            pt_h[pos0 / P] = -1;

            int32_t* d_pt[2] = {};
            int8_t *dk_ref[2] = {}, *dv_ref[2] = {}, *dk_dut[2] = {}, *dv_dut[2] = {};
            uint16_t *dks_ref[2] = {}, *dvs_ref[2] = {}, *dks_dut[2] = {}, *dvs_dut[2] = {};
            int8_t *hk_ref[2] = {}, *hv_ref[2] = {}, *hk_dut[2] = {}, *hv_dut[2] = {};
            uint16_t *hks_ref[2] = {}, *hvs_ref[2] = {}, *hks_dut[2] = {}, *hvs_dut[2] = {};
            QsaKvLayers kl{};
            for (int l = 0; l < L; ++l) {
                d_pt[l] = dev<int32_t>(pages); up(d_pt[l], pt_h);
                dk_ref[l] = dev<int8_t>(pool_e * D); dv_ref[l] = dev<int8_t>(pool_e * D);
                dk_dut[l] = dev<int8_t>(pool_e * D); dv_dut[l] = dev<int8_t>(pool_e * D);
                dks_ref[l] = dev<uint16_t>(pool_e * G); dvs_ref[l] = dev<uint16_t>(pool_e * G);
                dks_dut[l] = dev<uint16_t>(pool_e * G); dvs_dut[l] = dev<uint16_t>(pool_e * G);
                check(cudaMemset(dk_ref[l], 0, pool_e * D), "m"); check(cudaMemset(dv_ref[l], 0, pool_e * D), "m");
                check(cudaMemset(dk_dut[l], 0, pool_e * D), "m"); check(cudaMemset(dv_dut[l], 0, pool_e * D), "m");
                check(cudaMemset(dks_ref[l], 0, pool_e * G * 2), "m"); check(cudaMemset(dvs_ref[l], 0, pool_e * G * 2), "m");
                check(cudaMemset(dks_dut[l], 0, pool_e * G * 2), "m"); check(cudaMemset(dvs_dut[l], 0, pool_e * G * 2), "m");

                hk_ref[l] = dev<int8_t>(pool_e * D); hv_ref[l] = dev<int8_t>(pool_e * D);
                hk_dut[l] = dev<int8_t>(pool_e * D); hv_dut[l] = dev<int8_t>(pool_e * D);
                hks_ref[l] = dev<uint16_t>(pool_e * G); hvs_ref[l] = dev<uint16_t>(pool_e * G);
                hks_dut[l] = dev<uint16_t>(pool_e * G); hvs_dut[l] = dev<uint16_t>(pool_e * G);
                check(cudaMemset(hk_ref[l], 0, pool_e * D), "m"); check(cudaMemset(hv_ref[l], 0, pool_e * D), "m");
                check(cudaMemset(hk_dut[l], 0, pool_e * D), "m"); check(cudaMemset(hv_dut[l], 0, pool_e * D), "m");
                check(cudaMemset(hks_ref[l], 0, pool_e * G * 2), "m"); check(cudaMemset(hvs_ref[l], 0, pool_e * G * 2), "m");
                check(cudaMemset(hks_dut[l], 0, pool_e * G * 2), "m"); check(cudaMemset(hvs_dut[l], 0, pool_e * G * 2), "m");

                kl.k_q[l] = dk_dut[l]; kl.v_q[l] = dv_dut[l];
                kl.k_scale[l] = dks_dut[l]; kl.v_scale[l] = dvs_dut[l];
                kl.page_table[l] = d_pt[l];
                kl.host[l].k_q = hk_dut[l];
                kl.host[l].v_q = hv_dut[l];
                kl.host[l].k_scale = hks_dut[l];
                kl.host[l].v_scale = hvs_dut[l];
            }

            const int32_t win_rows[4] = {0, 1, 3, 4};
            const int nk = 4;
            int32_t* d_step = dev<int32_t>(kStepCount);
            float* d_kc = dev<float>(H * D); float* d_vc = dev<float>(H * D);
            for (int l = 0; l < L; ++l) {
                KvHostPools href{};
                href.k_q = hk_ref[l]; href.v_q = hv_ref[l]; href.k_scale = hks_ref[l]; href.v_scale = hvs_ref[l];
                KvHostPools hdut{};
                hdut.k_q = hk_dut[l]; hdut.v_q = hv_dut[l]; hdut.k_scale = hks_dut[l]; hdut.v_scale = hvs_dut[l];
                for (int t = 0; t < T; ++t) {
                    int32_t hs[kStepCount]; qsa_step_fill(hs, pos0 + t, qs);
                    up(d_step, std::vector<int32_t>(hs, hs + kStepCount));
                    const size_t off = ((size_t) l * T + t) * H * D;
                    check(cudaMemcpyAsync(d_kc, k_in.data() + off, (size_t) H * D * 4, cudaMemcpyHostToDevice, s), "kc");
                    check(cudaMemcpyAsync(d_vc, v_in.data() + off, (size_t) H * D * 4, cudaMemcpyHostToDevice, s), "vc");
                    kv_append_q8_step(dk_dut[l], dv_dut[l], dks_dut[l], dvs_dut[l], d_pt[l], d_step, d_kc, d_vc, qs, s, &hdut);
                }
                for (int st_i = 0; st_i < nk; ++st_i) {
                    int32_t hs[kStepCount]; qsa_step_fill(hs, pos0 + st_i, qs);
                    up(d_step, std::vector<int32_t>(hs, hs + kStepCount));
                    const size_t off = ((size_t) l * T + win_rows[st_i]) * H * D;
                    check(cudaMemcpyAsync(d_kc, k_in.data() + off, (size_t) H * D * 4, cudaMemcpyHostToDevice, s), "kc2");
                    check(cudaMemcpyAsync(d_vc, v_in.data() + off, (size_t) H * D * 4, cudaMemcpyHostToDevice, s), "vc2");
                    kv_append_q8_step(dk_ref[l], dv_ref[l], dks_ref[l], dvs_ref[l], d_pt[l], d_step, d_kc, d_vc, qs, s, &href);
                }
            }
            int32_t* d_nk = dev<int32_t>(1); up(d_nk, std::vector<int32_t>{nk});
            int32_t* d_p0w = dev<int32_t>(2);
            const uint64_t p0u = (uint64_t) (int64_t) pos0;
            up(d_p0w, std::vector<int32_t>{(int32_t) (uint32_t) p0u, (int32_t) (uint32_t) (p0u >> 32)});
            int32_t* d_rows = dev<int32_t>(nk); up(d_rows, std::vector<int32_t>(win_rows, win_rows + nk));

            kv_copy_tree_layers(kl, L, d_nk, d_p0w, d_rows, qs, s);
            check(cudaStreamSynchronize(s), "kv_copy_tree sync");

            for (int l = 0; l < L; ++l) {
                auto hr_k = down(hk_ref[l], pool_e * D), hd_k = down(hk_dut[l], pool_e * D);
                auto hr_v = down(hv_ref[l], pool_e * D), hd_v = down(hv_dut[l], pool_e * D);
                auto hs_rk = down(hks_ref[l], pool_e * G), hs_dk = down(hks_dut[l], pool_e * G);
                auto hs_rv = down(hvs_ref[l], pool_e * G), hs_dv = down(hvs_dut[l], pool_e * G);
                auto dr_k = down(dk_ref[l], pool_e * D), dd_k = down(dk_dut[l], pool_e * D);
                auto dr_v = down(dv_ref[l], pool_e * D), dd_v = down(dv_dut[l], pool_e * D);
                auto ds_rk = down(dks_ref[l], pool_e * G), ds_dk = down(dks_dut[l], pool_e * G);
                auto ds_rv = down(dvs_ref[l], pool_e * G), ds_dv = down(dvs_dut[l], pool_e * G);
                for (int st_i = 0; st_i < nk; ++st_i) {
                    const int pos = pos0 + st_i;
                    for (int h = 0; h < H; ++h) {
                        const int64_t hrow = ((int64_t) (pos / P) * H + h) * P + (pos % P);
                        if (std::memcmp(&hr_k[hrow * D], &hd_k[hrow * D], D) != 0 ||
                            std::memcmp(&hr_v[hrow * D], &hd_v[hrow * D], D) != 0 ||
                            std::memcmp(&hs_rk[hrow * G], &hs_dk[hrow * G], (size_t) G * 2) != 0 ||
                            std::memcmp(&hs_rv[hrow * G], &hs_dv[hrow * G], (size_t) G * 2) != 0)
                            ++kv_bad;
                        if (pt_h[pos / P] >= 0) {
                            const int64_t prow = ((int64_t) pt_h[pos / P] * H + h) * P + (pos % P);
                            if (std::memcmp(&dr_k[prow * D], &dd_k[prow * D], D) != 0 ||
                                std::memcmp(&dr_v[prow * D], &dd_v[prow * D], D) != 0 ||
                                std::memcmp(&ds_rk[prow * G], &ds_dk[prow * G], (size_t) G * 2) != 0 ||
                                std::memcmp(&ds_rv[prow * G], &ds_dv[prow * G], (size_t) G * 2) != 0)
                                ++kv_bad;
                        }
                    }
                }
            }
            for (int l = 0; l < L; ++l) {
                (void) cudaFree(d_pt[l]); (void) cudaFree(dk_ref[l]); (void) cudaFree(dv_ref[l]);
                (void) cudaFree(dk_dut[l]); (void) cudaFree(dv_dut[l]); (void) cudaFree(dks_ref[l]); (void) cudaFree(dvs_ref[l]);
                (void) cudaFree(dks_dut[l]); (void) cudaFree(dvs_dut[l]);
                (void) cudaFree(hk_ref[l]); (void) cudaFree(hv_ref[l]); (void) cudaFree(hk_dut[l]); (void) cudaFree(hv_dut[l]);
                (void) cudaFree(hks_ref[l]); (void) cudaFree(hvs_ref[l]); (void) cudaFree(hks_dut[l]); (void) cudaFree(hvs_dut[l]);
            }
            (void) cudaFree(d_step); (void) cudaFree(d_kc); (void) cudaFree(d_vc);
            (void) cudaFree(d_nk); (void) cudaFree(d_p0w); (void) cudaFree(d_rows);
        }

        const int64_t HCN = 4 * 2560;
        std::vector<float> h_R((size_t) 5 * HCN);
        for (float& x : h_R) x = nd(rng);
        float* d_R = dev<float>(h_R.size());
        up(d_R, h_R);
        int32_t* d_nk = dev<int32_t>(1); up(d_nk, std::vector<int32_t>{4});
        int32_t* d_wr = dev<int32_t>(4); up(d_wr, std::vector<int32_t>{0, 1, 3, 4});
        copy_tree_R_rows(d_R, HCN, d_nk, d_wr, s);
        check(cudaStreamSynchronize(s), "copy R");
        auto got_R = down(d_R, h_R.size());
        const int wr[4] = {0, 1, 3, 4};
        for (int k_i = 0; k_i < 4; ++k_i)
            if (std::memcmp(&got_R[(size_t) k_i * HCN], &h_R[(size_t) wr[k_i] * HCN], (size_t) HCN * 4) != 0) ++kv_bad;
        up(d_R, h_R);
        up(d_nk, std::vector<int32_t>{3});
        up(d_wr, std::vector<int32_t>{0, 1, 2});
        copy_tree_R_rows(d_R, HCN, d_nk, d_wr, s);
        check(cudaStreamSynchronize(s), "copy R noop");
        if (down(d_R, h_R.size()) != h_R) ++kv_bad;
        (void) cudaFree(d_R); (void) cudaFree(d_nk); (void) cudaFree(d_wr);

        std::printf("kv_copy_tree_layers + copy_tree_R_rows: %s\n", kv_bad ? "FAIL" : "bitwise");
        bad += kv_bad;
    }

    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") do_bench = true;
        else if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: verify_parity [--selftest] [--bench]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260925);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    bad += test_argmax(rng, s);
    bad += test_mapped_multi(rng, s);
    bad += test_quantize_q8_0_scaled(rng, s);
    bad += test_mtp_tp2_head(rng, s);
    bad += test_tree_ops(rng, s);
    {   using namespace strata::kernels; // tp2_resident_plan: bitwise plan parity against host GpuPlanSink replay
        const int NE = 512, K = 10, capx = 80;
        const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const int plan_i32 = (int) (ptr_off + 4 * capx + capx + 1);
        int32_t* d_ids = dev<int32_t>(capx);
        int32_t* d_r0 = dev<int32_t>(NE);
        int32_t* d_r1 = dev<int32_t>(NE);
        int32_t* d_pl0 = dev<int32_t>(plan_i32);
        int32_t* d_pl1 = dev<int32_t>(plan_i32);
        uint32_t* d_skip = dev<uint32_t>(1);
        std::vector<uint64_t> slot_off(1024);
        for (size_t i = 0; i < slot_off.size(); ++i) slot_off[i] = (uint64_t) i * 4096ull;
        unsigned long long* d_off = dev<unsigned long long>(1024);
        up(d_off, std::vector<unsigned long long>(slot_off.begin(), slot_off.end()));
        const uint8_t* fake_base = (const uint8_t*) (uintptr_t) 0x1000000ull;
        int plan_bad = 0;
        for (int mode = 0; mode < 4; ++mode) {
            for (int n_tok = 1; n_tok <= 8; ++n_tok) {
                const int n = n_tok * K;
                std::vector<int32_t> h_ids((size_t) n);
                std::vector<int32_t> h_r0(NE, -1), h_r1(NE, -1);
                for (int e = 0; e < NE; ++e) {
                    if (mode == 0) { if (e % 3 == 0) h_r0[e] = e; else if (e % 3 == 1) h_r1[e] = e + 100; }
                    else if (mode == 1) { if (e % 2 == 0) h_r0[e] = e; else h_r1[e] = e + 200; } // 0 misses
                    else if (mode == 2) { if (e % 5 == 0) h_r1[e] = e; } // 0 GPU0 hits
                    else { if (e % 4 == 0) h_r0[e] = e; } // 0 GPU1 hits
                }
                for (int i = 0; i < n; ++i) h_ids[i] = (int32_t) (rng() % NE);
                up(d_ids, h_ids); up(d_r0, h_r0); up(d_r1, h_r1);
                check(cudaMemsetAsync(d_pl0, 0xcd, (size_t) plan_i32 * 4, s), "z0");
                check(cudaMemsetAsync(d_pl1, 0xcd, (size_t) plan_i32 * 4, s), "z1");
                check(cudaMemsetAsync(d_skip, 0xcd, 4, s), "zs");
                const uint32_t ring = 17u + (uint32_t) n_tok;
                tp2_resident_plan(d_ids, n, K, d_r0, d_r1, NE, fake_base, d_off, 4096, d_pl0, capx, d_skip, ring, s);
                tp2_resident_plan(d_ids, n, K, d_r1, nullptr, NE, fake_base, d_off, 4096, d_pl1, capx, nullptr, 0, s);
                check(cudaStreamSynchronize(s), "plan sync");
                auto g0 = down(d_pl0, (size_t) plan_i32);
                auto g1 = down(d_pl1, (size_t) plan_i32);
                uint32_t got_skip = down(d_skip, 1)[0];

                // Host replay matching expert_source.cpp
                int first_of[80], distinct[80], nd = 0;
                for (int i = 0; i < n; ++i) {
                    first_of[i] = i;
                    for (int j = 0; j < i; ++j) if (h_ids[j] == h_ids[i]) { first_of[i] = first_of[j]; break; }
                    if (first_of[i] == i) distinct[nd++] = i;
                }
                std::vector<int32_t> exp_start0(81), exp_dst0, exp_tok0, exp_start1(81), exp_dst1, exp_tok1, exp_hit1;
                std::vector<unsigned long long> exp_ptr0, exp_ptr1;
                for (int q = 0; q < nd; ++q) {
                    int i0 = distinct[q], e = h_ids[i0];
                    if (h_r0[e] >= 0) {
                        exp_ptr0.push_back((unsigned long long) (uintptr_t) fake_base + slot_off[(size_t) h_r0[e]]);
                        exp_start0[exp_ptr0.size() - 1] = (int32_t) exp_dst0.size();
                        for (int i = i0; i < n; ++i) if (first_of[i] == i0) { exp_dst0.push_back(i); exp_tok0.push_back(i / K); }
                    } else if (h_r1[e] >= 0) {
                        exp_ptr1.push_back((unsigned long long) (uintptr_t) fake_base + slot_off[(size_t) h_r1[e]]);
                        exp_start1[exp_ptr1.size() - 1] = (int32_t) exp_dst1.size();
                        for (int i = i0; i < n; ++i) if (first_of[i] == i0) { exp_dst1.push_back(i); exp_tok1.push_back(i / K); }
                    }
                }
                for (int i = 0; i < n; ++i) if (h_r0[h_ids[i]] < 0 && h_r1[h_ids[i]] >= 0) exp_hit1.push_back(i);
                int nh0 = (int) exp_dst0.size(), nh1 = (int) exp_hit1.size();
                uint32_t exp_skip = (nh0 + nh1 == n) ? ring : 0u;
                if (got_skip != exp_skip) ++plan_bad;
                if (g0[0] != (int32_t) exp_ptr0.size() || g0[1] != nh0 || g0[2] != 0 || g0[3] != nh0 + nh1) ++plan_bad;
                const int32_t* gs0 = g0.data() + 4;
                const int32_t* gd0 = gs0 + capx + 1;
                const int32_t* gt0 = gd0 + capx;
                const unsigned long long* gp0 = (const unsigned long long*) (g0.data() + ptr_off);
                for (int g = 0; g < g0[0]; ++g) if (gs0[g] != exp_start0[g] || gp0[g] != exp_ptr0[g]) ++plan_bad;
                if (gs0[g0[0]] != nh0) ++plan_bad;
                for (int i = 0; i < nh0; ++i) if (gd0[i] != exp_dst0[i] || gt0[i] != exp_tok0[i]) ++plan_bad;
                std::vector<int32_t> got_h1(gd0 + nh0, gd0 + nh0 + nh1);
                if (got_h1 != exp_hit1) ++plan_bad;
                if (g1[0] != (int32_t) exp_ptr1.size() || g1[1] != nh1 || g1[3] != nh1) ++plan_bad;
                const int32_t* gs1 = g1.data() + 4;
                const int32_t* gd1 = gs1 + capx + 1;
                const int32_t* gt1 = gd1 + capx;
                const unsigned long long* gp1 = (const unsigned long long*) (g1.data() + ptr_off);
                for (int g = 0; g < g1[0]; ++g) if (gs1[g] != exp_start1[g] || gp1[g] != exp_ptr1[g]) ++plan_bad;
                for (int i = 0; i < nh1; ++i) if (gd1[i] != exp_dst1[i] || gt1[i] != exp_tok1[i]) ++plan_bad;
            }
        }
        (void) cudaFree(d_ids); (void) cudaFree(d_r0); (void) cudaFree(d_r1);
        (void) cudaFree(d_pl0); (void) cudaFree(d_pl1); (void) cudaFree(d_skip); (void) cudaFree(d_off);
        std::printf("tp2_resident_plan == host GpuPlanSink: %s (32 cases)\n", plan_bad ? "MISMATCH" : "bitwise");
        bad += plan_bad;
    }
    {   using namespace strata::kernels; // native_router_fused_bf16_bell_plan: bitwise parity vs unfused (T=1..8, both bell=true & bell=false)
        const int N = 2560, NE = 512, K = 10, capx = 80;
        const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
        std::vector<uint16_t> hw((size_t) NE * N), haw((size_t) N);
        for (auto& v : hw) { float f = 0.05f * std::normal_distribution<float>(0.0f, 1.0f)(rng); uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
        for (auto& v : haw) { float f = 0.05f * std::normal_distribution<float>(0.0f, 1.0f)(rng); uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
        std::vector<float> hx((size_t) 8 * N);
        for (auto& v : hx) v = std::normal_distribution<float>(0.0f, 1.0f)(rng);
        std::vector<int32_t> hr0(NE), hr1(NE);
        for (int e = 0; e < NE; ++e) { hr0[e] = (e % 3 == 0) ? (e * 7) % 1000 : -1; hr1[e] = (e % 5 == 0) ? (e * 11) % 1000 : -1; }
        std::vector<unsigned long long> hoff(1024);
        for (int i = 0; i < 1024; ++i) hoff[i] = (unsigned long long) i * 4096ull + 64ull;
        uint16_t *dw, *daw;
        float *dx, *dl1, *dl2, *da1, *da2, *dw1, *dw2, *hmx1, *hmx2, *dmx1, *dmx2;
        int32_t *di1, *di2, *dr0, *dr1, *dpl1, *dpl2, *hmi1, *hmi2, *dmi1, *dmi2;
        uint32_t *hmq1, *hmq2, *dmq1, *dmq2, *dsk1, *dsk2, *dctr;
        unsigned long long *doff;
        check(cudaMalloc(&dw, hw.size() * 2), "mw"); check(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice), "mcw");
        check(cudaMalloc(&daw, haw.size() * 2), "maw"); check(cudaMemcpy(daw, haw.data(), haw.size() * 2, cudaMemcpyHostToDevice), "mcaw");
        check(cudaMalloc(&dx, hx.size() * 4), "mx"); check(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "mcx");
        check(cudaMalloc(&dl1, 8 * NE * 4), "ml1"); check(cudaMalloc(&dl2, 8 * NE * 4), "ml2");
        check(cudaMalloc(&da1, 8 * 4), "ma1"); check(cudaMalloc(&da2, 8 * 4), "ma2");
        check(cudaMalloc(&dw1, 8 * K * 4), "mw1"); check(cudaMalloc(&dw2, 8 * K * 4), "mw2");
        check(cudaMalloc(&di1, 8 * K * 4), "mi1"); check(cudaMalloc(&di2, 8 * K * 4), "mi2");
        check(cudaMalloc(&dr0, NE * 4), "mr0"); check(cudaMemcpy(dr0, hr0.data(), NE * 4, cudaMemcpyHostToDevice), "mcr0");
        check(cudaMalloc(&dr1, NE * 4), "mr1"); check(cudaMemcpy(dr1, hr1.data(), NE * 4, cudaMemcpyHostToDevice), "mcr1");
        check(cudaMalloc(&dpl1, plan_words * 4), "mpl1"); check(cudaMalloc(&dpl2, plan_words * 4), "mpl2");
        check(cudaMalloc(&dsk1, 64), "msk1"); check(cudaMalloc(&dsk2, 64), "msk2");
        check(cudaMalloc(&dctr, 64), "mctr"); check(cudaMemset(dctr, 0, 64), "mcctr");
        check(cudaMalloc(&doff, 1024 * 8), "moff"); check(cudaMemcpy(doff, hoff.data(), 1024 * 8, cudaMemcpyHostToDevice), "mcoff");
        check(cudaHostAlloc(&hmx1, 8 * N * 4, cudaHostAllocMapped | cudaHostAllocPortable), "hmx1");
        check(cudaHostAlloc(&hmx2, 8 * N * 4, cudaHostAllocMapped | cudaHostAllocPortable), "hmx2");
        check(cudaHostAlloc(&hmi1, 8 * K * 4, cudaHostAllocMapped | cudaHostAllocPortable), "hmi1");
        check(cudaHostAlloc(&hmi2, 8 * K * 4, cudaHostAllocMapped | cudaHostAllocPortable), "hmi2");
        check(cudaHostAlloc(&hmq1, 64, cudaHostAllocMapped | cudaHostAllocPortable), "hmq1");
        check(cudaHostAlloc(&hmq2, 64, cudaHostAllocMapped | cudaHostAllocPortable), "hmq2");
        check(cudaHostGetDevicePointer(&dmx1, hmx1, 0), "dmx1"); check(cudaHostGetDevicePointer(&dmx2, hmx2, 0), "dmx2");
        check(cudaHostGetDevicePointer(&dmi1, hmi1, 0), "dmi1"); check(cudaHostGetDevicePointer(&dmi2, hmi2, 0), "dmi2");
        check(cudaHostGetDevicePointer(&dmq1, hmq1, 0), "dmq1"); check(cudaHostGetDevicePointer(&dmq2, hmq2, 0), "dmq2");
        uint8_t* const fake_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));
        int mega_bad = 0;
        for (int n = 1; n <= 8; ++n) {
            for (int bell = 0; bell <= 1; ++bell) {
                for (int has_aux = 0; has_aux <= 1; ++has_aux) {
                    check(cudaMemsetAsync(dpl1, 0xAB, plan_words * 4, s), "zp1");
                    check(cudaMemsetAsync(dpl2, 0xAB, plan_words * 4, s), "zp2");
                    check(cudaMemsetAsync(dsk1, 0, 64, s), "zsk1");
                    check(cudaMemsetAsync(dsk2, 0, 64, s), "zsk2");
                    *hmq1 = 0; *hmq2 = 0;
                    const uint32_t ring = (uint32_t) (n * 10 + bell * 3 + has_aux + 1);
                    if (bell) copy_from_mapped(dmx1, dx, (int64_t) n * N, s);
                    bf16_gemv_fp32_mmvf_multi(dx, N, dw, dl1, NE, N, NE, n, s);
                    if (has_aux) bf16_gemv_fp32_mmvf_multi(dx, N, daw, da1, 1, N, 1, n, s);
                    if (bell) {
                        native_router_top10_bell(dl1, di1, dw1, nullptr, dr0, N, n, nullptr, dmi1, dmq1, ring, true, s);
                        tp2_resident_plan(di1, n * K, K, dr0, dr1, NE, fake_base, doff, 4096, dpl1, capx, dsk1, ring, s);
                    } else {
                        native_router_top10_multi(dl1, di1, dw1, n, s);
                        tp2_resident_plan(di1, n * K, K, dr0, nullptr, NE, fake_base, doff, 4096, dpl1, capx, nullptr, 0, s);
                    }
                    native_router_fused_bf16_bell_plan(
                        dx, N, dw, dl2, NE, n, has_aux ? daw : nullptr, has_aux ? da2 : nullptr, 1,
                        bell ? dmx2 : nullptr, di2, dw2, bell ? dmi2 : nullptr, bell ? dmq2 : nullptr, ring,
                        dr0, bell ? dr1 : nullptr, fake_base, doff, 4096, dpl2, capx, bell ? dsk2 : nullptr, dctr, bell != 0, s);
                    check(cudaStreamSynchronize(s), "msync");
                    auto l1 = down(dl1, (size_t) n * NE), l2 = down(dl2, (size_t) n * NE);
                    auto w1v = down(dw1, (size_t) n * K), w2v = down(dw2, (size_t) n * K);
                    auto i1v = down(di1, (size_t) n * K), i2v = down(di2, (size_t) n * K);
                    auto p1v = down(dpl1, plan_words), p2v = down(dpl2, plan_words);
                    uint32_t sk1v = 0, sk2v = 0;
                    check(cudaMemcpy(&sk1v, dsk1, 4, cudaMemcpyDeviceToHost), "sk1d");
                    check(cudaMemcpy(&sk2v, dsk2, 4, cudaMemcpyDeviceToHost), "sk2d");
                    if (std::memcmp(l1.data(), l2.data(), l1.size() * 4) != 0 ||
                        std::memcmp(w1v.data(), w2v.data(), w1v.size() * 4) != 0 ||
                        std::memcmp(i1v.data(), i2v.data(), i1v.size() * 4) != 0 ||
                        std::memcmp(p1v.data(), p2v.data(), plan_words * 4) != 0 ||
                        (has_aux && std::memcmp(down(da1, (size_t) n).data(), down(da2, (size_t) n).data(), (size_t) n * 4) != 0) ||
                        (bell && (std::memcmp(hmx1, hmx2, (size_t) n * N * 4) != 0 ||
                                  std::memcmp(hmi1, hmi2, (size_t) n * K * 4) != 0 ||
                                  *hmq1 != *hmq2 || sk1v != sk2v))) {
                        ++mega_bad;
                    }
                }
            }
        }
        (void) cudaFree(dw); (void) cudaFree(daw); (void) cudaFree(dx);
        (void) cudaFree(dl1); (void) cudaFree(dl2); (void) cudaFree(da1); (void) cudaFree(da2);
        (void) cudaFree(dw1); (void) cudaFree(dw2); (void) cudaFree(di1); (void) cudaFree(di2);
        (void) cudaFree(dr0); (void) cudaFree(dr1); (void) cudaFree(dpl1); (void) cudaFree(dpl2);
        (void) cudaFree(dsk1); (void) cudaFree(dsk2); (void) cudaFree(dctr); (void) cudaFree(doff);
        (void) cudaFreeHost(hmx1); (void) cudaFreeHost(hmx2); (void) cudaFreeHost(hmi1); (void) cudaFreeHost(hmi2);
        (void) cudaFreeHost(hmq1); (void) cudaFreeHost(hmq2);
        std::printf("native_router_fused_bf16_bell_plan == unfused: %s (32 cases)\n", mega_bad ? "MISMATCH" : "bitwise");
        bad += mega_bad;
    }
    if (do_bench && bad == 0) bench_head_ops(s);
    (void) cudaStreamDestroy(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
