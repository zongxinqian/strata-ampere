// src/kernels/cpu/kq_avx2.cpp - Zen2-tuned AVX2 expert kernels for Unsloth UD-Q4_K_XL
// (Q4_K & Q5_K gate/up against Q8_K; Q5_1 & Q8_0 down against Q8_1 / Q8_0).
// Pure single-row linear DRAM streaming (hardware-prefetcher friendly) + precomputed token block scalars + zero YMM spills.
// Bit-exact against ggml-cpu / strata::quant AVX2 vec_dot for every token (nt >= 1).
#include "strata/kernels/cpu/kq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int kSmallNbK = 16;   // up to 4096 cols on fast 1 KB stack
constexpr int kSmallNb32 = 128; // up to 4096 cols on fast 4 KB stack

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

inline void half_pair(const void* block, float& d, float& m) {
    uint32_t dm;
    std::memcpy(&dm, block, 4);
    const __m128 f = _mm_cvtph_ps(_mm_cvtsi32_si128((int) dm));
    d = _mm_cvtss_f32(f);
    m = _mm_cvtss_f32(_mm_shuffle_ps(f, f, 1));
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

alignas(32) constexpr uint64_t kShufBits32[4] = {
    0x0000000000000000ull, 0x0101010101010101ull,
    0x0202020202020202ull, 0x0303030303030303ull
};
alignas(32) constexpr uint64_t kBitMask32[4] = {
    0x7fbfdfeff7fbfdfeull, 0x7fbfdfeff7fbfdfeull,
    0x7fbfdfeff7fbfdfeull, 0x7fbfdfeff7fbfdfeull
};

inline __m256i bytes_from_bits_32(const uint8_t* x, const __m256i shuf_mask, const __m256i bit_mask, const __m256i minus1) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, minus1);
}

inline __m256i bytes_from_nibbles_32(const uint8_t* rsi, const __m256i low_mask) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    return _mm256_and_si256(low_mask, bytes);
}

inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy, const __m256i ones) {
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax, sy)));
}

alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

inline void unpack_k4_scales_mins(const uint8_t* raw, __m256i& scales, __m128i& mins) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, raw, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
        _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
    mins = _mm256_extracti128_si256(mins_and_scales, 1);
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
}

inline void precompute_q8s(const void* const* act, int nt, int nb, __m128i q8s[][kSmallNbK], float yd[][kSmallNbK]) {
    for (int t = 0; t < nt; ++t) {
        const block_q8_K* y = (const block_q8_K*) act[t];
        for (int i = 0; i < nb; ++i) {
            yd[t][i] = y[i].d;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
            q8s[t][i] = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        }
    }
}

// ---- Q4_K x Q8_K: single-row linear DRAM stream, zero YMM spills for NT = 1..4
template <int NT>
void q4k_dot_nt(const block_q4_K* x, int nb, const void* const* act,
                const __m128i q8s[][kSmallNbK], const float yd[][kSmallNbK], float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    alignas(32) float acc_mem[NT][8];
    __m256 acc_reg[NT <= 4 ? NT : 1];
    if constexpr (NT <= 4) {
        for (int t = 0; t < NT; ++t) acc_reg[t] = _mm256_setzero_ps();
    } else {
        for (int t = 0; t < NT; ++t) _mm256_store_ps(acc_mem[t], _mm256_setzero_ps());
    }
    alignas(16) __m128 acc_m[NT];
    for (int t = 0; t < NT; ++t) acc_m[t] = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &x[i + 2] + 64, _MM_HINT_T0);
        }
        float xd, xdmin;
        half_pair(&x[i], xd, xdmin);
        __m256i scales;
        __m128i mins;
        unpack_k4_scales_mins(x[i].scales, scales, mins);
        for (int t = 0; t < NT; ++t) {
            const float dmin = -yd[t][i] * xdmin;
            const __m128i prod = _mm_madd_epi16(mins, q8s[t][i]);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
        }
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        const uint8_t* q4 = x[i].qs;
        const int off = i * (int) sizeof(block_q8_K) + (int) offsetof(block_q8_K, qs);
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const int qoff = off + 64 * j;
            {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                for (int t = 0; t < NT; ++t) {
                    const __m256i q8l = _mm256_loadu_si256((const __m256i*) ((const uint8_t*) act[t] + qoff));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l)));
                }
            }
            {
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                for (int t = 0; t < NT; ++t) {
                    const __m256i q8h = _mm256_loadu_si256((const __m256i*) ((const uint8_t*) act[t] + qoff + 32));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h)));
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float d = yd[t][i] * xd;
            if constexpr (NT <= 4) {
                acc_reg[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc_reg[t]);
            } else {
                const __m256 prev = _mm256_load_ps(acc_mem[t]);
                _mm256_store_ps(acc_mem[t], _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), prev));
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        const __m256 a = (NT <= 4) ? acc_reg[t] : _mm256_load_ps(acc_mem[t]);
        res[t] = hsum_float_8(a) + _mm_cvtss_f32(m);
    }
}

void q4k_dot_any(const block_q4_K* x, int nb, const void* const* act, int m,
                 const __m128i q8s[][kSmallNbK], const float yd[][kSmallNbK], float* res) {
    switch (m) {
        case 1: q4k_dot_nt<1>(x, nb, act, q8s, yd, res); break;
        case 2: q4k_dot_nt<2>(x, nb, act, q8s, yd, res); break;
        case 3: q4k_dot_nt<3>(x, nb, act, q8s, yd, res); break;
        case 4: q4k_dot_nt<4>(x, nb, act, q8s, yd, res); break;
        case 5: q4k_dot_nt<5>(x, nb, act, q8s, yd, res); break;
        default: q4k_dot_nt<6>(x, nb, act, q8s, yd, res); break;
    }
}

// ---- Q5_K x Q8_K
template <int NT>
void q5k_dot_nt(const block_q5_K* x, int nb, const void* const* act,
                const __m128i q8s[][kSmallNbK], const float yd[][kSmallNbK], float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    alignas(32) float acc_mem[NT][8];
    __m256 acc_reg[NT <= 4 ? NT : 1];
    float summs[NT];
    if constexpr (NT <= 4) {
        for (int t = 0; t < NT; ++t) { acc_reg[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    } else {
        for (int t = 0; t < NT; ++t) { _mm256_store_ps(acc_mem[t], _mm256_setzero_ps()); summs[t] = 0.0f; }
    }
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &x[i + 2] + 64, _MM_HINT_T0);
        }
        float xd, xdmin;
        half_pair(&x[i], xd, xdmin);
        __m256i scales;
        __m128i mins;
        unpack_k4_scales_mins(x[i].scales, scales, mins);
        for (int t = 0; t < NT; ++t) {
            const float dmin = -yd[t][i] * xdmin;
            const __m128i prod = _mm_madd_epi16(mins, q8s[t][i]);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * (float) _mm_extract_epi32(hsum, 0);
        }
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        const uint8_t* q5 = x[i].qs;
        const int off = i * (int) sizeof(block_q8_K) + (int) offsetof(block_q8_K, qs);
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5);
            q5 += 32;
            const int qoff = off + 64 * j;
            {
                const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
                const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
                const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                hmask = _mm256_slli_epi16(hmask, 1);
                for (int t = 0; t < NT; ++t) {
                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) ((const uint8_t*) act[t] + qoff));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0)));
                }
            }
            {
                const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
                const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                hmask = _mm256_slli_epi16(hmask, 1);
                for (int t = 0; t < NT; ++t) {
                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) ((const uint8_t*) act[t] + qoff + 32));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1)));
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float d = yd[t][i] * xd;
            if constexpr (NT <= 4) {
                acc_reg[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc_reg[t]);
            } else {
                const __m256 prev = _mm256_load_ps(acc_mem[t]);
                _mm256_store_ps(acc_mem[t], _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), prev));
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m256 a = (NT <= 4) ? acc_reg[t] : _mm256_load_ps(acc_mem[t]);
        res[t] = hsum_float_8(a) + summs[t];
    }
}

void q5k_dot_any(const block_q5_K* x, int nb, const void* const* act, int m,
                 const __m128i q8s[][kSmallNbK], const float yd[][kSmallNbK], float* res) {
    switch (m) {
        case 1: q5k_dot_nt<1>(x, nb, act, q8s, yd, res); break;
        case 2: q5k_dot_nt<2>(x, nb, act, q8s, yd, res); break;
        case 3: q5k_dot_nt<3>(x, nb, act, q8s, yd, res); break;
        case 4: q5k_dot_nt<4>(x, nb, act, q8s, yd, res); break;
        case 5: q5k_dot_nt<5>(x, nb, act, q8s, yd, res); break;
        default: q5k_dot_nt<6>(x, nb, act, q8s, yd, res); break;
    }
}

// ---- Q5_1 x Q8_1: single-row sequential DRAM stream, hoisted yd/ys + single vcvtph2ps + hoisted masks
template <int NT>
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act,
              const float yd[][kSmallNb32], const float ys[][kSmallNb32], float* res) {
    const __m256i shuf_mask = _mm256_load_si256((const __m256i*) kShufBits32);
    const __m256i bit_mask = _mm256_load_si256((const __m256i*) kBitMask32);
    const __m256i minus1 = _mm256_set1_epi64x(-1);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i m10 = _mm256_set1_epi8(0x10);
    const __m256i ones = _mm256_set1_epi16(1);

    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        float xd, xm;
        half_pair(&x[ib], xd, xm);
        const __m256 dx = _mm256_set1_ps(xd);
        const __m256i qx = _mm256_or_si256(bytes_from_nibbles_32(x[ib].qs, low_mask),
                                           _mm256_and_si256(bytes_from_bits_32(x[ib].qh, shuf_mask, bit_mask, minus1), m10));
        for (int t = 0; t < NT; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * ys[t][ib];
            const __m256 dy = _mm256_set1_ps(yd[t][ib]);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy, ones);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0: single-row sequential DRAM stream, hoisted yd + hoisted ones
template <int NT>
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, const float yd[][kSmallNb32], float* res) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * yd[t][ib]);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy, ones);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}


inline __m256 fast_exp_neg_avx2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    __m256 z = _mm256_fnmadd_ps(fx, c1, x);
    z = _mm256_fnmadd_ps(fx, c2, z);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(z, z), z);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}

template <bool SwiGLU>
inline void kq256_k_tile(int type, const uint8_t* w, size_t row_bytes, int nb, const void* const* act_t0, int m,
                         const __m128i q8s[][kSmallNbK], const float yd[][kSmallNbK], float* const* out_t0,
                         int rb, int rb1) {
    float v[6];
    float tmp[6][16];
    for (int r = rb; r < rb1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        if (type == 12) q4k_dot_any((const block_q4_K*) row, nb, act_t0, m, q8s, yd, v);
        else q5k_dot_any((const block_q5_K*) row, nb, act_t0, m, q8s, yd, v);
        if constexpr (SwiGLU) {
            for (int t = 0; t < m; ++t) tmp[t][r - rb] = v[t];
        } else {
            for (int t = 0; t < m; ++t) out_t0[t][r] = v[t];
        }
    }
    if constexpr (SwiGLU) {
        const int n_r = rb1 - rb;
        const __m256 one = _mm256_set1_ps(1.0f);
        for (int t = 0; t < m; ++t) {
            int k = 0;
            for (; k + 8 <= n_r; k += 8) {
                const __m256 g = _mm256_loadu_ps(out_t0[t] + rb + k);
                const __m256 u = _mm256_loadu_ps(tmp[t] + k);
                const __m256 en = fast_exp_neg_avx2(_mm256_sub_ps(_mm256_setzero_ps(), g));
                const __m256 res = _mm256_mul_ps(_mm256_div_ps(g, _mm256_add_ps(one, en)), u);
                _mm256_storeu_ps(out_t0[t] + rb + k, res);
            }
            for (; k < n_r; ++k) {
                const float g = out_t0[t][rb + k];
                out_t0[t][rb + k] = (g / (1.f + std::exp(-g))) * tmp[t][k];
            }
        }
    }
}

void kq256_rows_k(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                  int r0, int r1) {
    const int nb = n / QK_K;
    if (nb <= 0 || nb > kSmallNbK) return;
    __m128i q8s[6][kSmallNbK];
    float yd[6][kSmallNbK];
    const int tile_rows = 16;
    if (tile_rows <= 0) return;

    for (int t0 = 0; t0 < nt; t0 += 6) {
        const int m = nt - t0 < 6 ? nt - t0 : 6;
        precompute_q8s(act + t0, m, nb, q8s, yd);
        for (int rb = r0; rb < r1; rb += tile_rows) {
            const int rb1 = std::min(r1, rb + tile_rows);
            kq256_k_tile<false>(type, w, row_bytes, nb, act + t0, m, q8s, yd, out + t0, rb, rb1);
        }
    }
}

}  // namespace

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        const float* x0 = x + (size_t) t0 * cols;
        const float* x1 = x0 + cols;
        const float* x2 = x1 + cols;
        const float* x3 = x2 + cols;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
                   a3 = _mm256_setzero_ps();
            for (int c = 0; c + 8 <= cols; c += 8) {
                const __m256 wf = _mm256_castsi256_ps(
                    _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
                a0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x0 + c), a0);
                if (m > 1) a1 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x1 + c), a1);
                if (m > 2) a2 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x2 + c), a2);
                if (m > 3) a3 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x3 + c), a3);
            }
            out[(size_t) t0 * rows + r] = hsum_float_8(a0);
            if (m > 1) out[(size_t) (t0 + 1) * rows + r] = hsum_float_8(a1);
            if (m > 2) out[(size_t) (t0 + 2) * rows + r] = hsum_float_8(a2);
            if (m > 3) out[(size_t) (t0 + 3) * rows + r] = hsum_float_8(a3);
        }
    }
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c)));
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h, 16)), _mm256_loadu_ps(x + c), acc);
        }
        float s = hsum_float_8(acc);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    if (nb <= 0 || nb > kSmallNbK) return;
    __m128i q8s[6][kSmallNbK];
    float yd[6][kSmallNbK];
    const int tile_rows = 16;
    if (tile_rows <= 0) return;

    for (int t0 = 0; t0 < nt; t0 += 6) {
        const int m = nt - t0 < 6 ? nt - t0 : 6;
        precompute_q8s(act + t0, m, nb, q8s, yd);
        for (int rb = r0; rb < r1; rb += tile_rows) {
            const int rb1 = std::min(r1, rb + tile_rows);
            kq256_k_tile<false>(type, blob, gu_row, nb, act + t0, m, q8s, yd, ff + t0, rb, rb1);
            kq256_k_tile<true>(type, blob + up_off, gu_row, nb, act + t0, m, q8s, yd, ff + t0, rb, rb1);
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    if (type == 12 || type == 13) {
        kq256_rows_k(type, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
    const int nb = n / 32;
    if (nb <= 0 || nb > kSmallNb32) return;
    float yd[6][kSmallNb32];
    float ys[6][kSmallNb32];
    const int tile_rows = nt <= 6 ? (r1 - r0) : 32;
    if (tile_rows <= 0) return;

    for (int t0 = 0; t0 < nt; t0 += 6) {
        const int m = nt - t0 < 6 ? nt - t0 : 6;
        if (type == 7) {
            for (int t = 0; t < m; ++t) {
                const block_q8_1* y = (const block_q8_1*) act[t0 + t];
                for (int ib = 0; ib < nb; ++ib) half_pair(&y[ib], yd[t][ib], ys[t][ib]);
            }
        } else {
            for (int t = 0; t < m; ++t) {
                const block_q8_0* y = (const block_q8_0*) act[t0 + t];
                for (int ib = 0; ib < nb; ++ib) yd[t][ib] = h2f(y[ib].d);
            }
        }
        float s[6];
        for (int rb = r0; rb < r1; rb += tile_rows) {
            const int rb1 = std::min(r1, rb + tile_rows);
            for (int r = rb; r < rb1; ++r) {
                const uint8_t* row = w + (size_t) r * row_bytes;
                if (type == 7) {
                    const auto* x = (const block_q5_1*) row;
                    switch (m) {
                        case 1: q5_1_dot<1>(x, nb, act + t0, yd, ys, s); break;
                        case 2: q5_1_dot<2>(x, nb, act + t0, yd, ys, s); break;
                        case 3: q5_1_dot<3>(x, nb, act + t0, yd, ys, s); break;
                        case 4: q5_1_dot<4>(x, nb, act + t0, yd, ys, s); break;
                        case 5: q5_1_dot<5>(x, nb, act + t0, yd, ys, s); break;
                        default: q5_1_dot<6>(x, nb, act + t0, yd, ys, s); break;
                    }
                } else {
                    const auto* x = (const block_q8_0*) row;
                    switch (m) {
                        case 1: q8_0_dot<1>(x, nb, act + t0, yd, s); break;
                        case 2: q8_0_dot<2>(x, nb, act + t0, yd, s); break;
                        case 3: q8_0_dot<3>(x, nb, act + t0, yd, s); break;
                        case 4: q8_0_dot<4>(x, nb, act + t0, yd, s); break;
                        case 5: q8_0_dot<5>(x, nb, act + t0, yd, s); break;
                        default: q8_0_dot<6>(x, nb, act + t0, yd, s); break;
                    }
                }
                for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
            }
        }
    }
}

}  // namespace strata::kernels::cpu
