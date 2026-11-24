// include/strata/spec/rolling_suffix.hpp - Persistent cross-request rolling suffix cache (zero GPU cost, lossless).
//
// Unlike the per-request `SuffixDrafter` (which resets on every new request), `RollingSuffixSource` persists committed
// tokens (prompts and accepted outputs) across requests in a bounded ring-compacted trigram index (~10 MiB RAM at
// 256K tokens). In coding / agent / multi-turn workflows it lets both primary lookup and `--lookup-chain` reuse
// identifiers, boilerplate, JSON schemas, and recent file edits even on the first turn of a new request.
//
// Verification still accepts only exact target-model matches, so output tokens remain 100% strictly lossless.
#pragma once

#include "strata/spec/draft_source.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::spec {

class RollingSuffixSource final : public DraftSource {
public:
    explicit RollingSuffixSource(int min_match = 4, int max_match = 64, size_t capacity_tokens = 1u << 18)
        : min_match_(std::max(3, min_match)),
          cap_(std::max<size_t>(capacity_tokens, 8192)),
          d_(std::max(3, min_match), max_match, std::max<size_t>(capacity_tokens, 8192)) {}

    const char* name() const override { return "rolling-suffix"; }

    /// Persists across requests (intentional no-op on request reset).
    void reset() override {}

    /// Intentionally ignores per-token streaming inside extra_sources_append; committed turns are bulk-committed via
    /// `commit_turn()` at request end so current-request `SuffixDrafter` handles in-request recency with zero duplication.
    void append(const int32_t* tokens, size_t n) override { (void) tokens; (void) n; }

    /// Bulk-append committed tokens from a finished request (prompt + accepted outputs), compacting to the most recent
    /// half whenever the index approaches capacity so load factor stays <= 0.5 indefinitely.
    void commit_turn(const int32_t* tokens, size_t n) {
        if (tokens == nullptr || n == 0) return;
        const size_t headroom = std::min<size_t>(4096, cap_ / 8);
        if (n >= cap_ - headroom) {
            d_.reset();
            const size_t keep = cap_ / 2;
            d_.append(tokens + (n - keep), keep);
            return;
        }
        if (d_.size() + n >= cap_ - headroom) {
            const size_t keep = std::min(d_.size(), cap_ / 2);
            d_.compact_keep_recent(keep);
        }
        d_.append(tokens, n);
    }

    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        (void) n_pending;
        const int k = d_.propose_query(tail, n_tail, max_k, out);
        *match = k > 0 ? d_.last_match() : 0;
        return k;
    }

    /// Direct query for primary suffix lookup fallback (when current-request suffix lookup misses or has a shorter match).
    int propose_direct(const int32_t* tail, int n_tail, int max_k, int32_t* out, int* match) {
        const int k = d_.propose_query(tail, n_tail, max_k, out);
        const int m = k > 0 ? d_.last_match() : 0;
        if (m < min_match_) {
            if (match) *match = 0;
            return 0;
        }
        if (match) *match = m;
        return k;
    }

    size_t size() const { return d_.size(); }
    size_t capacity() const { return cap_; }
    int min_match() const { return min_match_; }

private:
    int min_match_;
    size_t cap_;
    SuffixDrafter d_;
};

}  // namespace strata::spec
