// #4590: which tokens the repeat / frequency / presence penalties count.  Header-only, CPU, no model.
//
// Default: the current reply only.  The prompt (the new task's digits, a quoted earlier answer, earlier assistant turns)
// is not in the window; the last prompt token, which the first decode step feeds in, is.  STRATA_PENALTY_PROMPT=1 is the
// old window: the last penalty_last_n tokens of prompt + reply.
#include "strata/kernels/sampler.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::kernels::penalty_counts_prompt;
using strata::kernels::penalty_rows;
using strata::kernels::penalty_rows_tree;
using strata::kernels::penalty_window_from;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr int kH = 64;           // penalty_last_n 64 (the server's default window)
constexpr int32_t kTaskTok = 12; // a token of the new task ("12"), in the last 64 prompt tokens
constexpr int32_t kOldAnswer = 7777;   // a token of an earlier assistant turn, quoted near the end of the prompt

// A request's prompt: filler, an earlier assistant answer (kOldAnswer tokens), then the new task (kTaskTok), then the
// assistant header whose last token is the fed-back head x.
std::vector<int32_t> prompt() {
    std::vector<int32_t> ids;
    for (int i = 0; i < 300; ++i) ids.push_back(1000 + i);
    for (int i = 0; i < 20; ++i) ids.push_back(kOldAnswer + (i % 3));
    for (int i = 0; i < 10; ++i) ids.push_back(kTaskTok);
    for (int i = 0; i < 4; ++i) ids.push_back(50 + i);
    return ids;
}

// The row the engine stages for the pick after `reply` has written its tokens: the tokens `consumed` holds when decode
// began (the prompt without its last token) followed by the reply, read from penalty_window_from().
std::vector<int32_t> row_after(const std::vector<int32_t>& ids, const std::vector<int32_t>& reply, bool counts_prompt) {
    std::vector<int32_t> consumed(ids.begin(), ids.end() - 1);
    const size_t from = penalty_window_from(consumed.size(), counts_prompt);
    const int32_t x = ids.back();
    // window = [x, reply...]: the single row of the last one is "the pick after the last reply token"
    std::vector<int32_t> window = {x};
    window.insert(window.end(), reply.begin(), reply.end());
    const int T = (int) window.size();
    std::vector<int32_t> rows((size_t) T * kH);
    penalty_rows(consumed.data() + from, (int64_t) (consumed.size() - from), window.data(), T, kH, rows.data());
    return std::vector<int32_t>(rows.end() - kH, rows.end());   // the last row
}

int count_of(const std::vector<int32_t>& row, int32_t tok) {
    int c = 0;
    for (int32_t t : row) c += t == tok;
    return c;
}
int filled(const std::vector<int32_t>& row) {
    int c = 0;
    for (int32_t t : row) c += t >= 0;
    return c;
}
}  // namespace

int main() {
    // ---- the switch: unset, empty, 0 and anything but "1" is the default (reply only)
    check(!penalty_counts_prompt(nullptr), "env unset: reply only");
    check(!penalty_counts_prompt(""), "env empty: reply only");
    check(!penalty_counts_prompt("0"), "env 0 (the old opt-in value): reply only");
    check(!penalty_counts_prompt("2"), "env 2: reply only");
    check(penalty_counts_prompt("1"), "env 1: the old window");

    const std::vector<int32_t> ids = prompt();
    const int32_t x = ids.back();

    // ---- first pick (nothing written yet)
    {
        const std::vector<int32_t> r = row_after(ids, {}, false);
        check(filled(r) == 1 && r[kH - 1] == x, "default: the first pick counts only the fed-back head x");
        check(count_of(r, kTaskTok) == 0 && count_of(r, kOldAnswer) == 0, "default: no prompt task / earlier-answer token");
        const std::vector<int32_t> o = row_after(ids, {}, true);
        check(filled(o) == kH && o[kH - 1] == x, "opt-out: a full window of prompt tokens ends in x");
        check(count_of(o, kTaskTok) == 10, "opt-out: the new task's token is in the window (the old behaviour)");
    }

    // ---- mid reply: the reply's own tokens are counted in both modes, a loop still gets penalised
    {
        const std::vector<int32_t> reply = {900, 901, 900, 902, 900};
        const std::vector<int32_t> r = row_after(ids, reply, false);
        check(filled(r) == 1 + (int) reply.size(), "default: the window is x + the reply");
        check(count_of(r, 900) == 3 && count_of(r, 901) == 1 && count_of(r, 902) == 1, "default: reply tokens counted");
        check(count_of(r, kTaskTok) == 0 && count_of(r, kOldAnswer) == 0, "default: still no prompt token mid reply");
        const std::vector<int32_t> o = row_after(ids, reply, true);
        check(count_of(o, 900) == 3 && count_of(o, kTaskTok) > 0, "opt-out: reply and prompt tokens both counted");
    }

    // ---- a long reply: the window slides over the reply alone, the prompt never comes back
    {
        std::vector<int32_t> reply;
        for (int i = 0; i < 200; ++i) reply.push_back(2000 + (i % 7));
        const std::vector<int32_t> r = row_after(ids, reply, false);
        check(filled(r) == kH, "default: a reply longer than the window fills it");
        bool prompt_token = false;
        for (int32_t t : r) prompt_token |= (t >= 0 && t < 2000);
        check(!prompt_token, "default: after 63 reply tokens no prompt token (not even x) is in the window");
        const std::vector<int32_t> o = row_after(ids, reply, true);
        check(filled(o) == kH && count_of(o, kTaskTok) == 0, "opt-out: the same long reply has pushed the prompt out too");
    }

    // ---- a restored conversation: the session comes back from the conversation cache or a saved slot with its state, but
    // the history is built from the request's ids and the tokens written since, so it is the same as a cold read of the
    // same ids.  A second turn that quotes the first answer in the prompt does not carry the first reply's window over.
    {
        const std::vector<int32_t> cold = row_after(ids, {900, 901}, false);
        const std::vector<int32_t> restored = row_after(ids, {900, 901}, false);
        check(cold == restored, "restored session: same window as a cold read of the same ids");
        std::vector<int32_t> next = ids;   // turn 2: the prompt is turn 1's prompt + its reply + a new task
        next.insert(next.end(), {900, 901, 902, 99});
        const std::vector<int32_t> r2 = row_after(next, {}, false);
        check(filled(r2) == 1 && r2[kH - 1] == 99, "turn 2 default: turn 1's reply tokens in the prompt are not counted");
    }

    // ---- 2-branch speculative tree window (`penalty_rows_tree`):
    // Branch A [0..fork) follows window[0..t]; Branch B [fork..T) follows window[0..parent] + window[fork..t].
    {
        const std::vector<int32_t> tail = {500, 501, 502};
        // T=8, fork=5, parent=1:
        // Branch A: [10, 11, 12, 13, 14]
        // Branch B: starts after window[1]=11 -> [22, 23, 24]
        const std::vector<int32_t> win = {10, 11, 12, 13, 14, 22, 23, 24};
        constexpr int T = 8, fork = 5, parent = 1, H = 6;
        std::vector<int32_t> tree_rows((size_t) T * H, -99);
        penalty_rows_tree(tail.data(), (int64_t) tail.size(), win.data(), T, fork, parent, H, tree_rows.data());

        // Compare Branch A rows 0..4 against linear penalty_rows on [10, 11, 12, 13, 14]
        std::vector<int32_t> lin_a((size_t) fork * H, -99);
        penalty_rows(tail.data(), (int64_t) tail.size(), win.data(), fork, H, lin_a.data());
        for (int t = 0; t < fork; ++t) {
            std::vector<int32_t> got(tree_rows.begin() + t * H, tree_rows.begin() + (t + 1) * H);
            std::vector<int32_t> exp(lin_a.begin() + t * H, lin_a.begin() + (t + 1) * H);
            check(got == exp, "tree penalty rows: Branch A matches linear penalty_rows");
        }

        // Compare Branch B rows 5..7 against linear penalty_rows on [10, 11, 22, 23, 24]
        const std::vector<int32_t> win_b = {10, 11, 22, 23, 24};
        std::vector<int32_t> lin_b(win_b.size() * H, -99);
        penalty_rows(tail.data(), (int64_t) tail.size(), win_b.data(), (int) win_b.size(), H, lin_b.data());
        for (int t = fork; t < T; ++t) {
            const int b_idx = (parent + 1) + (t - fork);
            std::vector<int32_t> got(tree_rows.begin() + t * H, tree_rows.begin() + (t + 1) * H);
            std::vector<int32_t> exp(lin_b.begin() + b_idx * H, lin_b.begin() + (b_idx + 1) * H);
            check(got == exp, "tree penalty rows: Branch B matches linear path from parent without Branch A leakage");
            check(count_of(got, 12) == 0 && count_of(got, 13) == 0 && count_of(got, 14) == 0,
                  "tree penalty rows: Branch B excludes unshared Branch A siblings");
        }

        // Degenerate fork=0 falls back cleanly to linear penalty_rows
        std::vector<int32_t> deg((size_t) T * H, -99), lin_all((size_t) T * H, -99);
        penalty_rows_tree(tail.data(), (int64_t) tail.size(), win.data(), T, 0, -1, H, deg.data());
        penalty_rows(tail.data(), (int64_t) tail.size(), win.data(), T, H, lin_all.data());
        check(deg == lin_all, "tree penalty rows: fork=0 falls back to linear penalty_rows");
    }

    std::printf("penalty_window_test: %d checks passed\n", checks);
    return 0;
}
