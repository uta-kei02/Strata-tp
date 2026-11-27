// M3-2's decode ruler: FNX_DEC_STATS=1 says, every so many windows, what a decode round really spends its time on.
//
// Off (the default, and any value but exactly "1") every call below returns at once: no line is printed, nothing is
// stored, and the only cost is a branch. That matters because these calls sit in the per-layer host loop of both
// cards (engine.cu) and in the expert tier's round (l2_dec.cu), which run tens of thousands of times a second; a
// ruler that clocked or printed while off would move the very number it exists to measure.
//
// Why the host manager is the one that counts the picks: the router sorts every window's picks three ways
// (moe.cuh:274-322) and bumps two counters, each only by the side that has picks - req_seq for the host worker's,
// l2_req_seq for the tier's (moe.cuh:345-346). The manager is the one thread that reads both (engine.cu:541-552),
// so it is the one place that can see a window the way the router saw it. Nothing is added to the box, so CUDA
// Graph capture and the two answer counters stay exactly as they are.
//
// Two honest limits, both a result of reading rather than adding:
//   - The cards' own pick count is an upper bound. It never reaches the box (only n_miss and n_l2 do), so it is
//     printed as the window's T * TOPK minus the tier's and the host worker's picks. The rows of one expert
//     merge, so the real share is at most that number and never less.
//   - A window nobody waits for can be rewritten before this thread reads it: a card spins on an answer counter
//     only for the side that fired, so the host manager may coalesce two such windows into one. The window count
//     is therefore a floor, not a census. The time is not affected - it is measured on the windows that were seen.
//
// One report is three lines, every FNX_DEC_STATS_EVERY windows (2048 by default), printed by card 0's manager only
// and reset as it is read, so each line describes the stretch since the last one:
//
//   fnx: decstats: <windows> over <s>: who fired, and how many experts, rows and picks each side took
//   fnx: decstats: how long the host worker took, split into the steps that can be slow
//   fnx: decstats: how long the expert tier took, per card, with its hits and rows
//
// Three lines rather than the two a first cut printed, because the host worker and the expert tier answer different
// window counts and one shared line would hide that. The card split is printed on every line: an imbalance between
// card 0 and card 1 is one of the things M3-2 is looking for, and it is invisible in a total.
#ifndef SPLIT_DEC_STATS_HPP
#define SPLIT_DEC_STATS_HPP

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include "kernels.hpp"

namespace fnx {

namespace dec_stats_detail {

// One shared set of counters: both cards' host managers and the expert tier's worker add into it, and one thread
// reads it once per sample. Relaxed everywhere - these are a ruler, not a protocol; no other thread waits on them
// becoming visible in any particular order, and nothing here is read back by the engine.
struct Counters {
    std::atomic<long long> win{0};             // windows the managers picked up, both cards together
    std::atomic<long long> win_tier{0};        // ... of which the router sent picks to the expert tier
    std::atomic<long long> win_host{0};        // ... of which it sent picks to the host worker (an L3 miss)
    std::atomic<long long> win_both{0};        // ... of which it sent picks to both at once
    std::atomic<long long> win_card[NCARD];    // per card: a load skew between the two shows up here
    std::atomic<long long> picks_tier{0};      // distinct experts handed to each side
    std::atomic<long long> picks_host{0};
    std::atomic<long long> rows_tier{0};       // activation rows each side has to run for them
    std::atomic<long long> rows_host{0};
    std::atomic<long long> rows_all{0};        // T * TOPK of every window; the cards' share is this minus the two
    std::atomic<long long> ns_host{0};         // a window the host worker answered: first step to done_seq
    std::atomic<long long> ns_quant{0};        // ... and its four steps apart, so the slow one is named
    std::atomic<long long> ns_gu{0};           //     phase 1 (gate|up) and phase 2 (down) include their barrier
    std::atomic<long long> ns_hq{0};
    std::atomic<long long> ns_dn{0};
    std::atomic<long long> ns_host_card[NCARD];
    std::atomic<long long> ns_tier{0};         // a round the expert tier answered: lists taken to counter posted
    std::atomic<long long> ns_tier_card[NCARD];
    std::atomic<long long> hits_tier[NCARD];   // what the tier really ran (a list that broke runs nothing)
    std::atomic<long long> rows_tier_card[NCARD];
    std::atomic<long long> origin{0};          // the first window of the sample being built
};

inline Counters& ctr() {
    static Counters c;
    return c;
}

inline long long clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void add(std::atomic<long long>& a, long long v) { a.fetch_add(v, std::memory_order_relaxed); }

// read and clear in one step, so a report leaves a clean sample behind without a lock
inline long long take(std::atomic<long long>& a) { return a.exchange(0, std::memory_order_relaxed); }

inline double to_ms(long long ns) { return (double) ns / 1e6; }
inline double pct(double part, double whole) { return whole > 0.0 ? 100.0 * part / whole : 0.0; }

}  // namespace dec_stats_detail

// read once: the operator sets it in the environment, never from the code
inline bool dec_stats_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_DEC_STATS");
        return e && std::strcmp(e, "1") == 0;
    }();
    return on;
}

// how many windows a sample covers; FNX_DEC_STATS_EVERY=1 reports every window, which is what a short repro wants
inline long long dec_stats_every() {
    static const long long n = [] {
        const char* e = std::getenv("FNX_DEC_STATS_EVERY");
        if (!e || !*e) return 2048LL;
        char* end = nullptr;
        const long long v = std::strtoll(e, &end, 10);
        return (end != e && v > 0) ? v : 2048LL;
    }();
    return n;
}

// One window, exactly as the router left it in the mapped box. Each card's host manager calls it once per window it
// picks up, before deciding whether that window is one it has to answer itself - so a window only the expert tier
// has picks for is counted as well, which is the point: it is a layer whose L3 wait this round did not have.
inline void dec_stats_window(int card, const CpuMoeBox* box) {
    if (!dec_stats_on()) return;
    using namespace dec_stats_detail;
    Counters& c = ctr();
    const int T = std::max(1, std::min(MAXT, (int) box->T));
    const int n_host = std::min(std::max((int) box->n_miss, 0), MAXT * TOPK);
    const int n_tier = std::min(std::max((int) box->n_l2, 0), MAXT * TOPK);
    long long r_tier = 0, r_host = 0;
    for (int m = 0; m < n_tier; ++m) r_tier += std::max((int) box->l2_n[m], 0);
    for (int m = 0; m < n_host; ++m) r_host += std::max((int) box->miss_n[m], 0);
    // the first window of a sample starts its clock; the report takes the origin, so the next one starts fresh
    if (c.win.fetch_add(1, std::memory_order_relaxed) == 0) c.origin.store(clock_ns(), std::memory_order_relaxed);
    if (n_tier != 0) add(c.win_tier, 1);
    if (n_host != 0) add(c.win_host, 1);
    if (n_tier != 0 && n_host != 0) add(c.win_both, 1);
    add(c.picks_tier, n_tier);
    add(c.picks_host, n_host);
    add(c.rows_tier, r_tier);
    add(c.rows_host, r_host);
    add(c.rows_all, (long long) T * TOPK);
    if (card >= 0 && card < NCARD) add(c.win_card[card], 1);
}

// One window the host worker answered, its four steps timed apart: the activation rows it quantizes for the experts,
// phase 1 (gate|up across the pool), the intermediate quantize, phase 2 (down) - and the window whole, which is a
// little more than the four because of the pool's bookkeeping. The caller passes spans, never the clock.
inline void dec_stats_host(int card, long long ns_quant, long long ns_gu, long long ns_hq, long long ns_dn, long long ns_all) {
    if (!dec_stats_on()) return;
    using namespace dec_stats_detail;
    Counters& c = ctr();
    add(c.ns_host, ns_all);
    add(c.ns_quant, ns_quant);
    add(c.ns_gu, ns_gu);
    add(c.ns_hq, ns_hq);
    add(c.ns_dn, ns_dn);
    if (card >= 0 && card < NCARD) add(c.ns_host_card[card], ns_all);
}

// One round the expert tier answered: took_ns from taking the lists to posting the counter, and what it ran. A round
// whose list broke runs nothing and answers with zeros, so hits stays behind the windows the manager saw.
inline void dec_stats_tier(int card, long long took_ns, long long jobs, long long rows) {
    if (!dec_stats_on()) return;
    using namespace dec_stats_detail;
    Counters& c = ctr();
    add(c.ns_tier, took_ns);
    if (card < 0 || card >= NCARD) return;
    add(c.ns_tier_card[card], took_ns);
    add(c.hits_tier[card], jobs);
    add(c.rows_tier_card[card], rows);
}

// The report, at most once every FNX_DEC_STATS_EVERY windows. Only card 0's host manager calls it: two threads
// printing the same totals would print them twice, and naming both cards in one line says the same thing anyway.
// Everything is read-and-cleared, so each line describes the stretch since the last one.
inline void dec_stats_report_maybe() {
    using namespace dec_stats_detail;
    if (!dec_stats_on()) return;
    Counters& c = ctr();
    if (c.win.load(std::memory_order_relaxed) < dec_stats_every()) return;
    const long long w = take(c.win), w_tier = take(c.win_tier), w_host = take(c.win_host), w_both = take(c.win_both);
    long long wc[NCARD], host_c[NCARD], tier_c[NCARD], hits[NCARD], rows_t[NCARD];
    for (int t = 0; t < NCARD; ++t) {
        wc[t] = take(c.win_card[t]);
        host_c[t] = take(c.ns_host_card[t]);
        tier_c[t] = take(c.ns_tier_card[t]);
        hits[t] = take(c.hits_tier[t]);
        rows_t[t] = take(c.rows_tier_card[t]);
    }
    const long long p_tier = take(c.picks_tier), p_host = take(c.picks_host);
    const long long r_tier = take(c.rows_tier), r_host = take(c.rows_host), r_all = take(c.rows_all);
    const long long ns_h = take(c.ns_host), n_q = take(c.ns_quant), n_gu = take(c.ns_gu), n_hq = take(c.ns_hq), n_dn = take(c.ns_dn);
    const long long ns_t = take(c.ns_tier), origin = take(c.origin);
    const double dur = (double) (clock_ns() - origin) / 1e9;
    const double span = dur > 0.0 ? dur * 1e3 : 1e-9;
    const long long cards_rows = r_all > r_tier + r_host ? r_all - r_tier - r_host : 0;
    std::fprintf(stderr,
                 "fnx: decstats: %lld windows over %.3f s (card 0 %lld, card 1 %lld): tier fired %lld (%.1f%%), host worker %lld (%.1f%%), "
                 "both %lld; picks tier %lld experts / %lld rows, host %lld / %lld, cards at most %lld rows%c",
                 w, dur, wc[0], wc[1], w_tier, pct((double) w_tier, (double) w), w_host, pct((double) w_host, (double) w), w_both, p_tier,
                 r_tier, p_host, r_host, cards_rows, 10);
    std::fprintf(stderr,
                 "fnx: decstats: host worker %.1f ms of the %.1f ms sample (%.1f%%, both cards): rows %.1f, gate|up %.1f, h quant %.1f, "
                 "down %.1f ms; card 0 %.1f ms over %lld windows, card 1 %.1f ms over %lld%c",
                 to_ms(ns_h), span, pct(to_ms(ns_h), span), to_ms(n_q), to_ms(n_gu), to_ms(n_hq), to_ms(n_dn), to_ms(host_c[0]), wc[0],
                 to_ms(host_c[1]), wc[1], 10);
    std::fprintf(stderr, "fnx: decstats: expert tier %.1f ms: card 0 %lld hits / %lld rows in %.1f ms, card 1 %lld / %lld in %.1f ms%c",
                 to_ms(ns_t), hits[0], rows_t[0], to_ms(tier_c[0]), hits[1], rows_t[1], to_ms(tier_c[1]), 10);
}

}  // namespace fnx

#endif  // SPLIT_DEC_STATS_HPP
