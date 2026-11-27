// The tier card as the decode path's middle tier: see l2_dec.hpp.
#include "l2_dec.hpp"

#include "env.hpp"
#include "kernels.hpp"
#include "l2_offload.hpp"
#include "model.hpp"
#include "dec_stats.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fnx {
namespace l2 {
namespace dec {
namespace {

constexpr int ROWS = MAXT * TOPK;                        // the pairs one window can route: the box's lists are this wide
constexpr size_t XQ_BYTES = (size_t) MAXT * q8_1_bytes(N);   // the window's activation rows, q8_1, as the card holds them

// What the worker uploads, in one copy: the job count and both job tables. Job is ints only, so the device layout is
// this layout; &cnt doubles as the njobs the two kernels read, which is why the tables and the count travel together.
struct Blob {
    int cnt;
    Job gu[ROWS];
    Job dn[ROWS];
};

struct Card {
    CpuMoeBox* box = nullptr;         // the card's mapped page-locked box (host side); set last, it starts the work
    int dev = -1;
    char* map = nullptr;              // [NL * NEXP] on the card: the experts the arena holds
    cudaStream_t clr = nullptr;       // a non-blocking stream on this card, only to zero its map after a failure
    const char* const* ep_gu = nullptr;   // [NL * NEXP] on the tier: where this card's gate|up of an expert is
    const char* const* ep_dn = nullptr;   // and its down
    DW gu_e[NL];                      // the card's own expert matrices, copied so a launch needs no Model
    DW dn_e[NL];
    size_t gu_stride[NL] = {}, dn_stride[NL] = {};
    unsigned long long last = 0;      // the last req_seq this card's round was answered
    long hits = 0, rows = 0, reqs = 0, shown = 0;   // shown: how many of this card's rounds the trace has printed
    double ms = 0;
};

struct S {
    bool on = false;
    bool dead = false;                // a tier failure: every later pick goes back to the cards, this thread zeroes the rest
    bool started = false;
    std::string why;
    int dev = -1;
    int ffs = 0;
    int cap = ROWS;
    bool stats = false;
    bool trace = false;               // FNX_L2_DEC_TRACE: the rounds around a failure, so the log says which layer it died in
    int stage = 0;                    // FNX_L2_DEC_STAGE: which of those rounds' steps to print (1 lists, 2 upload, 4 kernels, 8 answer copy, 16 counter)
    cudaStream_t st = nullptr;        // the tier's decode stream: one, so one MoE runs at a time on that card
    Blob* hblob = nullptr;            // page-locked mirror of what goes up
    Blob* dblob = nullptr;
    void* xq = nullptr;               // tier: the window's activation rows, as the card wrote them
    float* gu = nullptr;              // tier: [ROWS][2 * ffs]
    float* part = nullptr;            // tier: [ROWS][N]
    Card card[NCARD];
    tpopt::WaitSlot wake;
};

S& s() {
    static S v;
    return v;
}

// CUDA is strict about which device a handle belongs to, and this thread's device is the tier from start to finish
// except while a card's map is being cleared: that guard is what puts it back, on every way out.
struct DevGuard {
    int prev = 0;
    bool ok = false;

    explicit DevGuard(int dev) { ok = cudaGetDevice(&prev) == cudaSuccess && cudaSetDevice(dev) == cudaSuccess; }
    ~DevGuard() {
        if (ok) cudaSetDevice(prev);
    }
    DevGuard(const DevGuard&) = delete;
    DevGuard& operator=(const DevGuard&) = delete;
};

// the answer: the same fence, counter pair the host worker uses (engine.cu), on the box's own second counter so a
// round the tier answered never waits on the host and the other way round
void answer(Card& cd, unsigned long long cur) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *(volatile unsigned long long*) &cd.box->l2_done_seq = cur;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// A round whose rows the tier did not compute still has its picks out of the card's job list, so its rows have to be
// something: zero, which the ring sums as nothing. The whole array, so a pick this round did not count stays covered.
void zero_rows(Card& cd) { std::memset((void*) cd.box->l2_y_out, 0, sizeof(cd.box->l2_y_out)); }

// stop using the tier: zero every card's map so later layers stop sending picks, without waiting on the cards (the
// round that just failed has a kernel spinning on this answer, so a device sync here could never come back). The map
// is only a placement choice, so whenever the zero lands the sum stays whole.
void retire(const char* what) {
    S& g = s();
    if (g.dead) return;
    g.dead = true;
    std::fprintf(stderr, "fnx: l2dec: off: %s (later picks go back to the cards and the host worker)%c", what, 10);
    for (int t = 0; t < NCARD; ++t) {
        if (!g.card[t].map) continue;
        DevGuard on_card(g.card[t].dev);
        if (on_card.ok && g.card[t].clr) cudaMemsetAsync(g.card[t].map, 0, (size_t) NL * NEXP, g.card[t].clr);
    }
}

// one card's round. Everything here belongs to the tier: the worker's device is the tier's, so the uploads, both
// kernels and the sync are one context, and the only host-side write that matters is the counter at the end.
void run(int t, unsigned long long cur) {
    S& g = s();
    Card& cd = g.card[t];
    CpuMoeBox* bx = cd.box;
    const long long t0 = tpopt::now_ns();
    // Take the lists before anything else. The card runs its next layer while this round's kernels are still going,
    // and that layer's router rewrites the same lists (the host worker has exactly the same window); copying first
    // keeps the exposure to the microseconds between the counter and the copy instead of the milliseconds below.
    const int layer = bx->layer;
    const int T = std::max(1, std::min(MAXT, bx->T));
    const int n_l2 = std::min(std::max(bx->n_l2, 0), g.cap);
    bool bad = layer < 0 || layer >= NL;
    int njobs = 0, used = 0;
    // which number broke, so the log can say it: the card's merge reads the very same list, and one out-of-range
    // entry there is a write outside the rows of part it owns (merge_rows, kernels.cu)
    int badm = -1, badw = -1, badn = 0, bady = -1, bado = -1;
    const l2::Ctx& lc = l2::ctx();
    for (int m = 0; !bad && m < n_l2; ++m) {
        const int n = bx->l2_n[m], w = bx->l2_w[m];
        const size_t slot = (size_t) layer * NEXP + (size_t) w;
        if (n <= 0 || n > MAXT || used + n > ROWS || w < 0 || w >= NEXP || slot >= lc.at.size() || lc.at[slot] < 0) {
            bad = true; badm = m; badw = w; badn = n;   // the card wrote the lists; a shape or an expert out of range
            break;                                       // is a bug, not a thing to guess at: answer the round with zeros
        }
        Job& gu = g.hblob->gu[njobs];
        Job& dn = g.hblob->dn[njobs];
        gu.w = dn.w = w;
        gu.n = dn.n = n;
        for (int q = 0; q < n; ++q) {
            const int y = bx->l2_y[m][q], o = bx->l2_out[m][q];
            if (y < 0 || y >= T || o < 0 || o >= ROWS) { bad = true; badm = m; badw = w; badn = n; bady = y; bado = o; break; }
            gu.y[q] = y;                                  // the window row this pick reads, in the copy above
            gu.out[q] = dn.y[q] = dn.out[q] = used + q;   // compact: gate|up row, down's rows, and the D2H's order
        }
        if (bad) break;
        used += n;
        ++njobs;
    }
    // FNX_L2_DEC_TRACE=1 prints each card's first 64 rounds and every round that broke: a decode is thousands of
    // rounds a second, so "the first 64 plus every bad one" is what fits in a log, and it is enough to say which
    // layer a window died in. FNX_L2_DEC_STAGE adds one line per step below it. Both default off and print nothing.
    const bool dbg = g.trace && (cd.shown < 64 || bad);
    if (dbg) ++cd.shown;
    const auto stage = [&](int bit, const char* what) {
        if (dbg && (g.stage & bit) != 0)
            std::fprintf(stderr, "fnx: l2dec: card %d layer %d: %s%c", cd.dev, layer, what, 10);
    };
    if (dbg)
        std::fprintf(stderr, "fnx: l2dec: card %d layer %d T %d: %d picks for the tier, %d jobs, %d rows%s%c", cd.dev, layer, T, n_l2,
                     njobs, used, bad ? " (the list broke, answered with zeros)" : "", 10);
    if (bad || njobs == 0) {
        // Nothing of ours this round, or a round we cannot answer: either way no launch, no event and no stream sync,
        // which is what a captured graph needs. Only a round the card really dropped picks needs the zero fill; a
        // round with no picks leaves the rows to whoever has them.
        if (bad || n_l2 > 0) zero_rows(cd);
        ++cd.reqs;
        // FNX_DEC_STATS (dec_stats.hpp): a round that ran nothing still cost the walk over its lists, so it is timed
        // with zero jobs and zero rows. Off, this is a branch and nothing else.
        if (fnx::dec_stats_on()) fnx::dec_stats_tier(t, tpopt::now_ns() - t0, 0, 0);
        answer(cd, cur);
        if (bad) {
            char why[192];
            std::snprintf(why, sizeof why,
                          "a decode pick list was out of range (card %d layer %d T %d n_l2 %d: m %d w %d n %d y %d out %d)",
                          cd.dev, layer, T, bx->n_l2, badm, badw, badn, bady, bado);
            retire(why);
        }
        return;
    }
    g.hblob->cnt = njobs;
    const int Pwin = T * TOPK;   // the rows the kernels index their outputs by: the same convention the cards use
    bool err = false;
    stage(2, "uploading the jobs and this card's activation rows");
    try {
        if (cudaMemcpyAsync(g.dblob, g.hblob, sizeof(Blob), cudaMemcpyHostToDevice, g.st) != cudaSuccess) err = true;
        if (!err && cudaMemcpyAsync(g.xq, bx->x_q8, (size_t) T * q8_1_bytes(N), cudaMemcpyHostToDevice, g.st) != cudaSuccess)
            err = true;
        if (!err) {
            // the tier's own table for this layer: where this card's slice of each expert sits in the arena, at the
            // offset the arena was filled at (l2_offload.cu) and the same one the prompt pass reads (l2_pf.cu). The
            // card's DW says the shape, so the rows produced are this card's slice of the expert and nothing else.
            DW gu_e = cd.gu_e[layer], dn_e = cd.dn_e[layer];
            gu_e.ep = cd.ep_gu + (size_t) layer * NEXP;
            dn_e.ep = cd.ep_dn + (size_t) layer * NEXP;
            const int* njobs_dev = &g.dblob->cnt;
            stage(4, "gate|up and down, on the tier's own stream");
            moe_gate_up(gu_e, cd.gu_stride[layer], g.dblob->gu, njobs_dev, Pwin, (const void*) g.xq, g.gu, g.st);
            moe_down(dn_e, cd.dn_stride[layer], g.dblob->dn, njobs_dev, Pwin, g.gu, g.part, g.st);
            if (cudaMemcpyAsync(bx->l2_y_out, g.part, (size_t) used * N * sizeof(float), cudaMemcpyDeviceToHost, g.st) != cudaSuccess)
                err = true;
            stage(8, "waiting for the answer copy to land in the box");
            {
                tpopt::WaitGuard guard("the expert tier's decode answer");
                if (cudaStreamSynchronize(g.st) != cudaSuccess) err = true;
            }
        }
    } catch (const std::exception& e) {
        retire(e.what());
        err = true;
    }
    // the error by name, not just "an error": a sticky illegal access and a launch that did not fit are different
    // faults with different fixes, and which one it was is the whole answer to the next question
    if (const cudaError_t kerr = cudaGetLastError(); kerr != cudaSuccess) {
        char why[160];
        std::snprintf(why, sizeof why, "a tier kernel reported an error: %s", cudaGetErrorName(kerr));
        retire(why);
        err = true;
    }
    if (err) zero_rows(cd);
    else {
        cd.hits += njobs;
        cd.rows += used;
    }
    ++cd.reqs;
    if (fnx::dec_stats_on()) {
        const long long took = tpopt::now_ns() - t0;
        cd.ms += (double) took / 1e6;
        // what the tier actually ran: a round that reported an error answered with zeros, so it owns no rows here
        fnx::dec_stats_tier(t, took, err ? 0 : njobs, err ? 0 : used);
    } else {
        cd.ms += (double) (tpopt::now_ns() - t0) / 1e6;
    }
    stage(16, "the counter, so the card's merge may read the rows");
    answer(cd, cur);
    if (g.stats && cd.reqs % 512 == 0)
        std::fprintf(stderr, "fnx: l2dec card %d: %ld requests, %ld hits, %ld rows, %.3f ms a request on the tier%c", cd.dev, cd.reqs,
                     cd.hits, cd.rows, cd.ms / (double) cd.reqs, 10);
}

// The tier's worker: one thread, one stream, the cards taken in turn. One MoE at a time per device is not a choice
// here but a fact of the kernels (moe.cuh:385's shared h buffer is per device), and the cards' answers are tiny; a
// second stream would buy nothing and could race that buffer.
void worker() {
    S& g = s();
    if (cudaSetDevice(g.dev) != cudaSuccess) {
        retire("the tier card could not be made current for its worker");
        return;
    }
    tpopt::HotWait hw;
    while (true) {
        bool any = false;
        for (int t = 0; t < NCARD; ++t) {
            if (!g.card[t].box) continue;   // attached later, while its card was still being built
            const unsigned long long cur = *(volatile const unsigned long long*) &g.card[t].box->l2_req_seq;
            if (cur == g.card[t].last) continue;
            g.card[t].last = cur;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (g.dead) {
                zero_rows(g.card[t]);
                answer(g.card[t], cur);
            } else {
                run(t, cur);
            }
            any = true;
        }
        if (any) {
            hw.reset();
            continue;
        }
        tpopt::wait_for_work(g.wake, hw, [&g] {
            for (int t = 0; t < NCARD; ++t) {
                if (!g.card[t].box) continue;
                if (*(volatile const unsigned long long*) &g.card[t].box->l2_req_seq != g.card[t].last) return true;
            }
            return false;
        });
    }
}

}  // namespace

bool open(int card, Model& m, CpuMoeBox* hbox) {
    S& g = s();
    const l2::Ctx& lc = l2::ctx();
    if (!lc.ok) return false;   // no arena at all: silent, and the decode path is exactly what it was before M2b-2
    if (card < 0 || card >= NCARD || !hbox) return false;
    if (g.card[card].box) return g.on;
    if (!g.on) {
        const char* why = nullptr;
        if (env_int("FNX_L2_DEC", 1) == 0) why = "FNX_L2_DEC=0";
        else if (NCARD < 2) why = "a decode round needs two cards to hand a pick to";
        else if (lc.arena == nullptr || lc.at.size() < (size_t) NL * NEXP) why = "the arena holds no experts";
        else if (env_flag("FNX_PF_MOE_Q8"))
            why = "FNX_PF_MOE_Q8: the cards' experts run q8_1, and the arena still holds what the model loaded";
        for (int l = 0; !why && l < NL; ++l)
            if (m.cards[0].L[(size_t) l].gu_e.s.q || m.cards[0].L[(size_t) l].down_e.s.q)
                why = "its experts carry split Q8_0 scales, which the arena copy does not carry";
        int ffs = 0;
        for (int t = 0; t < NCARD && !why; ++t) {
            const int slice = (int) (m.cards[(size_t) t].ff1 - m.cards[(size_t) t].ff0);
            if (slice <= 0) why = "a card has no routed-expert slice (ff1 - ff0)";
            ffs = std::max(ffs, slice);
        }
        const int cap_want = (int) env_int("FNX_L2_DEC_MAX", 0);
        if (!why) g.cap = (cap_want > 0 && cap_want < ROWS) ? cap_want : ROWS;
        if (why) {
            if (card == 0) std::fprintf(stderr, "fnx: l2dec: off: %s%c", why, 10);
            return false;
        }
        g.dev = lc.dev;
        g.ffs = ffs;
        g.stats = env_flag("FNX_L2_DEC_STATS");
        g.trace = env_flag("FNX_L2_DEC_TRACE");
        g.stage = (int) env_int("FNX_L2_DEC_STAGE", 0);
        DevGuard on_tier(g.dev);   // everything below is the tier's memory, until a card's map says otherwise
        bool bad = !on_tier.ok;
        if (!bad && cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking) != cudaSuccess) bad = true;
        if (!bad && cudaMalloc((void**) &g.dblob, sizeof(Blob)) != cudaSuccess) bad = true;
        if (!bad && cudaMalloc(&g.xq, XQ_BYTES) != cudaSuccess) bad = true;
        if (!bad && cudaMalloc(&g.gu, (size_t) ROWS * 2 * g.ffs * sizeof(float)) != cudaSuccess) bad = true;
        if (!bad && cudaMalloc(&g.part, (size_t) ROWS * N * sizeof(float)) != cudaSuccess) bad = true;
        if (!bad && cudaHostAlloc((void**) &g.hblob, sizeof(Blob), cudaHostAllocDefault) != cudaSuccess) bad = true;
        if (bad) {
            std::fprintf(stderr, "fnx: l2dec: off: no room on the tier card for its decode tables%c", 10);
            return false;
        }
        // one expert table pair per card, on the tier: the arena's layout is [card 0 gate|up | card 0 down | card 1
        // gate|up | ...], so a card's offset is the bytes of the cards before it, and its down starts at the pool's
        // dn_off. The same lines the prompt pass uses (l2_pf.cu:895-916): fill both tables per layer without touching
        // the rows another layer already filled, and upload each table once. One table per card because a card's
        // slice is another card's offset.
        // Filling one shared array and shifting it in place (the first version) shifted the rows of every layer it
        // had already filled: layer 0 ended up 47 gate|up strides and 48 down strides past its own experts, so the
        // tier read the wrong weights, and the NaN that came back was what the two Turing cards faulted on (Xid 31,
        // a write to an unmapped address) one layer later. The bounds check below is why that cannot happen quietly
        // again: every entry of either table has to land inside the arena it came from.
        std::vector<const char*> ht_gu((size_t) NL * NEXP), ht_dn((size_t) NL * NEXP);
        for (int t = 0; t < NCARD && !bad; ++t) {
            for (int q = 0; q < 2 && !bad; ++q) {
                const size_t tb = (size_t) NL * NEXP * sizeof(char*);
                if (cudaMalloc((void**) (q ? &g.card[t].ep_dn : &g.card[t].ep_gu), tb) != cudaSuccess) bad = true;
            }
            if (bad) break;
            for (size_t i = 0; i < (size_t) NL * NEXP; ++i) { ht_gu[i] = nullptr; ht_dn[i] = nullptr; }
            for (int l = 0; l < NL; ++l) {
                const int p = m.cards[(size_t) t].ec.pool_of[l];
                const size_t gs = m.cards[(size_t) t].ec.dn_off[p];
                size_t pre = 0;
                for (int u = 0; u < t; ++u) {
                    const CardW& ow = m.cards[(size_t) u];
                    pre += ow.ec.slot_bytes[ow.ec.pool_of[l]];
                }
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    if (lc.at[i] < 0) continue;
                    ht_gu[i] = lc.arena + lc.at[i] + pre;
                    ht_dn[i] = ht_gu[i] + gs;
                }
            }
            // neither table may point outside the arena: a stride applied to the wrong layer's rows is caught here,
            // at start-up, rather than as a garbage weight a layer later.
            for (size_t i = 0; i < (size_t) NL * NEXP && !bad; ++i) {
                for (int q = 0; q < 2; ++q) {
                    const char* p = q ? ht_dn[i] : ht_gu[i];
                    if (!p) continue;
                    if (p < lc.arena || p >= lc.arena + lc.bytes) {
                        std::fprintf(stderr, "fnx: l2dec: off: card %d layer %zu expert %zu's %s pointer %p is outside the arena [%p, %p)%c",
                                     t, i / NEXP, i % NEXP, q ? "down" : "gate|up", (const void*) p, (const void*) lc.arena,
                                     (const void*) (lc.arena + lc.bytes), 10);
                        bad = true;
                    }
                }
            }
            for (int q = 0; q < 2 && !bad; ++q) {
                const char** src = q ? ht_dn.data() : ht_gu.data();
                char** dst = q ? (char**) g.card[t].ep_dn : (char**) g.card[t].ep_gu;
                if (cudaMemcpy(dst, src, (size_t) NL * NEXP * sizeof(char*), cudaMemcpyHostToDevice) != cudaSuccess) bad = true;
            }
        }
        if (bad) {
            std::fprintf(stderr, "fnx: l2dec: off: the tier's expert tables would not upload%c", 10);
            return false;
        }
        for (int t = 0; t < NCARD; ++t) {
            for (int l = 0; l < NL; ++l) {
                g.card[t].gu_e[l] = m.cards[(size_t) t].L[(size_t) l].gu_e;
                g.card[t].dn_e[l] = m.cards[(size_t) t].L[(size_t) l].down_e;
                g.card[t].gu_stride[l] = m.cards[(size_t) t].L[(size_t) l].gu_e_stride;
                g.card[t].dn_stride[l] = m.cards[(size_t) t].L[(size_t) l].down_e_stride;
            }
        }
        g.on = true;
        std::fprintf(stderr, "fnx: l2dec: on: card %d (%s): %d experts resident, %d picks a window at most, beside the host worker%c",
                     lc.dev, lc.name.c_str(), lc.experts, g.cap, 10);
    }

    // this card's pick map, in its own memory: one byte per expert, read by its router as it splits the window
    std::vector<char> ht((size_t) NL * NEXP, 0);
    for (size_t i = 0; i < ht.size(); ++i) ht[i] = lc.at[i] >= 0 ? 1 : 0;
    bool bad = false;
    {
        // the map and the stream that clears it both belong to this card: CUDA refuses a handle used from another
        // device, and a clear that never landed would leave the router sending picks to a tier that stopped answering
        // them (the same handle-shape mistake as Phase 14's events)
        DevGuard on_card(m.cards[(size_t) card].dev);
        if (!on_card.ok || cudaStreamCreateWithFlags(&g.card[card].clr, cudaStreamNonBlocking) != cudaSuccess) bad = true;
        else if (cudaMalloc((void**) &g.card[card].map, (size_t) NL * NEXP) != cudaSuccess) bad = true;
        else if (cudaMemcpy(g.card[card].map, ht.data(), (size_t) NL * NEXP, cudaMemcpyHostToDevice) != cudaSuccess) bad = true;
    }
    if (bad) {
        std::fprintf(stderr, "fnx: l2dec: off: card %d's pick map would not fit%c", card, 10);
        return false;
    }
    g.card[card].dev = m.cards[(size_t) card].dev;
    g.card[card].box = hbox;   // last: from here the worker may take this card's rounds
    if (!g.started) {
        g.started = true;
        std::thread(worker).detach();
    }
    return true;
}

const char* map(int card) { return card >= 0 && card < NCARD ? s().card[card].map : nullptr; }

int cap() { return s().on ? s().cap : 0; }

tpopt::WaitSlot* wake() { return &s().wake; }

}  // namespace dec
}  // namespace l2
}  // namespace fnx
