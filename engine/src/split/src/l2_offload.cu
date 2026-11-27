// What M2a does with the spare card: budget its arena, fill it with the hottest experts host RAM holds, and prove the
// copy (and the card's own code) before anything depends on it. l2_offload.hpp says why the tier exists.
#include "l2_offload.hpp"

#include "model.hpp"
#include "../../tpopt/tp_opt.hpp"
#include "l2_dec.hpp"
#include "l2_pf.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace fnx {
namespace l2 {

namespace {

constexpr double GIB = 1024.0 * 1024.0 * 1024.0;
constexpr size_t MIB = 1u << 20;
constexpr size_t PROBE_PAGE_BYTES = 64 * MIB;    // the pageable compare, so the pinned number can be read against it
constexpr size_t PROBE_D2H_BYTES = 256 * MIB;    // arena -> pinned host

// one card's slice of one expert, straight out of its page-locked copy. The host side of an expert is one contiguous
// [gate|up | down] block (model.cpp:971-974 writes it that way), so this is normally one DMA per card per expert.
struct Piece {
    const char* src;
    size_t len;
};

// the arena half of a piece: the same bytes as they sit on the card
struct Seg {
    unsigned long long off, len;
};

// The fold both sides agree on: every 4-byte little-endian word of a byte range added (mod 2^64) and xored, plus the
// bytes a whole word does not cover, each added and xored as itself. Both operations commute, so the host may walk a
// range in order over as many threads as it likes while the card takes the words strided by thread: they land on the
// same pair as long as each byte is covered once, which is what the segment list (not the arena's padding) decides.
__global__ void k_fold_one(const char* arena, const Seg* segs, int nseg, unsigned long long* out) {
    __shared__ unsigned long long red[2];
    if (threadIdx.x == 0) { red[0] = 0; red[1] = 0; }
    __syncthreads();
    unsigned long long s = 0, x = 0;
    for (int i = blockIdx.x; i < nseg; i += gridDim.x) {
        const Seg g = segs[i];
        const unsigned char* p = (const unsigned char*) (arena + g.off);
        const unsigned long long nw = g.len >> 2, tail = g.len & 3;
        for (unsigned long long w = threadIdx.x; w < nw; w += blockDim.x) {
            const unsigned long long v = (unsigned long long) p[4 * w] | ((unsigned long long) p[4 * w + 1] << 8) |
                                         ((unsigned long long) p[4 * w + 2] << 16) | ((unsigned long long) p[4 * w + 3] << 24);
            s += v;
            x ^= v;
        }
        if ((unsigned long long) threadIdx.x < tail) {
            const unsigned long long v = p[4 * nw + threadIdx.x];
            s += v;
            x ^= v;
        }
    }
    atomicAdd(&red[0], s);
    atomicXor(&red[1], x);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(out, red[0]);
        atomicXor(out + 1, red[1]);
    }
}

inline double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void host_fold(const std::vector<Piece>& pieces, unsigned long long& sum, unsigned long long& xr) {
    sum = 0;
    xr = 0;
    const int nth = std::min<int>(8, (int) pieces.size());
    if (nth <= 0) return;
    std::vector<unsigned long long> ps((size_t) nth, 0), px((size_t) nth, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t)
        th.emplace_back([&, t] {
            const size_t a = pieces.size() * (size_t) t / (size_t) nth, b = pieces.size() * (size_t) (t + 1) / (size_t) nth;
            for (size_t k = a; k < b; ++k) {
                const unsigned char* p = (const unsigned char*) pieces[k].src;
                const size_t n = pieces[k].len, nw = n >> 2;
                for (size_t i = 0; i < nw; ++i) {
                    const unsigned long long w = (unsigned long long) p[4 * i] | ((unsigned long long) p[4 * i + 1] << 8) |
                                                 ((unsigned long long) p[4 * i + 2] << 16) | ((unsigned long long) p[4 * i + 3] << 24);
                    ps[(size_t) t] += w;
                    px[(size_t) t] ^= w;
                }
                for (size_t i = 4 * nw; i < n; ++i) {
                    ps[(size_t) t] += p[i];
                    px[(size_t) t] ^= (unsigned long long) p[i];
                }
            }
        });
    for (std::thread& x : th) x.join();
    for (int t = 0; t < nth; ++t) {
        sum += ps[(size_t) t];
        xr ^= px[(size_t) t];
    }
}

}  // namespace

Ctx& shared() {
    static Ctx c;
    return c;
}

const Ctx& ctx() { return shared(); }

void setup(Model& m) {
    Ctx& c = shared();
    const long long req = requested_device();
    if (req < 0) return;   // off, and no word printed: a run without the tier logs what it logged before
    c.dev = (int) req;
    // a malformed setting is the operator's mistake, not a reason to lose the run: say which variable and stay off
    double cap_env = 8.0, reserve = 2.0;
    long long fill_mib = 0;
    bool want_test = true;
    bool pf_on = true, pf_stream = true;
    long long pf_floor_mib = 320;
    try {
        cap_env = env_num("FNX_L2_GIB", 8.0);
        reserve = env_num("FNX_L2_RESERVE_GIB", 2.0);
        fill_mib = env_int("FNX_L2_FILL_MIB", 0);
        want_test = env_int("FNX_L2_SELFTEST", 1) != 0;
        // M2b-1's streaming ask, read here as well as in l2_pf.cu: this file decides how much of the envelope the
        // streamed slots get, and l2_pf may only spend what is left. Two readers of one knob must not disagree, so
        // the defaults are the same numbers; a malformed value still turns the tier off with the one line below.
        pf_on = env_int("FNX_L2_PF", 1) != 0;
        pf_stream = env_int("FNX_L2_PF_STREAM", 1) != 0;
        pf_floor_mib = env_int("FNX_L2_PF_STAGE_MIB", 320);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fnx: l2: card %d off: %s\n", c.dev, e.what());
        return;
    }

    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) prev = 0;
    auto back = [&] { cudaSetDevice(prev); };
    auto off = [&](const std::string& why) {
        c.ok = false;
        c.why = why;
        c.stage_bytes = 0;   // nothing was reserved: an off tier must not leave a slot ask behind for l2_pf to spend
        if (c.arena) {
            cudaFree(c.arena);
            c.arena = nullptr;
            c.bytes = 0;
        }
        back();
        std::fprintf(stderr, "fnx: l2: card %d off: %s\n", c.dev, why.c_str());
    };

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess) return off("cudaGetDeviceCount failed");
    if (c.dev >= ndev) return off("CUDA_VISIBLE_DEVICES names " + std::to_string(ndev) + " devices, FNX_L2_GPU=" + std::to_string(c.dev));
    // the tensor-parallel cards' devs are visible ordinals (model.cpp:1545), so a collision means the launcher put the
    // same index in both places; the tier is refused rather than carved out of a working card
    for (int t = 0; t < NCARD; ++t)
        if (m.cards[(size_t) t].dev == c.dev)
            return off("device " + std::to_string(c.dev) + " is already tensor-parallel card " + std::to_string(t));
    if (cudaSetDevice(c.dev) != cudaSuccess) return off("cudaSetDevice(" + std::to_string(c.dev) + ") failed");
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, c.dev) != cudaSuccess) return off("cudaGetDeviceProperties failed");
    c.name = prop.name;
    c.major = prop.major;
    c.minor = prop.minor;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return off("cudaMemGetInfo failed");
    c.free_gib = (double) free_b / GIB;
    c.total_gib = (double) total_b / GIB;
    c.reserve_gib = reserve;
    const double budget = tpopt::expert_budget_gib(c.total_gib, c.free_gib, reserve);   // the same rule as the cards, FNX_VRAM_PCT and all
    c.cap_gib = cap_env < budget ? cap_env : budget;
    char num[128] = "";
    std::snprintf(num, sizeof num, "only %.2f GiB to spend (free %.2f, total %.2f, reserve %.2f, FNX_VRAM_PCT %.0f%%)", budget, c.free_gib,
                  c.total_gib, reserve, tpopt::vram_fraction() * 100.0);
    if (c.cap_gib < 2.0) return off(num);

    // the experts host RAM holds, hottest first (score is the profile's, model.cpp:1832-1840, and the cards left the
    // coldest of them there). An expert is a candidate when every card has a host copy of its slice: that is what
    // cpu_host_gu records (model.cpp:973 and :985), and it is the whole of it - one expert's weight is the sum of the
    // cards' slices, so a half-placed expert would be worth nothing.
    const size_t nidx = (size_t) NL * NEXP;
    c.at.assign(nidx, -1);
    c.hits.assign(nidx, 0);   // M3's slot counter: sized here so M2b never has to look at an empty vector
    std::vector<std::pair<double, int>> hot;
    for (int l = 0; l < NL; ++l)
        for (int e = 0; e < NEXP; ++e) {
            const size_t i = (size_t) l * NEXP + e;
            size_t tot = 0;
            for (int t = 0; t < NCARD; ++t) {
                const CardW& cw = m.cards[(size_t) t];
                const int p = cw.ec.pool_of[l];
                if (i >= cw.cpu_host_gu.size() || i >= cw.cpu_host_dn.size() || !cw.cpu_host_gu[i] || !cw.cpu_host_dn[i] || p < 0 ||
                    p >= EC_POOLS || cw.ec.slot_bytes[p] <= cw.ec.dn_off[p]) {
                    tot = 0;
                    break;
                }
                tot += cw.ec.slot_bytes[p];
            }
            if (!tot) continue;
            hot.push_back({m.score.size() == nidx ? m.score[i] : 0.0, (int) i});
        }
    std::stable_sort(hot.begin(), hot.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first;
    });
    if (hot.empty()) return off("no expert of this model is in host RAM (nothing to place)");

    // M2b-1's streamed slots are reserved here, before the arena asks for a byte: arena + slots x2 share one envelope,
    // and the envelope is `budget` - the number expert_budget_gib() already holds FNX_L2_RESERVE_GIB back of, so the
    // desktop's reserve can never be what the tier ends up eating. What one layer has to stream is the share of its
    // experts the arena will not hold, and that share depends on the arena, so the two sizes are settled together:
    // each round a bigger slot shrinks the arena, which needs a bigger slot, and six rounds is where it lands.
    double arena_gib = c.cap_gib;
    if (pf_on && pf_stream && pf_floor_mib >= 0) {
        size_t per_layer_max = 0, all_b = 0;
        {
            std::vector<size_t> per_layer((size_t) NL, 0);
            for (const std::pair<double, int>& hnd : hot) {
                const int l = (int) ((size_t) hnd.second / (size_t) NEXP);
                size_t need = 0;
                for (int t = 0; t < NCARD; ++t) need += m.cards[(size_t) t].ec.slot_bytes[m.cards[(size_t) t].ec.pool_of[l]];
                per_layer[(size_t) l] += need;
            }
            for (size_t b : per_layer) {
                all_b += b;
                if (b > per_layer_max) per_layer_max = b;
            }
        }
        const double floor_gib = (double) pf_floor_mib / 1024.0;
        size_t slot = 0;
        for (int it = 0; it < 6; ++it) {
            const double all_gib = (double) all_b / GIB;
            const double ram_share = all_gib > arena_gib ? 1.0 - arena_gib / all_gib : 0.0;   // what stays in RAM, and so must stream
            double one = (double) per_layer_max / GIB * ram_share;
            if (one < floor_gib) one = floor_gib;   // an operator who named FNX_L2_PF_STAGE_MIB gets at least that
            const double room = (budget - 2.0) / 2.0;   // two slots, with 2 GiB of arena still standing
            if (one > room) one = room;
            size_t s = one > 0.0 ? (size_t) ((((size_t) (one * GIB)) + MIB - 1) / MIB * MIB) : 0;
            double left = budget - 2.0 * ((double) s / GIB);
            if (left < 2.0) {
                s = 0;   // not even the floor fits next to a 2 GiB arena: the arena keeps the envelope, nothing streams
                left = budget;
            }
            const double a = c.cap_gib < left ? c.cap_gib : left;
            const double diff = a > arena_gib ? a - arena_gib : arena_gib - a;
            if (s == slot && diff < 0.002) break;
            slot = s;
            arena_gib = a;
        }
        c.stage_bytes = slot;
    }

    // half the ask until the card takes it: a desktop that grew into the free memory costs the tier, never the run
    size_t want = (size_t) (arena_gib * GIB);   // cap_env, less the streamed slots reserved just above
    while (want >= (size_t) (2.0 * GIB) && cudaMalloc((void**) &c.arena, want) != cudaSuccess) want /= 2;
    if (!c.arena) return off("no allocation of 2 GiB or more fits on it (free " + std::to_string((long long) c.free_gib) + " GiB)");
    c.bytes = want;
    c.gib = (double) want / GIB;

    // the probes go first, while the arena is still empty and nothing of it is needed
    {
        std::vector<char> page(PROBE_PAGE_BYTES, 0);
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        if (cudaMemcpy(c.arena, page.data(), PROBE_PAGE_BYTES, cudaMemcpyHostToDevice) == cudaSuccess)
            c.h2d_pageable_gbs = PROBE_PAGE_BYTES / seconds_since(t0) / 1e9;
        char* pinned = nullptr;
        if (cudaHostAlloc((void**) &pinned, PROBE_D2H_BYTES, cudaHostAllocDefault) == cudaSuccess && pinned) {
            const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            if (cudaMemcpy(pinned, c.arena, PROBE_D2H_BYTES, cudaMemcpyDeviceToHost) == cudaSuccess)
                c.d2h_gbs = PROBE_D2H_BYTES / seconds_since(t1) / 1e9;
            cudaFreeHost(pinned);
        }
    }

    std::vector<Piece> pieces;
    size_t at_b = 0;
    const size_t fill_limit = fill_mib > 0 ? (size_t) fill_mib * MIB : ~(size_t) 0;
    for (const std::pair<double, int>& hnd : hot) {
        const size_t i = (size_t) hnd.second;
        const int l = (int) (i / (size_t) NEXP);
        size_t need = 0;
        for (int t = 0; t < NCARD; ++t) need += m.cards[(size_t) t].ec.slot_bytes[m.cards[(size_t) t].ec.pool_of[l]];
        if (at_b + need > c.bytes || at_b >= fill_limit) {
            ++c.skipped;
            continue;
        }
        c.at[i] = (long long) at_b;   // the cards' slices of one expert next to each other: one launch can read the whole expert
        for (int t = 0; t < NCARD; ++t) {
            const CardW& cw = m.cards[(size_t) t];
            const int p = cw.ec.pool_of[l];
            const size_t gs = cw.ec.dn_off[p], ds = cw.ec.slot_bytes[p] - gs;
            if (cw.cpu_host_dn[i] == cw.cpu_host_gu[i] + gs) pieces.push_back({cw.cpu_host_gu[i], gs + ds});
            else {
                pieces.push_back({cw.cpu_host_gu[i], gs});
                pieces.push_back({cw.cpu_host_dn[i], ds});
            }
        }
        at_b += need;
        ++c.experts;
    }
    if (!c.experts) return off("its arena of " + std::to_string((long long) (c.gib * 100) / 100.0) + " GiB holds no whole expert");

    const std::chrono::steady_clock::time_point t_fill = std::chrono::steady_clock::now();
    size_t cursor = 0;
    bool copy_ok = true;
    for (const Piece& pc : pieces) {
        if (cudaMemcpy(c.arena + cursor, pc.src, pc.len, cudaMemcpyHostToDevice) != cudaSuccess) {
            copy_ok = false;
            break;
        }
        cursor += pc.len;
    }
    c.seconds = seconds_since(t_fill);
    c.moved = cursor;
    c.demote_room = c.bytes > c.moved ? c.bytes - c.moved : 0;   // the M3 demotion target: 0 when the arena came out full
    if (!copy_ok) return off("the host -> card copy failed after " + std::to_string(cursor >> 20) + " MiB");
    c.h2d_pinned_gbs = c.seconds > 0 ? (double) c.moved / c.seconds / 1e9 : 0;   // this fill is the pinned H2D measurement

    unsigned long long hsum = 0, hxor = 0;
    if (want_test) {
        host_fold(pieces, hsum, hxor);
        std::vector<Seg> segs(pieces.size());
        size_t cur = 0;
        for (size_t k = 0; k < pieces.size(); ++k) {
            segs[k] = {(unsigned long long) cur, (unsigned long long) pieces[k].len};
            cur += pieces[k].len;
        }
        Seg* dsegs = nullptr;
        unsigned long long* dout = nullptr;
        bool ok = cudaMalloc((void**) &dsegs, segs.size() * sizeof(Seg)) == cudaSuccess && cudaMalloc((void**) &dout, 2 * sizeof(unsigned long long)) == cudaSuccess;
        if (ok)
            ok = cudaMemcpy(dsegs, segs.data(), segs.size() * sizeof(Seg), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemset(dout, 0, 2 * sizeof(unsigned long long)) == cudaSuccess;
        if (ok) {
            const int grid = (int) std::min<size_t>(1024, segs.size());
            k_fold_one<<<grid, 256>>>(c.arena, dsegs, (int) segs.size(), dout);
            ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        }
        unsigned long long g[2] = {0, 0};
        if (ok) ok = cudaMemcpy(g, dout, sizeof g, cudaMemcpyDeviceToHost) == cudaSuccess;
        if (dsegs) cudaFree(dsegs);
        if (dout) cudaFree(dout);
        c.selftest_ok = ok && g[0] == hsum && g[1] == hxor;
    }

    c.ok = !want_test || c.selftest_ok;
    std::fprintf(stderr,
                 "fnx: l2: card %d %s (sm_%d%d): arena %.2f GiB (free %.2f, reserve %.2f, cap %.2f), experts %d (%.2f GiB, %d left in "
                 "RAM) in %.2f s = %.2f GB/s H2D pinned, %.2f GB/s pageable probe, %.2f GB/s D2H, selftest %s (sum 0x%016llx)"
                 ", slots %zu MiB x2\n",
                 c.dev, c.name.c_str(), c.major, c.minor, c.gib, c.free_gib, c.reserve_gib, c.cap_gib, c.experts, (double) c.moved / GIB,
                 c.skipped, c.seconds, c.h2d_pinned_gbs, c.h2d_pageable_gbs, c.d2h_gbs, want_test ? (c.selftest_ok ? "ok" : "FAILED") : "skipped",
                 hsum, c.stage_bytes >> 20);
    back();
}

// FNX_L2_DROP_HOST: for the experts the arena holds, the page-locked original in RAM buys nothing once both the
// prompt path and decode ask the tier for them. Give those copies back and the same bytes go to the table lock
// (MLOCK), whose misses are what a long decode round waits on. Safety comes from the tier then being the only place
// these experts live, which every reader has to agree with:
//   - decode: the router sends a pick that missed the first tier to the tier's box (moe.cuh:276-312), and only an
//     expert with a slot of its own stays in this card's jobs, so no Turing kernel is handed one;
//   - a long prompt: the same plan gives the tier these experts, and the cards read the host copy of an expert only
//     where the tier left it to them (prefill.cu:1513-1550);
//   - a short prompt: prefill.cu asks the tier for the resident ones alone (l2_pf::plan with stream=false);
//   - a first-tier promotion: cache_plan skips an expert whose host original is gone (moe.cuh:87), so k_cache_copy
//     never reads a null;
//   - and l2_dec's retire() (l2_dec.cu:106) stays the one way an expert can go missing after this, which is the
//     hole an expert outside the first tier was always allowed to be (the host worker answers 0 for it).
// Two rounds, because the swap has to be all or nothing: first each card reads its own tables and reserves the
// smaller blocks, and only when that has worked for every card does a card let its old blocks go. The layout check
// is the rest of the safety - the copy and the free both assume model.cpp's block [entry 0 | entry 1 | ...] of
// exp_b bytes per layer, mapped as one device pointer per block.
void drop_host(Model& m)
{
    Ctx& c = shared();
    if (env_int("FNX_L2_DROP_HOST", 0) == 0) return;
    const size_t nidx = (size_t) NL * NEXP;

    const char* why = nullptr;
    if (!c.ok || !c.arena || c.experts <= 0 || c.at.size() != nidx) why = "the tier's arena is not what this expects";
    else if (m.arena) why = "FNX_ARENA lends these very slots to K/V";
    else if (!pf::use()) why = "the prompt path is not on the tier (FNX_L2_PF=0)";
    else if (!dec::map(0) || !dec::map(1)) why = "decode is not on the tier (FNX_CPU_MOE=0 / FNX_L2_DEC=0)";
    else {
        const int dcap = dec::cap();
        if (dcap != 0 && dcap < MAXT * TOPK) why = "FNX_L2_DEC_MAX caps the tier below a whole window";
        for (int card = 0; card < NCARD && !why; ++card) {
            const CardW& cw = m.cards[card];
            if (cw.ec_owner0.size() != nidx || cw.ec_slot0.size() != nidx || cw.cpu_host_gu.size() != nidx ||
                cw.cpu_host_dn.size() != nidx || !cw.ec.ep_gu || !cw.ec.ep_dn || !cw.ec.host_gu || !cw.ec.host_dn)
                why = "a card's expert tables are not the whole grid this expects";
        }
    }
    if (why) {
        std::fprintf(stderr, "fnx: l2: drop: off: %s\n", why);
        return;
    }

    struct Card {
        bool ok = false, wrote = false;
        std::vector<char*> old_h, new_h;                   // per layer: the block, and the smaller one replacing it
        std::vector<size_t> exp_b, seen, keep;
        std::vector<unsigned char> gone;                   // [NL * NEXP]: this card gives this expert back
        std::vector<const char*> ogu, o_dn, ohu, ohd;      // what its tables said, read back
        std::vector<const char*> ngu, n_dn, nhu, nhd;      // and what they will say
        const char** t_gu = nullptr;
        const char** t_dn = nullptr;
        const char** th_gu = nullptr;
        const char** th_dn = nullptr;
        size_t freed = 0;
    };
    Card cd[NCARD];
    for (int card = 0; card < NCARD; ++card) {
        Card& p = cd[card];
        p.old_h.assign(NL, nullptr);
        p.new_h.assign(NL, nullptr);
        p.exp_b.assign(NL, 0);
        p.seen.assign(NL, 0);
        p.keep.assign(NL, 0);
        p.gone.assign(nidx, 0);
    }
    const auto t0 = std::chrono::steady_clock::now();

    // round 1: read the tables, check the layout, reserve the survivors' blocks. Nothing is published, so a card
    // that fails here costs nothing
    std::thread th[NCARD];
    for (int card = 0; card < NCARD; ++card)
        th[card] = std::thread([&, card] {
            Card& p = cd[card];
            CardW& cw = m.cards[card];
            p.ogu.assign(nidx, nullptr);
            p.o_dn.assign(nidx, nullptr);
            p.ohu.assign(nidx, nullptr);
            p.ohd.assign(nidx, nullptr);
            if (cudaSetDevice(cw.dev) != cudaSuccess) return;
            if (cudaDeviceSynchronize() != cudaSuccess) return;  // the engine is between builds: nothing is in flight
            if (cudaMemcpy(p.ogu.data(), cw.ec.ep_gu, nidx * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return;
            if (cudaMemcpy(p.o_dn.data(), cw.ec.ep_dn, nidx * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return;
            if (cudaMemcpy(p.ohu.data(), cw.ec.host_gu, nidx * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return;
            if (cudaMemcpy(p.ohd.data(), cw.ec.host_dn, nidx * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return;
            for (int l = 0; l < NL; ++l) {
                const size_t gs = cw.L[l].gu_e_stride, ds = cw.L[l].down_e_stride;
                if (gs == 0 || ds == 0) return;
                const size_t eb = gs + ds;
                // plan() only takes a layer's experts when both cards describe the layer's pool (l2_pf.cu:796-806).
                // Without that the layer keeps every expert on the cards, and an expert whose RAM went back would be
                // read from a null table entry - so a layer the tier cannot take gives nothing back either.
                for (int t = 0; t < NCARD; ++t) {
                    const CardW& ow = m.cards[(size_t) t];
                    const int pp = ow.ec.pool_of[l];
                    if (pp < 0 || pp >= EC_POOLS || ow.ec.slot_bytes[pp] <= ow.ec.dn_off[pp]) return;
                }
                char* ob = nullptr;
                char* mb = nullptr;
                size_t n_seen = 0, n_keep = 0;
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    const char* hp = cw.cpu_host_gu[i];
                    if (!hp) continue;                                  // a slot's expert, or one with no copy
                    if (!cw.cpu_host_dn[i]) return;
                    if (!ob) {
                        void* dm = nullptr;
                        if (cudaHostGetDevicePointer(&dm, (void*) hp, 0) != cudaSuccess) return;
                        ob = (char*) hp;
                        mb = (char*) dm;
                    }
                    const size_t off = (size_t)(hp - ob);
                    if (off % eb != 0) return;
                    const size_t rk = off / eb;
                    if (cw.cpu_host_dn[i] != hp + gs || p.ohu[i] != mb + rk * eb || p.ohd[i] != mb + rk * eb + gs) return;
                    ++n_seen;
                    if (c.at[i] >= 0 && cw.ec_slot0[i] < 0) { p.gone[i] = 1; continue; }  // the arena holds it, RAM is spare
                    ++n_keep;
                }
                p.exp_b[l] = eb;
                p.seen[l] = n_seen;
                p.keep[l] = n_keep;
                if (!ob || n_keep == n_seen) continue;                  // a layer with nothing to give back keeps its block
                void* nh = nullptr;
                if (cudaHostAlloc(&nh, n_keep * eb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return;
                p.old_h[l] = ob;
                p.new_h[l] = (char*) nh;
            }
            p.ok = true;
        });
    for (int card = 0; card < NCARD; ++card) th[card].join();

    bool all = true;
    for (int card = 0; card < NCARD; ++card) all = all && cd[card].ok;
    if (!all) {
        // nothing was published, so the only thing to undo is the reservation. Page-locked memory handed back this
        // way is the same memory on every context (Portable), so the main thread can free these
        for (int card = 0; card < NCARD; ++card)
            for (int l = 0; l < NL; ++l)
                if (cd[card].new_h[l]) cudaFreeHost(cd[card].new_h[l]);
        std::fprintf(stderr, "fnx: l2: drop: off: a card's host blocks are not the layout this expects\n");
        return;
    }

    // round 2: move the survivors, swap the tables, then let the old blocks go. The tables swap by pointer, so a
    // launch after this one sees the new grid; the old tables stay (a few MiB a card) rather than risk a captured
    // graph reading a freed one
    for (int card = 0; card < NCARD; ++card)
        th[card] = std::thread([&, card] {
            Card& p = cd[card];
            CardW& cw = m.cards[card];
            if (cudaSetDevice(cw.dev) != cudaSuccess) return;
            p.ngu = p.ogu;
            p.n_dn = p.o_dn;
            p.nhu = p.ohu;
            p.nhd = p.ohd;
            bool done = true;
            for (int l = 0; l < NL && done; ++l) {
                if (!p.new_h[l]) continue;
                void* dm = nullptr;
                if (cudaHostGetDevicePointer(&dm, p.new_h[l], 0) != cudaSuccess) { done = false; break; }
                const size_t eb = p.exp_b[l], gs = cw.L[l].gu_e_stride;
                size_t out = 0;
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    if (!cw.cpu_host_gu[i]) continue;
                    if (p.gone[i]) {
                        p.ngu[i] = p.n_dn[i] = p.nhu[i] = p.nhd[i] = nullptr;
                        // the mirrors go too: the host worker reads cpu_host_* per pick (engine.cu:478, :514, and a
                        // null there means "answer 0"), and the prompt path's staging tables read ec_host_* for the
                        // experts the tier did not take (prefill.cu:1541, :1550). Left as they were, both would read
                        // bytes this thread has just given back.
                        cw.cpu_host_gu[i] = cw.cpu_host_dn[i] = nullptr;
                        cw.ec_host_gu[i] = cw.ec_host_dn[i] = nullptr;
                        continue;
                    }
                    const size_t rk = (size_t)(cw.cpu_host_gu[i] - p.old_h[l]) / eb;
                    char* dst = p.new_h[l] + out * eb;
                    std::memcpy(dst, p.old_h[l] + rk * eb, eb);
                    const char* mg = (const char*) dm + out * eb;
                    const char* md = mg + gs;
                    if (p.ogu[i] == p.ohu[i]) p.ngu[i] = mg;             // it was read from RAM, so it follows the copy
                    if (p.o_dn[i] == p.ohd[i]) p.n_dn[i] = md;
                    p.nhu[i] = mg;
                    p.nhd[i] = md;
                    cw.cpu_host_gu[i] = (const char*) dst;
                    cw.cpu_host_dn[i] = dst + gs;
                    cw.ec_host_gu[i] = mg;
                    cw.ec_host_dn[i] = md;
                    ++out;
                }
                if (out != p.keep[l]) done = false;
            }
            if (!done) return;
            done = cudaMalloc((void**) &p.t_gu, nidx * sizeof(void*)) == cudaSuccess &&
                 cudaMalloc((void**) &p.t_dn, nidx * sizeof(void*)) == cudaSuccess &&
                 cudaMalloc((void**) &p.th_gu, nidx * sizeof(void*)) == cudaSuccess &&
                 cudaMalloc((void**) &p.th_dn, nidx * sizeof(void*)) == cudaSuccess &&
                 cudaMemcpy(p.t_gu, p.ngu.data(), nidx * sizeof(void*), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(p.t_dn, p.n_dn.data(), nidx * sizeof(void*), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(p.th_gu, p.nhu.data(), nidx * sizeof(void*), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(p.th_dn, p.nhd.data(), nidx * sizeof(void*), cudaMemcpyHostToDevice) == cudaSuccess;
            if (!done) return;
            cw.ec.ep_gu = p.t_gu;
            cw.ec.ep_dn = p.t_dn;
            cw.ec.host_gu = (const char* const*) p.th_gu;
            cw.ec.host_dn = (const char* const*) p.th_dn;
            // model.cpp:414-415 cached a row of the OLD table in every layer, and the non-staging prompt path reads
            // that cache (prefill.cu:1803 only re-points it for a staging pass). The grid moved, so the cached rows
            // are re-pointed here - the one place that knows the new tables are live.
            for (int l = 0; l < NL; ++l) {
                cw.L[(size_t) l].gu_e.ep = cw.ec.ep_gu + (size_t) l * NEXP;
                cw.L[(size_t) l].down_e.ep = cw.ec.ep_dn + (size_t) l * NEXP;
            }
            for (int l = 0; l < NL; ++l) {
                if (!p.new_h[l]) continue;
                cudaFreeHost(p.old_h[l]);
                p.freed += (p.seen[l] - p.keep[l]) * p.exp_b[l];
            }
            if (p.freed > cw.host_bytes) p.freed = cw.host_bytes;
            cw.host_bytes -= p.freed;
            cudaDeviceSynchronize();
            p.wrote = true;
        });
    for (int card = 0; card < NCARD; ++card) th[card].join();

    size_t freed = 0;
    int gone = 0, cards = 0;
    c.dropped_entry.assign(nidx, 0);
    for (size_t i = 0; i < nidx; ++i) {
        bool all_gone = true;
        for (int card = 0; card < NCARD; ++card) all_gone = all_gone && cd[card].gone[i];
        if (all_gone) { c.dropped_entry[i] = 1; ++gone; }
    }
    for (int card = 0; card < NCARD; ++card)
        if (cd[card].wrote) { freed += cd[card].freed; ++cards; }
    c.drop_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    c.dropped = gone > 0 && cards == NCARD;
    c.freed = freed;
    c.dropped_experts = gone;
    std::fprintf(stderr,
                 "fnx: l2: drop: %d experts gave their RAM copies back: %.2f GiB off the host (%d of %d cards, %.2f s); "
                 "the arena is their only copy now\n",
                 gone, (double) freed / GIB, cards, (int) NCARD, c.drop_seconds);
    for (int card = 0; card < NCARD; ++card)
        std::fprintf(stderr, "fnx: l2: drop: card %d %s: %.2f GiB back, %.2f GiB of copies left\n", card,
                     cd[card].wrote ? "gave" : "kept", (double) cd[card].freed / GIB, (double) m.cards[card].host_bytes / GIB);
}

}  // namespace l2
}  // namespace fnx
