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
// Value 1 hands back an arena expert that has no first-tier slot of its own. Value 2 also hands back the ones that
// do keep a slot: their RAM copy is only the landing place for a demotion, and after a demotion the tier answers
// them (the router sends a pick with no slot there), while cache_plan cannot promote them again. The arena is
// about a fifth of the experts, so 2 buys several GiB more of MLOCK - at the price of one more of these agreements.
// Two rounds: the first only reads the tables and checks the layout, so a card that does not match costs nothing.
// The second walks the layers and, one layer at a time, reserves that layer's smaller block, moves the survivors
// into it, republishes that layer's table rows and only then lets the old block go. One layer at a time is the
// point: the peak is the old grid plus a single layer (about 0.23 GiB a card), where reserving every smaller block
// up front asked for the whole new grid (21.66 GiB) on top of what was still locked, and that pushed the page
// cache out and set off the kernel's out-of-memory killer. The layout check is the rest of the safety - the copy
// and the free both assume model.cpp's block [entry 0 | entry 1 | ...] of exp_b bytes per layer, mapped as one
// device pointer per block.
void drop_host(Model& m)
{
    Ctx& c = shared();
    const int what = env_int("FNX_L2_DROP_HOST", 0);
    if (what == 0) return;
    const bool also_slot = what >= 2;
    const size_t nidx = (size_t) NL * NEXP;

    const char* why = nullptr;
    char detail[192] = {};
    if (!c.ok || !c.arena || c.experts <= 0 || c.at.size() != nidx) {
        std::snprintf(detail, sizeof(detail),
                      "the tier's arena is not what this expects (ok %d, arena %s, experts %d, at %zu entries of %zu)",
                      (int) c.ok, c.arena ? "set" : "null", c.experts, c.at.size(), nidx);
        why = detail;
    }
    else if (m.arena) why = "FNX_ARENA lends these very slots to K/V";
    else if (!pf::use()) why = "the prompt path is not on the tier (FNX_L2_PF=0)";
    else if (!dec::map(0) || !dec::map(1)) why = "decode is not on the tier (FNX_CPU_MOE=0 / FNX_L2_DEC=0)";
    else {
        const int dcap = dec::cap();
        if (dcap != 0 && dcap < MAXT * TOPK) why = "FNX_L2_DEC_MAX caps the tier below a whole window";
        for (int card = 0; card < NCARD && !why; ++card) {
            const CardW& cw = m.cards[card];
            // two widths here, and mixing them up turns the gate off for every run: ec_owner0 is one entry per VRAM
            // slot (model.cpp:372 sets n_slots to owner.size(), :399 hands that vector over), while the tables this
            // function walks are one entry per expert. Say which array fell short, and with how many rows.
            const char* bad = nullptr;
            long long have = 0, want = (long long) nidx;
            if (cw.ec.n_slots <= 0) { bad = "ec.n_slots"; have = cw.ec.n_slots; want = 1; }
            else if (cw.ec_owner0.size() != (size_t) cw.ec.n_slots) {
                bad = "ec_owner0"; have = (long long) cw.ec_owner0.size(); want = cw.ec.n_slots;
            } else if (cw.ec_slot0.size() != nidx) { bad = "ec_slot0"; have = (long long) cw.ec_slot0.size(); }
            else if (cw.ec_host_gu.size() != nidx) { bad = "ec_host_gu"; have = (long long) cw.ec_host_gu.size(); }
            else if (cw.ec_host_dn.size() != nidx) { bad = "ec_host_dn"; have = (long long) cw.ec_host_dn.size(); }
            else if (cw.cpu_host_gu.size() != nidx) { bad = "cpu_host_gu"; have = (long long) cw.cpu_host_gu.size(); }
            else if (cw.cpu_host_dn.size() != nidx) { bad = "cpu_host_dn"; have = (long long) cw.cpu_host_dn.size(); }
            else if (!cw.ec.ep_gu || !cw.ec.ep_dn || !cw.ec.host_gu || !cw.ec.host_dn) {
                bad = "ec.ep_gu / ep_dn / host_gu / host_dn"; have = 0;
            }
            if (bad) {
                std::snprintf(detail, sizeof(detail), "card %d reads %s with %lld rows where this expects %lld",
                              card, bad, have, want);
                why = detail;
            }
        }
    }
    if (why) {
        std::fprintf(stderr, "fnx: l2: drop: off: %s\n", why);
        return;
    }

    struct Card {
        bool ok = false, wrote = false;
        std::vector<char*> old_h, new_h;                   // per layer: the block, and the smaller one after it
        std::vector<size_t> exp_b, seen, keep;
        std::vector<unsigned char> gone;                   // [NL * NEXP]: this card gives this expert back
        std::vector<const char*> ogu, o_dn, ohu, ohd;      // what its tables said, read back
        int stopped = -1;                                  // the first layer this card could not move
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

    // round 1: read the tables, check the layout, and remember which block each layer would replace. Nothing is
    // reserved and nothing is published here, so a card that fails here costs nothing
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
                    // the arena holds it, so its RAM copy buys nothing; with value 2 that includes an expert that
                    // also holds a swap slot, whose RAM copy only ever caught a demotion (the tier answers it after)
                    if (c.at[i] >= 0 && (also_slot || cw.ec_slot0[i] < 0)) { p.gone[i] = 1; continue; }
                    ++n_keep;
                }
                p.exp_b[l] = eb;
                p.seen[l] = n_seen;
                p.keep[l] = n_keep;
                if (ob && n_keep < n_seen) p.old_h[l] = ob;   // round 2 replaces this layer's block with a smaller one
            }
            p.ok = true;
        });
    for (int card = 0; card < NCARD; ++card) th[card].join();

    bool all = true;
    for (int card = 0; card < NCARD; ++card) all = all && cd[card].ok;
    if (!all) {
        // round 1 locked nothing and published nothing, so there is nothing to undo
        std::fprintf(stderr, "fnx: l2: drop: off: a card's host blocks are not the layout this expects\n");
        return;
    }

    // round 2, one layer at a time: reserve this layer's smaller block, move the survivors, republish this layer's
    // four table rows over the old ones, and only then let the old block go. Nothing on the host side of a card
    // (cpu_host_*, ec_host_*) moves before its rows have landed, so a layer this card stops at is still the layer
    // it was: old block alive, old table rows, old mirrors. The tables keep their base address, so model.cpp's
    // cached rows (model.cpp:414-415), every captured graph and both mirrors keep reading the same array.
    for (int card = 0; card < NCARD; ++card)
        th[card] = std::thread([&, card] {
            Card& p = cd[card];
            CardW& cw = m.cards[card];
            if (cudaSetDevice(cw.dev) != cudaSuccess) return;
            // one layer of the six tables, filled here and pushed with four copies; the last two rows are what this
            // card's own host side will hold, and they go in only once the first four have landed
            std::vector<const char*> row_gu(NEXP), row_dn(NEXP), row_hgu(NEXP), row_hdn(NEXP);
            std::vector<char*> row_cg(NEXP), row_cd(NEXP);
            for (int l = 0; l < NL; ++l) {
                char* ob = p.old_h[l];
                if (!ob) continue;                                   // this layer has nothing to give back
                const size_t eb = p.exp_b[l], gs = cw.L[l].gu_e_stride, keep = p.keep[l];
                char* nh = nullptr;
                void* dm = nullptr;
                if (cudaHostAlloc((void**) &nh, keep * eb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                    p.stopped = l;
                    break;
                }
                if (cudaHostGetDevicePointer(&dm, (void*) nh, 0) != cudaSuccess) {
                    cudaFreeHost(nh);                                // nothing points at it yet
                    p.stopped = l;
                    break;
                }
                size_t out = 0;
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    row_gu[e] = p.ogu[i];
                    row_dn[e] = p.o_dn[i];
                    row_hgu[e] = p.ohu[i];
                    row_hdn[e] = p.ohd[i];
                    row_cg[e] = (char*) cw.cpu_host_gu[i];
                    row_cd[e] = (char*) cw.cpu_host_dn[i];
                    if (!cw.cpu_host_gu[i]) continue;                              // a slot's expert, or one with no copy
                    if (p.gone[i]) {
                        row_hgu[e] = row_hdn[e] = nullptr;
                        row_cg[e] = row_cd[e] = nullptr;
                        continue;
                    }
                    const size_t rk = (size_t)(cw.cpu_host_gu[i] - ob) / eb;
                    std::memcpy(nh + out * eb, ob + rk * eb, eb);
                    const char* mg = (const char*) dm + out * eb;
                    const char* md = mg + gs;
                    if (p.ogu[i] == p.ohu[i]) row_gu[e] = mg;        // it was read from RAM, so it follows the copy
                    if (p.o_dn[i] == p.ohd[i]) row_dn[e] = md;
                    row_hgu[e] = mg;
                    row_hdn[e] = md;
                    row_cg[e] = nh + out * eb;
                    row_cd[e] = nh + out * eb + gs;
                    ++out;
                }
                if (out != keep) {
                    cudaFreeHost(nh);
                    p.stopped = l;
                    break;
                }
                const auto push = [&](const void* from, void* into) {
                    return cudaMemcpy(into, from, NEXP * sizeof(void*), cudaMemcpyHostToDevice) == cudaSuccess;
                };
                bool pub = push(row_gu.data(), (void*) (cw.ec.ep_gu + (size_t) l * NEXP)) &&
                           push(row_dn.data(), (void*) (cw.ec.ep_dn + (size_t) l * NEXP)) &&
                           push(row_hgu.data(), (void*) (cw.ec.host_gu + (size_t) l * NEXP)) &&
                           push(row_hdn.data(), (void*) (cw.ec.host_dn + (size_t) l * NEXP));
                if (!pub) {
                    // the rows did not land, so the old block is still what the cards read. Keep both blocks: the
                    // survivors are already in the new one, and freeing the old would pull the table's target out
                    p.new_h[l] = nh;
                    p.stopped = l;
                    break;
                }
                // the layer reads the new block on both sides now: the mirrors first, then the old block may go
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    cw.cpu_host_gu[i] = row_cg[e];
                    cw.cpu_host_dn[i] = row_cd[e];
                    cw.ec_host_gu[i] = row_hgu[e];
                    cw.ec_host_dn[i] = row_hdn[e];
                }
                cudaFreeHost(ob);
                p.old_h[l] = nullptr;
                p.new_h[l] = nh;
                p.freed += (p.seen[l] - keep) * eb;
            }
            if (p.stopped >= 0) {
                // a layer this card never published still holds every expert of its own, so only what really went
                // back may be called dropped
                for (size_t i = (size_t) p.stopped * NEXP; i < nidx; ++i) p.gone[i] = 0;
            }
            if (p.freed > cw.host_bytes) p.freed = cw.host_bytes;
            cw.host_bytes -= p.freed;
            cudaDeviceSynchronize();
            p.wrote = true;
        });
    for (int card = 0; card < NCARD; ++card) th[card].join();

    size_t freed = 0;
    int gone = 0, with_slot = 0, cards = 0;
    c.dropped_entry.assign(nidx, 0);
    for (size_t i = 0; i < nidx; ++i) {
        bool all_gone = true, all_slot = true;
        for (int card = 0; card < NCARD; ++card) {
            all_gone = all_gone && cd[card].gone[i];
            all_slot = all_slot && (m.cards[card].ec_slot0[i] >= 0);
        }
        if (all_gone) {
            c.dropped_entry[i] = 1;
            ++gone;
            if (all_slot) ++with_slot;
        }
    }
    for (int card = 0; card < NCARD; ++card)
        if (cd[card].wrote) { freed += cd[card].freed; ++cards; }
    c.drop_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    c.dropped = gone > 0 && cards == NCARD;
    c.freed = freed;
    c.dropped_experts = gone;
    std::fprintf(stderr,
                 "fnx: l2: drop: %d experts gave their RAM copies back: %.2f GiB off the host (%d of %d cards, %.2f s), "
                 "%d of them still with a first-tier slot: the arena is where their weights live now\n",
                 gone, (double) freed / GIB, cards, (int) NCARD, c.drop_seconds, with_slot);
    for (int card = 0; card < NCARD; ++card)
        std::fprintf(stderr, "fnx: l2: drop: card %d %s: %.2f GiB back, %.2f GiB of copies left\n", card,
                     cd[card].wrote ? "gave" : "kept", (double) cd[card].freed / GIB, (double) m.cards[card].host_bytes / GIB);
}

}  // namespace l2
}  // namespace fnx
