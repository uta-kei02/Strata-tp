// M2b-1: the prompt pass's routed experts on the tier card. l2_pf.hpp says what it is for; the numbers it replaces are
// the staging copies of prefill.cu (10449 MiB a card a pass, bench_8002.log:7189) against the pinned H2D the M2a probe
// measured (37.70 GB/s, strata-tp.log:615).
#include "l2_pf.hpp"

#include "l2_offload.hpp"
#include "model.hpp"
#include "kernels.hpp"
#include "../../tpopt/tp_opt.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fnx {
namespace l2 {
namespace pf {
namespace {

constexpr size_t MIB = 1u << 20;
constexpr int PF_PASS_MAX = 8;      // Engine::PF_PASS (engine.hpp:396): a pass has at most this many chunks
constexpr int PF_FILL = 256;        // prefill.cu's PF_FILL_THREADS: the tier fills its jobs the same way
constexpr int STEPS = NL * PF_PASS_MAX;   // a pass's largest step index (l * nk + j)

// one host -> tier copy: a card's slice of one expert, straight out of its page-locked block (the same pieces
// l2_offload.cu:242-249 fills the arena from)
struct Piece {
    const char* src;
    char* dst;
    size_t len;
};

struct S {
    bool want = false, ok = false;
    std::string why;
    Model* mp = nullptr;        // the engine's model, for the cards' devices and layer tables
    int dev = -1, max_rows = 0, ffs = 0, ring = 2;
    bool stream = true;         // the experts host RAM holds also go to the tier (FNX_L2_PF_STREAM)
    bool moe_q8 = false;        // FNX_PF_MOE_Q8: the cards' experts run q8_1, whose activation twin the tier has not got
    char* slot[2] = {};
    float* part = nullptr;      // [C * TOPK][N]
    float* gu = nullptr;        // [C * TOPK][2 * ffs]
    uint16_t* h16 = nullptr;    // [C * TOPK][ffs]
    char* xin[2] = {};          // a chunk's activation rows, f16 (two, by step parity)
    int* topi[2] = {};
    float* topw[2] = {};        // masked by the run table, so the reduce needs no mask of its own
    int* cnt = nullptr, *jbase = nullptr, *njobs = nullptr;
    BigJob* jobs_gu = nullptr;
    BigJob* jobs_dn = nullptr;
    float* vec = nullptr;       // [card][ring slot][C * N]: one card's addend for a step, on the tier
    float* l2_in[NCARD] = {};   // the same, on the card that sums it
    char** ep[NCARD][2][2] = {};    // device [NL * NEXP] per card and kind, one set per slot parity
    char** hep[NCARD][2][2] = {};   // pinned staging for those tables
    int* run_tier = nullptr;    // device, on the tier: [NL * NEXP], 1 = the tier runs this expert
    int* run_host = nullptr;    // mapped pinned host [NL * NEXP], 1 = the card runs it (the cards read it directly)
    std::vector<int> take;      // host [NL * NEXP], 1 = the tier runs it
    std::vector<long long> off; // its byte offset in slot[l & 1] (-1: already in the arena)
    std::vector<Piece> pieces[NL];
    bool ltake[NL] = {};
    int taken = 0, n_arena = 0, n_stream = 0;
    size_t stream_bytes = 0;
    size_t slot_bytes = 0;      // one streamed-expert slot (two of them)
    size_t xin_bytes = 0;       // one card's activation rows for a chunk, as f16 or q8_1
    cudaStream_t s5 = nullptr, sc = nullptr;
    cudaEvent_t ev_in[STEPS] = {}, ev_done[STEPS] = {}, ev_xread[STEPS] = {};
    cudaEvent_t ev_layer[2] = {};
    cudaEvent_t ev_fill[2] = {};  // one slot fill, by parity: recorded on sc, waited by s5 before the layer's GEMM
    cudaEvent_t ev_cons[NCARD][PF_PASS_MAX] = {};
    bool pass = false;
    int nk = 0, rows = 0;
};

S& g() {
    static S s;
    return s;
}

inline int blocks_for(size_t n, int threads = 256) {
    const size_t b = (n + (size_t) threads - 1) / (size_t) threads;
    return (int) (b > 65535u ? 65535u : b);
}

// pair p's expert count (the cards' k_pf_count, for the tier's own copy of topi)
__global__ void k_l2_count(const int* __restrict__ topi, int P, int* __restrict__ cnt) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < P; p += gridDim.x * blockDim.x) atomicAdd(&cnt[topi[p]], 1);
}
// the jobs, of the experts that run here only: an expert the tier does not own gets no job slot at all, so the blocks
// past *njobs return before they touch a weight (moe_mma.cuh:311, moe.cuh:468)
__global__ void k_l2_jobbase(const int* __restrict__ cnt, int* __restrict__ jbase, int* __restrict__ njobs,
                             const int* __restrict__ run) {
    __shared__ int s[NEXP];
    const int e = threadIdx.x;
    const int nj = (run && !run[e]) ? 0 : (cnt[e] + PF_JOB - 1) / PF_JOB;
    s[e] = nj;
    __syncthreads();
    for (int o = 1; o < NEXP; o <<= 1) {
        const int a = e >= o ? s[e - o] : 0;
        __syncthreads();
        s[e] += a;
        __syncthreads();
    }
    jbase[e] = s[e] - nj;
    if (e == NEXP - 1) njobs[0] = s[e];
}
// the same fill as the cards' k_pf_fill (same pair order, so a pair lands in the same job either way), stopped short
// of the experts that do not run here
__global__ void __launch_bounds__(PF_FILL) k_l2_fill(const int* __restrict__ topi, int P, const int* __restrict__ cnt,
                                                     const int* __restrict__ jbase, BigJob* __restrict__ gu,
                                                     BigJob* __restrict__ dn, const int* __restrict__ run) {
    const int e = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (cnt[e] == 0 || (run && !run[e])) return;
    __shared__ int wsum[PF_FILL / 32];
    int base = 0;
    for (int p0 = 0; p0 < P; p0 += PF_FILL) {
        const int p = p0 + tid;
        const bool mine = p < P && topi[p] == e;
        const unsigned m = __ballot_sync(0xffffffffu, mine);
        if (lane == 0) wsum[warp] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
#pragma unroll
        for (int w = 0; w < PF_FILL / 32; ++w) {
            before += w < warp ? wsum[w] : 0;
            total += wsum[w];
        }
        if (mine) {
            const int s = base + before + __popc(m & ((1u << lane) - 1u)), j = jbase[e] + s / PF_JOB, k = s % PF_JOB;
            gu[j].y[k] = p / TOPK;
            gu[j].out[k] = p;
            dn[j].y[k] = p;
            dn[j].out[k] = p;
            if (k == 0) {
                gu[j].w = dn[j].w = e;
                gu[j].n = dn[j].n = min(PF_JOB, cnt[e] - (s / PF_JOB) * PF_JOB);
            }
        }
        base += total;
        __syncthreads();
    }
}
// topw with the experts that do not run here zeroed: their pair rows of part stay whatever they were, and a zero
// weight makes them worth nothing (the same trick both sides use, so no pair is counted twice or missed)
__global__ void k_l2_topw_mask(const int* __restrict__ topi, const float* __restrict__ topw, float* __restrict__ out, int P,
                               const int* __restrict__ run) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < P; p += gridDim.x * blockDim.x)
        out[p] = run[topi[p]] ? topw[p] : 0.f;
}
// one card's addend: its own part rows, weighted. Rows of `vec` are written, never added to, so no clear is needed
__global__ void k_l2_reduce(const float* __restrict__ part, const int* __restrict__ topi, const float* __restrict__ topw,
                            float* __restrict__ vec, int C) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) C * N; e += (size_t) gridDim.x * blockDim.x) {
        const int t = (int) (e / N), i = (int) (e % N);
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int p = t * TOPK + k;
            // a pair the cards run is masked to 0 here and has no part row on this card: skipping it rather than
            // multiplying it keeps every finite sum bit-identical and never reads a row this card did not write
            const float wv = topw[p];
            if (wv != 0.f) s = fmaf(wv, part[((size_t) p) * N + i], s);
        }
        vec[e] = s;
    }
}

// The tier is a second device, and CUDA is strict about which one a handle belongs to: cudaEventRecord needs the
// event and the stream in the same CUDA context, so an event is made on the device whose stream records it and dies
// there too. Every entry point that switches to the tier switches back with this, on every path it can leave by.
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

void free_rows(S& s) {
    if (s.part) cudaFree(s.part);
    if (s.gu) cudaFree(s.gu);
    if (s.h16) cudaFree(s.h16);
    for (char*& p : s.xin) {
        if (p) cudaFree(p);
        p = nullptr;
    }
    for (int q = 0; q < 2; ++q) {
        if (s.topi[q]) cudaFree(s.topi[q]);
        s.topi[q] = nullptr;
        if (s.topw[q]) cudaFree(s.topw[q]);
        s.topw[q] = nullptr;
    }
    if (s.cnt) cudaFree(s.cnt);
    if (s.jbase) cudaFree(s.jbase);
    if (s.njobs) cudaFree(s.njobs);
    if (s.jobs_gu) cudaFree(s.jobs_gu);
    if (s.jobs_dn) cudaFree(s.jobs_dn);
    if (s.vec) cudaFree(s.vec);
    s.part = s.gu = nullptr;
    s.h16 = nullptr;
    s.cnt = s.jbase = s.njobs = nullptr;
    s.jobs_gu = s.jobs_dn = nullptr;
    s.vec = nullptr;
    for (int t = 0; t < NCARD; ++t) {   // the landing place of an addend, freed where it was made: on that card
        DevGuard on_card(s.mp ? s.mp->cards[(size_t) t].dev : s.dev);
        if (s.l2_in[t]) cudaFree(s.l2_in[t]);
        s.l2_in[t] = nullptr;
    }
}

void free_all(S& s) {
    DevGuard on_tier(s.dev);   // what is left here is the tier's: its rows, slots, tables, events and streams
    free_rows(s);
    for (char*& p : s.slot) {
        if (p) cudaFree(p);
        p = nullptr;
    }
    if (s.run_tier) cudaFree(s.run_tier);
    if (s.run_host) cudaFreeHost(s.run_host);
    s.run_tier = nullptr;
    s.run_host = nullptr;
    for (int t = 0; t < NCARD; ++t)
        for (int q = 0; q < 2; ++q)
            for (int r = 0; r < 2; ++r) {
                if (s.ep[t][q][r]) cudaFree(s.ep[t][q][r]);
                s.ep[t][q][r] = nullptr;
                if (s.hep[t][q][r]) cudaFreeHost(s.hep[t][q][r]);
                s.hep[t][q][r] = nullptr;
            }
    // an event dies where it was made: ev_in with the card that records it, ev_cons with the card that sums an
    // addend, the tier's own four with the tier. Destroying one from another device is not what keeps it correct.
    for (int t = 0; t < NCARD; ++t) {
        DevGuard on_card(s.mp ? s.mp->cards[(size_t) t].dev : s.dev);
        if (t == 0)
            for (int q = 0; q < STEPS; ++q) {
                if (s.ev_in[q]) cudaEventDestroy(s.ev_in[q]);
                s.ev_in[q] = nullptr;
            }
        for (int r = 0; r < PF_PASS_MAX; ++r) {
            if (s.ev_cons[t][r]) cudaEventDestroy(s.ev_cons[t][r]);
            s.ev_cons[t][r] = nullptr;
        }
    }
    for (cudaEvent_t* ev : {s.ev_done, s.ev_xread})
        for (int q = 0; q < STEPS; ++q) {
            if (ev[q]) cudaEventDestroy(ev[q]);
            ev[q] = nullptr;
        }
    for (cudaEvent_t& ev : s.ev_layer) {
        if (ev) cudaEventDestroy(ev);
        ev = nullptr;
    }
    for (cudaEvent_t& ev : s.ev_fill) {
        if (ev) cudaEventDestroy(ev);
        ev = nullptr;
    }
    if (s.s5) cudaStreamDestroy(s.s5);
    if (s.sc) cudaStreamDestroy(s.sc);
    s.s5 = s.sc = nullptr;
    s.slot_bytes = 0;
}

// the row-sized buffers, at most `rows` chunk rows; freed and tried again at half the size when the card says no
bool alloc_rows(S& s, size_t rows) {
    const size_t P = rows * TOPK;
    size_t got = 0;
    auto try_alloc = [&](void** p, size_t n) {
        if (cudaMalloc(p, n ? n : 1) != cudaSuccess) {
            *p = nullptr;
            return false;
        }
        got += n;
        return true;
    };
    s.xin_bytes = std::max(rows * N * sizeof(__half), rows * q8_1_bytes(N));
    bool all = try_alloc((void**) &s.part, P * N * sizeof(float)) &&
               try_alloc((void**) &s.gu, P * 2 * s.ffs * sizeof(float)) && try_alloc((void**) &s.h16, P * s.ffs * sizeof(__half)) &&
               try_alloc((void**) &s.xin[0], s.xin_bytes) && try_alloc((void**) &s.xin[1], s.xin_bytes) &&
               try_alloc((void**) &s.topi[0], P * sizeof(int)) && try_alloc((void**) &s.topi[1], P * sizeof(int)) &&
               try_alloc((void**) &s.topw[0], P * sizeof(float)) && try_alloc((void**) &s.topw[1], P * sizeof(float)) &&
               try_alloc((void**) &s.cnt, NEXP * sizeof(int)) && try_alloc((void**) &s.jbase, NEXP * sizeof(int)) &&
               try_alloc((void**) &s.njobs, sizeof(int)) && try_alloc((void**) &s.jobs_gu, (P / PF_JOB + NEXP) * sizeof(BigJob)) &&
               try_alloc((void**) &s.jobs_dn, (P / PF_JOB + NEXP) * sizeof(BigJob)) &&
               try_alloc((void**) &s.vec, (size_t) s.ring * NCARD * rows * N * sizeof(float));
    if (!all) return false;
    // the pair rows of an expert the tier does not run are read with a zero weight: zero them once, so 0 * stale is 0
    // and never whatever cudaMalloc left there (a NaN would reach the sum)
    if (cudaMemsetAsync(s.part, 0, P * N * sizeof(float), s.s5) != cudaSuccess) return false;
    for (int t = 0; t < NCARD; ++t) {
        DevGuard on_card(s.mp->cards[(size_t) t].dev);   // the addend landing place lives on the card that sums it
        if (!on_card.ok) return false;
        if (cudaMalloc((void**) &s.l2_in[t], (size_t) s.ring * rows * N * sizeof(float)) != cudaSuccess) {
            s.l2_in[t] = nullptr;
            return false;
        }
    }
    s.max_rows = (int) rows;
    return true;
}

}  // namespace

const Ctx& ctx() {
    static Ctx c;
    return c;
}

bool use() {
    const S& s = g();
    return s.ok;
}

void setup(Model& m, int max_rows) {
    Ctx& c = const_cast<Ctx&>(ctx());
    S& s = g();
    if (s.ok) return;
    const l2::Ctx& lc = l2::ctx();
    if (!lc.ok) return;   // no arena: M2b-1 has nothing to say, M2a already printed why the tier is out
    if (!m.cards[0].ec.n_slots) return;   // no expert slots to share the layer with: the cards run all of it
    try {
        s.want = env_int("FNX_L2_PF", 1) != 0;
        s.ring = (int) env_int("FNX_L2_PF_RING", 2);
        s.stream = env_int("FNX_L2_PF_STREAM", 1) != 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", e.what());
        return;
    }
    if (!s.want) return;
    if (max_rows < MAXT) return;
    if (s.ring < 1 || s.ring > PF_PASS_MAX) s.ring = 2;
    s.moe_q8 = env_flag("FNX_PF_MOE_Q8");   // the q8_1 activation path has a split twin the tier copy does not have
    const long long stage_mib = env_int("FNX_L2_PF_STAGE_MIB", 320);
    if (stage_mib < 0) {
        std::fprintf(stderr, "fnx: l2pf: off: FNX_L2_PF_STAGE_MIB: want MiB >= 0\n");
        return;
    }
    long long ask = max_rows;
    if (const long long cap = env_int("FNX_L2_PF_ROWS", max_rows); cap > 0 && cap < ask) ask = cap;

    s.dev = lc.dev;
    s.mp = &m;
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) prev = 0;
    auto off = [&](const char* why) {
        s.ok = false;
        s.why = why;
        cudaSetDevice(s.dev);
        free_all(s);
        cudaSetDevice(prev);
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", why);
    };
    if (cudaSetDevice(s.dev) != cudaSuccess) return off("cudaSetDevice on the tier failed");
    // one card's slice of a routed expert's intermediate width: what a gate|up row is once the shard is taken
    s.ffs = (int) (m.cards[1].ff1 - m.cards[1].ff0);
    if (s.ffs <= 0) return off("card 1 has no routed-expert slice (ff1 - ff0)");
    for (int l = 0; l < NL; ++l)
        if (m.cards[0].L[(size_t) l].gu_e.s.q || m.cards[0].L[(size_t) l].down_e.s.q)
            return off("its experts carry split Q8_0 scales, which the tier copy does not carry");
    if (s.moe_q8) return off("FNX_PF_MOE_Q8: the q8_1 expert path has no twin for the tier's activation copy");
    if (cudaStreamCreateWithFlags(&s.s5, cudaStreamNonBlocking) != cudaSuccess) return off("no stream for the tier");
    if (cudaStreamCreateWithFlags(&s.sc, cudaStreamNonBlocking) != cudaSuccess) return off("no copy stream for the tier");

    // the streamed experts' two slots: l2_offload sized them and took that out of the card's envelope before the arena
    // asked for anything (l2_offload.cu), so what stays unspent is the desktop's FNX_L2_RESERVE_GIB and not the tier's.
    // Half the ask until the card really gives it, because the tier must never be why a working server stopped; no
    // reservation at all means no streaming, and the arena's experts still go to the tier.
    if (s.stream && stage_mib > 0 && lc.stage_bytes > 0) {
        size_t want = lc.stage_bytes;
        while (want >= MIB && cudaMalloc((void**) &s.slot[0], want) != cudaSuccess) want /= 2;
        if (s.slot[0] && cudaMalloc((void**) &s.slot[1], want) == cudaSuccess) s.slot_bytes = want;
        else {
            if (s.slot[0]) cudaFree(s.slot[0]);
            s.slot[0] = nullptr;
            s.slot[1] = nullptr;
            s.stream = false;   // no room to stream: the arena's experts still go to the tier
        }
    } else
        s.stream = false;

    // the run tables: the tier's own in device memory, the cards' in mapped pinned RAM they can read from any card
    if (cudaMalloc((void**) &s.run_tier, (size_t) NL * NEXP * sizeof(int)) != cudaSuccess)
        return off("no room on the tier for its expert table");
    if (cudaHostAlloc((void**) &s.run_host, (size_t) NL * NEXP * sizeof(int), cudaHostAllocMapped) != cudaSuccess)
        return off("no pinned host memory for the cards' expert table");
    const size_t tbytes = (size_t) NL * NEXP * sizeof(char*);
    for (int t = 0; t < NCARD; ++t)
        for (int q = 0; q < 2; ++q)
            for (int r = 0; r < 2; ++r) {
                if (cudaMalloc((void**) &s.ep[t][q][r], tbytes) != cudaSuccess) return off("no room on the tier for its expert maps");
                if (cudaMallocHost((void**) &s.hep[t][q][r], tbytes) != cudaSuccess) return off("no pinned host memory for its expert maps");
            }
    // Where an event is made is decided by the stream that records it, not by who owns the work: cudaEventRecord wants
    // event and stream in the same CUDA context. ev_in is recorded by card 0's stream and ev_cons by the stream of the
    // card that sums the addend, so both are made there; ev_done / ev_xread / ev_layer / ev_fill go on the tier's own
    // streams and stay on the tier. Only waiting on an event crosses devices, which cudaStreamWaitEvent allows.
    for (int q = 0; q < STEPS; ++q) {
        if (cudaEventCreateWithFlags(&s.ev_done[q], cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
        if (cudaEventCreateWithFlags(&s.ev_xread[q], cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    }
    for (cudaEvent_t& ev : s.ev_layer)
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    for (cudaEvent_t& ev : s.ev_fill)
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    const char* ev_off = nullptr;
    for (int t = 0; t < NCARD && !ev_off; ++t) {
        DevGuard on_card(m.cards[(size_t) t].dev);
        if (!on_card.ok) ev_off = "a card would not take the events its own stream records";
        if (t == 0 && !ev_off)
            for (int q = 0; q < STEPS && !ev_off; ++q)
                if (cudaEventCreateWithFlags(&s.ev_in[q], cudaEventDisableTiming) != cudaSuccess) ev_off = "no events for the card that records them";
        for (int r = 0; r < s.ring && !ev_off; ++r)
            if (cudaEventCreateWithFlags(&s.ev_cons[t][r], cudaEventDisableTiming) != cudaSuccess) ev_off = "no events for the card that records them";
    }
    if (ev_off) return off(ev_off);   // after the guard: off() hands the caller its own device back

    bool got_rows = false;
    for (size_t rows = (size_t) ask; rows >= (size_t) MAXT; rows /= 2) {
        free_rows(s);
        if (alloc_rows(s, rows)) {
            got_rows = true;
            break;
        }
    }
    if (!got_rows) return off("no room left on the card after its arena for a chunk of its expert rows");
    cudaSetDevice(prev);

    c.ok = s.ok = true;
    c.dev = s.dev;
    c.max_rows = s.max_rows;
    c.ring = s.ring;
    c.stream = s.stream;
    c.slot_bytes = s.slot_bytes;
    c.xin_bytes = s.xin_bytes;
    c.why.clear();
    if (env_flag("FNX_PF_TIMING"))
        std::fprintf(stderr, "fnx: l2pf: card %d (%s): prompt path on, chunks of %d rows, %zu MiB streamed slots x2, %zu MiB of rows, ring %d\n",
                     s.dev, lc.name.c_str(), s.max_rows, s.slot_bytes >> 20, (size_t) (s.max_rows * N * 4) >> 20, s.ring);
}

bool take(size_t i) {
    const S& s = g();
    return s.pass && i < s.take.size() && s.take[i] != 0;
}

bool layer_taken(int l) {
    const S& s = g();
    return s.pass && l >= 0 && l < NL && s.ltake[l];
}

const int* run_card(int l) {
    const S& s = g();
    if (!s.pass || l < 0 || l >= NL || !s.ltake[l]) return nullptr;
    return s.run_host + (size_t) l * NEXP;
}

void wait_xread(int k, void* st) {
    const S& s = g();
    if (!s.pass || k < 1 || k >= STEPS || !s.ev_xread[k - 1]) return;
    // the tier's read event, waited by a card: cross-device is allowed here, and losing this wait would let the card
    // write over rows the tier has not read yet, which is a wrong number rather than a missing one
    if (cudaStreamWaitEvent((cudaStream_t) st, s.ev_xread[k - 1], 0) != cudaSuccess)
        throw std::runtime_error("l2pf: the card could not wait for the tier's read of its rows");
}

void record_in(int k, void* st_in) {
    S& s = g();
    if (!s.pass || k < 0 || k >= STEPS || !s.ev_in[k]) return;
    // recorded on the card's stream, which is the device ev_in[k] was made on (setup): an error here means the tier
    // would read rows it was never told were written, so say so at once instead of leaving it for the next launch
    if (cudaEventRecord(s.ev_in[k], (cudaStream_t) st_in) != cudaSuccess)
        throw std::runtime_error("l2pf: the card could not mark its rows ready for the tier");
}

bool plan(Model& m, int rows, int nk) {
    S& s = g();
    Ctx& c = const_cast<Ctx&>(ctx());
    if (!s.ok) return false;
    s.pass = false;
    if (nk < 1 || nk > PF_PASS_MAX || rows < MAXT || rows > s.max_rows) return false;
    const l2::Ctx& lc = l2::ctx();
    s.rows = rows, s.nk = nk;
    s.mp = &m;
    s.take.assign((size_t) NL * NEXP, 0);
    s.off.assign((size_t) NL * NEXP, -1);
    for (int l = 0; l < NL; ++l) s.pieces[(size_t) l].clear(), s.ltake[l] = false;

    // which experts the tier may run at all: every card has a page-locked copy of its slice and a pool that describes
    // it (l2_offload.cu:206-224, the same test that filled the arena). Everything else stays on the cards' tables
    // A streamed expert is only worth taking when both cards really hold its slice page-locked: the run table would
    // otherwise name a slot that no copy ever fills, and the GEMM would compute from whatever was left in it.
    auto in_host = [&m](size_t i) {
        for (int t = 0; t < NCARD; ++t) {
            const CardW& cw = m.cards[(size_t) t];
            if (i >= cw.cpu_host_gu.size() || i >= cw.cpu_host_dn.size() || !cw.cpu_host_gu[i] || !cw.cpu_host_dn[i]) return false;
        }
        return true;
    };
    std::vector<size_t> stride((size_t) NL, 0);
    for (int l = 0; l < NL; ++l) {
        size_t tot = 0;
        for (int t = 0; t < NCARD; ++t) {
            const CardW& cw = m.cards[(size_t) t];
            const int p = cw.ec.pool_of[l];
            if (p < 0 || p >= EC_POOLS || cw.ec.slot_bytes[p] <= cw.ec.dn_off[p]) {
                tot = 0;
                break;
            }
            tot += cw.ec.slot_bytes[p];
        }
        stride[(size_t) l] = tot;
        if (!tot) continue;
        const bool str_ok = s.stream && s.slot_bytes > 0;
        int k = 0;
        for (int e = 0; e < NEXP; ++e) {
            const size_t i = (size_t) l * NEXP + e;
            const bool resident = i < lc.at.size() && lc.at[i] >= 0;
            // The arena's experts cost no copy, so they always go. The ones host RAM still holds go too, in this same
            // expert order, while the layer's slot has room: the first expert that would run past the end of the slot
            // ends the layer's streamed set, and everything after it keeps take = 0. The cards copy and compute those
            // exactly as they did before, which is why a partly filled slot neither double-counts a pair nor drops one.
            if (!resident) {
                if (!str_ok || (size_t) (k + 1) * tot > s.slot_bytes) continue;
                if (!in_host(i)) continue;   // nothing to stream: the cards keep this one, so no hole appears
                s.off[i] = (long long) ((size_t) k++ * tot);
            }
            s.take[i] = 1;
        }
    }
    // A slot filled in order can never pass its own end, so this guard no longer fires on sizes alone. It stays for the
    // one case that can still come out uneven: the card would not hand over the reserved bytes, so setup halved them
    // after plan's predecessor sized the layer. Handing the surplus back to the cards is the only way to keep every
    // expert that stays there in the sum, and nothing has been dropped yet at this point in the pass.
    for (int l = 0; l < NL; ++l) {
        const size_t tot = stride[(size_t) l];
        if (!tot) continue;
        size_t n_res = 0, n_str = 0;
        for (int e = 0; e < NEXP; ++e) {
            const size_t i = (size_t) l * NEXP + e;
            if (!s.take[i]) continue;
            if (s.off[i] < 0) ++n_res;
            else ++n_str;
        }
        if (n_str * tot > s.slot_bytes)   // too many to stream: leave the whole layer to the cards
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                if (s.off[i] >= 0) {
                    s.take[i] = 0;
                    s.off[i] = -1;
                }
            }
        size_t left = 0;
        for (int e = 0; e < NEXP; ++e)
            if (s.take[(size_t) l * NEXP + e]) ++left;
        s.ltake[l] = left > 0;
        (void) n_res;
    }
    s.taken = s.n_arena = s.n_stream = 0;
    s.stream_bytes = 0;
    for (int l = 0; l < NL; ++l) {
        if (!s.ltake[l]) continue;
        const size_t tot = stride[(size_t) l];
        for (int e = 0; e < NEXP; ++e) {
            const size_t i = (size_t) l * NEXP + e;
            if (!s.take[i]) continue;
            ++s.taken;
            if (s.off[i] < 0) ++s.n_arena;
            else {
                ++s.n_stream;
                s.stream_bytes += tot;
            }
        }
    }
    if (!s.taken) return false;

    // the run tables: the tier runs what take says, the cards run everything else
    std::vector<int> ht((size_t) NL * NEXP), hc((size_t) NL * NEXP);
    for (size_t i = 0; i < s.take.size(); ++i) ht[i] = s.take[i], hc[i] = s.take[i] ? 0 : 1;
    DevGuard on_tier(s.dev);   // every way out of plan(), this function's last line included, gives the device back
    if (!on_tier.ok) return false;
    bool bad = cudaMemcpy(s.run_tier, ht.data(), ht.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess;
    std::memcpy(s.run_host, hc.data(), hc.size() * sizeof(int));
    for (int par = 0; par < 2 && !bad; ++par)
        for (int t = 0; t < NCARD && !bad; ++t) {
            const CardW& cw = m.cards[(size_t) t];
            for (int q = 0; q < 2; ++q) std::memset(s.hep[t][q][par], 0, (size_t) NL * NEXP * sizeof(char*));
            for (int l = 0; l < NL; ++l) {
                if ((l & 1) != par || !s.ltake[l]) continue;
                const int p = cw.ec.pool_of[l];
                const size_t gs = cw.ec.dn_off[p];
                size_t pre = 0;
                for (int u = 0; u < t; ++u) {
                    const CardW& ow = m.cards[(size_t) u];
                    pre += ow.ec.slot_bytes[ow.ec.pool_of[l]];
                }
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    if (!s.take[i]) continue;
                    char* at = s.off[i] < 0 ? lc.arena + lc.at[i] : s.slot[par] + s.off[i];
                    s.hep[t][0][par][i] = at + pre;
                    s.hep[t][1][par][i] = at + pre + gs;
                }
            }
            for (int q = 0; q < 2 && !bad; ++q)
                if (cudaMemcpy(s.ep[t][q][par], s.hep[t][q][par], (size_t) NL * NEXP * sizeof(char*), cudaMemcpyHostToDevice) != cudaSuccess)
                    bad = true;
        }
    if (bad) {
        std::fprintf(stderr, "fnx: l2pf: off: the pass's tables would not upload\n");
        s.ok = false;
        return false;
    }

    // the streamed experts of each layer, as the copies that fill its slot
    int unfilled = 0;
    for (int l = 0; l < NL; ++l) {
        if (!s.ltake[l]) continue;
        std::vector<Piece>& pv = s.pieces[(size_t) l];
        for (int e = 0; e < NEXP; ++e) {
            const size_t i = (size_t) l * NEXP + e;
            if (!s.take[i] || s.off[i] < 0) continue;
            for (int t = 0; t < NCARD; ++t) {
                const CardW& cw = m.cards[(size_t) t];
                const int p = cw.ec.pool_of[l];
                const size_t gs = cw.ec.dn_off[p], ds = cw.ec.slot_bytes[p] - gs;
                size_t pre = 0;
                for (int u = 0; u < t; ++u) {
                    const CardW& ow = m.cards[(size_t) u];
                    pre += ow.ec.slot_bytes[ow.ec.pool_of[l]];
                }
                char* dst = s.slot[l & 1] + s.off[i] + pre;
                const char* gu = i < cw.cpu_host_gu.size() ? cw.cpu_host_gu[i] : nullptr;
                const char* dn = i < cw.cpu_host_dn.size() ? cw.cpu_host_dn[i] : nullptr;
                if (!gu || !dn) { ++unfilled; continue; }
                if (dn == gu + gs) pv.push_back({gu, dst, gs + ds});   // one block [gate|up | down], as model.cpp wrote it
                else {
                    pv.push_back({gu, dst, gs});
                    pv.push_back({dn, dst + gs, ds});
                }
            }
        }
    }
    // take means a copy: if one streamed expert lost its page-locked pair between the take and here, its table entry
    // would point at bytes nobody wrote, so the pass stays on the cards instead (nothing has run yet to lose).
    if (unfilled) {
        std::fprintf(stderr, "fnx: l2pf: off: %d streamed experts have no page-locked copy\n", unfilled);
        return false;
    }

    s.pass = true;
    c.taken = s.taken;
    c.n_arena = s.n_arena;
    c.n_stream = s.n_stream;
    c.stream_bytes = s.stream_bytes;
    c.slot_bytes = s.slot_bytes;
    c.xin_bytes = s.xin_bytes;
    c.stream = s.stream;
    c.ring = s.ring;
    ++c.passes;
    if (env_flag("FNX_PF_TIMING"))
        std::fprintf(stderr,
                     "fnx: l2pf: pass %d of %d rows x %d chunks: %d experts to card %d (%d resident, %d streamed %zu MiB into %zu MiB x2, "
                     "ring %d)\n",
                     c.passes, rows, nk, s.taken, s.dev, s.n_arena, s.n_stream, s.stream_bytes >> 20, s.slot_bytes >> 20, s.ring);
    return true;
}

// A layer whose streamed experts would not fill their slot is safe to hand back: run_card(l) is read when the layer's
// chunk runs (prefill.cu), which is after this call, so pass = false makes the cards mask nothing and compute all of
// it again. Nothing this layer dropped is lost, which is why layer_in bails and step does not.
void layer_in(int l) {
    S& s = g();
    if (!s.pass || l < 0 || l >= NL || !s.ltake[l] || s.pieces[(size_t) l].empty()) return;
    tpopt::WaitGuard guard("the expert tier's slot fills");
    DevGuard on_tier(s.dev);
    if (!on_tier.ok) {
        s.pass = false;
        s.why = "cudaSetDevice on the tier failed in a slot fill";
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", s.why.c_str());
        return;
    }
    // slot l & 1 is free once the layer two before has read it; an event nothing recorded is no wait at all
    if (s.ev_layer[l & 1]) cudaStreamWaitEvent(s.sc, s.ev_layer[l & 1], 0);
    bool undelivered = false;
    for (const Piece& pc : s.pieces[(size_t) l])
        if (cudaMemcpyAsync(pc.dst, pc.src, pc.len, cudaMemcpyHostToDevice, s.sc) != cudaSuccess) {
            std::fprintf(stderr, "fnx: l2pf: off: layer %d slot fill failed\n", l);
            s.pass = false;
            s.why = "a slot fill failed";
            undelivered = true;
            break;
        }
    // and the fill has to be over before the layer's GEMM reads the slot. s.sc and s.s5 are both non-blocking streams,
    // so nothing but this event puts the copy ahead of the launch: without it the kernel reads bytes that are still on
    // their way, and the card's fp16 range check reports what it finds as `prefill: ring error 2`. Bailing here is
    // safe the same way a short fill is - run_card(l) is read later, so the cards take the whole layer back.
    if (!undelivered && s.ev_fill[l & 1] && cudaEventRecord(s.ev_fill[l & 1], s.sc) != cudaSuccess) {
        s.pass = false;
        s.why = "a slot fill would not mark itself done";
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", s.why.c_str());
    }
}

// What a tier-side CUDA failure costs, and what it may not: run_card() goes null with s.pass, so the layers that have
// not run yet mask nothing and the cards compute every expert of theirs again. That hand-back is only safe while the
// drop itself has not happened, which is true of layer_in and false of step: by the time a step runs, both cards have
// already built this chunk's jobs with the tier's table and left the tier's experts out of them. Continuing there
// would leave those rows out of the sum with nothing said about it, so a failed step turns the tier off behind it and
// throws instead. DevGuard puts the caller's device back whichever way this function leaves.
void step(int l, int j, int C, int k, const void* xin, const int* topi, const float* topw) {
    S& s = g();
    if (!s.pass || !s.ltake[l] || C < 1 || C > s.max_rows || k < 0 || k >= STEPS) return;
    auto lost = [&](const char* why) {
        s.pass = false;
        s.why = why;
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", why);
        throw std::runtime_error(std::string("l2pf: a step lost work: ") + why);
    };
    const int par = l & 1, sp = k % s.ring, P = C * TOPK;
    Model& m = *s.mp;
    tpopt::WaitGuard guard("the expert tier's step");
    DevGuard on_tier(s.dev);
    if (!on_tier.ok) lost("cudaSetDevice on the tier failed in a step");
    // streamed weights come first: their copies went out on sc, and every launch below reads them from s5
    if (s.ev_fill[par] && !s.pieces[(size_t) l].empty() && cudaStreamWaitEvent(s.s5, s.ev_fill[par], 0) != cudaSuccess)
        lost("a step would not wait for its layer's slot fill");
    const size_t in_bytes = (size_t) C * N * sizeof(__half);   // the chunk's rows, not the buffer's whole size
    bool bad = false;
    if (s.ev_in[k] && cudaStreamWaitEvent(s.s5, s.ev_in[k], 0) != cudaSuccess) bad = true;   // card 0's rows are ready
    if (cudaMemcpyAsync(s.xin[k & 1], xin, in_bytes, cudaMemcpyDefault, s.s5) != cudaSuccess) bad = true;
    if (!bad && cudaMemcpyAsync(s.topi[k & 1], topi, (size_t) P * sizeof(int), cudaMemcpyDefault, s.s5) != cudaSuccess)
        bad = true;
    if (!bad && cudaMemcpyAsync(s.topw[k & 1], topw, (size_t) P * sizeof(float), cudaMemcpyDefault, s.s5) != cudaSuccess)
        bad = true;
    // the cards may overwrite their activation rows as soon as this copy is off their memory
    if (!bad && s.ev_xread[k] && cudaEventRecord(s.ev_xread[k], s.s5) != cudaSuccess) bad = true;
    if (bad) lost("a step could not copy the chunk to the tier");
    const int* run = s.run_tier + (size_t) l * NEXP;
    try {
        for (int t = 0; t < NCARD; ++t) {
            cudaMemsetAsync(s.cnt, 0, NEXP * sizeof(int), s.s5);
            k_l2_count<<<blocks_for((size_t) P), 256, 0, s.s5>>>(s.topi[k & 1], P, s.cnt);
            k_l2_jobbase<<<1, NEXP, 0, s.s5>>>(s.cnt, s.jbase, s.njobs, run);
            k_l2_fill<<<NEXP, PF_FILL, 0, s.s5>>>(s.topi[k & 1], P, s.cnt, s.jbase, s.jobs_gu, s.jobs_dn, run);
            k_l2_topw_mask<<<blocks_for((size_t) P), 256, 0, s.s5>>>(s.topi[k & 1], s.topw[k & 1], s.topw[k & 1], P, run);
            const LayerW& L = m.cards[(size_t) t].L[(size_t) l];
            DW gu_e = L.gu_e, dn_e = L.down_e;
            gu_e.ep = s.ep[t][0][par] + (size_t) l * NEXP;
            dn_e.ep = s.ep[t][1][par] + (size_t) l * NEXP;
            const int cap = P / PF_JOB + NEXP;
            moe_gate_up_mma(gu_e, L.gu_e_stride, s.jobs_gu, s.njobs, cap, reinterpret_cast<const uint16_t*>(s.xin[k & 1]), s.gu, s.s5);
            swiglu_h16(s.gu, s.ffs, P, s.h16, s.s5);
            moe_down_mma(dn_e, L.down_e_stride, s.jobs_dn, s.njobs, cap, s.h16, s.part, s.s5);
            // the slot's earlier addend must be off the card before this one lands on it
            if (s.ev_cons[t][sp] && cudaStreamWaitEvent(s.s5, s.ev_cons[t][sp], 0) != cudaSuccess) bad = true;
            k_l2_reduce<<<blocks_for((size_t) C * N), 256, 0, s.s5>>>(s.part, s.topi[k & 1], s.topw[k & 1],
                                                                     s.vec + ((size_t) t * s.ring + sp) * (size_t) C * N, C);
        }
    } catch (const std::exception& e) {
        s.pass = false;
        s.why = e.what();
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", e.what());
        throw;
    }
    if (bad) lost("a step could not wait for a card to leave its addend slot");
    if (j == s.nk - 1 && s.ev_layer[par] && cudaEventRecord(s.ev_layer[par], s.s5) != cudaSuccess) lost("a step could not close a layer slot");
    if (s.ev_done[k] && cudaEventRecord(s.ev_done[k], s.s5) != cudaSuccess) lost("a step could not mark its addend ready");
}


float* collect(int card, int l, int k, int C, void* st) {
    S& s = g();
    if (!s.pass || !s.ltake[l] || card < 0 || card >= NCARD || C < 1 || C > s.max_rows || k < 0 || k >= STEPS) return nullptr;
    if (!s.l2_in[card] || !s.ev_done[k]) return nullptr;
    const int sp = k % s.ring;
    cudaStream_t stn = (cudaStream_t) st;
    float* dst = s.l2_in[card] + (size_t) sp * (size_t) C * N;
    tpopt::WaitGuard guard("the expert tier's addend");
    // Past the two nulls above a null is not "this layer stayed on the cards": its experts are already off the card's
    // tables, so the sum would just miss them. Hence errors go up rather than becoming a quiet missing addend.
    // Nothing here switches device: waiting ev_done (the tier's) from a card stream is what cudaStreamWaitEvent is for,
    // and ev_cons[card] was made on this card, whose stream records it.
    auto gone = [&](const char* why) {
        s.pass = false;
        s.why = why;
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", why);
        throw std::runtime_error(std::string("l2pf: the tier's addend ") + why);
    };
    if (cudaStreamWaitEvent(stn, s.ev_done[k], 0) != cudaSuccess) gone("would not wait for the card");
    if (cudaMemcpyAsync(dst, s.vec + ((size_t) card * s.ring + sp) * (size_t) C * N, (size_t) C * N * sizeof(float), cudaMemcpyDefault, stn) !=
        cudaSuccess)
        gone("would not copy to the card");
    if (s.ev_cons[card][sp] && cudaEventRecord(s.ev_cons[card][sp], stn) != cudaSuccess) gone("would not mark its slot as read");
    return dst;
}

}  // namespace pf
}  // namespace l2
}  // namespace fnx
