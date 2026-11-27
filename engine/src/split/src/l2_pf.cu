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
// how many streamed-expert slots one pass may overlap (FNX_L2_PF_PIPE). 2 is the rev.15-M2b-1 shape: fill layer l+1
// while layer l computes, so the link and the GEMM never meet. 3 or 4 keep two or three fills in flight, which is
// what the layer's whole 281 MiB needs (l2_offload reserves one envelope for all of them, l2_offload.cu:245)
constexpr int PF_PIPE_MAX = 4;
// how many chunk-row buffers the tier keeps (FNX_L2_PF_XRING). A rows copy is 21 MiB for a 2048-row chunk, and left
// on s5 it cannot start while the step before it computes; on its own lane (sx) it needs a slot of its own to land in
// while the GEMM still reads the previous one. 2 is the rev.15-M2b-1u shape, 3 lets one copy run a whole step ahead
constexpr int XRING_MAX = 4;

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
    int dev = -1, max_rows = 0, ffs = 0, ring = 2, npipe = 2, xring = 2;
    bool stream = true;         // the experts host RAM holds also go to the tier (FNX_L2_PF_STREAM)
    bool auto_slots = true;     // FNX_L2_PF_AUTO: size the slots by what a layer really streams, rather than by npipe alone
    bool pull = true;           // FNX_L2_PF_PULL: the addend travels to the card on a stream of its own
    bool xcopy = true;          // FNX_L2_PF_XCOPY: a chunk's rows travel on a lane of their own, ahead of the GEMMs
    bool timing = false;        // FNX_L2_PF_TIMING: the tier's own breakdown, one line a pass
    bool add_h = true;          // FNX_L2_PF_FP16: an addend comes back as f16 rows, half the bytes over the host relay
    int add_bytes = 4;          // one addend row in bytes: 2 with the above, 4 with rev.15-M2b-1v's f32 rows
    bool early_in = true;       // FNX_L2_PF_EARLY_IN: card 0 marks its rows ready before its router runs
    int dst_ring = 2;           // landing slots per card (>= ring), so a pull does not queue behind the sum it follows
    bool moe_q8 = false;        // FNX_PF_MOE_Q8: the cards' experts run q8_1, whose activation twin the tier has not got
    char* slot[PF_PIPE_MAX] = {};
    char* env = nullptr;        // the envelope those slots are one cut out of (l2_offload.cu:238-259 sized it)
    size_t env_bytes = 0;
    float* part = nullptr;      // [C * TOPK][N]
    float* gu = nullptr;        // [C * TOPK][2 * ffs]
    uint16_t* h16 = nullptr;    // [C * TOPK][ffs]
    char* xin[XRING_MAX] = {};  // a chunk's activation rows, f16 (s.xring of them, by step modulo)
    int* topi[XRING_MAX] = {};
    float* topw[XRING_MAX] = {};  // masked by the run table, so the reduce needs no mask of its own
    int prev_read[XRING_MAX] = {};  // the last step that read a rows slot: the lane waits for exactly that one, so a
                                    // layer that stayed on the cards cannot leave the wait naming a stale slot
    int* cnt = nullptr, *jbase = nullptr, *njobs = nullptr;
    BigJob* jobs_gu = nullptr;
    BigJob* jobs_dn = nullptr;
    char* vec = nullptr;        // [card][ring slot][C * N * add_bytes]: one card's addend for a step, on the tier
    char* l2_in[NCARD] = {};    // the same, on the card that sums it (f16 rows with FNX_L2_PF_FP16, f32 rows without)
    char** ep[NCARD][2][PF_PIPE_MAX] = {};   // device [NL * NEXP] per card and kind, one set per streamed slot
    char** hep[NCARD][2][PF_PIPE_MAX] = {};  // pinned staging for those tables
    int* run_tier = nullptr;    // device, on the tier: [NL * NEXP], 1 = the tier runs this expert
    int* run_host = nullptr;    // mapped pinned host [NL * NEXP], 1 = the card runs it (the cards read it directly)
    std::vector<int> take;      // host [NL * NEXP], 1 = the tier runs it
    std::vector<long long> off; // its byte offset in slot[l % npipe] (-1: already in the arena)
    std::vector<Piece> pieces[NL];
    bool ltake[NL] = {};
    int taken = 0, n_arena = 0, n_stream = 0;
    size_t stream_bytes = 0;
    size_t slot_bytes = 0;      // one streamed-expert slot (s.npipe of them)
    size_t xin_bytes = 0;       // one card's activation rows for a chunk, as f16 or q8_1
    cudaStream_t s5 = nullptr, sc = nullptr, sx = nullptr;
    cudaStream_t pull_st[NCARD] = {};   // the card's own addend lane: made and destroyed on that card, never on the tier
    cudaEvent_t ev_in[STEPS] = {}, ev_xread[STEPS] = {};
    cudaEvent_t ev_pair[STEPS] = {};     // card 0's top-k pairs are ready: the rows lane waits it before copying them
    cudaEvent_t ev_done[NCARD][STEPS] = {};  // one per card: its own shard's addend is ready before the other's GEMM ran
    cudaEvent_t ev_rows[XRING_MAX] = {};     // rows landed on the tier: the lane records it, the GEMM stream waits for it
    cudaEvent_t ev_layer[PF_PIPE_MAX] = {};   // one slot read, by slot: recorded on s5, waited by sc before that slot is filled again
    cudaEvent_t ev_fill[PF_PIPE_MAX] = {};    // one slot fill, by slot: recorded on sc, waited by s5 before the layer's GEMM
    cudaEvent_t ev_cons[NCARD][PF_PASS_MAX] = {};
    cudaEvent_t ev_pull[NCARD][PF_PASS_MAX] = {};   // an addend has landed in a card's landing slot (the pull lane records it)
    cudaEvent_t ev_read[NCARD][PF_PASS_MAX] = {};   // ... and the card's sum has read it again (that card records it)
    bool pass = false;
    int nk = 0, rows = 0;
    int filled_to = -1;         // the deepest layer whose slot fill this pass has handed to sc (layer_in runs each one once)
    // FNX_L2_PF_TIMING only: four marks per step (tm_a / tm_b on whatever lane carries the rows, tm_d / tm_c on the
    // GEMM stream), two per fill on sc, two per step on the lane that carries an
    // addend to a card. They are made with timing (a timing event costs a little, a disabled one does not answer an
    // elapsed query), and only the ones this pass recorded count (tm_step / tm_fill), so a layer that stayed on the
    // cards cannot feed the totals with an older pass's pair.
    cudaEvent_t tm_a[STEPS] = {}, tm_b[STEPS] = {}, tm_c[STEPS] = {}, tm_d[STEPS] = {};
    cudaEvent_t tm_f0[NL] = {}, tm_f1[NL] = {};
    cudaEvent_t tm_q0[NCARD][STEPS] = {}, tm_q1[NCARD][STEPS] = {};
    std::vector<unsigned char> tm_step, tm_fill;
    int tm_pass = 0;            // the pass number those marks belong to, so one pass reports exactly once
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
template <bool H16>   // rev.15-M2b-1w: f16 rows halve what a card has to take back over its link (FNX_L2_PF_FP16)
__global__ void k_l2_reduce(const float* __restrict__ part, const int* __restrict__ topi, const float* __restrict__ topw,
                            void* __restrict__ vec, int C) {
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
        // rows of `vec` are written, never added to, so no clear is needed; f16 is one rounding of a sum the card
        // rounds to f16 again when it packs its side of the AllReduce (prefill.cu's k_pfr_pack), and a weight that
        // large would already have packed to inf on its own
        if constexpr (H16) reinterpret_cast<__half*>(vec)[e] = __float2half_rn(s);
        else reinterpret_cast<float*>(vec)[e] = s;
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
    for (int q = 0; q < XRING_MAX; ++q) {
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
    if (s.env) cudaFree(s.env);
    s.env = nullptr;
    s.env_bytes = 0;
    if (s.run_tier) cudaFree(s.run_tier);
    if (s.run_host) cudaFreeHost(s.run_host);
    s.run_tier = nullptr;
    s.run_host = nullptr;
    for (int t = 0; t < NCARD; ++t)
        for (int q = 0; q < 2; ++q)
            for (int r = 0; r < PF_PIPE_MAX; ++r) {
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
                if (s.ev_pair[q]) cudaEventDestroy(s.ev_pair[q]);   // made here too, so it dies here too
                s.ev_pair[q] = nullptr;
            }
        for (int r = 0; r < PF_PASS_MAX; ++r) {
            if (s.ev_cons[t][r]) cudaEventDestroy(s.ev_cons[t][r]);
            s.ev_cons[t][r] = nullptr;
            if (s.ev_pull[t][r]) cudaEventDestroy(s.ev_pull[t][r]);
            s.ev_pull[t][r] = nullptr;
            if (s.ev_read[t][r]) cudaEventDestroy(s.ev_read[t][r]);
            s.ev_read[t][r] = nullptr;
        }
        for (int q = 0; q < STEPS; ++q) {
            if (s.tm_q0[t][q]) cudaEventDestroy(s.tm_q0[t][q]);
            s.tm_q0[t][q] = nullptr;
            if (s.tm_q1[t][q]) cudaEventDestroy(s.tm_q1[t][q]);
            s.tm_q1[t][q] = nullptr;
        }
        if (s.pull_st[t]) cudaStreamDestroy(s.pull_st[t]);
        s.pull_st[t] = nullptr;
    }
    for (int t = 0; t < NCARD; ++t)
        for (int q = 0; q < STEPS; ++q) {
            if (s.ev_done[t][q]) cudaEventDestroy(s.ev_done[t][q]);
            s.ev_done[t][q] = nullptr;
        }
    for (cudaEvent_t* ev : {s.ev_xread})
        for (int q = 0; q < STEPS; ++q) {
            if (ev[q]) cudaEventDestroy(ev[q]);
            ev[q] = nullptr;
        }
    for (cudaEvent_t& ev : s.ev_rows) {
        if (ev) cudaEventDestroy(ev);
        ev = nullptr;
    }
    for (cudaEvent_t& ev : s.ev_layer) {
        if (ev) cudaEventDestroy(ev);
        ev = nullptr;
    }
    for (cudaEvent_t& ev : s.ev_fill) {
        if (ev) cudaEventDestroy(ev);
        ev = nullptr;
    }
    for (cudaEvent_t* ev : {s.tm_a, s.tm_b, s.tm_c, s.tm_d})
        for (int q = 0; q < STEPS; ++q) {
            if (ev[q]) cudaEventDestroy(ev[q]);
            ev[q] = nullptr;
        }
    for (cudaEvent_t* ev : {s.tm_f0, s.tm_f1})
        for (int q = 0; q < NL; ++q) {
            if (ev[q]) cudaEventDestroy(ev[q]);
            ev[q] = nullptr;
        }
    if (s.s5) cudaStreamDestroy(s.s5);
    if (s.sc) cudaStreamDestroy(s.sc);
    if (s.sx) cudaStreamDestroy(s.sx);
    s.s5 = s.sc = s.sx = nullptr;
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
    // the rows slots: s.xring of them, so the lane may keep one copy landing while the GEMM reads the previous one.
    // setup narrows s.xring before it narrows the chunk, so a tier with little room left keeps the lane rather than
    // losing the tier
    auto try_xrows = [&]() {
        for (int q = 0; q < s.xring; ++q)
            if (!try_alloc((void**) &s.xin[q], s.xin_bytes)) return false;
        for (int q = 0; q < s.xring; ++q)
            if (!try_alloc((void**) &s.topi[q], P * sizeof(int))) return false;
        for (int q = 0; q < s.xring; ++q)
            if (!try_alloc((void**) &s.topw[q], P * sizeof(float))) return false;
        return true;
    };
    bool all = try_alloc((void**) &s.part, P * N * sizeof(float)) &&
               try_alloc((void**) &s.gu, P * 2 * s.ffs * sizeof(float)) && try_alloc((void**) &s.h16, P * s.ffs * sizeof(__half)) &&
               try_xrows() &&
               try_alloc((void**) &s.cnt, NEXP * sizeof(int)) && try_alloc((void**) &s.jbase, NEXP * sizeof(int)) &&
               try_alloc((void**) &s.njobs, sizeof(int)) && try_alloc((void**) &s.jobs_gu, (P / PF_JOB + NEXP) * sizeof(BigJob)) &&
               try_alloc((void**) &s.jobs_dn, (P / PF_JOB + NEXP) * sizeof(BigJob)) &&
               try_alloc((void**) &s.vec, (size_t) s.ring * NCARD * rows * N * (size_t) s.add_bytes);
    if (!all) return false;
    // the pair rows of an expert the tier does not run are read with a zero weight: zero them once, so 0 * stale is 0
    // and never whatever cudaMalloc left there (a NaN would reach the sum)
    if (cudaMemsetAsync(s.part, 0, P * N * sizeof(float), s.s5) != cudaSuccess) return false;
    for (int t = 0; t < NCARD; ++t) {
        DevGuard on_card(s.mp->cards[(size_t) t].dev);   // the addend landing place lives on the card that sums it
        if (!on_card.ok) return false;
        // dst_ring, not ring: with FNX_L2_PF_PULL the lane that carries an addend runs ahead of the card's sum, and the
        // slots are what lets it (collect waits ev_read before it reuses one, so the depth only buys slack)
        if (cudaMalloc((void**) &s.l2_in[t], (size_t) s.dst_ring * rows * N * (size_t) s.add_bytes) != cudaSuccess) {
            s.l2_in[t] = nullptr;
            return false;
        }
    }
    s.max_rows = (int) rows;
    return true;
}

// How the streamed-expert slots are cut. One layer's streamed experts all live in one slot, so a slot has to be as
// wide as the widest layer a pass wants to stream; plan counts that (need_max, in the same expert order it fills the
// slot with) before it cuts anything, and calls this with it. The envelope is the one envelope l2_offload reserved
// for streaming and took out of the card's budget before the arena asked for a byte (l2_offload.cu:238-259, "two
// slots, with 2 GiB of arena still standing"), so cutting the same envelope into three or four slots spends nothing
// the desktop has not already given up, and a pass whose layer fits into one slot copies nothing up the link at all
// (the "copied in" line of the pass says when it does).
//
// Only when even two slots of that envelope are too narrow does the envelope itself grow, and then only out of the
// memory free above FNX_L2_RESERVE_GIB (l2_offload.hpp's reserve, never below 2 GiB): the wider one is taken before
// the old one goes, so the reserve is measured against a tier that still holds both, never against a half-freed one.
// A card that will not give it keeps the slots as they are and plan hands the tail of that layer back to the cards,
// which is the same short fill a too-small slot always was. The caller's device comes back either way.
static void split_slots(S& s, size_t need_max) {
    if (!s.env || s.env_bytes < 2 * MIB) return;
    tpopt::WaitGuard guard("the expert tier's slots");
    DevGuard on_tier(s.dev);
    if (!on_tier.ok) return;
    auto cut = [&](int n) { return s.env_bytes / (size_t) n / MIB * MIB; };
    int want = s.npipe;
    if (s.auto_slots) {
        for (int n = s.npipe; n >= 2; --n)
            if (cut(n) >= need_max) {
                want = n;
                break;
            }
    }
    if (s.auto_slots && need_max > cut(want)) {   // the envelope is the narrow thing, not the cut: widen it, carefully
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            const double r = l2::ctx().reserve_gib > 2.0 ? l2::ctx().reserve_gib : 2.0;
            const size_t keep = (size_t) (r * 1024.0 * 1024.0 * 1024.0);
            const size_t room = free_b > keep ? free_b - keep : 0;
            const size_t ask = 2 * need_max > s.env_bytes ? 2 * need_max - s.env_bytes : 0;
            const size_t grow = ask < room ? ask : room;
            if (grow >= MIB) {
                char* bigger = nullptr;
                if (cudaMalloc((void**) &bigger, s.env_bytes + grow) == cudaSuccess) {
                    cudaStreamSynchronize(s.s5);
                    cudaStreamSynchronize(s.sc);
                    cudaFree(s.env);
                    s.env = bigger;
                    s.env_bytes += grow;
                    for (int n = s.npipe; n >= 2; --n)
                        if (cut(n) >= need_max) {
                            want = n;
                            break;
                        }
                }
            }
        }
    }
    const size_t slot = cut(want);
    if (!slot || (slot == s.slot_bytes && want == s.npipe && s.slot[0] == s.env)) return;   // nothing moved
    // nothing of this pass has run yet, and the pass before it drained (prefill waits every card stream, and every
    // tier answer is on one); these two syncs are what makes that fact enough to free the old envelope
    cudaStreamSynchronize(s.s5);
    cudaStreamSynchronize(s.sc);
    for (int q = 0; q < PF_PIPE_MAX; ++q) s.slot[q] = (q < want ? s.env + (size_t) q * slot : nullptr);
    s.slot_bytes = slot;
    s.npipe = want;
    if (s.timing)
        std::fprintf(stderr, "fnx: l2pf: slots: a layer streams %zu MiB, so %zu MiB x %d of %zu MiB, reserve kept\n",
                     need_max >> 20, slot >> 20, want, s.env_bytes >> 20);
}

// FNX_L2_PF_TIMING only: add one span to a total when this pass really recorded both marks of it. An event pair left
// over from an earlier pass is not this pass's time, so `live` says which pairs are; nothing is guessed at here.
static void span_ms(cudaEvent_t a, cudaEvent_t b, bool live, double* into) {
    if (!live || !a || !b) return;
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, a, b) != cudaSuccess) return;
    if (!(ms >= 0.f) || ms > 1e7f) return;
    *into += (double) ms;
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
        // how many streamed slots one pass may overlap; 0 (or 1) asks for the rev.15-M2b-1 shape back, two slots and
        // one layer of lookahead, which is the switch a pipelined run is compared against
        s.npipe = (int) env_int("FNX_L2_PF_PIPE", PF_PIPE_MAX);
        // how the slot envelope is cut (split_slots): 1 sizes it by the widest layer of the pass, 0 keeps the fixed
        // npipe-way split the first rev.15-M2b-1 runs used. And whether an addend travels to a card on a lane of its
        // own (collect), 0 putting it back on the card's stream where the card's own work has to wait for it
        s.auto_slots = env_int("FNX_L2_PF_AUTO", 1) != 0;
        s.pull = env_int("FNX_L2_PF_PULL", 1) != 0;
        // whether a chunk's rows ride a lane of their own (so a 21 MiB copy overlaps the tier's own GEMMs instead of
        // sitting in front of it), and how many rows slots that lane may keep filled ahead of the GEMMs
        s.xcopy = env_int("FNX_L2_PF_XCOPY", 1) != 0;
        s.xring = (int) env_int("FNX_L2_PF_XRING", 3);
        // how an addend comes back and when card 0 lets the tier read its rows. The addend is the bigger half of what
        // crosses: 5,120 MiB a card a pass as f32 rows (2048 rows x N x 4 x 128 steps), against the 10-12.5 GB/s a
        // 2080 Ti's link answers, so f16 rows is what halves it. The early mark buys the same on the other 21 MiB a
        // step. FNX_L2_PF_FP16=0 puts those bytes back to f32 (byte for byte what rev.15-M2b-1v sent);
        // FNX_L2_PF_EARLY_IN=0 moves the rows mark back after the chunk's top-k, which is a few launches later
        // than the early mark and leaves every copy reading the same bytes
        s.add_h = env_int("FNX_L2_PF_FP16", 1) != 0;
        s.early_in = env_int("FNX_L2_PF_EARLY_IN", 1) != 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", e.what());
        return;
    }
    if (!s.want) return;
    if (max_rows < MAXT) return;
    if (s.ring < 1 || s.ring > PF_PASS_MAX) s.ring = 2;
    if (s.npipe <= 1) s.npipe = 2;
    if (s.npipe > PF_PIPE_MAX) s.npipe = PF_PIPE_MAX;
    if (s.xring < 2 || s.xring > XRING_MAX) s.xring = 3;
    if (!s.xcopy) s.xring = 2;   // rows back on s5: parity is all the rev.15-M2b-1u shape had, and all it needs
    // the width of an addend row, settled before any addend buffer is sized (alloc_rows and collect below)
    s.add_bytes = s.add_h ? 2 : 4;
    // the landing slots a card keeps for addends: deeper than the tier's own addend ring, which is what lets the pull
    // run ahead of the sum. It costs the card (dst_ring - ring) * rows * N * add_bytes bytes and nothing else
    s.dst_ring = s.pull ? (s.ring + 2 < PF_PASS_MAX ? s.ring + 2 : PF_PASS_MAX) : s.ring;
    s.timing = env_flag("FNX_L2_PF_TIMING");
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
    if (cudaStreamCreateWithFlags(&s.sx, cudaStreamNonBlocking) != cudaSuccess) return off("no rows lane for the tier");

    // the streamed experts' slots: one envelope, which plan cuts by what a layer really streams (split_slots).
    // l2_offload sized that envelope and took it out of the card's budget before the arena asked for anything
    // (l2_offload.cu:238-259, "two slots, with 2 GiB of arena still standing"), so the cut spends no extra byte and
    // leaves the desktop's FNX_L2_RESERVE_GIB alone. The envelope is halved before it is refused: a narrower one still
    // streams, it just hands a few more experts of a layer back to the cards (plan). The tier must never be why a
    // working server stopped, and with no room at all the arena's experts still go to the tier.
    if (s.stream && stage_mib > 0 && lc.stage_bytes > 0) {
        const size_t total = 2 * lc.stage_bytes;   // what was reserved, whatever the slots are cut into
        size_t got = 0;
        for (size_t n = total; n >= 2 * MIB && !s.env; n /= 2)
            if (cudaMalloc((void**) &s.env, n) == cudaSuccess) got = n;
            else s.env = nullptr;
        if (got) {
            s.env_bytes = got;
            split_slots(s, 0);   // the first cut: as many even slots as the pass may overlap; plan recuts by need
        } else {
            s.stream = false;   // no room to stream: the arena's experts still go to the tier
            s.npipe = 2;
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
            for (int r = 0; r < PF_PIPE_MAX; ++r) {
                if (cudaMalloc((void**) &s.ep[t][q][r], tbytes) != cudaSuccess) return off("no room on the tier for its expert maps");
                if (cudaMallocHost((void**) &s.hep[t][q][r], tbytes) != cudaSuccess) return off("no pinned host memory for its expert maps");
            }
    // Where an event is made is decided by the stream that records it, not by who owns the work: cudaEventRecord wants
    // event and stream in the same CUDA context. ev_in is recorded by card 0's stream and ev_cons by the stream of the
    // card that sums the addend, so both are made there; ev_done / ev_xread / ev_rows / ev_layer / ev_fill go on the tier's own
    // streams and stay on the tier. Only waiting on an event crosses devices, which cudaStreamWaitEvent allows.
    for (int q = 0; q < STEPS; ++q) {
        for (int t = 0; t < NCARD; ++t)
            if (cudaEventCreateWithFlags(&s.ev_done[t][q], cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
        if (cudaEventCreateWithFlags(&s.ev_xread[q], cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    }
    for (cudaEvent_t& ev : s.ev_rows)
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    for (cudaEvent_t& ev : s.ev_layer)
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    for (cudaEvent_t& ev : s.ev_fill)
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return off("no events for the tier");
    if (s.timing) {   // timing events, and only now: on the tier they mark s5 and sc, both of which live here
        for (int q = 0; q < STEPS; ++q) {
            if (cudaEventCreateWithFlags(&s.tm_a[q], cudaEventDefault) != cudaSuccess) return off("no timing events for the tier");
            if (cudaEventCreateWithFlags(&s.tm_b[q], cudaEventDefault) != cudaSuccess) return off("no timing events for the tier");
            if (cudaEventCreateWithFlags(&s.tm_c[q], cudaEventDefault) != cudaSuccess) return off("no timing events for the tier");
            if (cudaEventCreateWithFlags(&s.tm_d[q], cudaEventDefault) != cudaSuccess) return off("no timing events for the tier");
        }
        for (int q = 0; q < NL; ++q) {
            if (cudaEventCreateWithFlags(&s.tm_f0[q], cudaEventDefault) != cudaSuccess) return off("no timing events for the tier");
            if (cudaEventCreateWithFlags(&s.tm_f1[q], cudaEventDefault) != cudaSuccess) return off("no timing events for the tier");
        }
        s.tm_step.assign(STEPS, 0);
        s.tm_fill.assign(NL, 0);
    }
    const char* ev_off = nullptr;
    for (int t = 0; t < NCARD && !ev_off; ++t) {
        DevGuard on_card(m.cards[(size_t) t].dev);
        if (!on_card.ok) ev_off = "a card would not take the events its own stream records";
        if (t == 0 && !ev_off)   // ev_pair with ev_in: card 0's stream is what records both
            for (int q = 0; q < STEPS && !ev_off; ++q) {
                if (cudaEventCreateWithFlags(&s.ev_in[q], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&s.ev_pair[q], cudaEventDisableTiming) != cudaSuccess)
                    ev_off = "no events for the card that records them";
            }
        for (int r = 0; r < s.ring && !ev_off; ++r)
            if (cudaEventCreateWithFlags(&s.ev_cons[t][r], cudaEventDisableTiming) != cudaSuccess) ev_off = "no events for the card that records them";
        // the addend lane and its two events belong to this card: its own stream records ev_pull and ev_read, and the
        // card pays for the wider landing ring. A card that will not take them puts the addend back on its stream
        if (s.pull) {
            bool lane = cudaStreamCreateWithFlags(&s.pull_st[t], cudaStreamNonBlocking) == cudaSuccess;
            for (int r = 0; lane && r < s.dst_ring; ++r)
                lane = cudaEventCreateWithFlags(&s.ev_pull[t][r], cudaEventDisableTiming) == cudaSuccess &&
                       cudaEventCreateWithFlags(&s.ev_read[t][r], cudaEventDisableTiming) == cudaSuccess;
            if (!lane) {
                s.pull = false;
                s.dst_ring = s.ring;   // no lane: the landing slots are the addend ring again, byte for byte
            }
        }
        if (s.timing && !ev_off)
            for (int q = 0; q < STEPS && !ev_off; ++q) {
                if (cudaEventCreateWithFlags(&s.tm_q0[t][q], cudaEventDefault) != cudaSuccess) ev_off = "no timing events for the card";
                else if (cudaEventCreateWithFlags(&s.tm_q1[t][q], cudaEventDefault) != cudaSuccess) ev_off = "no timing events for the card";
            }
    }
    if (ev_off) return off(ev_off);   // after the guard: off() hands the caller its own device back

    bool got_rows = false;
    for (size_t rows = (size_t) ask; rows >= (size_t) MAXT && !got_rows; rows /= 2)
        for (int xr = s.xring; xr >= 2 && !got_rows; --xr) {   // fewer rows slots first, a narrower chunk only after
            free_rows(s);
            s.xring = xr;
            if (alloc_rows(s, rows)) got_rows = true;
        }
    if (!got_rows) return off("no room left on the card after its arena for a chunk of its expert rows");
    cudaSetDevice(prev);

    c.ok = s.ok = true;
    c.dev = s.dev;
    c.max_rows = s.max_rows;
    c.ring = s.ring;
    c.stream = s.stream;
    c.slot_bytes = s.slot_bytes;
    c.env_bytes = s.env_bytes;
    c.xin_bytes = s.xin_bytes;
    c.npipe = s.npipe;
    c.auto_slots = s.auto_slots;
    c.pull = s.pull;
    c.dst_ring = s.dst_ring;
    c.timing = s.timing;
    c.why.clear();
    if (env_flag("FNX_PF_TIMING"))
        std::fprintf(stderr, "fnx: l2pf: card %d (%s): prompt path on, chunks of %d rows, %zu MiB streamed slots x%d, %zu MiB of rows, ring %d\n",
                     s.dev, lc.name.c_str(), s.max_rows, s.slot_bytes >> 20, s.npipe, (size_t) (s.max_rows * N * 4) >> 20, s.ring);
    if (s.timing)   // what a timing run can tell apart: the addend lane, how deep each card's landing ring is, and
                    // whether a chunk's rows ride their own lane (the thing that lets a copy overlap a GEMM)
        std::fprintf(stderr,
                     "fnx: l2pf: timing on: addends %s %s, rows marked %s, %d landing slots a card, rows %s in %d slots, slots out of %zu MiB\n",
                     s.add_h ? "f16" : "f32", s.pull ? "on their own lane" : "on the card's own stream",
                     s.early_in ? "before the router" : "after its top-k", s.dst_ring,
                     s.xcopy ? "their own lane" : "the GEMM stream", s.xring, s.env_bytes >> 20);
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

// card 0's activation rows are written (the hc read, or the hyper-connection join, made them), so the tier may read
// them. This is what FNX_L2_PF_EARLY_IN moves up: on the old position a 21 MiB copy queued behind the router, the
// top-k, the count and the job tables of the same chunk - seven launches of card 0 work the copy has no part in.
// Recorded on card 0's stream, the device ev_in was made on (setup); an error means the tier would read rows nobody
// told it were written, so it goes up at once rather than being left for the next launch.
void record_rows(int k, void* st_in) {
    S& s = g();
    if (!s.pass || !s.early_in || k < 0 || k >= STEPS || !s.ev_in[k]) return;
    if (cudaEventRecord(s.ev_in[k], (cudaStream_t) st_in) != cudaSuccess)
        throw std::runtime_error("l2pf: the card could not mark its rows ready for the tier");
}

// card 0's top-k pairs are written. The lane waits this immediately before the two copies that read them (step), so
// marking the rows early cannot send a chunk up whose weights the router has not chosen yet. With
// FNX_L2_PF_EARLY_IN off, ev_in is marked here as well: rows and pairs were both written by that point of card 0's
// stream; rev.15-M2b-1v recorded its single event a few launches later, after the count and the job tables, and
// nothing between the two points writes the rows or the pairs.
void record_pairs(int k, void* st_in) {
    S& s = g();
    if (!s.pass || k < 0 || k >= STEPS) return;
    if (s.ev_pair[k] && cudaEventRecord(s.ev_pair[k], (cudaStream_t) st_in) != cudaSuccess)
        throw std::runtime_error("l2pf: the card could not mark its top-k pairs for the tier");
    if (!s.early_in && s.ev_in[k] && cudaEventRecord(s.ev_in[k], (cudaStream_t) st_in) != cudaSuccess)
        throw std::runtime_error("l2pf: the card could not mark its rows ready for the tier");
}

bool plan(Model& m, int rows, int nk, bool stream) {
    S& s = g();
    Ctx& c = const_cast<Ctx&>(ctx());
    if (!s.ok) return false;
    // stream=false is the resident-only ask (M3-1): the tier takes what its arena holds and nothing else, so a short
    // prompt can run with the RAM copies given back. The slots keep whatever the last pass sized them to; nothing
    // streams into them this time, so a stale size cannot move a byte.
    bool do_stream = stream && s.stream;
    // rev.15-M3-1b: with the RAM copies handed back the cards cannot read the arena's experts at all, so a pass that
    // is too short to stream - the 12-row tail of a long prompt, staged like the rest of it - still has to hand them
    // over. The rows bound below then gives up streaming, never the tier: resident-only only needs rows >= 1. Only the
    // dropped case changes, so a pass asked for while the copies are still in place is bit for bit what it was.
    if (do_stream && rows < MAXT && l2::ctx().dropped) do_stream = false;
    s.pass = false;
    s.filled_to = -1;   // a new pass fills its slots from the first layer again
    for (int q = 0; q < XRING_MAX; ++q) s.prev_read[q] = -1;   // ... and every rows slot starts unread
    if (s.timing) {   // only this pass's own marks count (split_slots' line aside), so start the pass clean
        std::fill(s.tm_step.begin(), s.tm_step.end(), (unsigned char) 0);
        std::fill(s.tm_fill.begin(), s.tm_fill.end(), (unsigned char) 0);
    }
    if (nk < 1 || nk > PF_PASS_MAX || rows < (do_stream ? MAXT : 1)) return false;
    if (rows > s.max_rows) {   // wider than the rows buffers were sized for: the cards run this pass, and say so once
        static int told = 0;   // or it would fall back silently, with nothing in the log to compare a bench against
        if (env_flag("FNX_PF_TIMING") || told < 3) {
            ++told;
            std::fprintf(stderr, "fnx: l2pf: skip: a pass of %d rows does not fit the tier's buffers of %d rows\n", rows,
                         s.max_rows);
        }
        return false;
    }
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
    size_t need_max = 0;
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
        // What this layer would stream if its slot were wide enough: the experts host RAM still holds, counted in the
        // very order the packing below fills a slot with, so the number is a real prefix of the slot and never a wish.
        // The arena's experts cost no copy and always go, so they are not in it. This is what split_slots sizes the
        // slots by: measure every layer first, cut once, and a layer whose need fits its slot hands the cards nothing
        // at all to copy up the link (a pass line of "0 MiB copied in"). The cut below only bites when the tier
        // really has no room for a layer, which is the same short fill a too-small slot always was.
        if (do_stream) {
            size_t need = 0;
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                if (i < lc.at.size() && lc.at[i] >= 0) continue;
                if (in_host(i)) need += tot;
            }
            if (need > need_max) need_max = need;
        }
    }
    // one pass, one cut: nothing of it has been enqueued yet and no device memory has been touched here, so moving the
    // slots now costs the pass nothing and every table below is built against the slots that will really hold them
    if (do_stream) split_slots(s, need_max);
    for (int l = 0; l < NL; ++l) {
        const size_t tot = stride[(size_t) l];
        if (!tot) continue;
        const bool str_ok = do_stream && s.slot_bytes > 0;
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
    for (int par = 0; par < s.npipe && !bad; ++par)
        for (int t = 0; t < NCARD && !bad; ++t) {
            const CardW& cw = m.cards[(size_t) t];
            for (int q = 0; q < 2; ++q) std::memset(s.hep[t][q][par], 0, (size_t) NL * NEXP * sizeof(char*));
            for (int l = 0; l < NL; ++l) {
                if ((l % s.npipe) != par || !s.ltake[l]) continue;
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
                char* dst = s.slot[l % s.npipe] + s.off[i] + pre;
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
                     "fnx: l2pf: pass %d of %d rows x %d chunks: %d experts to card %d (%d resident, %d streamed %zu MiB into %zu MiB x%d, "
                     "ring %d)\n",
                     c.passes, rows, nk, s.taken, s.dev, s.n_arena, s.n_stream, s.stream_bytes >> 20, s.slot_bytes >> 20, s.npipe,
                     s.ring);
    return true;
}

// One slot fill: layer q's streamed experts into slot q % npipe, then the event that closes it. The ordering is the
// reason this path is exact at all, so it lives in one place however many fills a pass has queued:
//   wait ev_layer[q % npipe] (s5 read that slot out) -> copies on sc -> record ev_fill[q % npipe] (step's GEMM waits
// for it on s5). Both streams are non-blocking, so without those two events the copy and the GEMM race, and the card's
// fp16 range check reports what it finds as `prefill: ring error 2`.
static bool fill_slot(int q) {
    S& s = g();
    const int par = q % s.npipe;
    tpopt::WaitGuard guard("the expert tier's slot fills");
    DevGuard on_tier(s.dev);
    if (!on_tier.ok) {
        s.pass = false;
        s.why = "cudaSetDevice on the tier failed in a slot fill";
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", s.why.c_str());
        return false;
    }
    // the slot is free once the layer that used it last has read it; an event nothing recorded is no wait at all
    if (s.ev_layer[par]) cudaStreamWaitEvent(s.sc, s.ev_layer[par], 0);
    // FNX_L2_PF_TIMING: after the wait, so the span says what the copies themselves cost and not how long the tier
    // spent queueing for the layer before it to finish reading the slot out
    if (s.timing && s.tm_f0[q]) cudaEventRecord(s.tm_f0[q], s.sc);
    bool undelivered = false;
    for (const Piece& pc : s.pieces[(size_t) q])
        if (cudaMemcpyAsync(pc.dst, pc.src, pc.len, cudaMemcpyHostToDevice, s.sc) != cudaSuccess) {
            std::fprintf(stderr, "fnx: l2pf: off: layer %d slot fill failed\n", q);
            s.pass = false;
            s.why = "a slot fill failed";
            undelivered = true;
            break;
        }
    // and the fill has to be over before the layer's GEMM reads the slot. Bailing here is safe the same way a short
    // fill is - run_card(q) is read when the layer's chunk runs, which is later, so the cards take the whole layer back.
    if (!undelivered && s.ev_fill[par] && cudaEventRecord(s.ev_fill[par], s.sc) != cudaSuccess) {
        s.pass = false;
        s.why = "a slot fill would not mark itself done";
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", s.why.c_str());
        undelivered = true;
    }
    if (s.timing) {
        if (!undelivered && s.timing && s.tm_f1[q]) cudaEventRecord(s.tm_f1[q], s.sc);
        if (!undelivered) s.tm_fill[q] = 1;   // only a fill that really went out is this pass's time
    }
    return !undelivered;
}

// A layer whose streamed experts would not fill their slot is safe to hand back: run_card(l) is read when the layer's
// chunk runs (prefill.cu), which is after this call, so pass = false makes the cards mask nothing and compute all of
// it again. Nothing this layer dropped is lost, which is why layer_in bails and step does not.
//
// Fills go out in layer order and every layer exactly once (filled_to), so naming any layer ahead of the running one
// is safe: the queue behind s5 holds at most npipe - 1 fills, which is what FNX_L2_PF_PIPE = 3 or 4 uses to keep two
// or three of them in flight while s5 and the cards compute. The caller's device comes back either way.
void layer_in(int l) {
    S& s = g();
    if (!s.pass || l < 0) return;
    if (l >= NL) l = NL - 1;
    while (s.filled_to < l) {
        const int q = ++s.filled_to;
        if (!s.ltake[q] || s.pieces[(size_t) q].empty()) continue;
        if (!fill_slot(q)) break;
    }
}

// how many layers ahead of the running one layer_in may name: the slots that may hold a fill before one is read out
int lookahead() {
    const S& s = g();
    return s.npipe - 1;
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
    // a layer whose slot fill never reached sc would read the previous occupant of that slot, so check the queue
    // covers this layer before anything is launched; layer_in fills every layer up to the one it names
    if (!s.pieces[(size_t) l].empty() && s.filled_to < l) lost("a step ran a layer whose slot was never filled");
    const int par = l % s.npipe, sp = k % s.ring, xs = k % s.xring, P = C * TOPK;
    const cudaStream_t lane = s.xcopy ? s.sx : s.s5;   // the lane that carries this chunk's rows
    Model& m = *s.mp;
    tpopt::WaitGuard guard("the expert tier's step");
    DevGuard on_tier(s.dev);
    if (!on_tier.ok) lost("cudaSetDevice on the tier failed in a step");
    if (s.timing && s.tm_a[k]) cudaEventRecord(s.tm_a[k], lane);   // where this step's own time starts
    // streamed weights come first: their copies went out on sc, and every launch below reads them from s5
    if (s.ev_fill[par] && !s.pieces[(size_t) l].empty() && cudaStreamWaitEvent(s.s5, s.ev_fill[par], 0) != cudaSuccess)
        lost("a step would not wait for its layer's slot fill");
    const size_t in_bytes = (size_t) C * N * sizeof(__half);   // the chunk's rows, not the buffer's whole size
    bool bad = false;
    // FNX_L2_PF_XCOPY: the rows ride a lane of their own. Left on s5 this copy could only start behind the step
    // before it, so every card's wait on ev_xread reached back through a whole step of GEMMs: 128 copies sat in
    // front of 128 GEMMs and a timing run read rows 2486 ms + compute 1912 ms, the sum of a pass that did one thing
    // at a time. Here the copy of step k runs on sx while s5 still computes step k-1. What it waits for is exactly
    // two things: card 0's rows (ev_in, recorded by the card), and the slot it lands in - which is free once the
    // last kernel that read it is over, that kernel being the last card's reduce of the step that used this slot
    // before (prev_read, because a layer that stayed on the cards leaves no event of its own to name). Nothing on
    // sx waits for s5, so this cannot stick: s5 is always behind the copy it waits for, never in front of it.
    if (s.ev_in[k] && cudaStreamWaitEvent(lane, s.ev_in[k], 0) != cudaSuccess) bad = true;   // card 0's rows are ready
    if (!bad && s.xcopy && s.prev_read[xs] >= 0 && s.ev_done[NCARD - 1][s.prev_read[xs]] &&
        cudaStreamWaitEvent(s.sx, s.ev_done[NCARD - 1][s.prev_read[xs]], 0) != cudaSuccess) bad = true;
    if (cudaMemcpyAsync(s.xin[xs], xin, in_bytes, cudaMemcpyDefault, lane) != cudaSuccess) bad = true;
    // FNX_L2_PF_EARLY_IN: ev_in says the rows are written, not that the router has answered. The rows copy above is
    // what the early mark is for and goes first; the pairs wait here, in front of the two copies that read them. The
    // order stays provable: both events are card 0's, this lane waits card 0 only, and card 0 waits only ev_xread of
    // the step before (wait_xread), so neither lane can reach around the other
    if (!bad && s.early_in && s.ev_pair[k] && cudaStreamWaitEvent(lane, s.ev_pair[k], 0) != cudaSuccess) bad = true;
    if (!bad && cudaMemcpyAsync(s.topi[xs], topi, (size_t) P * sizeof(int), cudaMemcpyDefault, lane) != cudaSuccess)
        bad = true;
    if (!bad && cudaMemcpyAsync(s.topw[xs], topw, (size_t) P * sizeof(float), cudaMemcpyDefault, lane) != cudaSuccess)
        bad = true;
    // the cards may overwrite their activation rows as soon as this copy is off their memory
    if (!bad && s.ev_xread[k] && cudaEventRecord(s.ev_xread[k], lane) != cudaSuccess) bad = true;
    // ... and only then may the GEMM stream read them: on s5 the order alone says it, on sx an event has to
    if (!bad && s.xcopy && s.ev_rows[xs] && cudaEventRecord(s.ev_rows[xs], s.sx) != cudaSuccess) bad = true;
    if (bad) lost("a step could not copy the chunk to the tier");
    if (s.timing && s.tm_b[k]) cudaEventRecord(s.tm_b[k], lane);   // the rows are on the tier: past here is compute
    if (s.timing) s.tm_step[k] = 1;
    if (s.xcopy) s.prev_read[xs] = k;
    if (s.xcopy && s.ev_rows[xs] && cudaStreamWaitEvent(s.s5, s.ev_rows[xs], 0) != cudaSuccess)
        lost("a step would not wait for its rows to land on the tier");
    if (s.timing && s.tm_d[k]) cudaEventRecord(s.tm_d[k], s.s5);   // the GEMMs start here; tm_a..tm_b is only the lane
    const int* run = s.run_tier + (size_t) l * NEXP;
    try {
        // The layer's table and its masked weights, once per step instead of once per card: both come from the run
        // table (one per layer, the same for the two cards) and from card 0's rows (which hold the same numbers on
        // every card, the thing that makes one tier pass per card correct), and the job buffers they write are shared
        // too. Two cards building them twice was 4 more launches in front of every card's GEMMs, on the step's own
        // stream; the GEMMs below read the same jobs either way, so the numbers are bit for bit what they were.
        cudaMemsetAsync(s.cnt, 0, NEXP * sizeof(int), s.s5);
        k_l2_count<<<blocks_for((size_t) P), 256, 0, s.s5>>>(s.topi[xs], P, s.cnt);
        k_l2_jobbase<<<1, NEXP, 0, s.s5>>>(s.cnt, s.jbase, s.njobs, run);
        k_l2_fill<<<NEXP, PF_FILL, 0, s.s5>>>(s.topi[xs], P, s.cnt, s.jbase, s.jobs_gu, s.jobs_dn, run);
        k_l2_topw_mask<<<blocks_for((size_t) P), 256, 0, s.s5>>>(s.topi[xs], s.topw[xs], s.topw[xs], P, run);
        for (int t = 0; t < NCARD; ++t) {
            const LayerW& L = m.cards[(size_t) t].L[(size_t) l];
            DW gu_e = L.gu_e, dn_e = L.down_e;
            gu_e.ep = s.ep[t][0][par] + (size_t) l * NEXP;
            dn_e.ep = s.ep[t][1][par] + (size_t) l * NEXP;
            const int cap = P / PF_JOB + NEXP;
            moe_gate_up_mma(gu_e, L.gu_e_stride, s.jobs_gu, s.njobs, cap, reinterpret_cast<const uint16_t*>(s.xin[xs]), s.gu, s.s5);
            swiglu_h16(s.gu, s.ffs, P, s.h16, s.s5);
            moe_down_mma(dn_e, L.down_e_stride, s.jobs_dn, s.njobs, cap, s.h16, s.part, s.s5);
            // the slot's earlier addend must be off the card before this one lands on it
            if (s.ev_cons[t][sp] && cudaStreamWaitEvent(s.s5, s.ev_cons[t][sp], 0) != cudaSuccess) bad = true;
            const size_t voff = ((size_t) t * s.ring + sp) * (size_t) s.max_rows * N * (size_t) s.add_bytes;
            if (s.add_h)
                k_l2_reduce<true><<<blocks_for((size_t) C * N), 256, 0, s.s5>>>(s.part, s.topi[xs], s.topw[xs], s.vec + voff, C);
            else
                k_l2_reduce<false><<<blocks_for((size_t) C * N), 256, 0, s.s5>>>(s.part, s.topi[xs], s.topw[xs], s.vec + voff, C);
            // this card's addend is ready here, before the other card's gate_up / down even launch. One event per
            // card instead of one per step is what lets card 0's sum start while the tier still works on card 1, and
            // the rows lane waits the last of them by name (NCARD - 1), which is the whole slot-free proof it needs
            if (s.ev_done[t][k] && cudaEventRecord(s.ev_done[t][k], s.s5) != cudaSuccess) bad = true;
        }
    } catch (const std::exception& e) {
        s.pass = false;
        s.why = e.what();
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", e.what());
        throw;
    }
    if (bad) lost("a step could not hand its addend to a card");
    if (j == s.nk - 1 && s.ev_layer[par] && cudaEventRecord(s.ev_layer[par], s.s5) != cudaSuccess) lost("a step could not close a layer slot");
    if (s.timing && s.tm_c[k]) cudaEventRecord(s.tm_c[k], s.s5);
}


void* collect(int card, int l, int k, int C, void* st) {
    S& s = g();
    if (!s.pass || !s.ltake[l] || card < 0 || card >= NCARD || C < 1 || C > s.max_rows || k < 0 || k >= STEPS) return nullptr;
    if (!s.l2_in[card] || !s.ev_done[card][k]) return nullptr;
    const int sp = k % s.ring;         // the tier's addend slot for this step, which this card alone reads
    const int dsp = k % s.dst_ring;    // the card's landing slot for it, deeper than the tier's ring on purpose
    cudaStream_t stn = (cudaStream_t) st;
    // FNX_L2_PF_PULL: the addend crosses on a lane of the card's own instead of sitting in front of the card's work.
    // Nine steps' worth of 20 MiB a pass rode the stream that also does the attention, the FFN and the sum, and the
    // card could not start its sum until it had landed; here it lands while the card computes, and only the sum waits.
    // What keeps the arithmetic exact is the pair of events around it: ev_read (recorded by read_done, on the stream
    // that summed) says the card has left a landing slot behind, ev_pull says this one has landed before that stream
    // reads it again, and ev_cons keeps the meaning the tier relies on - the addend has left the tier's slot, so a
    // later step may write that slot again. Landing slots are per card, so neither card's copy touches the other's.
    cudaStream_t ps = (s.pull && s.pull_st[card]) ? s.pull_st[card] : stn;
    char* dst = s.l2_in[card] + (size_t) dsp * (size_t) s.max_rows * N * (size_t) s.add_bytes;
    tpopt::WaitGuard guard("the expert tier's addend");
    // Past the two nulls above a null is not "this layer stayed on the cards": its experts are already off the card's
    // tables, so the sum would just miss them. Hence errors go up rather than becoming a quiet missing addend.
    // Nothing here switches device: waiting ev_done (the tier's) from a card stream is what cudaStreamWaitEvent is for,
    // and ev_cons / ev_pull / ev_read[card] were made on this card, whose streams record them - the lane is the card's
    // own stream, made there, so recording on it needs no device switch either.
    auto gone = [&](const char* why) {
        s.pass = false;
        s.why = why;
        std::fprintf(stderr, "fnx: l2pf: off: %s\n", why);
        throw std::runtime_error(std::string("l2pf: the tier's addend ") + why);
    };
    if (s.timing && s.tm_q0[card][k]) cudaEventRecord(s.tm_q0[card][k], ps);   // from here to tm_q1 is wait plus the copy
    // a landing slot the card has not summed yet is not free: an event nothing recorded is no wait at all, which is
    // what a pass's first (dst_ring) steps want. Every collect of a pass goes in step order, so the wait here always
    // names the read that came before this copy, never one that has still to come
    if (ps != stn && s.ev_read[card][dsp] && cudaStreamWaitEvent(ps, s.ev_read[card][dsp], 0) != cudaSuccess)
        gone("would not wait for the card to leave its landing slot");
    if (cudaStreamWaitEvent(ps, s.ev_done[card][k], 0) != cudaSuccess) gone("would not wait for the card");
    if (cudaMemcpyAsync(dst, s.vec + ((size_t) card * s.ring + sp) * (size_t) s.max_rows * N * (size_t) s.add_bytes,
                        (size_t) C * N * (size_t) s.add_bytes, cudaMemcpyDefault, ps) != cudaSuccess)
        gone("would not copy to the card");
    if (s.ev_cons[card][sp] && cudaEventRecord(s.ev_cons[card][sp], ps) != cudaSuccess) gone("would not mark its slot as read");
    if (s.timing && s.tm_q1[card][k]) cudaEventRecord(s.tm_q1[card][k], ps);
    if (ps != stn) {
        if (s.ev_pull[card][dsp] && cudaEventRecord(s.ev_pull[card][dsp], ps) != cudaSuccess) gone("would not mark its addend as landed");
        // the sum runs on the card's stream and reads the rows from there: the wait is only the order between the two
        // streams of one card, and it can never stick - the lane carries nothing but addends
        if (cudaStreamWaitEvent(stn, s.ev_pull[card][dsp], 0) != cudaSuccess) gone("would not hand the addend to the sum");
    }
    return dst;
}

// The card's sum has read what collect() landed for step k, so that landing slot is free for the next one. Call it on
// the stream that summed, which is the card's own and where ev_read was made; every addend read of a pass goes through
// one of those (kernels.cu:876, k_ring / k_ring_big / k_pfr_addend, all launched on the card's stream). Skipping it is
// not wrong, it only parks the lane until the wait's event happens to be over - hence a throw, not a silent widening.
void read_done(int card, int k, void* st) {
    S& s = g();
    if (!s.pass || !s.pull || card < 0 || card >= NCARD || !s.pull_st[card]) return;
    if (k < 0 || k >= STEPS) return;
    const int dsp = k % s.dst_ring;
    if (!s.ev_read[card][dsp]) return;
    if (cudaEventRecord(s.ev_read[card][dsp], (cudaStream_t) st) != cudaSuccess)
        throw std::runtime_error("l2pf: the card could not mark the tier's addend as summed");
}

// Which of a RingMoe's two addend pointers the bytes collect() landed belong to: add_h for f16 rows, add for f32
// rows, neither for null. One width per run (settled at setup), so a consumer reads one pointer and the f32 path is
// bit for bit what it was before the f16 one existed; a null is a layer that stayed entirely on the cards, which is
// the no-tier case the cards have always handled.
void set_add(RingMoe& m, void* landed) {
    const S& s = g();
    m.add = nullptr;
    m.add_h = nullptr;
    if (!landed) return;
    if (s.add_h) m.add_h = reinterpret_cast<const __half*>(landed);
    else m.add = reinterpret_cast<const float*>(landed);
}

// FNX_L2_PF_TIMING: one line for the pass that has just been enqueued and drained. Nothing is created, recorded or
// printed unless the knob asked, so an ordinary run pays nothing for it. prefill calls it after it has waited every
// card's stream, which is after every addend and therefore after every tier launch and every slot fill as well - so
// the marks are all over, and an elapsed query answers instead of guessing. A pair that does not answer (a layer that
// stayed on the cards, a card that refused its timing events) is skipped, never counted as zero.
void timing_report() {
    S& s = g();
    Ctx& c = const_cast<Ctx&>(ctx());
    if (!s.ok || !s.timing || s.nk < 1 || s.tm_pass == c.passes) return;
    s.tm_pass = c.passes;
    const int steps = NL * s.nk;
    double copy_ms = 0, gemm_ms = 0, fill_ms = 0, wait_ms[NCARD] = {};
    {
        DevGuard on_tier(s.dev);
        if (on_tier.ok) {
            for (int k = 0; k < steps; ++k) {
                const bool live = k < (int) s.tm_step.size() && s.tm_step[(size_t) k] != 0;
                span_ms(s.tm_a[k], s.tm_b[k], live, &copy_ms);   // the chunk's rows, onto the tier (its own lane)
                span_ms(s.tm_d[k], s.tm_c[k], live, &gemm_ms);   // the tier's own GEMMs, from rows landed to addends sent
            }
            for (int l = 0; l < NL; ++l)
                span_ms(s.tm_f0[l], s.tm_f1[l], l < (int) s.tm_fill.size() && s.tm_fill[(size_t) l] != 0, &fill_ms);
        }
    }
    for (int t = 0; t < NCARD; ++t) {
        DevGuard on_card(s.mp ? s.mp->cards[(size_t) t].dev : s.dev);
        if (!on_card.ok) continue;
        for (int k = 0; k < steps; ++k)
            span_ms(s.tm_q0[t][k], s.tm_q1[t][k], k < (int) s.tm_step.size() && s.tm_step[(size_t) k] != 0, &wait_ms[t]);
    }
    std::fprintf(stderr,
                 "fnx: l2pf: pass %d timing: tier rows %.1f ms, tier compute %.1f ms, slot fills %.1f ms, addends %.1f ms and %.1f ms\n",
                 c.passes, copy_ms, gemm_ms, fill_ms, wait_ms[0], wait_ms[1]);
}

void end_pass() {
    S& s = g();
    s.pass = false;
}

}  // namespace pf
}  // namespace l2
}  // namespace fnx
