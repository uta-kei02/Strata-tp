// What M2b-1 does with the tier card (docs/m2_design_rtx5070_l2.md §4): the prompt pass stops copying the routed
// experts host RAM holds up the thin OCuLink and asks the tier to compute them instead. Two sets of experts are
// involved and only one of them moves: the ones l2_offload put in the card's arena (already there, no copy at all),
// and the ones still in page-locked host RAM, streamed through FNX_L2_PF_PIPE slots at the pinned rate the M2a probe
// (37.70 GB/s, strata-tp.log:615) while the layer before it computes.
//
// TP stays exact. The tier runs each hosted expert's FFN twice, once per card, because the expert's weight is the sum
// of the cards' slices: card t's pass reads card t's bytes and writes card t's addend. Nothing crosses the AllReduce
// except that addend, added to the card's own part before its chunk sum, so a pair row is counted once and once only
// (docs/design_spec.md §4.16). The cards drop the same experts from their jobs by the same table, so neither side
// computes what the other already did.
//
// Off unless the tier is on and FNX_L2_PF says otherwise, and every part of the prompt path is guarded by `use()`:
// a run without FNX_L2_GPU enqueues exactly what it enqueued before.
#pragma once

#include <string>
#include <vector>

namespace fnx {

class Model;
struct RingMoe;   // kernels.hpp: the sum's addend, which collect() hands back and set_add() fills

namespace l2 {
namespace pf {

struct Ctx {
    bool want = false;    // the tier is up and FNX_L2_PF did not say no
    bool ok = false;      // its buffers are ready, so a pass may hand experts to it
    std::string why;      // why not, when it is not (printed once)
    int dev = -1;         // the tier's visible ordinal (l2::ctx().dev)
    int max_rows = 0;     // the largest chunk the buffers were sized for
    int ring = 2;         // addend slots per card (FNX_L2_PF_RING)
    bool stream = true;   // FNX_L2_PF_STREAM: the experts host RAM holds also go to the tier
    int npipe = 2;        // FNX_L2_PF_PIPE: how many streamed slots a pass overlaps (2 = the rev.15-M2b-1 shape)
    bool auto_slots = true;   // FNX_L2_PF_AUTO: cut the slot envelope as wide as the widest layer of this pass needs
    bool pull = true;     // FNX_L2_PF_PULL: an addend comes to the card on a stream of its own, off the card's critical path
    int dst_ring = 2;     // landing slots per card for those addends (>= ring, so a pull never waits on a sum)
    bool timing = false;  // FNX_L2_PF_TIMING: one line per pass, where the tier's milliseconds went
    int ffs = 0;          // one card's slice of a routed expert's intermediate width
    size_t slot_bytes = 0;        // one streamed-expert slot (two of them)
    size_t env_bytes = 0;         // the envelope those slots are cut out of (l2_offload.cu:238-259 reserved it)
    size_t xin_bytes = 0;         // one card's activation rows for a chunk, as f16 or q8_1
    size_t stream_bytes = 0;      // what the last pass streamed
    int taken = 0, n_arena = 0, n_stream = 0;   // the last pass's experts
    int passes = 0;
};

const Ctx& ctx();

// the prompt path asks this before it changes anything
bool use();

// the tier's buffers, sized for a chunk of max_rows (Engine::pf_c_). Never throws: every failure leaves the tier out
// of the prompt path with one line, the pass then exactly what it was without M2b-1
void setup(Model& m, int max_rows);

// which experts the tier computes for this pass (rows x nk chunks). Called once, before the pass builds its expert
// tables, so the cards can leave those bytes in host RAM instead of copying them. False: the pass is today's
bool plan(Model& m, int rows, int nk);

bool take(size_t i);          // [NL * NEXP]: the tier computes this expert this pass
bool layer_taken(int l);      // and this layer's worth of them fits
// the cards' own table for layer l (mapped pinned RAM: 0 = skip this expert, 1 = run it), null when the layer stays
// entirely on them; the same table both sides mask with, which is what keeps a pair counted once
const int* run_card(int l);

// a chunk's work. `k` is the pass step (l * nk + j); xin / topi / topw are card 0's rows for the chunk (every card
// holds the same numbers, which is what makes one tier pass per card correct), `st_in` the stream that produced them.
// Every handle belongs to one device: an event is made on the device whose stream records it (cudaEventRecord wants
// event and stream in the same CUDA context), only waiting on one crosses devices. ev_in is made on card 0 because
// its stream records it, ev_cons on the card that sums the addend, the rest on the tier. Each of these switches to
// the tier and back; the caller's device is what they leave, on every way out.
// card 0's activation rows are written, so the tier may copy them. With FNX_L2_PF_EARLY_IN (the default) this marks
// ev_in right there, which lets the 21 MiB copy cross while card 0 still runs its router, its top-k and its job
// tables; with the knob off it marks nothing and record_pairs marks both at rev.15-M2b-1v's point.
void record_rows(int k, void* st_in);
// card 0's top-k pairs (topi / topw) are written: the rows lane waits this just before it copies them, so an early
// rows mark can never send a pair out before the router's answer exists. Without FNX_L2_PF_EARLY_IN this is where
// ev_in gets marked too, a few launches before where the copy's single mark stood at rev.15-M2b-1v (the count and
// the job tables came after it) and reading the same bytes either way.
void record_pairs(int k, void* st_in);
// card 0 may overwrite its activation rows again only once step k-1's copy of them has left its memory
void wait_xread(int k, void* st);     // throws when the wait would not hold
// stream layer l's experts, and every layer of this pass whose fill has not gone out yet, into their slots and mark
// each fill done, so step's launches wait for it: sc and s5 are both non-blocking streams, and those events are the
// only order between the copies and the GEMMs. Naming a layer ahead of the running one is how a pass keeps fills in
// flight (lookahead); each layer is filled exactly once per pass. A layer that would not fill is handed back to the
// cards: its run table is read when the layer's chunk runs, which is later.
void layer_in(int l);
// how many layers ahead of the running one layer_in may name without a slot being overwritten before s5 read it
int lookahead();
// A tier failure here throws: the cards built this chunk's jobs with the tier's table before it ran, so continuing
// quietly would leave the tier's rows out of the sum. Layers that have not run go back to the cards (run_card goes
// null), which is the only hand-back that is safe.
void step(int l, int j, int C, int k, const void* xin, const int* topi, const float* topw);
// wait for step k and bring its addend to `card`; its card-local buffer, null when the layer did not go to the tier.
// Null means that layer; a tier failure after the drop throws, because a null then would be a hole in the sum.
// The bytes are f16 rows when FNX_L2_PF_FP16 (the default) and f32 rows when it is off: one width, chosen once at
// setup, so a pass never mixes them. l2_pf.hpp's set_add() is what puts them into a RingMoe, and it fills exactly
// one of the two addend pointers.
void* collect(int card, int l, int k, int C, void* st);
// which addend pointer of the card's RingMoe the bytes collect() returned belong to: add_h for f16 rows, add for f32
// rows, and neither when it returned null (the layer stayed on the cards). Both null is what a run without a tier has
// always had, so nothing about the cards' own sum changes.
void set_add(RingMoe& m, void* landed);

// the card is done reading the addend collect() handed it: call it once the layer's sum has consumed it, on the stream
// that summed it (every such stream is the card's own, which is where ev_read was made). Only FNX_L2_PF_PULL uses it;
// it lets the next addend land in that landing slot without waiting for the sum, and losing it only stops a later copy.
void read_done(int card, int k, void* st);

// FNX_L2_PF_TIMING: one line for the pass that has just finished, with the tier's copy / GEMM / slot-fill / pull time.
// Nothing is recorded, and no line printed, unless the knob asked; a run that did not pays nothing at all.
void timing_report();
void end_pass();

}  // namespace pf
}  // namespace l2
}  // namespace fnx
