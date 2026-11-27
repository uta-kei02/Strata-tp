// The expert tier on a spare card (rev.15-M2, docs/m2_design_rtx5070_l2.md). The split reads its routed experts from
// two places today: a slot in a card's VRAM (ExpertCache, model.hpp) and a page-locked copy in host RAM. A card left
// over - prod's RTX 5070, which is not a tensor-parallel member and shares no P2P path with the 2080 Tis - is the
// middle of the three: it holds experts the cards gave up and runs their FFN on its own, reached over PCIe from the
// host's pinned copies instead of the thin OCuLink the cards swap over.
//
// M2a is the ground only: ask for the card, budget its arena, fill it with the hottest experts host RAM holds, and
// prove the copy and the card's own code with a byte self-test and timed copies. Nothing reads the arena yet, so an
// unset FNX_L2_GPU leaves the engine on the path it has always had (and prints nothing).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "env.hpp"

namespace fnx {

class Model;

namespace l2 {

// which visible device holds the arena (-1: off). Reads only the environment, so the model can ask before any CUDA
// context exists (model.cpp's card count check runs first); FNX_L2_GPU is a visible ordinal, not an nvidia-smi index
inline long long requested_device() { return env_int("FNX_L2_GPU", -1); }

// M3-1's two thresholds: the env names are frozen now so the launcher and the docs stop moving later, but M2b reads
// neither of them (an unread env_int cannot throw). promote = host RAM -> arena, demote = arena -> host RAM
inline long long promote_thresh() { return env_int("FNX_L2_PROMOTE_THRESH", 3); }
inline long long demote_thresh() { return env_int("FNX_L1_DEMOTE_THRESH", 0); }

// M3-2 (docs/m3_plan.md sec.9): the arena stops being a fixed set and becomes a dynamic first-tier cache. The steps
// are measured before they move anything, so 1 is step 0 (read-only probes plus the dry-run of one pass's picks) and
// 2 and above are the re-lay itself, which is not implemented yet. 0 means none of it runs: at[], the cards' expert
// tables and both cards' tier maps keep what startup wrote, exactly as rev.15-M3-1b. A value that is not a number is
// off too - a measurement knob must never be the reason a working server stops.
inline long long dyn_mode() {
    try {
        return env_int("FNX_L2_DYN", 0);
    } catch (const std::exception&) {
        return 0;
    }
}

// how many experts one re-lay may move; what step 0's dry-run sizes itself with. 0 = as many as the arena holds now
inline long long dyn_swap_n() {
    try {
        return env_int("FNX_L2_DYN_N", 0);
    } catch (const std::exception&) {
        return 0;
    }
}

struct Ctx {
    int dev = -1;                      // the visible ordinal the arena lives on (-1: off)
    int major = 0, minor = 0;          // its compute capability: the cards' own check only looks at dev < NCARD
    bool ok = false;                   // the arena is allocated, filled and (when asked) verified
    bool selftest_ok = false;          // the host and the card agreed on the same byte fold
    std::string name;                  // the card's name, so the log shows which card the ordinal was
    std::string why;                   // why the tier is off, when it is
    double gib = 0, reserve_gib = 0, cap_gib = 0, free_gib = 0, total_gib = 0;
    double seconds = 0;                // the fill, copies included
    double h2d_pinned_gbs = 0;         // what the fill really moved (its source is the pinned host copies)
    double h2d_pageable_gbs = 0;       // the same size through an ordinary buffer, for comparison
    double d2h_gbs = 0;                // arena -> pinned host
    size_t bytes = 0;                  // arena bytes
    // M2b-1's streamed slots (l2_pf.cu): the bytes of one of its two slots. Sized here, because the arena and the
    // slots share one envelope and the envelope is the number that keeps FNX_L2_RESERVE_GIB for the desktop; asking
    // for the slots before the arena is what protects it. 0: nothing streams, and the arena keeps its full cap.
    size_t stage_bytes = 0;
    size_t moved = 0;                  // bytes actually placed
    int experts = 0;                   // host-tier experts now resident in the arena
    int skipped = 0;                   // host-tier experts that did not fit
    char* arena = nullptr;             // the card's memory
    // [NL * NEXP]: the expert's byte offset in arena, -1 when it stayed in host RAM. The weights are read-only, so a
    // copy is enough for M2a; M2b reads this to decide which experts the card computes instead of the host
    std::vector<long long> at;

    // --- M3-1 (FNX_L2_DROP_HOST, off by default): the arena's experts no longer need their page-locked RAM copy.
    //     drop_host() gives those bytes back and rewrites the cards' host mirrors and expert tables around what is
    //     left, so the tier is the only place an arena expert's weight exists. `dropped` is the flag every other path
    //     checks before it reads a host pointer that used to be there; false means nothing here was touched.
    bool dropped = false;            // at least one host expert block was handed back
    size_t freed = 0;                // page-locked bytes given back in total
    int dropped_experts = 0;         // [NL * NEXP] entries whose RAM copy is gone (the same on every card)
    double drop_seconds = 0;         // the compaction itself, so a slow knob is visible in the startup log
    std::vector<unsigned char> dropped_entry;   // [NL * NEXP]: 1 = this entry has no host copy any more

    // --- the M3 hooks (docs/m2_design_rtx5070_l2.md sec.6): nothing writes them yet, so M2b/M2c stay as they are.
    //     They exist so M3's two-step gravity swap, and the demotion of a card's low-score slots onto this tier, do
    //     not have to reshape what M2b/M2c put in place.
    std::vector<unsigned int> hits;   // [NL * NEXP]: per-slot hit count (decode will count it / 0 until then)
    size_t demote_room = 0;           // arena bytes still free = how much a card may demote onto the tier (0: none)

    // --- M3-2's step 0 (FNX_L2_DYN, off by default): can this context reach the two kinds of page-locked block a
    //     re-lay would have to read and write, and what would one pass's picks have covered after a re-lay. Sizes,
    //     measurements and a dry-run only: none of it moves a weight, an entry of at[], a card table or a map.
    std::vector<size_t> expert_bytes;   // [NL]: one expert of this layer, every card's slice of it together
    std::vector<char*> new_host;        // [NCARD * NL]: drop_host's block for this card's layer (null: its old block stayed)
    std::vector<size_t> new_host_bytes;  // [NCARD * NL]: the bytes of that block, so a row inside it can be measured

    // one answer to one question: is a page-locked block that another context laid down usable from the tier?
    struct Probe {
        bool ran = false;             // the knob asked and the tier is up (otherwise the probe never ran at all)
        bool found = false;           // a live block of this kind exists to sample
        bool map_ok = false;          // cudaHostGetDevicePointer named it inside the tier's own context
        bool h2d_ok = false, d2h_ok = false;
        double h2d_gbs = 0, d2h_gbs = 0;   // what the tier reads out of it, and what it writes into it
        int card = -1;                // whose block it sampled, since that card's context owns the allocation
        size_t bytes = 0;             // the size it measured with
        bool raw_ok = false;          // the raw host address worked too, so a re-lay would not need the mapped name
        std::string why;              // what stopped it, when something did
    };
    Probe probe_mapped, probe_portable;   // model.cpp's Mapped block / drop_host's Mapped | Portable one
    // at[] as a tier number in one call (0: still in host RAM, 1: resident in the arena). M3's victim choice goes
    // through this, so adding a tier later does not touch the callers
    int tier_of(size_t i) const { return i < at.size() && at[i] >= 0 ? 1 : 0; }
};

const Ctx& ctx();

// takes the requested card, fills it, checks it and prints one line (off: none). Never throws: every refusal leaves
// the tier off and the run on the old path, because losing the spare card must not stop a server that works without it
void setup(Model& m);

// M3-1: hand the page-locked RAM copies of the arena's experts back, once the tier is up and both the prompt and the
// decode path agree they will never be read. Off unless FNX_L2_DROP_HOST says otherwise, and every gate below is a
// no: a run that does not ask for it (or that fails one gate) keeps every byte exactly where M2b-2 left it. Call it
// after l2::setup / l2::pf::setup / l2::dec::open and before anything computes.
void drop_host(Model& m);

// M3-2 step 0 (docs/m3_plan.md sec.9.5 / 9.8 / 9.9), behind FNX_L2_DYN: ask the tier's own context to name, read and
// write one live page-locked block of the kind named. 0 is the block model.cpp mapped for the cards' host copies
// (cudaHostAllocMapped, no Portable flag - sec.9.5's open question), 1 is the smaller block drop_host built from the
// survivors (Mapped | Portable). A probe only ever reads a live block, and its write lands in a fresh buffer of the
// same flags laid down by the same card's context, so no weight can be corrupted and no table is touched. What M3-2
// needs to know before it moves a byte is exactly these two numbers; with the knob off nothing runs and nothing prints.
void probe_pinned(Model& m, int kind);

// M3-2 step 0's other half: the picks of the passes so far, as card 0's [decode | prompt][NL][NEXP]
// counters (empty when FNX_ROUTE_COUNT is off, which the first line then says once). The dry-run only compares this
// pass's hottest experts outside the arena with its coldest residents and prints what a 1-for-1 re-lay would have
// covered. No weight, no at[], no expert table and no tier map moves, so the answers of the run do not change.
void note_route_counts(Model& m, const unsigned* counts, size_t n);

}  // namespace l2
}  // namespace fnx
