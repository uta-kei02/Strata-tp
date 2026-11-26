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

    // --- the M3 hooks (docs/m2_design_rtx5070_l2.md sec.6): nothing writes them yet, so M2b/M2c stay as they are.
    //     They exist so M3's two-step gravity swap, and the demotion of a card's low-score slots onto this tier, do
    //     not have to reshape what M2b/M2c put in place.
    std::vector<unsigned int> hits;   // [NL * NEXP]: per-slot hit count (decode will count it / 0 until then)
    size_t demote_room = 0;           // arena bytes still free = how much a card may demote onto the tier (0: none)
    // at[] as a tier number in one call (0: still in host RAM, 1: resident in the arena). M3's victim choice goes
    // through this, so adding a tier later does not touch the callers
    int tier_of(size_t i) const { return i < at.size() && at[i] >= 0 ? 1 : 0; }
};

const Ctx& ctx();

// takes the requested card, fills it, checks it and prints one line (off: none). Never throws: every refusal leaves
// the tier off and the run on the old path, because losing the spare card must not stop a server that works without it
void setup(Model& m);

}  // namespace l2
}  // namespace fnx
