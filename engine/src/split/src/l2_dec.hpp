// What M2b-2 does with the tier card at decode (docs/m2_design_rtx5070_l2.md sec.5.8). M2b-1 gave the prompt pass the
// arena; the decode round went on sending every expert a card had no slot for to the host worker and left that arena
// idle between prompts. This is the handoff the host worker already runs - the router writes a list into a mapped
// page-locked box, a worker answers it - done a second time beside the first: the router sorts each window's picks
// into the card's own slots (L1), the arena (L2) and host RAM (L3) in one pass, and the two workers answer their
// lists at the same time the cards run their own experts.
//
// The single-count guarantee is the router's, not a worker's: a pick that goes to the tier drops out of the card's
// job list in the same pass (moe.cuh), so exactly one of the three answers each pair row. Each worker computes only
// the picks its own card sent, over that card's slice of the arena (l2_offload.hpp holds one slice per card), and
// the card assigns those rows into its own part; the ring then sums every card once, as it always did. A card never
// answers another card's rows, so nothing is counted twice and nothing is missing.
//
// A layer with no tier picks launches nothing: no kernel, no event, no stream sync on either side, only a counter.
// That is what keeps a captured CUDA graph replayable on a run without a tier, and FNX_L2_DEC=0 asks for exactly
// that. Nothing here may stop a server that works without the extra card: the gates keep it off, and a tier that
// fails mid-decode hands the picks back to the cards and the host worker instead of throwing.
#pragma once

#include <string>

#include "kernels.hpp"
#include "../../tpopt/tp_opt.hpp"

namespace fnx {

class Model;

namespace l2 {
namespace dec {

// take the arena for decode on behalf of `card`, whose mapped page-locked box is hbox. False: this card answers every
// pick itself, as it did before. Never throws; the first refusal prints one line with the reason and stays refused.
bool open(int card, Model& m, CpuMoeBox* hbox);

// the card's [NL * NEXP] pick map (1: the arena holds this expert, so the router may send a pick that missed the
// card's slots there), in the card's own memory; null when the tier is off for it. Zeroing it at any moment is safe:
// it only moves picks between the three lists, it can neither duplicate one nor drop one.
const char* map(int card);

// the most picks one window may send to the tier (FNX_L2_DEC_MAX; 0: the whole box, which is what the lists hold)
int cap();

// the slot a card's CPU Hybrid TP manager publishes each round, so the tier's worker wakes with the host workers
// rather than polling for the work: req_seq is written from the CUDA side, so nobody else can reach it
tpopt::WaitSlot* wake();

}  // namespace dec
}  // namespace l2
}  // namespace fnx
