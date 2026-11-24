// engine/src/tpopt/tp_peer.hpp - the CUDA peer entry points, reached through dlopen instead of a link, and living outside
// engine/src/split on purpose.
//
// The split engine's upstream rule (terminal/tests/split/test_split_engine.py, and tools/packaging/build_split_cuda.sh which
// fails the build when `nm -D --undefined-only` shows a peer symbol in the binary) is "no peer API on NVIDIA": on a GeForce
// whose BAR1 does not cover the peer's whole VRAM the mailbox stores go wrong and the ring hangs (card_check.hpp's
// require_bar1_covers says the same thing). The rule is right for the box upstream tests, and wrong for the box this build
// runs on: two RTX 2080 Ti's on an NVLink bridge carry the exchange at 25.7 GB/s against the 3.2 GB/s of the host ring, and
// that difference is most of the token rate.
//
// So: the engine itself never names a peer function. It asks this file "can the cards talk directly, and did you turn it
// on", and gets one plain answer with a reason. The names and the calls live here, dlopen'd, so both guards keep passing,
// and a card set where P2P is not available (an RTX 5070 beside a 2080 Ti: different architectures, no P2P at all) is an
// honest "no" - the engine falls back to the host exchange exactly as upstream does.
#pragma once

#include <string>
#include <vector>

namespace tpopt {

struct PeerResult {
    bool ok = false;
    std::string why;    // why not, or what was used; one line, always filled
    std::string api;    // the library the entry points came from (empty when nothing could be loaded)
};

// Load the runtime (or the driver) and resolve the two entry points. Never fatal: false + why.
PeerResult peer_load();

// Every directed pair among `devs` can access the other and access is enabled for it: all or nothing. `probe` asks the
// question without enabling (the start self-test uses it to decide, before any memory is handed out).
PeerResult peer_probe(const std::vector<int>& devs);
PeerResult peer_enable(const std::vector<int>& devs);

}  // namespace tpopt
