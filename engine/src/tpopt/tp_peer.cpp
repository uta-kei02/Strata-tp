// engine/src/tpopt/tp_peer.cpp - see tp_peer.hpp. The whole point of this file: the split engine may not name a CUDA peer
// function (upstream rule, pinned twice by tests and by the packaging build), and still has to be able to use the NVLink
// bridge between two RTX 2080 Ti's. dlopen buys both: the binary carries no undefined peer symbol, and a machine where the
// entry points, the bridge or the architecture do not allow it gets one plain "no" and the host exchange.
#include "tp_peer.hpp"

#include <dlfcn.h>

#include <cstdio>
#include <mutex>

namespace tpopt {
namespace {

// cudaError_t is an enum and CUresult is one too, and a CUcontext handle is a pointer-sized value: declaring the entry
// points with these shapes is ABI-correct on x86-64 and keeps the CUDA headers out of this file. The argument order is
// the part that has to be spelled out, because getting it wrong is not a compile error here:
//   cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)      cuDeviceCanAccessPeer(int*, CUdevice, CUdevice)
//   cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags)                   cuCtxEnablePeerAccess(CUcontext, unsigned)
// The two "can" calls answer about a directed pair and write the answer through their FIRST argument (an int, which is
// address 0/1 when the order is swapped -> the call faults). The two "enable" calls take only the peer: they enable the
// path from whichever card is current on the calling thread, so the current card has to be switched for every row of the
// pair loop and put back afterwards.
typedef int (*can_access_t)(int*, int, int);
typedef int (*enable_t)(int, unsigned);
typedef int (*get_cur_dev_t)(int*);
typedef int (*set_dev_t)(int);
typedef int (*last_error_t)();
typedef int (*cu_get_dev_t)(int*, int);
typedef int (*cu_retain_t)(void**, int);
typedef int (*cu_can_t)(int*, int, int);
typedef int (*cu_enable_t)(void*, unsigned);
typedef int (*cu_ctx_get_t)(void**);
typedef int (*cu_ctx_set_t)(void*);

constexpr int SUCCESS = 0;
constexpr int ERR_PEER_ACCESS_ALREADY_ENABLED = 704;   // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED == cudaErrorPeerAccessAlreadyEnabled

struct Api {
    void* handle = nullptr;
    std::string src;
    bool driver = false;
    can_access_t can = nullptr;
    enable_t enable = nullptr;
    get_cur_dev_t gdev = nullptr;
    set_dev_t sdev = nullptr;
    last_error_t last = nullptr;
    cu_get_dev_t get_dev = nullptr;
    cu_retain_t retain = nullptr;
    cu_can_t dcan = nullptr;
    cu_enable_t denable = nullptr;
    cu_ctx_get_t cget = nullptr;
    cu_ctx_set_t cset = nullptr;

    // the two sets of entry points are checked on their own; ok() only reports the set that was actually chosen, and
    // `driver` is set at the moment the driver's set is accepted (checking through ok() before that is what made the
    // driver branch answer "no" even when all six cu* entry points resolved)
    bool runtime_ok() const { return can && enable; }
    bool driver_ok() const { return get_dev && retain && dcan && denable; }
    bool ok() const { return driver ? driver_ok() : runtime_ok(); }
    // turning access on is per current card (or per current context on the driver path), so those two entry points are
    // needed as well; probing can answer without them
    bool switchable() const { return driver ? (cget && cset) : (gdev && sdev); }
    template <typename T> static void bind(void* h, const char* name, T& into) { into = reinterpret_cast<T>(dlsym(h, name)); }
    void from_runtime(void* h) {
        can = nullptr;
        enable = nullptr;
        gdev = nullptr;
        sdev = nullptr;
        last = nullptr;
        bind(h, "cudaDeviceCanAccessPeer", can);
        bind(h, "cudaDeviceEnablePeerAccess", enable);
        bind(h, "cudaGetDevice", gdev);
        bind(h, "cudaSetDevice", sdev);
        bind(h, "cudaGetLastError", last);
    }
    void from_driver(void* h) {
        get_dev = nullptr;
        retain = nullptr;
        dcan = nullptr;
        denable = nullptr;
        cget = nullptr;
        cset = nullptr;
        bind(h, "cuDeviceGet", get_dev);
        bind(h, "cuDevicePrimaryCtxRetain", retain);
        bind(h, "cuDeviceCanAccessPeer", dcan);
        bind(h, "cuCtxEnablePeerAccess", denable);
        bind(h, "cuCtxGetCurrent", cget);
        bind(h, "cuCtxSetCurrent", cset);
    }
};

Api& api_slot() {
    static Api a;
    return a;
}

void load_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        Api& a = api_slot();
        // the runtime the engine already links shares its symbols with the global scope, so a lookup through RTLD_DEFAULT
        // finds them without this binary carrying an undefined reference of its own (that is what the nm guard looks at)
        a.can = reinterpret_cast<can_access_t>(dlsym(RTLD_DEFAULT, "cudaDeviceCanAccessPeer"));
        a.enable = reinterpret_cast<enable_t>(dlsym(RTLD_DEFAULT, "cudaDeviceEnablePeerAccess"));
        a.gdev = reinterpret_cast<get_cur_dev_t>(dlsym(RTLD_DEFAULT, "cudaGetDevice"));
        a.sdev = reinterpret_cast<set_dev_t>(dlsym(RTLD_DEFAULT, "cudaSetDevice"));
        a.last = reinterpret_cast<last_error_t>(dlsym(RTLD_DEFAULT, "cudaGetLastError"));
        if (a.can && a.enable) {
            a.src = "the libcudart the engine links";
            return;
        }
        a.can = nullptr;
        a.enable = nullptr;
        a.gdev = nullptr;
        a.sdev = nullptr;
        a.last = nullptr;
        const char* rts[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so"};
        for (const char* name : rts) {
            void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (!h) continue;
            a.from_runtime(h);
            if (a.runtime_ok()) {
                a.handle = h;
                a.src = name;
                return;
            }
            dlclose(h);
        }
        // no runtime by name (a static runtime, or a version this build does not know): the driver's pair answers the same
        // question. Enabling is still per current context, so cuCtxGetCurrent / cuCtxSetCurrent come with it and each
        // card's own primary context is put current for its row of the pair loop (see CurrentCard).
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!h) return;
        a.from_driver(h);
        if (a.driver_ok()) {
            a.handle = h;
            a.driver = true;
            a.src = "libcuda.so.1 (driver)";
            return;
        }
        dlclose(h);
        a.handle = nullptr;
    });
}

// One card of the pair loop: card `dev` is made current for the whole of its row, and leaving the scope - however it
// leaves, including a "no" answer mid-row - puts the thread back on the card (or context) it was on. Without this the
// enable calls all land on whatever happened to be current: card 0 enabling access to itself, which is an error, and the
// rows that actually need the bridge never enabled.
struct CurrentCard {
    const Api* a = nullptr;
    int prev_dev = -1;
    void* prev_ctx = nullptr;
    bool switched = false;
    std::string err;

    explicit CurrentCard(const Api& api) : a(&api) {}
    ~CurrentCard() { restore(); }
    CurrentCard(const CurrentCard&) = delete;
    CurrentCard& operator=(const CurrentCard&) = delete;

    bool enter(int dev) {
        if (a->driver) {
            if (a->cget(&prev_ctx) != SUCCESS) {
                err = "cuCtxGetCurrent returned an error";
                return false;
            }
            int own = 0;
            void* ctx = nullptr;
            if (a->get_dev(&own, dev) != SUCCESS) {
                err = "cuDeviceGet(" + std::to_string(dev) + ") returned an error";
                return false;
            }
            if (a->retain(&ctx, own) != SUCCESS) {
                err = "cuDevicePrimaryCtxRetain(" + std::to_string(own) + ") returned an error";
                return false;
            }
            if (a->cset(ctx) != SUCCESS) {
                err = "cuCtxSetCurrent(the primary context of card " + std::to_string(dev) + ") returned an error";
                return false;
            }
            // the reference taken above stays: the engine runs on these primary contexts for the whole session, and
            // releasing it here would tear down a context the enable call has just been made in
            switched = true;
            return true;
        }
        if (a->gdev(&prev_dev) != SUCCESS) {
            err = "cudaGetDevice returned an error";
            return false;
        }
        if (prev_dev != dev) {
            if (a->sdev(dev) != SUCCESS) {
                err = "cudaSetDevice(" + std::to_string(dev) + ") returned an error";
                return false;
            }
            switched = true;
        }
        return true;
    }
    void restore() {
        if (!switched) return;
        if (a->driver)
            a->cset(prev_ctx);
        else if (a->sdev)
            a->sdev(prev_dev);
        switched = false;
    }
};

PeerResult run(const std::vector<int>& devs, bool probe) {
    load_once();
    Api& a = api_slot();
    PeerResult r;
    if (!a.ok()) {
        const char* e = dlerror();
        r.why = std::string("no CUDA library with the peer entry points could be found") + (e && *e ? " (" + std::string(e) + ")" : "");
        return r;
    }
    if (devs.size() < 2) {
        r.ok = true;
        r.api = a.src;
        r.why = "one card: nothing to reach";
        return r;
    }
    // probing only asks the question (the answer comes as an argument, no current card needed); enabling writes into the
    // current card's state, so it needs the switch and the two entry points that go with it
    if (!probe && !a.switchable()) {
        r.why = std::string(a.driver ? "libcuda.so.1 has no cuCtxGetCurrent / cuCtxSetCurrent" : "no cudaGetDevice / cudaSetDevice") +
                ": turning peer access on is per current card, and this library cannot switch it";
        r.api = a.src;
        return r;
    }
    for (size_t i = 0; i < devs.size(); ++i) {
        // the whole row of card devs[i] runs with devs[i] current (a no-op while probing)
        CurrentCard here(a);
        if (!probe && !here.enter(devs[i])) {
            r.why = here.err;
            r.api = a.src;
            return r;
        }
        for (size_t j = 0; j < devs.size(); ++j) {
            if (i == j) continue;
            int can = 0;
            int e = 0;
            void* peer_ctx = nullptr;
            if (a.driver) {
                int pdev = 0;
                if ((e = a.get_dev(&pdev, devs[j])) != SUCCESS) {
                    r.why = "cuDeviceGet(" + std::to_string(devs[j]) + ") returned " + std::to_string(e);
                    r.api = a.src;
                    return r;
                }
                if ((e = a.retain(&peer_ctx, pdev)) != SUCCESS) {
                    r.why = "cuDevicePrimaryCtxRetain(" + std::to_string(devs[j]) + ") returned " + std::to_string(e);
                    r.api = a.src;
                    return r;
                }
                if ((e = a.dcan(&can, devs[i], devs[j])) != SUCCESS) {
                    r.why = "the driver call for card " + std::to_string(devs[i]) + " -> " + std::to_string(devs[j]) + " returned " + std::to_string(e);
                    r.api = a.src;
                    return r;
                }
            } else {
                if ((e = a.can(&can, devs[i], devs[j])) != SUCCESS) {
                    r.why = "the runtime call for card " + std::to_string(devs[i]) + " -> " + std::to_string(devs[j]) + " returned " + std::to_string(e);
                    r.api = a.src;
                    return r;
                }
            }
            if (!can) {
                r.why = "card " + std::to_string(devs[i]) + " cannot reach card " + std::to_string(devs[j]) +
                        " (no P2P path between them: not the same architecture, no bridge, or BAR1/IOMMU in the way)";
                r.api = a.src;
                return r;
            }
            if (probe) continue;
            if (a.driver) {
                e = a.denable(peer_ctx, 0);
                if (e == SUCCESS || e == ERR_PEER_ACCESS_ALREADY_ENABLED) continue;
            } else {
                e = a.enable(devs[j], 0);
                if (e == ERR_PEER_ACCESS_ALREADY_ENABLED) {
                    if (a.last) a.last();   // an "already enabled" answer is sticky on the runtime: clear it, or the next ck() trips on it
                    continue;
                }
                if (e == SUCCESS) continue;
                if (a.last) a.last();
            }
            r.why = "enabling the direct path from card " + std::to_string(devs[i]) + " to card " + std::to_string(devs[j]) + " returned " + std::to_string(e);
            r.api = a.src;
            return r;
        }
        here.restore();   // the next row switches to the next card; the destructor puts the thread back for every other exit
    }
    r.ok = true;
    r.api = a.src;
    r.why = probe ? "the cards can reach each other directly" : "the direct path between the cards is on";
    return r;
}

}  // namespace

PeerResult peer_load() {
    load_once();
    PeerResult r;
    Api& a = api_slot();
    r.ok = a.ok();
    r.api = a.src;
    r.why = r.ok ? "the peer entry points came from " + a.src : "no CUDA library with the peer entry points could be loaded";
    return r;
}

PeerResult peer_probe(const std::vector<int>& devs) { return run(devs, true); }

PeerResult peer_enable(const std::vector<int>& devs) { return run(devs, false); }

}  // namespace tpopt
