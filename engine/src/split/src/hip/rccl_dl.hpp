// RCCL for the prompt path's chunk sums (AMD, FNX_PF_RCCL=1 behind a passed peer probe): loaded at run time with dlopen, never linked, so a missing library can only mean
// "the host exchange for the prompt sums, one plain line" and the engine starts exactly as before. CPU only (no HIP header: a stream is an opaque pointer), so
// tools/rccl_policy_test.cpp holds the loader and the decision to their rules without a card. The library is the collective library of whichever box this is: NVIDIA NCCL
// (the libnccl.so.2 / libnccl.so.1 a CUDA install leaves on the library path, or a build outside it named by FNX_RCCL_LIB) or AMD RCCL (the rocm[libraries]
// wheel ships librccl.so.1); both are BSD-3-clause, in share/doc/<nccl|rccl>/LICENSE.txt of the SDK - nothing is copied or shipped here, and FNX_RCCL_LIB names one explicitly.
// The six functions and the two enum values are RCCL's (rccl.h: ncclSum = 0, ncclFloat16 = 6, ncclFloat32 = 7, ncclSuccess = 0).
#pragma once

#include <dlfcn.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace fnx {

constexpr int RCCL_SUCCESS = 0, RCCL_SUM = 0, RCCL_FLOAT16 = 6, RCCL_FLOAT32 = 7;
constexpr bool RCCL_DEFAULT_ON = true;   // the prompt sums through RCCL when FNX_PF_RCCL is unset: 103K prompt 33.3 -> 28.8 s, 61-task suite 56/61, first-token KL vs the layer split 0.0309; any failure is one line and the host sums (tests/split pins it)

struct RcclApi {
    void* handle = nullptr;
    int (*CommInitAll)(void** comms, int ndev, const int* devlist) = nullptr;
    int (*AllReduce)(const void* send, void* recv, size_t count, int dtype, int op, void* comm, void* stream) = nullptr;
    int (*GroupStart)() = nullptr;
    int (*GroupEnd)() = nullptr;
    int (*CommDestroy)(void* comm) = nullptr;
    const char* (*ErrorString)(int result) = nullptr;
};

// path: the library to load ("" = the defaults below; FNX_RCCL_LIB overrides everything, for tests and for a non-standard install).
// false with the reason in why.
// (tp-opt) the defaults are tried in order: the NVIDIA NCCL SONAMEs a CUDA install leaves on the library path first (libnccl.so.2 /
// libnccl.so.1, which on a box whose token exchange rides on NCCL over NVLink is the one that works), then AMD's librccl.so.1.
// A candidate that will not open is tried through to the next; when all of them fail the engine keeps the host
// exchange for the prompt sums with the one plain line the upstream loader already prints - and so is a candidate that opens but
// turns out not to carry the six entry points (an old SONAME, a stub, a renamed build): giving up there would throw away the next
// name on the list, which is exactly the one that works on a box where the first hit is a leftover. `why` then holds the reason of
// the last candidate tried. An explicit FNX_RCCL_LIB is taken alone:
// no fallback, so a pointed-at library says exactly what it is (tools/rccl_policy_test.cpp).
inline bool rccl_load(RcclApi& api, const std::string& path, std::string& why) {
    std::vector<std::string> tries;
    const char* over = std::getenv("FNX_RCCL_LIB");
    if (over && *over)
        tries.push_back(over);
    else if (!path.empty())
        tries.push_back(path);
    else
        tries = {"libnccl.so.2", "libnccl.so.1", "librccl.so.1"};
    for (const std::string& lib : tries) {
        api.handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!api.handle) {
            const char* e = dlerror();
            why = "cannot load " + lib + (e ? std::string(": ") + e : "");
            continue;
        }
        auto sym = [&](const char* name, auto& fn) {
            void* p = dlsym(api.handle, name);
            if (!p) {
                why = lib + " has no " + name;
                return false;
            }
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
            return true;
        };
        if (!sym("ncclCommInitAll", api.CommInitAll) || !sym("ncclAllReduce", api.AllReduce) || !sym("ncclGroupStart", api.GroupStart) ||
            !sym("ncclGroupEnd", api.GroupEnd) || !sym("ncclCommDestroy", api.CommDestroy) || !sym("ncclGetErrorString", api.ErrorString)) {
            dlclose(api.handle);
            api.handle = nullptr;
            continue;   // not this one: the next name on the list may be the real library
        }
        why = lib;
        return true;
    }
    return false;
}

// The decision: FNX_PF_RCCL=0 or unset (the default, until the screens say the prompt sums win): the host exchange, no line. =1: only with the peer push on (RCCL moves its
// data over the same peer links); open() loads the library and runs the self-test; any failure is the host exchange for the prompt sums and one plain line.
struct RcclDecision {
    bool use = false;
    std::string line;
};
inline RcclDecision rccl_decide(const char* env, bool peer_on, bool default_on, const std::function<bool(std::string&)>& open) {
    RcclDecision d;
    const bool wanted = env ? std::strcmp(env, "0") != 0 : default_on;
    if (!wanted) return d;
    if (!peer_on) {
        if (env) d.line = "RCCL is not used for the prompt sums: the peer push between the cards is off, so the host exchange stays.";   // (asked for: say so; the default stays quiet)
        return d;
    }
    std::string why;
    if (!open(why)) {
        d.line = "RCCL is not available here (" + why + "): the prompt sums use the host exchange.";
        return d;
    }
    d.use = true;
    d.line = "The prompt sums use RCCL (" + why + ").";
    return d;
}

}  // namespace fnx
