// engine/src/tpopt/tp_opt.hpp - the three limits this TP build has to live inside, in one place: the share of each card's
// VRAM the resident experts may take (MAX VRAM の 80% by default), the whole-box host RAM ceiling (OS 込み 56 GiB by
// default) and the CPU share (50% by default). Every one of them is a parameter, never a compiled-in constant: command line
// (the launcher), environment, and the engine's FNX_* variables all reach the same numbers.
//
// Why the VRAM share and not just "all that is free": the 5070 is an optional third card, and the hot experts that miss on
// the two 2080 Ti's are meant to land there. A card that grabs everything free leaves nothing for the KV cache, the arena
// and the desktop compositor, and the first big prompt OOMs or pushes the box into swap - the exact failures this build is
// supposed to stop. Capping at a share of the card's own total is also what makes an asymmetric set (22 GiB + 22 GiB + 12
// GiB) plan sanely: the small card is never asked to hold what the big ones hold.
//
// Why the CPU share is read here and not just set by the launcher: with CPU Hybrid TP on, the workers are barrier-synced,
// so 20 workers inside a 50%-of-32-threads quota means every worker runs at half speed and one stalled worker stalls the
// barrier (measured on the release: 10 workers per card, 4 threads left for the OS and the recorder). Clamping the worker
// count to the CPUs actually allowed is cheaper than paying that stall.
//
// Header only, no CUDA and no GPU: it is compiled into the engine and can be read by a plain C++ test.
#pragma once

#include <sched.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#if defined(__linux__)
#include <cerrno>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "strata/kernels/cpu/cgroup_cpu.hpp"

namespace tpopt {

inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// a number from the environment: unset, empty or unparseable keeps the default
inline double env_num(const char* name, double dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    if (!end || end == v) return dflt;
    return x;
}

// MAX VRAM の使い上限（割合 0..1）。FNX_VRAM_PCT wins, then STRATA_TP_VRAM_PCT, default 80%.
inline double vram_fraction() {
    const double pct = env_num("FNX_VRAM_PCT", env_num("STRATA_TP_VRAM_PCT", 80.0));
    return clampd(pct, 1.0, 100.0) / 100.0;
}

// What one card may spend on resident experts: never more than vram_fraction() of its own total (the rule the RTX 5070
// overflow tier runs by too), never more than what is actually free on it, and always with `reserve_gib` of KV cache and
// workspace left out of both. Below that floor it is better to keep experts in host RAM than to fail the load.
inline double expert_budget_gib(double total_gib, double free_gib, double reserve_gib) {
    if (total_gib <= 0.0 || free_gib <= 0.0) return 0.0;
    const double capped = total_gib * vram_fraction();
    const double room = (capped < free_gib ? capped : free_gib) - (reserve_gib > 0.0 ? reserve_gib : 0.0);
    return room < 2.0 ? 0.0 : room;
}

// The whole-box host RAM ceiling, OS and page cache included: STRATA_TP_MEM_CAP_GIB, then FNX_RAM_LIMIT_GB, default 56 GiB.
// (The launcher also sets a cgroup memory.max with the same number; this is the soft cap for when it cannot.)
inline double mem_cap_gib() {
    const double c = env_num("STRATA_TP_MEM_CAP_GIB", env_num("FNX_RAM_LIMIT_GB", 56.0));
    return c > 0.0 ? c : 56.0;
}

// /proc/meminfo: total and MemAvailable in GiB; false when the file cannot be read (a container without /proc)
inline bool meminfo(double& total_gib, double& avail_gib) {
    std::ifstream f("/proc/meminfo");
    if (!f.is_open()) return false;
    double total_kb = 0.0, avail_kb = 0.0;
    std::string key;
    long long val = 0;
    char unit[32] = {};
    while (f >> key >> val) {
        f.getline(unit, sizeof(unit));
        if (key == "MemTotal:") total_kb = (double) val;
        else if (key == "MemAvailable:") avail_kb = (double) val;
    }
    if (total_kb <= 0.0) return false;
    total_gib = total_kb / (1024.0 * 1024.0);
    avail_gib = avail_kb / (1024.0 * 1024.0);
    return true;
}

// How much RAM is still the engine's to use under the ceiling: the ceiling minus what the rest of the box is holding right
// now (MemTotal - MemAvailable: the kernel counts clean page cache as available, so a warm GGUF cache does not count here),
// and never more than MemAvailable itself. The second bound matters when the ceiling is above what the box has (a 62 GiB
// machine asked to live under 80 GiB, or a desktop eating half of it): the arithmetic alone would hand out RAM that is not
// there, and the host would pay for it in swap rather than the planner noticing.
inline double ram_allowance_gib() {
    double total = 0.0, avail = 0.0;
    if (!meminfo(total, avail)) return -1.0;   // unknown: the caller must not fail the start over a number it cannot read
    const double room = mem_cap_gib() - (total - avail);
    return avail < room ? avail : room;
}

// CPUs the box will really give this process: the affinity mask, the cgroup cpu quota (what systemd-run -p CPUQuota= or
// Docker --cpus set) and STRATA_TP_CPU_PCT / FNX_CPU_PCT (default 50% of the affinity mask), all as the smallest of them.
inline double cpu_allow() {
    double allow = 0.0;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) allow = (double) CPU_COUNT(&set);
    if (allow <= 0.0) allow = (double) std::thread::hardware_concurrency();
    if (allow <= 0.0) allow = 1.0;
    std::string self;
    if (std::ifstream("/proc/self/cgroup") >> std::noskipws) {
        std::string line;
        std::ifstream f("/proc/self/cgroup");
        while (std::getline(f, line)) self += line + "\n";
    }
    const double quota = strata::kernels::cpu::cgroup_cpu_quota_at("/sys/fs/cgroup", self);
    if (quota > 0.0 && quota < allow) allow = quota;
    const double pct = env_num("STRATA_TP_CPU_PCT", env_num("FNX_CPU_PCT", 50.0));
    const double by_pct = allow * clampd(pct, 1.0, 100.0) / 100.0;
    return by_pct < 1.0 ? 1.0 : by_pct;
}

// worker threads: never more than the wanted number, the hard ceiling, or the CPUs allowed
inline int clamp_threads(int want, int hard_max) {
    int n = want > 0 ? want : 1;
    if (hard_max > 0 && n > hard_max) n = hard_max;
    const int allowed = (int) cpu_allow();
    if (n > allowed) n = allowed < 1 ? 1 : allowed;
    return n < 1 ? 1 : n;
}

// FNX_CPU_THREADS is a budget for the whole process, not per card: the CPU Hybrid TP loop starts its workers once per card,
// so a wanted 8 across three cards is 24 threads inside a quota that was sized for 8, and every barrier pays for the
// difference (a throttled worker holds the others). Split the allowed count over the cards, and never below one.
inline int clamp_threads_per_card(int want, int hard_max, int ncards) {
    const int total = clamp_threads(want, hard_max);
    const int cards = ncards > 0 ? ncards : 1;
    const int each = total / cards;
    return each < 1 ? 1 : each;
}

// One step of waiting on a flag another thread will set. Pause is right for the tens of microseconds the handover really
// takes, and wrong for anything longer: inside a CPU quota a spinning thread is a running thread, so a waiter that keeps
// pausing takes CPU away from the thread that was about to release it. After `pause_limit` steps the wait sleeps instead.
inline void pause_cpu() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// How long a wait may burn CPU before it sleeps, and how deep the sleep is. A pause step is ~54 ns on the box this was
// measured on, so the default window of 20,000 steps is ~1.1 ms of wall clock: enough for the real handover, short enough
// that a stalled worker does not eat the quota. The window and the sleep are start-up parameters because the release
// (8000) waits with pure pause and comes back 47 ms a round faster on long prompts - sleeping at 1 ms granularity costs
// ~0.5 ms per barrier, twice per layer (measured with out/_work/barrier_wake_probe.c), and inside a CPU quota that sleep
// also delays the thread that has work. FNX_TP_SPIN_PAUSES is the window in pause steps, FNX_TP_SLEEP_MS is the sleep
// granularity in ms and 0 there means "never sleep, keep pausing" (the release's behaviour). Both are read once per
// process: they sit inside a spin loop, so getenv must not run on every step.
inline long long spin_pause_limit() {
    static const long long v = [] {
        const double x = env_num("FNX_TP_SPIN_PAUSES", 20000.0);
        return x < 1.0 ? 1LL : (long long) x;
    }();
    return v;
}

inline int spin_sleep_ms() {
    static const int v = (int) clampd(env_num("FNX_TP_SLEEP_MS", 1.0), 0.0, 1000.0);
    return v;
}

inline void backoff(long long& spins, long long pause_limit, int sleep_ms) {
    if (++spins < pause_limit || sleep_ms <= 0) {
        pause_cpu();
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
}

// every wait in the CPU Hybrid TP loop calls this, so the window and the granularity stay tunable without a rebuild
inline void backoff(long long& spins) { backoff(spins, spin_pause_limit(), spin_sleep_ms()); }

// ---------------------------------------------------------------- rev.14: 待機は「時間」と「notify」で決める
// Why this exists, with the numbers: the pause-count window above decides "the handover is late" after ~1.1 ms, but the
// gap a worker really waits through is not one layer. The manager only moves phase1_id/phase2_id on layers that miss
// (engine/src/split/src/engine.cu:512), so a phase-1 wait spans several layers, every miss event paid a ~0.5 ms wake-up,
// and a 3,019-token decode came out at 104.66 ms a round (out/logs/bench_8002.log:4181). Turning the sleep off reaches
// 30.32 ms a round but burns 700% of a core per idle worker, and inside a 50%-of-32 CPUQuota that spin crosses the frame
// (nr_throttled +645 / +57,517 ms in the same log, :4580).
// So the wait decides with wall clock, and past the window it gives the CPU back. For the two waits this process itself
// publishes (phase1_id / phase2_id) the wait is a futex on a companion word that the publisher wakes: measured p50 0.006 ms
// and worst 0.10 ms for the whole handover, against 0.5 ms for the sleep it replaces, and the idle cost of seven workers
// falls from 14,000 ms to 10 ms of CPU per 2 s (out/_work/probe3_rev14_v0b.txt). The waits whose writer is a CUDA kernel
// (hbox->req_seq) or a counter nobody watches cannot be woken, and those keep a timer park instead.
//
// "spin" puts rev.13 back exactly (pause-count window, FNX_TP_SLEEP_MS, 0 = pure pause) without a rebuild. That is the
// escape hatch if a kernel ever stops answering the publish path; it is also how 8000 (the release build, no quota) waits.
inline bool wait_mode_spin() {
    static const bool spin = [] {
        const char* v = std::getenv("FNX_TP_WAIT_MODE");
        if (!v || !*v) v = std::getenv("STRATA_TP_WAIT_MODE");
        return v && *v && std::string(v) == "spin";
    }();
    return spin;
}

// How long a wait may still burn CPU before it parks, in microseconds of wall clock (FNX_TP_HOT_US, default 2,000).
// A decode here advances a layer every ~397 us and a miss event every ~1.2 ms (30.32 ms a round / 1.59 tok / 48 layers),
// so 2 ms keeps the real handover inside the pause and adds nothing, while an idle engine parks after 2 ms and stops
// taking quota. Raising it buys nothing measurable: waking a parked worker is already ~0.01-0.10 ms.
inline long long hot_window_us() {
    static const long long v = (long long) clampd(env_num("FNX_TP_HOT_US", 2000.0), 0.0, 600000.0);
    return v;
}

// The park is bounded (FNX_TP_PARK_MS, default 50 ms). It is a safety valve, not the normal path: all 2,380 parks in the
// measurement ended on the notify, and without a bound a lost wake-up would hang a barrier instead of costing one gap.
inline int park_timeout_ms() {
    static const int v = (int) clampd(env_num("FNX_TP_PARK_MS", 50.0), 1.0, 5000.0);
    return v;
}

// The window for the waits that cannot be woken at all (FNX_TP_IDLE_US, default 20,000 us). A decode never reaches it, so
// those waits keep pausing exactly like the release while there is work, and only a stopped request stream sleeps.
inline long long idle_window_us() {
    static const long long v = (long long) clampd(env_num("FNX_TP_IDLE_US", 20000.0), 1000.0, 6000000.0);
    return v;
}

inline long long now_ns() {
    return (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A wait's clock. The first 512 steps are always "only just started" (a pause is ~54 ns, so ~28 us), and after that the
// monotonic clock is read once per 512 steps: now_ns() costs about as much as a pause, so reading it every step would
// double the spin it exists to cheapen. One HotWait lives per wait, not per step.
struct HotWait {
    long long spins = 0;
    long long t0 = 0;
    void reset() { spins = 0; t0 = 0; }
    bool inside(long long window_us) {
        if (window_us <= 0) return false;
        if (++spins < 512) return true;
        if (t0 == 0) t0 = now_ns();
        return (now_ns() - t0) / 1000 < window_us;
    }
};

// One parkable wait. The futex word has to be 32 bit, which is why the workers do not park on the 64 bit phase counter
// itself; `waiters` lets the publisher skip the syscall while nobody is parked.
struct WaitSlot {
    alignas(64) std::atomic<unsigned> epoch{0};
    std::atomic<long long> waiters{0};
};

// futex rather than std::condition_variable: a condition_variable cannot wait on an atomic counter without taking a
// mutex on every barrier, and these barriers run twice a layer for every token.
inline void futex_wait32(std::atomic<unsigned>* word, unsigned expect, int timeout_ms) {
#if defined(__linux__)
    struct timespec ts { timeout_ms / 1000, (timeout_ms % 1000) * 1000000L };
    ::syscall(SYS_futex, reinterpret_cast<const unsigned*>(word), FUTEX_WAIT_PRIVATE, static_cast<int>(expect), &ts, nullptr, 0);
    // EAGAIN (the word already moved) and EINTR are fine: the caller re-reads the work counter.
#else
    (void) word; (void) expect;
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
#endif
}

inline void futex_wake32(std::atomic<unsigned>* word) {
#if defined(__linux__)
    ::syscall(SYS_futex, reinterpret_cast<const unsigned*>(word), FUTEX_WAKE_PRIVATE, INT_MAX);
#else
    (void) word;
#endif
}

// The publisher: call immediately after the work counter moved. The seq_cst bump and the seq_cst read of `waiters` are one
// half of the Dekker handshake; wait_for_work() reading the work counter after it registered is the other half.
inline void publish(WaitSlot& slot) {
    slot.epoch.fetch_add(1, std::memory_order_seq_cst);
    if (slot.waiters.load(std::memory_order_seq_cst) > 0) futex_wake32(&slot.epoch);
}

// The wait on a counter this process publishes. has_work() must be a seq_cst read of the counter that decides the wait,
// and it is called here after registering as a waiter: if the publisher saw `waiters` as 0, the registration came after it
// in the total order, so this thread's read of the work counter must see that publish. That is the whole reason there is
// no 50 ms tail (a probe that parked on the futex word alone showed p99 51 ms; the re-check removed it).
template <typename HasWork>
inline void wait_for_work(WaitSlot& slot, HotWait& hw, HasWork has_work) {
    if (wait_mode_spin()) {
        backoff(hw.spins, spin_pause_limit(), spin_sleep_ms());
        return;
    }
    if (hw.inside(hot_window_us())) {
        pause_cpu();
        return;
    }
    const unsigned snap = slot.epoch.load(std::memory_order_relaxed);
    slot.waiters.fetch_add(1, std::memory_order_seq_cst);
    if (!has_work()) futex_wait32(&slot.epoch, snap, park_timeout_ms());
    slot.waiters.fetch_sub(1, std::memory_order_seq_cst);
}

// The waits with nobody to wake them (the manager's req_seq wait, whose writer is the CUDA side, and the phase barriers).
// Pause until FNX_TP_IDLE_US without progress, then sleep 1 ms at a time: measured 0.000 ms added at the 0.2 ms and 1.0 ms
// gaps a decode really has, and an idle process stops holding the quota.
inline void wait_for_progress(HotWait& hw) {
    if (wait_mode_spin()) {
        backoff(hw.spins, spin_pause_limit(), spin_sleep_ms());
        return;
    }
    if (hw.inside(idle_window_us())) {
        pause_cpu();
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// what the start-up line prints, so a log always records the policy the process is actually running
inline std::string wait_policy_text() {
    if (wait_mode_spin())
        return "spin (" + std::to_string(spin_pause_limit()) + " pauses then " + std::to_string(spin_sleep_ms()) +
               " ms sleep, 0 = pure pause)";
    return "park (pause " + std::to_string(hot_window_us()) + " us then futex park, " + std::to_string(park_timeout_ms()) +
           " ms safety valve; waits with no notifier: " + std::to_string(idle_window_us()) + " us then 1 ms)";
}

// ---------------------------------------------------------------- TP 同期 watchdog
// A collective or a stream sync that never comes back is the failure mode this cannot see from the outside: the process sits
// in cudaStreamSynchronize forever, the server answers nothing, and only a restart clears it. So the wait is armed with a
// deadline: when it passes, one plain line says which wait timed out and the process exits non-zero, which is something a
// supervisor (or a human) can act on. FNX_TP_WATCHDOG_S is the deadline (default 300 s); 0 turns the watchdog off entirely,
// which is what a long prefill on a cold disk may want.
inline double watchdog_seconds() { return env_num("FNX_TP_WATCHDOG_S", 300.0); }

inline std::atomic<long long>& wd_deadline_ms() {
    static std::atomic<long long> d{0};
    return d;
}
inline std::atomic<const char*>& wd_label() {
    static std::atomic<const char*> l{""};
    return l;
}

inline long long now_ms() {
    return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void watchdog_thread();

// start the checker once, only when the watchdog is on
inline void watchdog_start() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (watchdog_seconds() <= 0.0) return;
        std::thread(watchdog_thread).detach();
    });
}

inline void watchdog_arm(const char* what) {
    const double s = watchdog_seconds();
    if (s <= 0.0) return;
    watchdog_start();
    wd_label().store(what, std::memory_order_relaxed);
    wd_deadline_ms().store(now_ms() + (long long) (s * 1000.0), std::memory_order_release);
}

inline void watchdog_clear() { wd_deadline_ms().store(0, std::memory_order_release); }

// the wait armed for its whole scope: a throw out of the middle of it (a kernel that reported an error) must not leave the
// deadline running and kill a process that is already on its way out
struct WaitGuard {
    explicit WaitGuard(const char* what) { watchdog_arm(what); }
    ~WaitGuard() { watchdog_clear(); }
    WaitGuard(const WaitGuard&) = delete;
    WaitGuard& operator=(const WaitGuard&) = delete;
};

inline void watchdog_thread() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        const long long deadline = wd_deadline_ms().load(std::memory_order_acquire);
        if (deadline == 0 || now_ms() < deadline) continue;
        const char* what = wd_label().load(std::memory_order_relaxed);
        std::fprintf(stderr, "fnx: TP watchdog: %s did not return within %.0f s (the cards are not answering); exiting so the supervisor can restart%c",
                     what && *what ? what : "a card wait", watchdog_seconds(), 10);
        std::fflush(stderr);
        wd_deadline_ms().store(0, std::memory_order_release);
        std::_Exit(70);
    }
}

}  // namespace tpopt
