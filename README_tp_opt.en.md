# Strata NVLink TP optimization (rev.15 / 2080 Ti x2 + RTX 5070)

An English mirror of `README_tp_opt.md`. Instead of touching upstream behaviour, one spare
GPU (RTX 5070 / sm_120) is added to the 2-card tensor parallel setup (RTX 2080 Ti / sm_75)
as an **L2 = expert-only tier**, and the Host-spilled expert copies of Prefill are moved onto
compute that runs on the 5070. The changes stay inside `engine/src/split/src/` and
`engine/src/tpopt/`, and with the default (L2 off) the behaviour is identical to rev.15-M1b.

> **Note**: this strata-tp (strata fork) is **100 % written by AI** - the engine diff, the check scripts and this
> README included. What a human did was run the commands and collect the measured numbers.

| Item | Value |
|---|---|
| Upstream base SHA | `3142b66b96e9c0e775e30f620b11b191a1427c2a` (main) |
| Work branch | `feat/m3-1-l2-host-dedup` (rev.15-M3-1b) |
| Push remote (origin) | https://github.com/uta-kei02/Strata-tp.git |
| Upstream | push disabled (`git remote set-url --push upstream DISABLED`) |
| Diff | `git diff --binary 3142b66b9` (same content as the bundled `patches/strata_tp_opt.patch`) |

## Results (measured)

| Phase | 10K-class cold Prefill | Decode | Content |
|---|---|---|---|
| rev.15-M1b (2 cards) | 360.7 tok/s | 28.6 ms a round | Instrumentation only. Of 5,250 ms per pass, 4,255 ms `enqueued` = waiting on the OCuLink copy of 10,938 MiB x 2 Host-spilled experts |
| rev.15-M2a | - | 56.51 tok/s / 28.79 ms | 5070 reserved as the L2 tier (arena 7.88 GiB / selftest ok / pinned H2D 37.70 GB/s). Execution path still 2080 Ti only |
| rev.15-M2b-1 | 426.0 tok/s (+18.1 %) | 55.00 tok/s / 29.59 ms | Prefill computed on the 5070 (reuses the existing IQ3_S kernels / 2 pinned streaming slots) |
| rev.15-M2b-1 (zero-streamed fix) | 729.6 tok/s (+71.3 %) | 55.00 tok/s / 29.59 ms | card 0/1 experts **0 MiB copied in** / 10,938 MiB on the tier (the Host 17.6 GiB is swept by the 5070) |
| rev.15-M2b-1t | 832.7 tok/s (+14.1 %) | - | Four streamed slots with a 2-layer lookahead (Step B) / the 16-column chunked GDN variant (Step A) / a 4,096-row pass checked to fit (Step C) |
| rev.15-M2b-1u | 832.2 tok/s | 54.31 tok/s | Streamed slots split by what one layer needs (423 -> 852 MiB x 2) / a dedicated addend lane / the step table built before the layer / the `FNX_L2_PF_TIMING` breakdown |
| rev.15-M2b-1v | 988.1 tok/s (+18.7 %) | 54.62 tok/s | Chunk rows moved onto a 5070-only lane / a per-card addend event / the 2080 Ti and the 5070 run overlapped (pushed to origin) |
| rev.15-M2b-1w | **1,099.4 tok/s (+11.3 %)** / 1,248.1 tok/s for a 4,096-row pass alone | 55.04 tok/s / 28.82 ms | Addends come back as f16 rows (Host relay 1,920 -> 960 MiB per card and pass) / card 0 marks its rows before the router / `FNX_PF_PASS` is passed through (`PASS=4` fell to 913.0 tok/s and Decode 51.11, so it is not the default) |
| rev.15-M2b-2 | 1,099.4 tok/s (Prefill unchanged) | pending (was 55.04) | The Decode first tier is computed on the 5070 as well (one router pass distributes picks to L1 / L2 / L3 exclusively). Implemented and statically checked; the on-hardware bench is pending |
| rev.15-M2b-2 (measured) | 895.1 tok/s (`FNX_L2_DEC=1`) | **55.46 tok/s / 28.42 ms** | 5 of 5 passes, no Xid, no ring error, 0 MiB copied in on every pass; the 5070 answered 82,252 of 82,432 requests (card 0 / card 1 symmetric), 0 retires |
| rev.15-M2c-arch | **1,410.3 tok/s (+71 %)** | pending (was 55.46) | The MoE MMA picks its instruction width by device arch (the 5070 `sm_120` uses `HMMA.16816.F32`, the 2080 Ti keeps the old width): `tier compute` 1,912 -> 279 ms |
| rev.15-M3-1a | - | - | Turing gets its tensor cores back too: the width choice in `moe_mma.cuh` now opens at `__CUDA_ARCH__ >= 750`, and `sm_75` builds the K=16 dot from two `mma.m16n8k8`. SASS check: `sm_75` shows `HMMA.1688.F32` 640 times and no `HMMA.884`, `sm_120` keeps `HMMA.16816.F32` 320 times |
| rev.15-M3-1 (three measured runs) | 1,419.4 / 1,432.3 / 1,418.3 tok/s | 50.15 / 51.99 / **51.48 tok/s / 30.54 ms** | The page-locked Host originals of the experts resident in the 5070 arena are handed back one layer at a time (1,713 experts / 3.38 GiB, 2,622 / 5.26 GiB, 2,563 / 5.14 GiB). What is handed back only exists in the arena, so a short pass gives the tier only its resident experts (5 tok fixed 308.0 / 300.0 / 311.0 ms). Major faults 454.5 -> 344.0 per s. No Xid and no OOM in any run |
| rev.15-M3-1b (production 8010) | - | short round trips ok | A tail pass shorter than `MAXT` = 16 rows now falls back onto the tier instead of reading a handed-back original. The repro passes at 2,089 tokens (41 left, 3.07 s) and 4,110 tokens (14 left, 6.96 s) with 0 `l2pf: off` lines and 0 `stage copy` errors. On 8010 the drop returned 2,620 experts / 5.25 GiB (10.35 s) alongside `MLOCK_MAX_GIB=10` |

The fastest Prefill so far is M2c-arch at **1,410.3 tok/s**; the next **1,500 tok/s** (7,609 tokens
within 5.1 s, today 6.9 s) lives on the first-tier swap and the K/V side, not on the 5070's instruction
width. M3-1 is done up to handing the Host copies back; next is M3-2 (the 5070 arena as the entry-level
primary cache, a 3-tier promotion path). The design and the reserved hooks are in
`m2_design_rtx5070_l2.md` and `m3_plan.md` inside the bundled `docs/`.

## Switches read by the engine (default off)

| Purpose | env | Default |
|---|---|---|
| 5070 as L2 (visible ordinal) | `FNX_L2_GPU` | unset = no L2 |
| L2 tier size / desktop reserve | `FNX_L2_GIB` / `FNX_L2_RESERVE_GIB` | 8.0 / 2.0 GiB |
| L2 selftest / fill amount | `FNX_L2_SELFTEST` / `FNX_L2_FILL_MIB` | on / auto |
| Prefill on the 5070 | `FNX_L2_PF` / `FNX_L2_PF_STREAM` | 1 (when L2 is on) / 1 |
| Number of streamed slots (overlap) | `FNX_L2_PF_PIPE` | 4 (0 or 1 = 2 slots, the rev.15-M2b-1 shape) |
| Thickness of one slot (floor) | `FNX_L2_PF_STAGE_MIB` | 320 MiB (the real size is raised automatically from free VRAM) |
| addend ring depth / max rows per tier job | `FNX_L2_PF_RING` / `FNX_L2_PF_ROWS` | 2 / 0 = no limit |
| Rows per chunk / chunks per Prefill pass | `FNX_PF_ROWS` / `FNX_PF_PASS` | 1024 / 2 (the adopted shape is 2048 x 2 = 4,096 rows; `PASS=4` only with spare VRAM) |
| Addends returned as f16 rows (0 = f32, byte for byte rev.15-M2b-1v) | `FNX_L2_PF_FP16` | 1 |
| card 0 marks its rows before the router (0 = right after top-k) | `FNX_L2_PF_EARLY_IN` | 1 |
| The 5070 breakdown, one line a pass (H2D / GEMM / wait) | `FNX_L2_PF_TIMING` | 0 (no instrumentation overhead while off) |
| Decode also computed on the 5070 | `FNX_L2_DEC` | 1 (when L2 is on) / never exported with L2 off |
| Decode L2 hit statistics, one line per 512 requests | `FNX_L2_DEC_STATS` | 0 |
| Picks one window may send to the 5070 (retreat knob) | `FNX_L2_DEC_MAX` | 0 = no limit |
| Decode split: L1/L2/L3 counts and the host worker / 5070 ms (3 lines a sample) | `FNX_DEC_STATS` | 0 (a branch only while off) |
| Windows folded into one sample | `FNX_DEC_STATS_EVERY` | 2048 |
| Hand back the Host copy of the experts resident in the 5070 arena after load (M3-1) | `FNX_L2_DROP_HOST` | 0 (1 arms only when the arena, `FNX_L2_PF`, `FNX_L2_DEC` and both cards' tables are all in place; otherwise it warns and stays off; 2 also hands back the resident experts that keep a first-tier swap slot) |
| Pre-study for moving the 5070 arena around (host-pinned probe + a dry-run over the route counts, M3-2 Step 0) | `FNX_L2_DYN` | 0 (a branch only while off; 1 runs probes and a dry run that move no weights, tables or maps) |
| How many swaps the dry run should price | `FNX_L2_DYN_N` | 0 = price up to the resident count |
| Chunked GDN (sub-chunk width auto-selected from the card's opt-in shared memory) | `FNX_PF_GDN_CHUNK` / `FNX_GDN_FORCE_REC` | 1 (sm_75 = 16 columns = 57,988 B) / 0 |
| GDN numeric compare / timing | `FNX_PF_GDN_CHECK` / `FNX_GDN_TIME` | 0 |
| Diagnostics when Prefill fails | `FNX_PF_ERRDBG` | 0 |

The launch and measurement wrappers (`run_strata_tp.sh`, `bench_switch_8002.sh`,
`strata_tp.conf`) and the detailed docs live in the delivery directory that carries this tree
(the `out/` root). `STRATA_TP_*` are wrapper-side names and reach the engine as the `FNX_*`
names in the table above. `docs/` inside this repository is upstream documentation.

## Rebuild

```bash
cmake -S engine -B build_sm75 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_SPLIT=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES='75;120' -DSTRATA_SPLIT_ARCH=75 \
  -DSTRATA_SPLIT_FAT_ARCHS=120 '-DSTRATA_SPLIT_CARDS=2;3' \
  -DSTRATA_SPLIT_PF_JOB=64 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build_sm75 --target strata-split-2 strata-split-3 -j 6
```

`engine/strata-split-*-sm75` (116 MB each) exceed GitHub's 100 MB limit and are therefore
**not tracked by git** (`.gitignore`). The sm_120 SASS is already inside the fatbin, so nothing
has to be rebuilt for the 5070.

## Where the changes are

- `engine/src/split/src/l2_offload.{hpp,cu}` - the 5070 arena, pinned staging, M2a selftest
- `engine/src/split/src/l2_pf.{hpp,cu}` - Prefill on the 5070 (resident / streamed tier, slot overlap and its ordering invariants)
- `engine/src/split/src/l2_dec.{hpp,cu}` - the Decode merge onto the 5070 (exclusive 3-way routing, per-card slices, running beside the host worker, retire)
- `engine/src/split/src/prefill.cu` - zero-weight masking, hand-off of the addends, pass progress
- `engine/src/split/src/kernels.cu` / `kernels.hpp` and `gdn_prompt.cuh` - automatic sub-chunk width for chunked GDN, the `CpuMoeBox` L2 hand-off, the index clamp inside `merge_rows`
- `engine/src/split/src/engine.cu` and `engine_impl.hpp` - the mapped pinned `CpuMoeBox` (`cudaHostAllocMapped | cudaHostAllocPortable`) and its L2 members
- `engine/src/tpopt/` - the wait guard (`WaitGuard`) and peer synchronization
- `engine/src/split/src/hip/rccl_dl.hpp` - dlopen candidates reduced to generic names (a local NCCL comes from `FNX_RCCL_LIB`)
- The M3 hooks in `engine/src/split/src/l2_offload.hpp` (per-slot hit statistics, the drop-to-5070 receiver) are kept unused
- Checks: bundled `_work/guard_check.py` (source guards) / `_work/rc71_check.sh` (shutdown protocol) / `_work/refcheck_docs.py`

## Operational rules

- Never drop the running server (measurement switches to another port and recovers automatically).
- No GPU-occupying experiments (builds run under `nice`, the on-hardware bench is run by a human).
- No personal environment paths (a user's home directory and the like) inside `engine/src`.
