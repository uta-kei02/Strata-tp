# Strata NVLink TP 最適化（rev.15 / 2080 Ti x2 + RTX 5070）

本家 Strata に手を入れずに、テンソル並列 2 枚（RTX 2080 Ti / sm_75）へ余剰 1 枚
（RTX 5070 / sm_120）を **L2 = エキスパート専用枠**として足し、Prefill の Host 退避
expert コピーを 5070 側の演算へ載せ替えた改造です。変更は `engine/src/split/src/`
と `engine/src/tpopt/` に集約し、既定（L2 off）では rev.15-M1b と挙動が同一です。

> **注記**: この strata-tp（strata-fork）は **100% AI** で書かれています（engine 側の差分・検証スクリプト・この README を
> 含む。人が担当したのは実行コマンドと実測値の回収のみ）。

| 項目 | 値 |
|---|---|
| 本家 base SHA | `3142b66b96e9c0e775e30f620b11b191a1427c2a`（main） |
| 作業ブランチ | `feat/m3-2-dynamic-l2`（rev.15-M3-2 step 1。prev `feat/m3-1-l2-host-dedup`） |
| push 先（origin） | https://github.com/uta-kei02/Strata-tp.git |
| 本家（upstream） | push 禁止（`git remote set-url --push upstream DISABLED`） |
| 差分 | `git diff --binary 3142b66b9`（同梱の `patches/strata_tp_opt.patch` と同一内容） |

## 成果（実測）

| フェーズ | 10K 級 cold Prefill | Decode | 内容 |
|---|---|---|---|
| rev.15-M1b（2 枚） | 360.7 tok/s | 28.6 ms a round | 計装のみ。1 パス 5,250 ms のうち enqueued 4,255 ms = Host 退避 expert 10,938 MiB x 2 枚の OCuLink コピー待ち |
| rev.15-M2a | - | 56.51 tok/s / 28.79 ms | 5070 を L2 枠として確保（arena 7.88 GiB / selftest ok / pinned H2D 37.70 GB/s）。実行経路はまだ 2080 Ti のみ |
| rev.15-M2b-1 | 426.0 tok/s（+18.1 %） | 55.00 tok/s / 29.59 ms | Prefill を 5070 で演算（既存 IQ3_S カーネルを再利用 / 2 枠 pinned ストリーミング） |
| rev.15-M2b-1（0 streamed 解消） | 729.6 tok/s（+71.3 %） | 55.00 tok/s / 29.59 ms | card 0/1 experts **0 MiB copied in** / 10,938 MiB on the tier（Host 17.6 GiB を 5070 側へ掃引） |
| rev.15-M2b-1t | 832.7 tok/s（+14.1 %） | - | streamed tier の 4 枠化 + 2 層先読み（Step B）/ chunked GDN の 16 列変種（Step A）/ 1 pass 4,096 行の通過確認（Step C） |
| rev.15-M2b-1u | 832.2 tok/s | 54.31 tok/s | streamed slot を 1 層必要量で自動分割（423 → 852 MiB x 2）・addend 専属 lane・step 表の層前組み立て・内訳ログ `FNX_L2_PF_TIMING` |
| rev.15-M2b-1v | 988.1 tok/s（+18.7 %） | 54.62 tok/s | chunk 行を 5070 専属 lane へ / card 別の addend event / 2080 Ti と 5070 の重複実行（origin へ push 済み） |
| rev.15-M2b-1w | **1,099.4 tok/s（+11.3 %）** / 4,096 行 pass 単体 1,248.1 | 55.04 tok/s / 28.82 ms | 加算項を f16 行へ（host 中継 1,920 → 960 MiB / card・pass）・card 0 の行 mark を router 前へ・`FNX_PF_PASS` 透過（`PASS=4` は 913.0 tok/s・Decode 51.11 へ低下で不採用） |
| rev.15-M2b-2 | 1,099.4 tok/s（Prefill は不変） | 実測待ち（従来 55.04） | Decode の 1 軍を 5070 の L2 でも演算（router が 1 回で L1 / L2 / L3 へ排他分類）。実装・静的検証済み / 実機ベンチ待ち |
| rev.15-M2b-2（実測） | 895.1 tok/s（`FNX_L2_DEC=1`） | **55.46 tok/s / 28.42 ms** | 5/5 完走・Xid 0・ring error 0・全パス 0 MiB copied in。5070 の hit 82,252 / 82,432 request（card 0/1 対称）/ retire 0 件 |
| rev.15-M2c-arch | **1,410.3 tok/s（+71 %）** | 実測待ち（従来 55.46） | MoE MMA の命令幅を device arch で選択（5070 = `sm_120` は `HMMA.16816.F32`、2080 Ti = `sm_75` は従来どおり）。`tier compute` 約 1,912 → 279 ms |
| rev.15-M3-1a | - | - | Turing 側でも Tensor Core を使わせる（`moe_mma.cuh` の命令幅選択を `__CUDA_ARCH__ >= 750` へ広げ、`sm_75` は `mma.m16n8k8` を 2 命令で組む）。SASS 実測 = `sm_75` は `HMMA.1688.F32` 640 本で `HMMA.884` は 0 本、`sm_120` は `HMMA.16816.F32` 320 本のまま |
| rev.15-M3-1（実測 3 連） | 1,419.4 / 1,432.3 / 1,418.3 tok/s | 50.15 / 51.99 / **51.48 tok/s / 30.54 ms** | 5070 arena 常駐 expert の RAM 側 page-locked 原本を 1 層ずつ返却（1,713 exp / 3.38 GiB、2,622 exp / 5.26 GiB、2,563 exp / 5.14 GiB）。返した分は arena にしか無いので短文 pass は常駐分だけ tier へ渡す（5 tok fixed 308.0 / 300.0 / 311.0 ms）。majflt 454.5 → 344.0 回/s。全 run Xid 0・OOM 0 |
| rev.15-M3-1b（本番 8010） | - | 短文往復 OK | 長文末尾の端数 pass（残りが `MAXT` = 16 行に満たない）も、返した原本を読みに行かず tier へ回落。再現テスト 2,089 tok（余り 41）3.07 s と 4,110 tok（余り 14）6.96 s が通り、`l2pf: off` 0 件・`stage copy` error 0 件。本番では返却 2,620 exp / 5.25 GiB（10.35 s）で `MLOCK_MAX_GIB=10` と同居 |
| rev.15-M3-2 step 1（実測 3 連） | **1,445.4 tok/s（+2.5 %）** | **52.25 tok/s / 30.24 ms** | 5070 アレーナを入口の 1 次キャッシュへ。pass 境界で route 回数上位の expert を arena と 1-for-1 に入れ替え（`FNX_L2_DYN=2` / `N=64`）、追い出した expert へ空地の RAM 原本を戻すので RAM は増えない。カバー率は 10K prefill 帯 **15.0% → 23.4%**、decode 帯 **12.7% → 16.4%**。`5 tok fixed` 293.0 → 248.0 ms、majflt 411.2 → 399.3 回/s。`VERIFY=1` では 10 / 10 境界・640 ペアが一致（mismatch 0）。全 run 5/5 完走・Xid 0・OOM 0 |

Prefill の現最速は M3-2 step 1 の **1,445.4 tok/s**（M2c-arch の 1,410.3 から +2.5 %）で、次の
**1,500 tok/s**（7,609 tok を 5.1 s 以内 / 現在 5.3 s）は 1 軍 swap の被覆と K/V 側の話です。M3-1 は RAM
側の二重保持を返す段階まで、M3-2 step 1 は pass 境界の 1-for-1 入替まで完了しました。次は M3-2 step 2
（decode 中の hit 駆動で逐次昇格）と K/V 側。設計と拡張点は同梱 `docs/` の `m2_design_rtx5070_l2.md` と
`m3_plan.md` に置いてあります。

## engine が読むスイッチ（既定 off / 数値は実機で確認）

| 目的 | env | 既定 |
|---|---|---|
| 5070 を L2 に（可視 ordinal） | `FNX_L2_GPU` | 未設定 = L2 なし |
| L2 枠の厚み・デスクトップ予約 | `FNX_L2_GIB` / `FNX_L2_RESERVE_GIB` | 8.0 / 2.0 GiB |
| L2 selftest・充填量 | `FNX_L2_SELFTEST` / `FNX_L2_FILL_MIB` | on / 自動 |
| Prefill を 5070 で演算 | `FNX_L2_PF` / `FNX_L2_PF_STREAM` | 1（L2 有効時）/ 1 |
| streamed slot の数（重複化） | `FNX_L2_PF_PIPE` | 4（0・1 で 2 枠 = rev.15-M2b-1 の形） |
| slot 1 枠の厚み（床） | `FNX_L2_PF_STAGE_MIB` | 320 MiB（実枠は空き VRAM から自動引き上げ） |
| addend リング深さ・tier 1 job 行数 | `FNX_L2_PF_RING` / `FNX_L2_PF_ROWS` | 2 / 0 = 上限なし |
| 1 chunk の行数 / 1 pass の chunk 数 | `FNX_PF_ROWS` / `FNX_PF_PASS` | 1024 / 2（採用値は 2048 x 2 = 4,096 行、`PASS=4` は余力があるとき） |
| 加算項を f16 行で返す（0 で fp32 = rev.15-M2b-1v と同じ byte） | `FNX_L2_PF_FP16` | 1 |
| card 0 の行 mark を router 前へ（0 で top-k 直後） | `FNX_L2_PF_EARLY_IN` | 1 |
| 5070 側の内訳を 1 pass 1 行（H2D / GEMM / wait） | `FNX_L2_PF_TIMING` | 0（off のあいだ計装 overhead なし） |
| Decode を 5070 の L2 でも演算 | `FNX_L2_DEC` | 1（L2 有効時）/ L2 off では未 export |
| Decode の L2 hit 統計（512 request に 1 行） | `FNX_L2_DEC_STATS` | 0 |
| 1 window が 5070 へ送る pick 上限（撤退ノブ） | `FNX_L2_DEC_MAX` | 0 = 上限なし |
| Decode の L1/L2/L3 内訳（件数と host worker・5070 の ms、3 行/サンプル） | `FNX_DEC_STATS` | 0（off は分岐 1 回のみ） |
| 1 サンプルにまとめる window 数 | `FNX_DEC_STATS_EVERY` | 2048 |
| 5070 arena 常駐 expert の host 原本をロード後に返却（M3-1） | `FNX_L2_DROP_HOST` | 0（1 は arena + `FNX_L2_PF` + `FNX_L2_DEC` + 両 card の表が揃うときだけ発火。揃わなければ警告して off。2 は 1軍 swap slot を持つ arena 常駐分も返す） |
| 5070 アレーナの動的入替（M3-2） | `FNX_L2_DYN` | 0（off は分岐 1 回のみ。1 = host pinned 実測と route 回数の dry-run 試算だけ。2 = pass 境界で 1-for-1 の実入れ替え） |
| 1 境界あたりの入れ替えペア数の上限 N | `FNX_L2_DYN_N` | 64（0 = 上限なし） |
| 入れ替えを見送る picks 数（短文のスラッシング防止） | `FNX_L2_DYN_MIN_PICKS` | 16384 |
| 動いた行の check-sum 照合（不一致は安全停止） | `FNX_L2_DYN_VERIFY` | 0 |
| chunked GDN（sub-chunk 幅は card の opt-in 共有メモリから自動選択） | `FNX_PF_GDN_CHUNK` / `FNX_GDN_FORCE_REC` | 1（sm_75 は 16 列 = 57,988 B）/ 0 |
| GDN 数値比較・時間計測 | `FNX_PF_GDN_CHECK` / `FNX_GDN_TIME` | 0 |
| Prefill 失敗時の診断印字 | `FNX_PF_ERRDBG` | 0 |

起動・計測のラッパ（`run_strata_tp.sh` / `bench_switch_8002.sh` / `strata_tp.conf`）と
詳細 docs は、このツリーを配った同梱ディレクトリ（`out/` ルート）側にあります。
`STRATA_TP_*` はラッパ側の名前であり、engine へは上表の `FNX_*` として渡されます。
本リポジトリ内の `docs/` は本家ドキュメントです。

## 再ビルド

```bash
cmake -S engine -B build_sm75 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_SPLIT=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES='75;120' -DSTRATA_SPLIT_ARCH=75 \
  -DSTRATA_SPLIT_FAT_ARCHS=120 '-DSTRATA_SPLIT_CARDS=2;3' \
  -DSTRATA_SPLIT_PF_JOB=64 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build_sm75 --target strata-split-2 strata-split-3 -j 6
```

`engine/strata-split-*-sm75`（各 116 MB）は GitHub の 100 MB 制限を超えるため
**git 管理外**（`.gitignore`）です。sm_120 の SASS は fatbin に同梱済みなので、
5070 側の再ビルドは不要です。

## 変更の所在

- `engine/src/split/src/l2_offload.{hpp,cu}` — 5070 の arena・pinned 段・M2a selftest
- `engine/src/split/src/l2_pf.{hpp,cu}` — Prefill の 5070 演算（resident / streamed tier、slot 重複化と同期不変条件）
- `engine/src/split/src/l2_dec.{hpp,cu}` — Decode の 5070 L2 合流（router の 3 択排他・card 別 slice・host worker との並列・retire）
- `engine/src/split/src/prefill.cu` — 0 重みマスク・加算項の受け渡し・パス進行
- `engine/src/split/src/kernels.cu` / `kernels.hpp` と `gdn_prompt.cuh` — chunked GDN の幅自動選択、`CpuMoeBox` の L2 受け口、`merge_rows` の index clamp
- `engine/src/split/src/engine.cu` と `engine_impl.hpp` — `CpuMoeBox` の mapped pinned 確保（`cudaHostAllocMapped | cudaHostAllocPortable`）と L2 用メンバ
- `engine/src/tpopt/` — 待機ガード（`WaitGuard`）・ペア同期
- `engine/src/split/src/hip/rccl_dl.hpp` — dlopen 候補を汎用名のみへ（自前 NCCL は `FNX_RCCL_LIB`）
- `engine/src/split/src/l2_offload.hpp` の M3 拡張点（slot 別ヒット統計・5070 への 1 軍降格受け口）は未使用のまま保持
- 検証: 同梱 `_work/guard_check.py`（ソースガード）/ `_work/rc71_check.sh`（停止プロトコル）/ `_work/refcheck_docs.py`

## 運用制約

- 稼働中のサーバーを落とさないこと（計測は別ポートで張り、終了後に自動復旧）。
- GPU を占有する実験をしないこと（ビルドは `nice`、実機ベンチは人が回す）。
- `engine/src` 配下へ個人環境パス（ユーザーの home 配下など）を入れないこと。
