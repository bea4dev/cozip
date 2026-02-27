# GPU Optimization Candidates (Sequential Validation Plan)

Updated: 2026-02-26

このファイルは、現状の「ZIP 互換ハイブリッド圧縮」に対する GPU 最適化を
**1 項目ずつ実装してベンチで検証する**ための記録です。
前版の候補一覧は本ファイルで置き換えます。

## Current Baseline (Before Next Changes)

- Scheduler policy: **変更しない**（CZDF 実績ベースの単純スケジューラー維持）
- Current concern:
  - `CPU+GPU` が `CPU_ONLY` より遅くなるケースがある
  - `gpu_chunks` が増えても全体時間が悪化する場合がある
  - GPU 側処理時間（`gpu_worker_busy_ms`）が長く、CPU の steal が進みにくい

Representative log:
- `/home/bea4dev/Documents/development/cozip/bench_results/bench_ratio_20260226_212052.log`

## Sequential Work Items

### [1] GPUパスのコマンド提出回数削減（pass 融合）

Goal:
- コマンドエンコーダ提出回数と pass 境界を減らして GPU ドライバ overhead を削減

Do:
- dynamic path の phase ごとの encoder/submit を見直し
- 可能な範囲で copy/compute/copy をまとめ、不要な submit を削減
- 正しさ優先（出力互換、fallback 条件維持）

Success criteria:
- `comp_stage_ms` と `gpu_worker_busy_ms` の低下
- `gpu_chunks` 同等以上で `avg_comp_ms` 改善

Status: **Implemented (Validation Pending)**

Implementation note:
- dynamic path で、従来は
  - `pack submit` -> bits readback
  - `payload submit` -> payload readback
  の2段 submit だったものを、
  - `pack submit` 1回で `output_words_buffer` と `total_bits_buffer` の readback コピーを同時実行
  に変更。
- これにより dynamic 1バッチあたりの submit 回数を削減。

### [2] GPUバッチの in-flight 多重化（prepare/submit/readback 重畳強化）

Goal:
- GPU の空き時間を減らし、CPU prepare と GPU 実行をより重ねる

Do:
- in-flight バッチ数の制御を追加（2～3本を上限に段階評価）
- readback 待ちで全体停止しないようキューを明確化

Success criteria:
- `gpu_worker_busy_ms` の密度改善
- `comp_stage_ms` 低下、または同時間で `gpu_chunks` 増加

Status: **Implemented (Validation Pending)**

Implementation note:
- dynamic freq readback 回収を「先頭1件待ち（front-only）」から「ready になった任意件回収（ready-any）」へ変更。
- これにより、先頭 chunk の map 完了待ちで後続 ready chunk が詰まる
  head-of-line blocking を緩和。
- pack/readback 側も `ready-any` 逐次回収に変更。
- submit ごとに `Poll + 回収` を実施し、最後にまとめて `Wait` する構造を緩和。

### [3] dynamic header 生成の GPU 化（CPU 側整形削減）

Goal:
- CPU で行っている dynamic header 整形・詰め替えを GPU 側へ移す

Do:
- header metadata を GPU に渡し、出力バッファ先頭のヘッダ書き込みを GPU で実施
- CPU 側は最小限の descriptor 準備に限定

Success criteria:
- `comp_stage_ms` の CPU 寄与を削減
- small/medium chunk での改善確認

Status: **Implemented (Validation Pending)**

Implementation note:
- dynamic Huffman plan (`header_bytes/header_bits` 含む) を頻度表キーでキャッシュ化。
- 同一頻度表の chunk では plan 再構築を省略し、CPU 側 header 生成コストを削減。
- タイミングログに `freq_plan_cache_hits` / `freq_plan_cache_misses` を追加。
- cache hit 時の深い `clone` を削除（`Arc<DynamicHuffmanPlan>` 化）。
- `dyn_table` と `header_bytes_padded` を plan 側で事前保持し、chunk ごとの再構築/再パディングを削減。

Note (既視感について):
- 既に実施済みなのは「dynamic header 用 staging buffer の再利用（アロケーション削減）」。
- 今回の [3] はそれより大きい変更で、**header 自体の生成/書き込み処理を GPU に寄せる**ため別物。

## Measurement Protocol (固定)

同一条件比較で 1 項目ずつ効果を判定する:

1. Baseline を取得（変更前）
2. 項目を 1 つだけ実装
3. 3-run 以上で比較
4. 改善がなければ revert または feature flag 化

Suggested command:

```bash
./bench.sh --mode ratio --runs 3 --size-mib 1024 --chunk-mib 10 --gpu-slots 6 --stream-pipeline-depth 3
```

## Changelog

- 2026-02-26:
  - 本ドキュメントを「候補一覧」から「逐次最適化の実験計画」に全面更新。
  - 次アクションを [1]（pass 融合）に固定。
  - [1] の第1段を実装:
    - dynamic 経路の payload readback 用 submit を廃止
    - pack submit に payload + bits の readback コピーを同梱
  - ビルド/単体テストは通過。性能評価ログは未取得。
  - [2] の第1段を実装:
    - dynamic freq readback を front-only から ready-any 回収に変更
    - head-of-line blocking を緩和
  - [2] の第2段を実装:
    - dynamic pack/readback を ready-any 逐次回収に変更
    - submit 間で回収を進め、tail 待ちを削減
  - [3] の第1段を実装:
    - dynamic Huffman plan のキャッシュ化
    - `freq_plan_cache_hits/misses` の可視化
  - [3] の第2段を実装:
    - plan を `Arc` 化し cache hit 時の deep copy を排除
    - `dyn_table` / `header_bytes_padded` を plan 事前生成に変更
  - ビルド/単体テストは通過。性能評価ログは未取得。

- 2026-02-27:
  - 4GiB・`ratio`・`chunk=10MiB` 条件で、`CPU+GPU` が `CPU_ONLY` に対してほぼ頭打ち（約 `1.02x` 前後）となる事象を記録。
  - 代表ログ:
    - `/home/bea4dev/Documents/development/cozip/bench_results/bench_ratio_20260227_003031.log`
  - 観測値（代表）:
    - `CPU_ONLY avg_comp_ms=6774.401`
    - `CPU+GPU avg_comp_ms=6534.124`
    - `cpu_worker_busy_ms`: `134770.4 -> 114362.9`
    - `gpu_worker_busy_ms`: `7894.8`
    - `cpu_chunks/gpu_chunks`: `344/66`
  - 定量メモ:
    - CPU仕事量比: `114362.9 / 134770.4 = 0.8486`（約 `-15.1%`）
    - CPU実効並列度:
      - CPU_ONLY: `134770.4 / 6774.401 = 19.894`
      - CPU+GPU : `114362.9 / 6534.124 = 17.502`
    - この「CPU実効並列度低下」を織り込んで逆算すると、実測 `CPU+GPU avg_comp_ms` と一致（差分ほぼ 0）。
  - 結論:
    - この条件では、GPU経路の絶対速度不足だけでなく、Hybrid時のCPU実効稼働低下が支配的要因。
    - `t_freq_poll_wait_ms` は引き続き高く、GPU側待ちが長い。
  - ポリシー:
    - `ratio` の探索量（match scan / distance candidates）は下げない（過去に速度悪化・圧縮率悪化を確認済み）。
  - スケジューラー比較用に `scheduler_policy` を追加:
    - `legacy`（既存挙動維持）
    - `global-local`（global queue + CPU/GPU の小さな local buffer）
  - `bench.sh` / `bench_1gb` に `--scheduler legacy|global-local` を追加。

## Scheduler Redesign Memo (User Proposal)

提案:
- 共有グローバルキューから CPU/GPU が仕事を取得
- CPU/GPU はそれぞれ小さいローカルキューを持つ
- ローカルキュー間の相互stealはしない（溜め込みは少量に制限）

評価:
- 現実装（CPU/GPU分割キュー + CPUがGPUキューをsteal）より、タスク供給の偏りを抑えやすい。
- CPU実効並列度低下（特に4GiB条件）の改善余地がある。
- 一方で、GPU側が長時間バッチを保持すると同様の偏りが再発するため、ローカル保持上限は必須。

実装する場合の最小要件:
- `global_ready_queue`: 単一ソースオブトゥルース
- `cpu_local_limit`: 1〜2チャンク
- `gpu_local_limit`: `gpu_batch_chunks` 上限、ただしタイムスライス超過時は未着手分を global に返却
- `gpu_fraction` は「初期配分ヒント」のみに限定し、実行中は空き側優先で global から取得

注意:
- この設計はスケジューラー変更そのもののため、適用する場合は feature flag で A/B 比較する。
