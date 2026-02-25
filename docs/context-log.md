# Context Log

## 2026-02-24 - 初回設計ドラフト作成

### 決定事項

- まずは実装前に設計ドキュメントを整備する
- cozipは「CPU + GPU(WebGPU)協調」を前提にチャンク並列設計を採用する
- Deflateはチャンク独立性を高めるため、チャンク跨ぎ参照を禁止する方針を採用する
- ZIP互換は維持しつつ、並列解凍用索引は extra field で保持する

### 採用理由

- GPUへ割り当てるタスクを独立化しやすい
- CPU/GPUの同時実行で総スループットを上げやすい
- 標準ZIPとの互換を維持しつつ、cozip同士で高速解凍経路を持てる

### トレードオフ

- 圧縮率が通常Deflateより低下する可能性がある
- 小さな入力ではGPUオーバーヘッドが勝つ可能性がある

### 次アクション

1. Rustのモジュール設計(`chunk`, `scheduler`, `deflate`, `zip`, `gpu`)を `src/` に反映
2. M1としてCPUのみのチャンク独立Deflate(最小実装)を作る
3. ZIPエンコード/デコードの基礎(ローカルヘッダ + セントラルディレクトリ)を追加

---

## 2026-02-24 - workspace分割 + Deflate実装着手

### 決定事項

- ルート `cozip` を Cargo workspace 化し、`cozip_deflate` と `cozip_zip` に分割した
- `cozip_deflate` に以下を実装した
- 純CPUの raw Deflate 圧縮・解凍関数
- チャンク独立フォーマット(`CZDF`)による並列圧縮・解凍
- CPU/GPU 協調実行(圧縮: GPU解析 + CPU Deflate、解凍: GPU解析支援 + CPU Deflate復元)
- `cozip_zip` に最小の単一ファイル ZIP 圧縮・解凍を実装した

### 補足

- 現在のGPU経路は WebGPU でチャンク統計解析を実行し、圧縮レベル調整に利用している
- Deflate のビットストリーム生成/復元自体は現段階ではCPU実装

### 次アクション

1. GPU側でLZ候補探索を持てるように `cozip_deflate` を段階拡張する
2. CZDFメタデータをZIP extra fieldに載せる統合を `cozip_zip` へ追加する
3. 外部Deflate/ZIP実装との相互運用テストを増やす

---

## 2026-02-24 - CPU vs CPU+GPU 統合テスト追加

### 決定事項

- `cozip_deflate/tests/hybrid_integration.rs` を追加
- `-- --nocapture` 前提で CPU-only と CPU+GPU の比較ログを出力する
- 比較項目は圧縮/解凍時間、圧縮後サイズ、チャンク配分(stats)とした

### 仕様

- GPU利用可能なら `gpu_chunks > 0` を必須化
- GPUがない環境ではフォールバック実行としてテスト継続し、ログで明示
- 追加テスト:
1. `compare_cpu_only_vs_cpu_gpu_with_nocapture`
2. `hybrid_uses_both_cpu_and_gpu_when_gpu_is_available`

### 実行コマンド

1. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture`
2. `cargo test --workspace`

---

## 2026-02-24 - GPU実タスク設計(大チャンク転送 + GPU内細分化)

### 決定事項

- GPUは解析補助ではなく、Deflate圧縮/解凍の実タスクを担当する設計へ進める
- 2段階チャンク化を採用する
- `host_chunk`(1〜4MiB, 初期2MiB): CPU/GPUスケジューリング単位
- `gpu_subchunk`(64〜256KiB, 初期128KiB): GPU内部並列実行単位
- 静的50:50配分ではなく、共通キュー + EWMAによる動的配分を採用する

### 採用理由

- 転送オーバーヘッドを吸収しつつGPU並列度を確保できる
- CPU/GPUの処理能力差があっても、先に空いた側へ仕事を回せる
- 可変長出力の競合を2パス(長さ計算->prefix sum->emit)で管理しやすい

### 追記したドキュメント

1. `docs/gpu-full-task-design.md`
2. `docs/architecture.md` (v1 draftへ更新)
3. `docs/deflate-parallel-profile.md` (host/subchunk方針へ更新)

### 次アクション

1. `cozip_deflate` に `GpuContext` 使い回し機構を導入する
2. `HybridScheduler` を共通キュー + 動的配分に置き換える
3. GPU圧縮本体(`match_find`, `token_count`, `prefix_sum`, `token_emit`)を段階実装する

---

## 2026-02-24 - CPU全力 + GPU全力 実装(第1段)

### 決定事項

- `cozip_deflate` を動的スケジューラへ置き換えた
- CPUワーカー群とGPUワーカーが同時にキューを消費する実装にした
- `gpu_fraction` に基づくGPU予約チャンクを導入し、CPUが取り切らないようにした
- `GpuContext` はプロセス内で使い回す方式(遅延初期化)へ変更した

### GPU実タスク(圧縮/解凍)

- 圧縮時:
1. GPUで連続一致率(repeat ratio)を計算し圧縮レベルを調整
2. GPUで `EvenOdd` 可逆変換(サブチャンク単位)を実行
3. CPUでDeflateビットストリーム化

- 解凍時:
1. CPUでDeflate展開
2. GPU(またはCPUフォールバック)で `EvenOdd` 逆変換

### 互換性

- フレームバージョンを `CZDF v2` へ更新
- チャンクメタデータに `transform` フィールドを追加
- 旧 `v1` フレームの読み取りは継続対応

### 検証

1. `cargo test --workspace` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` で CPU と CPU+GPU 比較出力を確認
3. GPU利用可能環境で `cpu_chunks` / `gpu_chunks` 両方が配分されることを確認

---

## 2026-02-24 - 速度ベンチマーク実施(Release)

### 実行条件

- コマンド: `cargo run --release -p cozip_deflate --example bench_hybrid`
- 反復回数: 5
- データサイズ: 4MiB / 16MiB
- 方式比較:
1. CPU_ONLY (`prefer_gpu=false`, `gpu_fraction=0.0`)
2. CPU+GPU (`prefer_gpu=true`, `gpu_fraction=0.5`)

### 計測結果

- 4MiB:
1. CPU_ONLY: comp 72.977ms / decomp 4.276ms / comp 54.81MiB/s / decomp 935.43MiB/s
2. CPU+GPU : comp 156.421ms / decomp 6.821ms / comp 25.57MiB/s / decomp 586.41MiB/s

- 16MiB:
1. CPU_ONLY: comp 75.138ms / decomp 4.633ms / comp 212.94MiB/s / decomp 3453.26MiB/s
2. CPU+GPU : comp 682.850ms / decomp 7.577ms / comp 23.43MiB/s / decomp 2111.67MiB/s

### 所見

- 現状実装では CPU+GPU が CPU_ONLY を上回っていない
- 主因は GPU側タスクが「可逆変換 + 解析」に寄っており、Deflateビットストリーム本体がCPU実装のため
- 次段でGPU側のトークン生成本体(`match_find`/`token_count`/`prefix_sum`/`token_emit`)を強化する必要がある

---

## 2026-02-24 - 1GB比較用ベンチ追加

### 決定事項

- `cozip_deflate/examples/bench_1gb.rs` を追加した
- デフォルトを 1GiB 入力(`--size-mib 1024`)に設定した
- CPU_ONLY と CPU+GPU を同条件で比較出力する
- 引数でサイズ/反復/ウォームアップ/チャンク設定を変更可能にした

### 実行コマンド

1. 1GiB本番比較:
`cargo run --release -p cozip_deflate --example bench_1gb`
2. 軽量確認:
`cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0`

### 次アクション

1. 1GiB実測値を取得して共有
2. GPU側Deflate本体実装(`match_find/token_count/prefix_sum/token_emit`)へ着手

---

## 2026-02-24 - 1GBベンチのデフォルト実行回数を短縮

### 決定事項

- `bench_1gb` のデフォルト反復数を短縮した
- `iters: 3 -> 1`
- `warmups: 1 -> 0`

### 理由

- 1GiB比較は1回でも十分傾向確認でき、待ち時間を大幅に減らせるため

### 補足

- 厳密比較したい場合は明示的に `--iters` / `--warmups` を指定する

---

## 2026-02-24 - GPU Deflate本体設計(独立チャンク実行 + 連結)

### 決定事項

- 実装方針を明確化した:
1. 入力を独立チャンクへ分割
2. CPUとGPUがそれぞれDeflateを独立実行
3. `index` 順に戻して連結
- 圧縮率低下を許容し、並列スループットを優先する
- `Chunk-Member Profile (CMP)` を採用し、各チャンクを独立Deflate memberとして扱う

### 追記したドキュメント

1. `docs/gpu-deflate-chunk-pipeline.md`
2. `docs/architecture.md` (CMPを追記)
3. `docs/deflate-parallel-profile.md` (CMPを追記)

### 次アクション

1. `cozip_deflate` に `ChunkMember` データモデルを固定する
2. GPU圧縮本体(`match_find/token_count/prefix_sum/token_emit`)を実装する
3. GPU解凍(固定Huffman先行)を追加する

---

## 2026-02-24 - GPU Deflate本体実装(第2段)

### 実装内容

- `cozip_deflate/src/lib.rs` をCMP前提で更新した
- GPU経路に `match_find -> token_count -> prefix_sum -> token_emit` を実装した
- GPUで得た run 開始位置から、固定Huffman Deflate を生成する経路を追加した
- 各チャンクは独立 Deflate member として圧縮され、`index` 順で連結される

### 主要ポイント

1. GPU圧縮:
- `run_start_positions()` でGPUパイプライン実行
- `encode_deflate_fixed_from_runs()` で固定Huffman Deflate生成

2. CPU圧縮:
- 従来どおり `flate2` Deflate

3. 解凍:
- チャンク単位でCPU/GPUワーカーが分担
- Deflate展開は現段階ではCPU実装を利用
- GPU担当チャンクはGPUパイプラインを補助的に実行可能

### 検証

1. `cargo test -p cozip_deflate` 通過
2. `cargo test --workspace` 通過
3. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過

### 補足

- 現在のGPU Deflate経路は「runベース + 固定Huffman」のため、圧縮率はCPU経路より悪化しやすい
- 次段はGPU側のトークン品質(一般LZ候補)と bitpack の拡張が必要

---

## 2026-02-24 - 改善1/3実装 (GPUトークン品質 + 大きめチャンク)

### 実装内容

1. 改善1: GPUトークン品質向上
- `build_tokens_from_run_starts()` を run-only 方式から LZ77 greedy 方式へ更新
- ハッシュ候補 + runヒントの複数距離候補を評価し、最長一致を採用
- 既存GPUパイプライン(`match_find/token_count/prefix_sum/token_emit`)の出力をヒントとして利用

2. 改善3: チャンクサイズのデフォルト拡大
- `HybridOptions::default()`:
- `chunk_size: 2MiB -> 4MiB`
- `gpu_subchunk_size: 128KiB -> 256KiB`
- ベンチ既定値も同様に更新

### 計測(bench_hybrid, release, GPU実機)

- 4MiB:
1. CPU_ONLY ratio=0.3361
2. CPU+GPU ratio=0.3564

- 16MiB:
1. CPU_ONLY ratio=0.3364
2. CPU+GPU ratio=0.3465

### 所見

- 以前のGPU比率(約0.512)から大幅に改善
- ただし速度は依然CPU_ONLY優位で、GPU bitpack/候補品質の追加改善が必要

---

## 追記テンプレート

```
## YYYY-MM-DD - タイトル
### 決定事項
- ...
### 問題
- ...
### 対応
- ...
### 次アクション
1. ...
```

## 2026-02-24 - 改善2/3着手 (GPUバッチsubmit + 転送前処理削減)

### 実装内容

1. 改善3(実行バッチ化): GPU圧縮ワーカーを単一チャンク処理からバッチ処理へ変更
- `compress_gpu_worker()` で最大 `GPU_BATCH_CHUNKS` 件(既定8)をまとめて取得
- `compress_chunk_gpu_batch()` を追加し、複数チャンクのGPU補助処理を一括実行

2. 改善2(転送・同期オーバーヘッド削減): `run_start_positions_batch()` を追加
- 旧 `run_start_positions()` を単発ラッパにし、内部はバッチAPI経由へ統一
- 複数チャンクの compute pass を1つの command encoder に積み、`queue.submit()` をバッチ単位に削減
- `map_async + poll(wait)` もバッチ単位に集約し、同期回数を削減
- GPU入力を `u32` への1byte展開を廃止し、packed byte入力をWGSL側で参照する方式へ変更

### 検証

1. `cargo test -p cozip_deflate` 通過
2. `cargo run --release -p cozip_deflate --example bench_hybrid` 実行
3. `cargo test --workspace` 通過

### メモ

- 小規模ベンチ(4MiB/16MiB)では改善幅が見えづらい
- 効果確認は1GiB級で再評価するのが妥当

## 2026-02-24 - GPU Deflate完結経路(固定Huffman literal) + readback最小化

### 実装内容

1. GPU内Deflate完結経路を追加
- `GpuAssist::deflate_fixed_literals_batch()` を追加
- GPU上で以下を実行して、チャンクごとのDeflateバイト列を生成
  1. literal code/bitlen生成 (`litlen_pipeline`)
  2. bit offset prefix-sum (`prefix_pipeline` 再利用)
  3. bitpack (`bitpack_pipeline`)
- CPU側は中間トークンを組み立てず、GPU出力をそのままチャンク圧縮データとして採用

2. readback最小化
- 旧経路の `positions` 大量readbackを圧縮経路から除去
- 圧縮で戻すデータを「合計bit数 + 最終圧縮バイト列」に限定

3. 圧縮ワーカー接続変更
- `compress_chunk_gpu_batch()` は `run_start_positions_batch()` + CPU bitpack ではなく、
  `deflate_fixed_literals_batch()` を使用するよう変更

### 検証

1. `cargo test -p cozip_deflate` 通過
2. `cargo test --workspace` 通過
3. `cargo run --release -p cozip_deflate --example bench_hybrid` 実行

### 直近ベンチ(bench_hybrid)

- 4MiB:
  - CPU_ONLY comp_mib_s=38.82 ratio=0.3361
  - CPU+GPU  comp_mib_s=22.95 ratio=1.0181

- 16MiB:
  - CPU_ONLY comp_mib_s=122.75 ratio=0.3364
  - CPU+GPU  comp_mib_s=48.84 ratio=0.6771

### メモ

- 速度面は以前のGPU経路より改善傾向だが、CPU_ONLYをまだ下回る
- 現在のGPU完結経路はliteral主体のため圧縮率が悪化しやすい
- 次段はGPU側でmatch探索/トークン化を強化し、match token(長さ・距離)を実際に出力する必要がある

## 2026-02-24 - 追加改善(継続): 安全側の高速化調整

### 対応

1. 解凍時の不要GPU補助を削除
- `decode_descriptor_gpu()` で行っていた `run_start_positions()` 呼び出しを削除
- 展開後データに対する補助GPU実行は、現行仕様では性能メリットが薄くオーバーヘッドが勝るため無効化

2. GPUバッチ粒度の拡大
- `GPU_BATCH_CHUNKS: 8 -> 16`
- 1GiBクラスでGPU submit回数を減らしやすくする調整

### 重要メモ

- 共通バッファ再利用 + 単一 `queue.submit()` の試行は一旦見送り
- 理由: `queue.write_buffer()` は encoder内コマンドではないため、
  複数チャンク分の更新を先に積むと全dispatchが最終書き込み状態を参照し、データ破損が発生し得る
- 次に同方針を進める場合は、`copy_buffer_to_buffer` をencoder内に入れる staged upload 設計が必要

### 検証

1. `cargo test -p cozip_deflate` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration` 通過
3. `cargo run --release -p cozip_deflate --example bench_hybrid` 実行

## 2026-02-24 - GPU len/dist 実装 (RLE系 match token 追加)

### 実装内容

1. GPUトークン化パスを追加
- `tokenize_pipeline` を追加し、各バイトを `literal` / `match(dist=1)` / `skip` に分類
- run先頭バイトは常に literal にし、残りを `match(len, dist=1)` 化

2. GPUコード生成を拡張
- 既存 `litlen_pipeline` をトークン入力ベースに変更
- `literal` と `match(len/dist)` の固定HuffmanコードをGPU上で生成
- bit長配列を作成してprefix-sum -> bitpack で最終Deflateを構築

3. readback最小化は維持
- 返却は `total_bits + compressed bytes` のみ

### バグ修正

- 初版で `dist=1` match開始位置が不正（run先頭でmatch開始）になり、展開破損
- 修正後は run先頭をliteral化し、残りのみmatch化して整合性を回復

### 検証

1. `cargo test -p cozip_deflate` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo test --workspace` 通過
4. `cargo run --release -p cozip_deflate --example bench_hybrid` 実行

### 観測

- 圧縮率は literal-only より改善したケースがあるが、CPU_ONLYには依然届かない
- 現段階のmatchは dist=1 (RLE系) に限定されるため、一般データの圧縮効率はまだ不十分

## 2026-02-24 - GPU len/dist 強化 (candidate + finalize 2段)

### 実装内容

1. `dist>1` を扱うGPU Deflateトークン化へ更新
- `token_dist` バッファを追加し、match tokenに距離を保持
- `litlen` シェーダで distance symbol/extra bits を生成するよう拡張

2. トークン生成を2段化
- `tokenize_pipeline`:
  - 各index独立で `candidate(len, dist)` を並列算出
  - 候補距離は近距離優先の固定セット(1..1024の疎サンプル)
- `token_finalize_pipeline`:
  - 単一スレッドで候補列を走査し、非重複の最終token列へ確定
  - `literal` / `match(len,dist)` / `skip` を整合性付きで確定

3. Deflate実行順を変更
- tokenize(candidate) -> tokenize(finalize) -> token prefix -> litlen codegen -> bitlen prefix -> bitpack

### 目的と効果

- 以前の「各indexが独立にstart/skipを決める」方式で起きていた被覆競合を解消
- 圧縮長不整合(展開長ミスマッチ)を回避しつつ、GPUでlen/dist生成を継続

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
- `compare_cpu_only_vs_cpu_gpu_with_nocapture` 成功
- `hybrid_uses_both_cpu_and_gpu_when_gpu_is_available` 成功

### 現状メモ

- correctnessは回復
- 速度はまだ `CPU_ONLY` より遅いが、GPUが空ブロック化する不安定挙動は解消
- 次段は `tokenize_pipeline` の候補探索をさらにGPUフレンドリー化(探索候補とscan長の最適化、shared memory活用)が必要

### 追加ベンチ (2026-02-24 / bench_hybrid)

- 4MiB
  - CPU_ONLY: comp_mib_s=40.82 ratio=0.3361
  - CPU+GPU : comp_mib_s=4.11 ratio=0.6592

- 16MiB
  - CPU_ONLY: comp_mib_s=114.92 ratio=0.3364
  - CPU+GPU : comp_mib_s=8.06 ratio=0.4977

メモ: correctnessは改善したが、性能・圧縮率ともCPU_ONLY優位のまま。

## 2026-02-24 - finalize/prefix ボトルネック削減

### 対応

1. `token_finalize` を単一点逐次からセグメント並列へ変更
- 旧: `dispatch_workgroups(1,1,1)` で全体を1スレッド処理
- 新: `TOKEN_FINALIZE_SEGMENT_SIZE=4096` 単位で複数workgroupへ分割
- 各セグメント内でgreedy確定(セグメント境界を跨がないよう match 長をクランプ)

2. `prefix` を階層並列scanへ置換
- `scan_blocks_pipeline` で block 内 exclusive scan + block sums
- block sums を再帰的に同scanで prefix 化
- `scan_add_pipeline` で block offset を全要素へ加算
- これを token prefix / bitlen prefix の両方で利用

### 実装メモ

- 追加定数: `PREFIX_SCAN_BLOCK_SIZE=256`, `TOKEN_FINALIZE_SEGMENT_SIZE=4096`
- `GpuAssist::dispatch_parallel_prefix_scan()` を追加し、Deflate経路のprefix計算を統一

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo run --release -p cozip_deflate --example bench_hybrid` 実行

### 直近ベンチ (bench_hybrid)

- 4MiB
  - CPU_ONLY: comp_mib_s=36.15 ratio=0.3361
  - CPU+GPU : comp_mib_s=199.39 ratio=0.6593

- 16MiB
  - CPU_ONLY: comp_mib_s=112.39 ratio=0.3364
  - CPU+GPU : comp_mib_s=132.78 ratio=0.4978

補足: 圧縮スループットは大きく改善。圧縮率は依然CPU_ONLYより悪化しやすい。

## 2026-02-24 - Atomic状態ベースの適応スケジューラ

### 目的

- 固定比率(`gpu_fraction`)配分だけだとGPU遅い環境で全体が引きずられる問題を緩和
- GPU優先予約を維持しつつ、未着手予約をCPUへ再配分する

### 実装

1. 圧縮タスク状態をAtomic化
- `ScheduledCompressTask` を追加
- 状態: `Pending / ReservedGpu / RunningGpu / RunningCpu / Done`
- `reserved_at_ms` を保持し、GPU予約の鮮度を判定

2. GPU有効時の圧縮経路を新スケジューラへ切替
- `compress_hybrid()` でGPU有効時は `compress_hybrid_adaptive_scheduler()` を使用
- CPU-only時は既存キュー経路を維持

3. 監視スレッド(Watchdog)を追加
- `ReservedGpu` のまま一定時間(`GPU_RESERVATION_TIMEOUT_MS`)更新されないタスクを検出
- CPU空き数(`active_cpu`)ぶんだけ `Pending` へ降格し、CPUに実行機会を渡す

4. CPU/GPUワーカーはCASでタスク獲得
- CPUは `Pending -> RunningCpu`
- GPUは `ReservedGpu -> RunningGpu` を優先、その後 `Pending -> RunningGpu`
- 実行後は `Done` に遷移し `remaining` を減算

5. 待機は `Condvar + timeout` で実装
- 両者が取り合えない時は短時間sleepし、busy-spinを回避

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo run --release -p cozip_deflate --example bench_hybrid` 実行

### 直近ベンチ (bench_hybrid)

- 4MiB
  - CPU_ONLY: comp_mib_s=40.85 ratio=0.3361
  - CPU+GPU : comp_mib_s=190.71 ratio=0.6593

- 16MiB
  - CPU_ONLY: comp_mib_s=138.54 ratio=0.3364
  - CPU+GPU : comp_mib_s=149.78 ratio=0.4978

メモ: 圧縮率はまだGPU側が不利だが、スループット面は固定配分より改善。

### 追加ベンチ (2026-02-24 / 4GiB)

command:
`cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 4096 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256`

- CPU_ONLY:
  - comp_mib_s=514.72
  - decomp_mib_s=4656.01
  - ratio=0.3364
  - cpu_chunks=1024 gpu_chunks=0

- CPU+GPU:
  - comp_mib_s=587.09
  - decomp_mib_s=4282.01
  - ratio=0.4020
  - cpu_chunks=816 gpu_chunks=208

メモ:
- 圧縮はCPU+GPUが優位(+14%程度)
- 解凍はCPU_ONLYが優位
- speedup表記の注記(`>1.0 means CPU_ONLY faster`)は逆で、実際はCPU_ONLY/hybrid比なので>1はhybridが速い

## 2026-02-24 - GPU転送/readback最適化 (実装4)

### 対応

1. 不要なCPUゼロ書き込みを削減
- `deflate_fixed_literals_batch()` で以下の `queue.write_buffer(0埋め)` を削除:
  - `token_total_buffer`
  - `bitlens_buffer`
  - `total_bits_buffer`
  - `output_words_buffer` の全体ゼロ埋め
- `output_words_buffer` は先頭ヘッダ(0b011)のみ書き込み維持
- WebGPUのゼロ初期化前提を利用し、PCIe転送量を削減

2. readbackを単一バッファ化
- 旧: `total_readback` と `compressed_readback` を別々にmap
- 新: `readback` 1本に `total_bits(先頭4byte) + compressed payload` を集約
- map_async/map回数・チャネル処理を削減

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 1024 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256`
4. `cargo run --release -p cozip_deflate --example bench_hybrid`

### 直近ベンチ

- 1GiB (`bench_1gb`)
  - CPU_ONLY: comp_mib_s=476.38 decomp_mib_s=4146.80 ratio=0.3364
  - CPU+GPU : comp_mib_s=572.34 decomp_mib_s=5232.33 ratio=0.4373
  - chunk配分: cpu=176 gpu=80

- `bench_hybrid`
  - 4MiB:
    - CPU_ONLY comp_mib_s=39.12 ratio=0.3361
    - CPU+GPU  comp_mib_s=237.52 ratio=0.6593
  - 16MiB:
    - CPU_ONLY comp_mib_s=95.98 ratio=0.3364
    - CPU+GPU  comp_mib_s=155.51 ratio=0.4978

### メモ

- 圧縮スループットは改善傾向
- 圧縮率は依然としてCPU_ONLYより悪化しやすく、GPU match品質の改善が次段課題

## 2026-02-24 - 速度最適化(1+2継続): Deflateスロット再利用

### 実装

1. GPU Deflateバッファの永続化/再利用
- `GpuAssist` に `deflate_slots: Mutex<Vec<DeflateSlot>>` を追加
- チャンクごとの大量 `create_buffer/create_bind_group` を削減
- 必要容量を超えた場合のみスロットを再確保

2. per-slot bind group 再利用
- `litlen_bg` / `tokenize_bg` / `bitpack_bg` をスロット内に保持
- 毎チャンク再生成を廃止

3. 初期化のGPU側クリア
- 再利用バッファの初期化は `encoder.clear_buffer` を利用
- `output_words` ヘッダは専用 `deflate_header_buffer` から `copy_buffer_to_buffer` で設定

4. readbackは単一バッファ維持
- `total_bits + payload` を1本で回収

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 1024 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256`
4. `cargo run --release -p cozip_deflate --example bench_hybrid`

### 直近ベンチ

- 1GiB (`bench_1gb`)
  - CPU_ONLY: comp_mib_s=502.32 decomp_mib_s=4364.46 ratio=0.3364
  - CPU+GPU : comp_mib_s=625.10 decomp_mib_s=5463.49 ratio=0.4373
  - speedup(cpu/hybrid): compress=1.244x decompress=1.252x
  - 配分: cpu_chunks=176 gpu_chunks=80

- `bench_hybrid`
  - 4MiB:
    - CPU_ONLY comp_mib_s=33.06 ratio=0.3361
    - CPU+GPU  comp_mib_s=305.02 ratio=0.6593
  - 16MiB:
    - CPU_ONLY comp_mib_s=112.70 ratio=0.3364
    - CPU+GPU  comp_mib_s=133.84 ratio=0.4978

### メモ

- `1` は反映済み
- `2` の完全版(二重バッファ submit/collect 分離による upload/compute/readback 重畳)は次段実装候補

## 2026-02-24 - 速度最適化(2完全版): submit/collect 重畳の実装

### 実装

1. `deflate_fixed_literals_batch()` をスロットプール型に変更
- `chunk_index` 固定スロットではなく、`free_slots` + `pending` で再利用する方式へ移行
- プール上限: `GPU_DEFLATE_SLOT_POOL (= GPU_BATCH_CHUNKS)`

2. submit と collect の重畳
- `GPU_PIPELINED_SUBMIT_CHUNKS` ごとに `queue.submit` して `map_async` 登録
- 直後に `poll(Maintain::Poll)` + `try_recv` で回収可能分を先行 collect
- 空きスロットがない場合のみ `poll(Maintain::Wait)` で1件回収して再利用

3. readback復元処理の共通化
- `collect_deflate_readback()` を追加
- `total_bits + payload` 形式の readback から圧縮バイト列を復元し、`unmap` まで一元化

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo run --release -p cozip_deflate --example bench_1gb` 実行

### 直近ベンチ

- 1GiB (`bench_1gb`, default)
  - CPU_ONLY: comp_mib_s=499.68 decomp_mib_s=4031.88 ratio=0.3364
  - CPU+GPU : comp_mib_s=635.92 decomp_mib_s=5209.82 ratio=0.4574
  - speedup(cpu/hybrid): compress=1.273x decompress=1.292x
  - 配分: cpu_chunks=160 gpu_chunks=96

### メモ

- 前回実装の「部分的重畳」から、実際に submit/collect を分離したパイプラインへ移行
- 圧縮率は CPU_ONLY より悪いままだが、スループットは引き続き CPU+GPU が優位

## 2026-02-24 - モード共存実装 (Speed/Balanced/Ratio + codec_id)

### 実装

1. 圧縮モードを追加
- `HybridOptions` に `compression_mode: CompressionMode` を追加
- `CompressionMode`:
  - `Speed` (既存優先)
  - `Balanced` (GPU探索品質を中間設定)
  - `Ratio` (GPU探索品質を高設定)
- `Default` は `CompressionMode::Speed`

2. フレームメタへ `codec_id` を追加
- `FRAME_VERSION` を `3` へ更新
- chunk metadata に `codec_id` を追加 (`backend + transform + codec + raw_len + compressed_len`)
- 新形式書き出し時は v3
- 読み取りは v1/v2/v3 互換維持
  - v2は `backend` から codec を推定

3. チャンクcodec分岐を追加
- `ChunkCodec::{DeflateCpu, DeflateGpuFast}` を導入
- 圧縮時にチャンクごとに codec を格納
- 復号時は `codec` で分岐してinflate（現状どちらも deflate inflate だが将来拡張点を確保）

4. `Balanced` / `Ratio` の挙動
- speed と同様に GPU へタスクを割り当てる
- CPU再圧縮フォールバックではなく、GPU tokenize/finalize の探索品質をモード別に切り替える

### 検証

1. `cargo test -p cozip_deflate --lib` 通過
2. `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
3. `cargo test -p cozip_deflate` 通過
4. 追加テスト:
- `ratio_mode_roundtrip`
- `decode_v2_frame_compatibility`

### 運用メモ

- `bench_1gb` に `--mode speed|balanced|ratio` を追加済み

### 4GiB モード比較 (2026-02-24 / ローカル実測・更新版)

command (各モード):
`target/release/examples/bench_1gb --size-mib 4096 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256 --mode <speed|balanced|ratio>`

結果:

- `speed`
  - CPU_ONLY: comp_mib_s=514.68 decomp_mib_s=4328.25 ratio=0.3364
  - CPU+GPU : comp_mib_s=753.17 decomp_mib_s=4888.60 ratio=0.4625
  - speedup(cpu/hybrid): compress=1.463x decompress=1.129x
  - chunk配分: cpu_chunks=624 gpu_chunks=400
  - GPU使用率観測: ほぼ100%張り付き

- `balanced`
  - CPU_ONLY: comp_mib_s=518.28 decomp_mib_s=4339.05 ratio=0.3364
  - CPU+GPU : comp_mib_s=435.05 decomp_mib_s=5488.71 ratio=0.3364
  - speedup(cpu/hybrid): compress=0.839x decompress=1.265x
  - chunk配分: cpu_chunks=1024 gpu_chunks=0
  - GPU使用率観測: 16%前後

- `ratio`
  - CPU_ONLY: comp_mib_s=514.65 decomp_mib_s=4328.79 ratio=0.3364
  - CPU+GPU : comp_mib_s=527.86 decomp_mib_s=3767.50 ratio=0.3364
  - speedup(cpu/hybrid): compress=1.026x decompress=0.870x
  - chunk配分: cpu_chunks=1024 gpu_chunks=0
  - GPU使用率観測: 30%前後

所見:
- 圧縮速度最優先なら `speed` が最良。CPU+GPUで圧縮/解凍ともCPU_ONLYを上回る。
- 圧縮率最優先なら `balanced` が有効（CPU_ONLY同等の ratio を維持）。ただし圧縮スループットは低下。
- `ratio` は現状ほぼCPU実行だが、圧縮速度はCPU_ONLYと同等以上、解凍速度は低下傾向が見られる。

## 2026-02-24 - モード仕様修正 (GPU再計算フォールバック廃止)

方針:
- `balanced` / `ratio` でも `speed` と同様に GPU へ圧縮タスクを割当
- 圧縮率改善は CPU再圧縮でなく GPU側ロジック改善で行う

実装:
- `compress_hybrid()` の `Ratio` による GPU無効化を廃止
- GPU圧縮の CPU再圧縮比較ロジックを撤去
- tokenize shader:
  - mode別に `max_match_scan` / `max_match_len` / `distance candidate count` を切替
  - distance candidate を 32段階 (最大 32768) まで拡張
- token_finalize shader:
  - mode別 lazy matching (`speed=0`, `balanced=1`, `ratio=2`) を追加

mode別GPU品質パラメータ:
- `Speed`: scan=64, max_len=64, dist_slots=20
- `Balanced`: scan=128, max_len=128, dist_slots=28
- `Ratio`: scan=192, max_len=258, dist_slots=32

検証:
- `cargo test -p cozip_deflate --lib` 通過
- `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過

## 2026-02-24 - Ratioモード: GPU頻度集計 + CPU木生成(dynamic Huffman)

実装:
- `ratio` モードでも GPU タスク割当は維持（GPU無効化しない）
- `deflate_fixed_literals_batch()` に `ratio` 分岐を追加し、`deflate_dynamic_hybrid_batch()` を使用
- `deflate_dynamic_hybrid_batch()`:
  1. GPUで tokenize + finalize
  2. GPU frequency pass で `litlen(286)` / `dist(30)` 頻度を atomic 集計
  3. 頻度テーブルを readback
  4. CPUで Huffman木(符号長)生成 + canonical code生成
  5. CPUで dynamic header + token列を bitpack
- GPU側は mode に応じて探索品質を切替:
  - `Speed`: scan=64 / len=64 / dist_slots=20
  - `Balanced`: scan=128 / len=128 / dist_slots=28
  - `Ratio`: scan=192 / len=258 / dist_slots=32

補足:
- `balanced/ratio` のCPU再圧縮フォールバックは廃止
- 圧縮率改善はスケジューリングではなく GPU tokenization 品質 + dynamic Huffman で行う

検証:
- `cargo test -p cozip_deflate --lib` 通過
- `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
- `cargo check -p cozip_deflate --examples` 通過

## 2026-02-24 - Balanced/Ratio 実行時エラー修正

報告エラー:
1. `mode=balanced` で `InvalidFrame("raw chunk length mismatch in cpu path")`
2. `mode=ratio` で `Source buffer is missing the COPY_SRC usage flag`

修正:
- ratio用 dynamic path で readback コピーしているバッファに `COPY_SRC` を追加
  - `token_flags/token_kind/token_len/token_dist/token_lit`
- balanced/ratio の GPU 圧縮結果に対して整合性ガードを追加
  - GPU圧縮結果を inflate して元チャンク一致を検証
  - 不一致時のみ CPU圧縮へフォールバック（クラッシュ防止の保険）

検証:
- `cargo test -p cozip_deflate --lib` 通過
- `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過

## 2026-02-24 - Ratio 1+2 実装 (freq+final readback / GPU bitpack)

目的:
- ratio で readback を最小化
  - 旧: token配列 + freq + 最終出力
  - 新: freq + 最終出力のみ
- CPUは dynamic Huffman 木生成のみ
- GPUで token bitpack + EOB finalize を実行

主な変更:
- `deflate_dynamic_hybrid_batch()` を2段化
  1. GPU tokenize/finalize/freq → `litlen/dist` 頻度のみreadback
  2. CPUで dynamic Huffman plan 作成
  3. planをGPUへupload
  4. GPUで token map → prefix scan → bitpack → dynamic finalize(EOB)
  5. 最終圧縮バイト列のみreadback
- dynamic Huffman plan構築ヘルパーを追加
  - `build_dynamic_huffman_plan()`
- GPU dynamic map/finalize パイプラインを追加
  - `dyn_map_pipeline`
  - `dyn_finalize_pipeline`
- bitpack shader を拡張
  - base bit offset を `params._pad1` で可変化
  - 33bit以上コード用の high-lane (`dyn_overflow_buffer`) を追加

制約対応:
- `wgpu` の `max_storage_buffers_per_shader_stage=8` 制限へ対応
  - `dyn_map` のstorage bindingを8本以内に再設計
  - token compact index(`token_prefix`)依存を除去（lane indexで直接bitpack）
  - dynamic tableを単一storage buffer (`dyn_table_buffer`) に統合

バッファ/slot側:
- 追加: `dyn_table_buffer`, `dyn_meta_buffer`, `dyn_overflow_buffer`, `dyn_map_bg`, `dyn_finalize_bg`
- 出力上限を dynamic も見込んで拡張
  - `GPU_DEFLATE_MAX_BITS_PER_BYTE = 20`

検証:
- `cargo test -p cozip_deflate --lib --no-run` 通過
- `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
- `cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 512 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256 --mode ratio` 通過
  - CPU_ONLY ratio=0.3364, CPU+GPU ratio=0.3373 (512MiB, 1iter)

備考:
- 未使用関数/定数に関する warning は残る（既存設計由来）。
- 今回は panic/validation error を出さずに ratio 経路をGPU bitpack 化できた状態。

## 2026-02-24 - bench_1gb: gpu_fraction引数追加 + ratio 4GiB試験(1.0)

変更:
- `cozip_deflate/examples/bench_1gb.rs` に `--gpu-fraction <F>` を追加
- CPU+GPUケースの `HybridOptions.gpu_fraction` をCLI指定値で上書き
- ベンチ出力に `gpu_fraction` を表示

実行:
- `cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 4096 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256 --mode ratio --gpu-fraction 1.0`

結果:
- CPU_ONLY: comp=518.47 MiB/s, decomp=4412.30 MiB/s, ratio=0.3364
- CPU+GPU : comp=583.18 MiB/s, decomp=5803.31 MiB/s, ratio=0.3375
- speedup(cpu/hybrid): compress=1.125x, decompress=1.315x
- chunk配分: cpu_chunks=833, gpu_chunks=191

所見:
- `gpu_fraction=1.0` でも実配分は動的調整でCPU側へ多く戻る（予約比率=実配分ではない）。
- 圧縮率差は小さいまま、圧縮/解凍ともCPU+GPUが優位。

## 2026-02-24 - デフォルトgpu_fractionを1.0へ変更

変更:
- `HybridOptions::default().gpu_fraction` を `0.5 -> 1.0` に変更
- `bench_1gb` の `--gpu-fraction` デフォルト表示/実値を `1.0` に変更

確認:
- `cargo test -p cozip_deflate --lib` 通過

## 2026-02-24 - balanced低GPU利用の原因調査と修正

現象:
- `--mode balanced` で `gpu_chunks` が極端に少ない / 0 になるケースが発生
- 圧縮速度がCPU_ONLYより悪化

原因(確認済み):
1. dynamic Huffman計画の code-length 木生成失敗
- エラー: `failed to build codelen huffman lengths`
- これにより GPUバッチ全体がCPUフォールバックし、`gpu_chunks` が増えない

2. 予約降格タイミングが短く、GPU予約が早期にCPUへ流れる
- 旧設計は予約時刻が同時刻で、watchdogがまとめて降格しやすかった

実装修正:
- dynamic Huffman code-length木生成にフォールバックを追加
  - `fallback_codelen_lengths()`
  - 生成不能時は code-lengthシンボルに安全な固定長(5bit)を割当
- 予約降格のモード別チューニング
  - `GPU_RESERVATION_TIMEOUT_MS_DYNAMIC = 100`
  - `GPU_RESERVATION_STAGGER_MS_DYNAMIC = 8`
- 予約時刻を段階的にずらす初期化を追加
  - `reserved_at_ms = now + seq * stagger_ms`
  - 一斉降格を抑止し、GPUが連続してバッチ取得しやすくした
- `balanced` は引き続き dynamic Huffman + speed探索（tokenize modeはSpeed）
- GPU検証は `balanced/ratio` で有効維持（誤圧縮防止）

補助デバッグ:
- `COZIP_LOG_GPU_FALLBACK=1` で以下をstderr出力
  - GPUバッチエラー
  - GPUバッチサイズ不整合
  - GPU検証失敗によるCPUフォールバック

ローカル確認(1GiB, balanced, gpu_fraction=1.0):
- 修正前: gpu_chunks=0 相当のケースあり
- 修正後例: gpu_chunks=72 / cpu_chunks=184
- ただし圧縮率はまだ高め (`ratio=0.3967` 例) で、balancedの圧縮品質は引き続き改善余地あり

## 2026-02-24 - gpu-fractionフラグ再追加（再適用）

対応内容:
- `cozip_deflate/examples/bench_1gb.rs`
  - `--gpu-fraction <R>` を再追加（0.0..=1.0）
  - デフォルトを `1.0` に設定
  - CPU+GPU 実行時の `HybridOptions.gpu_fraction` に反映
  - ベンチ出力に `gpu_fraction=...` を表示
- `cozip_deflate/src/lib.rs`
  - `HybridOptions::default().gpu_fraction` を `1.0` に変更

確認:
- `cargo check -p cozip_deflate --example bench_1gb` 通過

## 2026-02-24 - GPU検出確認用 nocapture テスト追加

目的:
- `cargo test ... -- --nocapture` で、現在環境で見えているGPUと、PowerPreference別に選択されるGPUを確認できるようにする。

変更:
- `cozip_deflate/tests/hybrid_integration.rs`
  - `print_current_gpu_with_nocapture` テストを追加
  - 検出アダプタ一覧（name/vendor/device/type/backend/driver）を表示
  - `HighPerformance` / `LowPower` それぞれで `request_adapter` 結果を表示
  - それぞれ `request_device` の成否を表示
  - GPU未検出でも panic せずに情報出力して通る

ローカル実行結果:
- 検出:
  - NVIDIA GeForce RTX 5070 Laptop GPU (Vulkan)
  - AMD Radeon 890M Graphics (GL)
- 選択:
  - HighPerformance: NVIDIA
  - LowPower: NVIDIA

## 2026-02-24 - GPU dispatch 2D化 + submit/collect待機点削減 (着手: 1,2)

目的:
- 16MiB級チャンクで `dispatch_workgroups(x>65535)` に当たる問題を解消
- GPU圧縮バッチで待機点を減らし、submit/collectをより分離

実装:
1) 2D dispatch 対応
- 追加:
  - `MAX_DISPATCH_WORKGROUPS_PER_DIM = 65535`
  - `dispatch_grid_for_groups()`
  - `dispatch_grid_for_items()`
- 変更:
  - `pass.dispatch_workgroups(..)` を主要GPUパスで `(x,y,1)` に変更
  - 対象: tokenize/litlen/bitpack/freq/dyn_map/scan blocks/scan add/match/count/emit
- WGSL側 index を 2D flatten 対応
  - workgroup_size=128 系: `idx = id.x + id.y * 8388480`
  - workgroup_size=256 系: `idx = id.x + id.y * 16776960`
  - scan_blocks は `gid = wg_id.x + wg_id.y * 65535`
- scan_blocks の over-dispatch 対策
  - `block_sums[gid]` 書き込み時に `gid*256 < len` ガード追加

2) submit/collect の待機点削減（fixed GPU batch path）
- `deflate_fixed_literals_batch` で、slot枯渇時に `Wait` 後まとめて ready readback を回収
- 最終回収で `poll(Wait)` を1回化し、pendingを連続 drain
- 1件ずつ `Wait` していた箇所を削減

確認:
- `cargo check -p cozip_deflate` 通過
- `cargo test -p cozip_deflate --lib` 通過
- `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
- `cargo run --release -p cozip_deflate --example bench_hybrid` 実行
  - 4MiB: CPU+GPU comp ~11.679ms
  - 16MiB: CPU+GPU comp ~94.245ms
- `cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 16 --iters 1 --warmups 1 --chunk-mib 16 --gpu-subchunk-kib 256 --mode speed --gpu-fraction 1.0`
  - 16MiB単一チャンク (chunk_mib=16) が panicせず通ることを確認

メモ:
- 2D flatten の stride は `global_invocation_id.x` の性質上、workgroup_size込みで設定する必要がある。

## 2026-02-24 - dynamic側バッチ最適化（phase分離）

目的:
- dynamic Huffman 経路のチャンク毎 `submit->map->wait` を削減し、GPU待機点を減らす。

変更:
- `deflate_dynamic_hybrid_batch` を2段階バッチに再構成
  1. Phase1: tokenize/finalize/freq を複数チャンクまとめて submit
     - 周期: `GPU_PIPELINED_SUBMIT_CHUNKS`
     - submit後に freq readback を map 予約し、最後に一括 `poll(Wait)` で回収
  2. Phase2: CPUで Huffman plan 生成後、dyn_map/bitpack/finalize を複数チャンクまとめて submit
     - 同様に readback map を束ね、最後に一括回収
- 追加した内部構造体
  - `PendingDynFreqReadback`
  - `PreparedDynamicPack`
  - `PendingDynPackReadback`

期待効果:
- dynamic経路での同期オーバーヘッド低減
- チャンク数が増えたときの submit/wait の効率化

確認:
- `cargo check -p cozip_deflate` 通過
- `cargo test -p cozip_deflate --lib` 通過
- `cargo test -p cozip_deflate --test hybrid_integration -- --nocapture` 通過
- `cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 1 --chunk-mib 4 --gpu-subchunk-kib 256 --mode ratio --gpu-fraction 1.0`
  - 実行成功（roundtrip OK）

## 2026-02-25 - GPU圧縮の時間計測ログ追加（COZIP_PROFILE_TIMING）

目的:
- 大幅改善余地の有無を切り分けるため、GPU圧縮パスのどこで時間を使っているかを可視化。

実装:
- `cozip_deflate/src/lib.rs`
  - `COZIP_PROFILE_TIMING=1` で有効化される軽量タイミング計測を追加
  - 追加ヘルパー:
    - `timing_profile_enabled()`
    - `elapsed_ms()`
    - `GPU_TIMING_CALL_SEQ`（ログ追跡用call id）
  - fixed GPU path (`deflate_fixed_literals_batch`) サマリ出力:
    - `t_encode_submit_ms`
    - `t_bits_rb_ms`（total_bits readback待ち）
    - `t_payload_submit_ms`
    - `t_payload_rb_ms`
    - `t_cpu_fallback_ms`
    - `payload_chunks/fallback_chunks/readback量`
  - dynamic GPU path (`deflate_dynamic_hybrid_batch`) サマリ出力:
    - `t_freq_submit_ms`
    - `t_freq_wait_plan_ms`（freq回収+CPU木生成）
    - `t_pack_submit_ms`
    - `t_pack_bits_rb_ms`（total_bits回収）
    - `t_payload_submit_ms`
    - `t_payload_rb_ms`
    - `t_cpu_fallback_ms`
    - `payload_chunks/fallback_chunks/readback量`
  - adaptive scheduler (`compress_hybrid_adaptive_scheduler`) サマリ出力:
    - 総時間、CPU/GPUチャンク数、入出力サイズ

使い方:
- 例:
  - `COZIP_PROFILE_TIMING=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 4096 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256 --mode speed --gpu-fraction 1.0`

ローカル動作確認:
- `cargo check -p cozip_deflate` 通過
- `cargo check -p cozip_deflate --example bench_1gb` 通過
- `COZIP_PROFILE_TIMING=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 8 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 256 --mode speed --gpu-fraction 1.0`
  - `[cozip][timing][gpu-fixed] ...`
  - `[cozip][timing][scheduler] ...`
  が出力されることを確認

## 2026-02-25 - dynamic freq区間の犯人切り分け（poll/map/plan分離）

目的:
- `t_freq_wait_plan_ms` が大きい問題を、GPU待機かCPU木生成かまで分解して特定する。

実装:
- `GpuDynamicTiming` を分割:
  - `t_freq_poll_wait_ms`
  - `t_freq_recv_ms`
  - `t_freq_map_copy_ms`
  - `t_freq_plan_ms`
- `deflate_dynamic_hybrid_batch` で
  - `device.poll(Wait)` 区間
  - receiver `recv` 区間
  - map + copy + unmap 区間
  - `build_dynamic_huffman_plan` 区間
  を個別計測。

ローカル確認 (64MiB, ratio):
- `COZIP_PROFILE_TIMING=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 512 --mode ratio --gpu-fraction 1.0`
- 出力例:
  - `t_freq_poll_wait_ms=283.916`
  - `t_freq_recv_ms=0.001`
  - `t_freq_map_copy_ms=0.021`
  - `t_freq_plan_ms=0.386`

所見:
- 少なくともこの実行では、`freq`区間の大半は「GPU freqカーネル完了待ち (`poll wait`)」。
- CPU側の木生成は支配的でない。

## 2026-02-25 - ratio: freq集計をworkgroup局所化 + capped dispatch

目的:
- `freq` フェーズの global atomic 競合を下げ、`t_freq_poll_wait_ms` を短縮する。

実装:
- `cozip_deflate/src/lib.rs`
  - 追加: `GPU_FREQ_MAX_WORKGROUPS = 4096`
  - 追加: `dispatch_grid_for_items_capped(items, group_size, max_groups)`
  - dynamic の freq pass dispatch を
    - `dispatch_grid_for_items(len, 128)` から
    - `dispatch_grid_for_items_capped(len, 128, GPU_FREQ_MAX_WORKGROUPS)`
    に変更
  - `freq` WGSL を変更:
    - 直接 `litlen_freq/dist_freq` へ atomicAdd する方式を廃止
    - `var<workgroup> local_litlen_freq/local_dist_freq` に集計
    - workgroup内で集計後、非0 binのみ global freq へ atomicAdd
    - grid-stride loop (`idx += num_workgroups*workgroup_size`) で1スレッドが複数トークン処理

確認:
- `cargo check -p cozip_deflate` 通過
- `COZIP_PROFILE_TIMING=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 512 --mode ratio --gpu-fraction 1.0`
  - ローカル結果:
    - `t_freq_poll_wait_ms=269.699`（前回計測 283.916 から減少）
    - `t_freq_plan_ms=0.756`（CPU木生成は依然支配的でない）

メモ:
- 改善幅は限定的で、さらに詰めるには `GPU_FREQ_MAX_WORKGROUPS` のチューニング、
  または partial histogram バッファを使った完全2pass reduce（global atomic最小化）が候補。

## 2026-02-25 - dynamic Phase1 深掘りプローブ追加（tokenize/finalize/freq分離）

目的:
- Phase1(`tokenize + token_finalize + freq`)の真犯人を予測ではなく計測で特定する。

実装:
- `COZIP_PROFILE_DEEP=1` を追加（`COZIP_PROFILE_TIMING` と併用推奨）
- `GpuAssist::profile_dynamic_phase1_probe()` を追加し、dynamic pathで最初の非空chunkに対して
  - tokenize pass を単独submit+wait
  - token_finalize pass を単独submit+wait
  - freq pass を単独submit+wait
  を実行し、各msを出力
- 出力形式:
  - `[cozip][timing][gpu-dynamic-probe] ... t_tokenize_ms=... t_finalize_ms=... t_freq_ms=...`

確認:
- `cargo check -p cozip_deflate` 通過
- `COZIP_PROFILE_TIMING=1 COZIP_PROFILE_DEEP=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 512 --mode ratio --gpu-fraction 1.0`
- ローカル例:
  - `t_tokenize_ms=17.369`
  - `t_finalize_ms=2.386`
  - `t_freq_ms=0.171`

所見:
- 少なくともこの計測では、`freq` ではなく `tokenize` がPhase1支配要因。

## 2026-02-25 - tokenize内訳プローブ（literal/head/extend 差分計測）

目的:
- `tokenize` 内の真犯人を特定するため、処理内訳を差分で計測。

実装:
- tokenize WGSL に deep profile用 mode を追加
  - `100`: literal-only（候補探索なし）
  - `101`: head-only speed
  - `102`: head-only balanced
  - `103`: head-only ratio
- `profile_dynamic_phase1_probe()` を拡張
  - `tokenize` を 3回実行（lit/head/full）
  - `head_only = head_total - lit`
  - `extend_only = full - head_total`
  - 各モードは warmup 1回 + 計測1回でブレを低減
- 出力形式:
  - `[cozip][timing][gpu-dynamic-probe] ... t_tokenize_lit_ms=... t_tokenize_head_total_ms=... t_tokenize_full_ms=... t_tokenize_head_only_ms=... t_tokenize_extend_only_ms=...`

確認:
- `cargo check -p cozip_deflate` 通過
- `COZIP_PROFILE_TIMING=1 COZIP_PROFILE_DEEP=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 512 --mode ratio --gpu-fraction 1.0`
- ローカル例(4MiB probe):
  - `t_tokenize_lit_ms=0.313`
  - `t_tokenize_head_only_ms=0.486`
  - `t_tokenize_extend_only_ms=15.653`

所見:
- tokenize内では `extend_only`（一致後の長さ延長ループ）が圧倒的に支配的。

## 2026-02-25 - tokenize延長ループ最適化（4byte比較）

目的:
- 真犯人だった `tokenize_extend_only` を直接短縮する。

実装:
- tokenize WGSL に `load_u32_unaligned()` を追加
- 一致後延長ループを
  - 旧: 1byteずつ `byte_at(p) == byte_at(p-dist)` 比較
  - 新: 4byte比較 (`left4 == right4`) を先行し、ミスマッチ時は `countTrailingZeros(xor)>>3` で差分byte位置まで一気に進める
  - 残りは tail の1byteループで処理
- 先頭3byte判定の `byte_at(i), byte_at(i+1), byte_at(i+2)` を事前読み出しして再利用

確認:
- `cargo check -p cozip_deflate` 通過
- `COZIP_PROFILE_TIMING=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 512 --mode ratio --gpu-fraction 1.0`
  - `t_freq_poll_wait_ms`: 283ms級 -> 149ms級（ローカル）
- `COZIP_PROFILE_TIMING=1 COZIP_PROFILE_DEEP=1 ...` で深掘り
  - `t_tokenize_extend_only_ms`: 15.6ms級 -> 7.3ms級（4MiB probe, ローカル）

メモ:
- 深掘り (`COZIP_PROFILE_DEEP=1`) は計測用追加実行が入るため、実運用ベンチ比較には使わない。

## 2026-02-25 - tokenize延長ループの追加最適化（16byte先行比較）

目的:
- 4byte比較化の次段として、長い一致ランでの反復回数をさらに削減する。

実装:
- tokenize WGSL の延長ループに 16byte 先行比較を追加
  - 4byte×4本の `load_u32_unaligned` 比較
  - 途中不一致時は `xor + countTrailingZeros >> 3` で一致バイト数を即時反映
  - 16byte一致時のみ `mlen/p/scanned` を `+16`
  - その後は既存の4byteループ -> 1byte tail へフォールバック

確認:
- `cargo check -p cozip_deflate` 通過
- `COZIP_PROFILE_TIMING=1 COZIP_PROFILE_DEEP=1 cargo run --release -p cozip_deflate --example bench_1gb -- --size-mib 64 --iters 1 --warmups 0 --chunk-mib 4 --gpu-subchunk-kib 512 --mode ratio --gpu-fraction 1.0`
  - ローカル例(4MiB probe):
    - `t_tokenize_extend_only_ms`: 15.6ms級 -> 7.6ms級
    - `t_tokenize_full_ms`: 16.8ms級 -> 8.4ms級
  - dynamic全体:
    - `t_freq_poll_wait_ms`: 280-313ms級 -> 148-170ms級

メモ:
- `COZIP_PROFILE_DEEP=1` 実行では追加プローブ分のオーバーヘッドがあるため、最終スループット比較は
  `COZIP_PROFILE_TIMING=1` のみで行う。

## 2026-02-25 - deep計測の誤比較防止

実装:
- `COZIP_PROFILE_DEEP` 有効時に警告を1回だけ表示
  - `throughput numbers include probe overhead ...`
- deep probe 実行回数を「各dynamicバッチ」から「プロセス全体で1回」に変更
  - `DEEP_DYNAMIC_PROBE_TAKEN` で制御
- `COZIP_PROFILE_TIMING=1` 時に選択GPUアダプタ情報を出力
  - name/vendor/device/backend/type

目的:
- deep計測が有効なままベンチ比較してしまう事故を減らす
- ハイブリッドGPU環境でアダプタ揺れを観測しやすくする
