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
