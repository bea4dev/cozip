# GPU Full-Task Design (Deflate)

更新日: 2026-02-24

## 1. 目的

- CPUでチャンク分割し、GPUへ「解析補助」ではなく「実圧縮/実解凍タスク」を割り当てる
- CPUとGPUが同時に別チャンクを処理し、片側ボトルネックを緩和する
- ZIP/Deflate互換を維持しつつ、cozip並列プロファイルで高スループットを狙う

## 2. 主要アイデア

2段階チャンク化を採用する。

1. ホストチャンク (`host_chunk`)
- CPUスケジューラが扱う単位
- 目安: 1〜4MiB
- CPU/GPUどちらへ投げるかをここで決める

2. GPUサブチャンク (`gpu_subchunk`)
- GPUカーネルが並列処理する単位
- 目安: 64〜256KiB
- 1つの `host_chunk` をGPU内部で分割して実行する

狙いは「転送は大きく、演算は細かく」である。

## 3. 圧縮パイプライン設計

### 3.1 全体フロー

1. `ChunkPlanner` が入力を `host_chunk` に分割
2. `HybridScheduler` が各 `host_chunk` をCPU/GPUへ動的配分
3. CPUワーカーは通常のDeflateチャンク圧縮を実行
4. GPUワーカーは `host_chunk` を一括転送し、内部で `gpu_subchunk` 並列実行
5. 両者の結果を `DeflateAssembler` が順序通り連結

### 3.2 GPU圧縮カーネル(最小実装)

GPU経路は「Deflateトークン生成」まで担当する。

1. `match_find` パス
- 各 `gpu_subchunk` でLZ77候補探索
- まずはサブチャンク内参照に限定して独立性を維持

2. `token_count` パス
- 各位置で出力トークン数/ビット長を計算
- 可変長出力のため、実データはまだ書かない

3. `prefix_sum` パス
- `token_count` 結果をスキャンして出力オフセット確定

4. `token_emit` パス
- 確定オフセットへトークン列を書き込む
- Huffman用ヒストグラムも同時集計

5. CPU側 `bitpack`
- 初期フェーズはCPUでHuffman生成とビットパックを実施
- 段階的にGPU bitpackへ拡張可能

## 4. 解凍パイプライン設計

### 4.1 方針

解凍もGPUへ実タスクを割り当てる。ただし段階導入する。

1. D1: CPU/GPUでチャンク単位並列解凍(既存方針)
2. D2: GPUで固定Huffmanブロック解凍
3. D3: GPUで動的Huffmanブロック解凍(本命)

### 4.2 GPU解凍の実装上の注意

- Deflate解凍は依存が強く、圧縮より難しい
- `length/distance` 展開で書き込み競合が起きやすい
- 対策として、サブチャンクごとに出力領域を固定し、チャンク跨ぎ参照禁止を前提にする

## 5. スケジューラ設計(ボトルネック対策)

## 5.1 静的配分は避ける

- `50:50` 固定配分は遅い側を待つ時間が増える
- 実測性能の揺れに追従できない

## 5.2 動的ワークキュー

1. `ready_queue` に `host_chunk` を投入
2. CPUワーカーとGPUワーカーが空き次第 pop
3. 処理完了時に throughput を報告
4. `LoadBalancer` が次チャンク配分重みを更新

## 5.3 重み更新式(EWMA)

- `cpu_tp = ewma(cpu_tp, bytes / elapsed_ms)`
- `gpu_tp = ewma(gpu_tp, bytes / elapsed_ms)`
- 次配分率 `gpu_ratio = gpu_tp / (cpu_tp + gpu_tp)` (clampあり)

## 6. チャンクサイズ戦略

初期値:

- `host_chunk_size = 2MiB`
- `gpu_subchunk_size = 128KiB`
- `gpu_min_input = 1MiB` 未満はCPU優先

自動調整:

1. GPU待機時間が長い -> `host_chunk_size` を増やす
2. GPUカーネル占有が低い -> `gpu_subchunk_size` を小さくする
3. kernel launch過多 -> `gpu_subchunk_size` を大きくする

## 7. 想定される問題と対策

1. GPU初期化コスト
- 問題: 呼び出しごと初期化すると高コスト
- 対策: `GpuContext` を使い回し、セッション単位で保持

2. 転送コスト過多
- 問題: 小チャンク転送で逆効果
- 対策: 大きめ `host_chunk` に統一し、バッチ転送

3. 可変長出力の競合
- 問題: 同時書き込みでオフセット衝突
- 対策: 2パス(サイズ計算→prefix sum→emit)

4. 圧縮率低下
- 問題: 独立チャンク化で参照範囲が縮む
- 対策: `host_chunk` を適度に大きくし、サブチャンク境界で参照窓を工夫

5. 片側ボトルネック
- 問題: CPU or GPU が詰まる
- 対策: 動的ワークキュー + EWMA配分 + backpressure

## 8. 計測指標

必須メトリクス:

- `compress_throughput_cpu` / `compress_throughput_gpu`
- `decompress_throughput_cpu` / `decompress_throughput_gpu`
- `gpu_transfer_ms`, `gpu_kernel_ms`, `gpu_idle_ms`
- `chunk_wait_ms` (queue滞留)
- `compressed_size_ratio`

`-- --nocapture` では上記要約を標準出力へ表示する。

## 9. 実装フェーズ

1. P1: `GpuContext` 使い回し + 動的ワークキュー導入
2. P2: GPU `match_find` + `token_emit` 実装(圧縮本体をGPU化)
3. P3: GPU固定Huffman解凍
4. P4: GPU動的Huffman解凍
5. P5: ZIP extra field へ並列プロファイル索引統合
