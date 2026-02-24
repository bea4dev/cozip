# GPU Deflate Chunk Pipeline (CPU+GPU Full-Power)

更新日: 2026-02-24

## 1. 目的

- 入力を独立チャンクへ分割する
- CPU と GPU がそれぞれ Deflate を独立実行する
- 完了したチャンクを元の順序で連結して最終出力を作る
- 圧縮率よりも並列スループットを優先する

## 2. 基本方針

採用プロファイル: `Chunk-Member Profile (CMP)`

1. `host_chunk` 単位で完全独立
2. 各チャンクは「独立した Deflate member」として圧縮
3. メンバー間で辞書共有しない(チャンク跨ぎ参照禁止)
4. コンテナ側(CZDF)でチャンク境界とオフセットを保持

この方式では圧縮率が低下しうるが、CPU/GPU並列化は単純で堅牢になる。

## 3. データモデル

## 3.1 圧縮ジョブ

- `ChunkJob { index, input_offset, input_len, bytes }`
- `index` が最終連結順序を決める

## 3.2 圧縮結果

- `ChunkMember { index, backend(cpu|gpu), raw_len, cmp_len, payload, crc32 }`
- `payload` はチャンク単体で解凍可能な Deflate データ

## 3.3 フレーム

- `FrameHeader`
- `ChunkTable[]` (index順)
- `ChunkPayload[]`

`ChunkTable` には最低限以下を入れる:
- `index`
- `backend`
- `raw_len`
- `cmp_offset`
- `cmp_len`

## 4. 圧縮パイプライン

1. `ChunkPlanner`: 入力を `host_chunk` に分割
2. `Scheduler`: 共通キューへ全チャンク投入
3. CPUワーカー: `cpu_deflate(job)`
4. GPUワーカー: `gpu_deflate(job)` (失敗時はCPUフォールバック)
5. `Assembler`: `index` 順に `ChunkMember` を連結してフレーム化

## 4.1 スケジューリング

- CPUとGPUは同じ `ready_queue` を取り合う
- ただし `gpu_fraction` に基づくGPU予約を許可
- 予約し過ぎてGPU待ちが出る場合はCPU横取りを許可(タイムアウト付き)

## 4.2 GPU Deflate 最小実装

`gpu_deflate(job)` の中身:

1. `match_find` (GPU): LZ候補探索
2. `token_count` (GPU): 出力サイズ算出
3. `prefix_sum` (GPU): 出力オフセット確定
4. `token_emit` (GPU): LZトークン出力
5. `huffman + bitpack`:
- Phase-1: CPU
- Phase-2: GPU化

## 5. 解凍パイプライン

1. `FrameParser` が `ChunkTable` を読む
2. CPU/GPU が `ChunkMember` を独立解凍
3. `index` 順に展開データを連結
4. 最終バッファを返す

`decompress` も完全にチャンク独立なので、CPU/GPU の同時実行が容易。

## 6. チャンクサイズ指針

- `host_chunk_size`: 1〜8MiB (初期 2MiB)
- `gpu_subchunk_size`: 64〜256KiB (初期 128KiB)
- 小入力(例 1MiB未満)は CPU 優先

## 7. 想定リスク

1. 圧縮率低下
- 原因: チャンク独立化で参照範囲縮小
- 対策: `host_chunk_size` を大きめにする

2. GPU転送/同期コスト
- 原因: 小チャンク過多
- 対策: 大きめ `host_chunk` + バッチ転送

3. 片側ボトルネック
- 原因: 静的配分
- 対策: 動的キュー + EWMA重み更新

## 8. 実装フェーズ

1. F1: `ChunkMember` 形式を `cozip_deflate` へ固定
2. F2: GPU `match_find/token_count/prefix_sum/token_emit` を圧縮経路へ導入
3. F3: GPU解凍(固定Huffman先行)を追加
4. F4: GPU bitpack 追加
5. F5: ZIP統合時のCMPメタデータマッピング
