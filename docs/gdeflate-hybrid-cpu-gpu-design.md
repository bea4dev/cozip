# GDeflate CPU+GPUハイブリッド設計（独自形式先行）

更新日: 2026-02-28

## 1. 目的

- 当面は互換性より実装速度を優先し、独自GDeflate形式を導入する。
- CPUを管理専用にせず、CPUとGPUの両方が実圧縮・実解凍タスクを消化する。
- 既存 `CoZipDeflate` のスケジューラー（`GlobalQueueLocalBuffers`）を参照し、アルゴリズムは原則変更しない。

## 2. 非スコープ

- DirectStorage / MS GDeflate との完全互換（後続フェーズで対応）。
- 既存ZIP経路への即時統合（まずは独自コンテナで成立させる）。
- スケジューラー方針の刷新（新規アルゴリズム導入は行わない）。

## 3. 設計原則

1. 実処理のCPU参加
- CPUワーカーは「前処理だけ」でなく、実際に圧縮・解凍チャンクを処理する。

2. キュー共有
- CPU/GPUワーカーは同一のグローバルキューからタスクを取得する。
- 取り出しと再順序化は `CoZipDeflate` と同等の構造を使う。

3. チャンク独立
- 1チャンク単位で圧縮・解凍が完結する形式にする。
- ストリーム内依存を持ち込まない（CPU/GPUどちらでも処理可能にする）。

4. Writer順序保証
- 出力は `index` 順で確定。`ready_state` 経由で順序を整える。

## 4. CoZipDeflateスケジューラー流用ポイント

参照元（実装）:
- `src/cozip_deflate/src/lib.rs`
  - `compress_cpu_stream_worker_continuous`
  - `compress_gpu_stream_worker_continuous`
  - `decode_cpu_indexed_worker`（構造参照）
  - `cpu_worker_count`

流用する要素:
- `queue_state = (Mutex<QueueState>, Condvar)` + `ready_state` + `error` の3共有状態
- CPU複数ワーカー + GPUワーカーが同一キュー消費
- GPU側のバッチ提出 (`gpu_batch_chunks`, `gpu_pipelined_submit_chunks`)
- tail制御 (`gpu_tail_stop_ratio`) とGPU配分 (`gpu_fraction`)
- writer側のHOL待ち計測を含む統計収集

固定する方針:
- スケジューラーのアルゴリズム自体（global queue + local buffering）は変更しない。
- 新規最適化は「タスク中身」「GPUカーネル」「バッチ上限調整」に限定する。

## 5. 独自GDeflate v0 データモデル

```rust
pub struct GdFrameHeaderV0 {
    pub magic: [u8; 4],       // "CGDF"
    pub version: u16,         // 0
    pub chunk_size: u32,      // default 4 MiB
    pub chunk_count: u32,
    pub flags: u32,           // mode bits
}

pub struct GdChunkEntryV0 {
    pub comp_off: u64,
    pub comp_len: u32,
    pub raw_len: u32,
    pub codec_flags: u32,     // 予約
}
```

- チャンク索引はフレーム末尾に集約。
- まずは「1チャンク = 1独立圧縮データ」前提で実装する。

## 6. 圧縮フロー（Hybrid）

1. Readerが入力を `chunk_size` ごとに分割し `ChunkTask{index, raw}` を投入
2. CPUワーカー: `compress_chunk_cpu(task)` 実行
3. GPUワーカー: 複数チャンクをまとめて `compress_chunk_gpu_batch(tasks)` 実行
4. `ready_state[index]` へ格納
5. Writerが index順にフレームへ書き出し、`GdChunkEntryV0` を構築

要点:
- CPU/GPUとも同一の `ChunkMember` 形式を返す。
- chunk境界は常に独立なので、CPU/GPU混在でも結合が容易。

## 7. 解凍フロー（Hybrid）

1. フレームヘッダ・索引を読み `DecodeTask{index, comp_off, comp_len, raw_len}` を投入
2. CPUワーカー: `decode_chunk_cpu(task)` 実行
3. GPUワーカー: バッチ化して `decode_chunk_gpu_batch(tasks)` 実行
4. `ready_state[index]` に `DecodedChunk` を置く
5. Writerが index順に出力先へ書き込む

要点:
- 解凍でもCPUは実処理に参加する。
- GPU不利チャンク（小さい・失敗しやすい）はCPUへ回る前提。

## 8. API草案

```rust
pub struct GdHybridOptions {
    pub chunk_size: usize,
    pub prefer_gpu: bool,
    pub gpu_fraction: f32,
    pub gpu_tail_stop_ratio: f32,
    pub gpu_batch_chunks: usize,
    pub gpu_pipelined_submit_chunks: usize,
}

pub fn gdeflate_compress_hybrid<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &GdHybridOptions,
) -> Result<GdHybridStats, GdError>;

pub fn gdeflate_decompress_hybrid<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &GdHybridOptions,
) -> Result<GdHybridStats, GdError>;
```

## 9. 統計・計測（最低限）

- `cpu_chunks`, `gpu_chunks`, `chunk_count`
- `cpu_worker_busy_ms`, `gpu_worker_busy_ms`
- `cpu_wait_for_task_ms`, `gpu_wait_for_task_ms`
- `gpu_batches`, `gpu_batch_avg_ms`
- writer HOL系 (`writer_hol_wait_ms`, `writer_hol_ready_avg/max`)

`CoZipDeflate` の既存ログ項目と揃えることで、比較と回帰検知を容易にする。

## 10. 実装フェーズ

### G0: 形式とCPU基準実装
- `CGDF v0` ヘッダ/索引/エンコード・デコード
- CPU-only 圧縮/解凍でラウンドトリップ成立

### G1: ハイブリッド圧縮
- `CoZipDeflate` 型の queue/ready/worker 構造を流用
- CPU+GPU圧縮を同時稼働

### G2: ハイブリッド解凍
- 同じスケジューラー構造で decode worker を追加
- CPU+GPU解凍で同時稼働

### G3: 安定化
- batchサイズ・tail stopの調整
- 計測ログを基にホットスポット最適化

### G4: 互換性フェーズ（後日）
- MS GDeflate互換の差分吸収
- 必要ならコンテナ/ビットストリーム変換レイヤ追加

## 11. 受け入れ基準

- 圧縮/解凍ともにCPU+GPU両方の `*_chunks > 0` を確認できる。
- 1GiB級ベンチでCPU-onlyに対して圧縮スループットが改善する。
- 解凍でCPU-onlyより遅い場合でも、原因を `busy/wait/batch` 指標で説明可能。
- スケジューラーは `GlobalQueueLocalBuffers` 準拠のまま維持される。
