# CoZipDeflate Hybrid解凍 + ZIPメタデータ設計

更新日: 2026-02-27

## 1. 目的

- `CoZipDeflate` に CPU+GPU のハイブリッド解凍経路を追加する。
- 最終的に ZIP エントリのメタデータにチャンク索引を格納し、索引がある場合のみ高速解凍経路を使う。
- 既存の Hybrid圧縮で最適化済みのスケジューラ方針（`global-local`）はアルゴリズムを変更せず再利用する。

## 2. スコープ / 非スコープ

スコープ:
- Deflate(Method 8) のみ。
- CoZip が生成する「チャンク独立制約付き」ストリームの高速解凍。
- ZIP64 互換は維持（既存実装のまま）。

非スコープ:
- 任意の汎用ZIPを完全に並列解凍すること（メタデータ無しはCPU逐次フォールバック）。
- スケジューラの配分アルゴリズム変更。
- 実行時の重い検証（デバッグ時テストで担保する）。

## 3. 現状整理

- 圧縮は `deflate_compress_stream_hybrid_zip_compatible*` で CPU+GPU 実行済み。
- 解凍は `deflate_decompress_stream_on_cpu` のみ。
- ZIP側は ZIP64 extra のみ扱っており、CoZip独自の解凍索引は未実装。

## 4. 設計原則

1. 互換性優先
- 独自メタデータが無い/壊れている場合は必ず既存CPU経路へフォールバックする。

2. スケジューラ再利用
- 圧縮側の `global queue + local buffers` のタスク取り出し方・待ち方・カウンタ設計を解凍側でも踏襲する。

3. フォールバック責務の分離
- `CoZipDeflate` 層は「要求された実行モードを満たせない場合はエラーを返す」。
- CPUフォールバック可否の判断は `cozip` ラッパー層で行う。

4. 実行時コスト最小化
- メタデータ検証は最小限（境界、件数、サイズ一致）。
- 重い整合性検証はテスト・デバッグフラグで実施。

5. 段階導入
- いきなりGPUフル解凍にせず、CPU indexed解凍 -> GPU対象拡大の順で進める。

## 5. ZIPメタデータ仕様（CoZip Deflate Index v1）

## 5.1 置き場所

- 第1候補: Central Directory extra field に格納する。
- 第2候補: extra field に収まらない場合は ZIP64 EOCD の extensible data sector に格納する。
- Local Header extra への重複格納は行わない（可変長大索引と相性が悪いため）。

理由:
- Central Directory は圧縮完了後に書くため、ストリーミング圧縮でも索引を確定して書ける。
- Local Header は先に出るため、seek不要設計だと後書きが難しい。
- extra field の 64KiB 制限を超える大規模入力でも sidecar なしで運用できる。

## 5.2 Header ID

- 暫定 Header ID: `0x435A` (private use)
- data 先頭に magic/version を置き、将来のID変更や誤認を防ぐ。

## 5.3 データ構造（v1）

```text
magic[4]              = "CZDI"
u8  version           = 1
u8  flags             (bit0: indexed-deflate, bit1: table-varint)
u16 reserved          = 0
u32 chunk_size        (基本チャンクサイズ)
u32 chunk_count
u64 uncompressed_size
u64 compressed_size
u32 chunk_table_len
u32 chunk_table_crc32
u8  chunk_table[chunk_table_len]
```

`chunk_table` は `uLEB128` の繰り返し:
- `delta_comp_bit_off`
- `comp_bit_len`
- `final_header_rel_bit`
- `raw_len`

補足:
- `comp_*` は bit 単位で持つ（byte境界前提を置かない）。
- `final_header_rel_bit` を持つことで、非最終化済みチャンクでも解凍時に `BFINAL=1` へ戻せる。

## 5.4 64KiB制限への対応

- ZIP extra field 全体は 65535 bytes 制限がある。
- v1は以下の優先順で格納する。
  1. Central Directory extra field（優先）
  2. ZIP64 EOCD extensible data sector（overflow時）
  3. どちらにも格納できない場合のみ「索引なし」としてCPUフォールバック

実装方針:
- `extra` には小さな locator のみを置けるようにする。
  - `storage = 0`: inline（extraに本体格納）
  - `storage = 1`: eocd64（EOCD領域へ格納）
  - `storage = 2`: none
- EOCD64側には `CZDI` blob を複数エントリ分まとめて保持し、`entry_ordinal` で参照する。

## 6. CoZipDeflate 側アーキテクチャ

## 6.1 新規データモデル

```rust
pub struct DeflateChunkIndex {
    pub chunk_size: u32,
    pub chunk_count: u32,
    pub uncompressed_size: u64,
    pub compressed_size: u64,
    pub entries: Vec<DeflateChunkIndexEntry>,
}

pub struct DeflateChunkIndexEntry {
    pub comp_bit_off: u64,
    pub comp_bit_len: u32,
    pub final_header_rel_bit: u32,
    pub raw_len: u32,
}
```

## 6.2 圧縮API拡張

- 既存APIは保持。
- 索引が必要な呼び出し向けに、索引付き結果を返すAPIを追加。

例:
```rust
pub struct DeflateHybridCompressResult {
    pub stats: DeflateCpuStreamStats,
    pub index: Option<DeflateChunkIndex>,
}
```

## 6.3 解凍API追加

```rust
pub fn deflate_decompress_stream_hybrid_indexed<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    index: &DeflateChunkIndex,
    options: &HybridOptions,
) -> Result<DeflateCpuStreamStats, CozipDeflateError>;
```

## 6.4 スケジューラ再利用方針（重要）

- 圧縮側の以下を解凍側へそのまま再利用する。
  - `queue_state` / `ready_state` / `error` の共有構造
  - CPU/GPU worker + writer の3役
  - `gpu_batch_chunks`, `gpu_pipelined_submit_chunks`, `gpu_tail_stop_ratio`
  - `global-local` の取り出しポリシー

- 変更するのは「タスク本体」のみ:
  - 圧縮: `ChunkTask(raw) -> ChunkMember(compressed)`
  - 解凍: `DecodeTask(comp_range) -> DecodedChunk(raw)`

## 7. Hybrid解凍フロー

1. ZIP側で metadata(index) を取得。
2. Deflate payload を `Arc<[u8]>` として保持（必要なら mmap 化）。
3. `DecodeTask` を index 順にキュー投入。
4. CPU worker / GPU worker がタスク処理。
5. `ready_state` は index 順に writer が排出し、CRCと出力サイズを更新。

### GPU worker の段階導入

- `CoZipDeflate` 層では、GPU要求時に未対応ブロックが含まれる場合は `Unsupported` / `GpuExecution` 系エラーを返す。
- `cozip` 層はそのエラーを受けてCPU経路へ切替える（必要な場合のみ）。

## 8. cozip 側統合

## 8.1 圧縮時

- `stream_deflate_from_reader` で索引付き結果を受け取る。
- まず Central Directory extra field へ `CZDI v1` の inline格納を試みる。
- 収まらない場合は ZIP64 EOCD extensible data sector へ退避し、extraには locator のみ格納する。
- 既存 ZIP64 extra との共存を維持。

## 8.2 解凍時

- `read_central_directory_entries` で `CZDI` locator を parse。
- 解凍時の索引探索順:
  1. extra inline `CZDI`
  2. extra locator が指す EOCD64 `CZDI`
  3. 見つからなければ索引なし
- `extract_entry_to_writer` で以下分岐:
  - `method == DEFLATE` かつ `CZDI有効` -> `deflate_decompress_stream_hybrid_indexed`
  - それ以外 -> 既存 `deflate_decompress_stream_on_cpu`

## 9. テスト戦略

1. 単体テスト
- `CZDI` encode/decode roundtrip
- bit range / BFINAL patch の境界テスト
- chunk-mib 4/5 など過去に壊れた境界条件

2. 統合テスト
- ZIP roundtrip（single/directory）
- metadata有り/無しの両経路
- metadata破損時CPUフォールバック確認
- extra overflow時に EOCD64 退避されること
- EOCD64メタデータ欠損時のCPUフォールバック確認

3. 性能テスト
- 既存 `bench.sh` に解凍 speedup 指標を追加
- `speed/balanced/ratio` 各モードで `gpu_chunks` と `decomp_mib_s` を比較

## 10. フェーズ計画

### D0: 仕様と型
- `DeflateChunkIndex` 型追加
- `CZDI v1` encode/decode 実装
- `extra inline + EOCD64退避` の2段格納実装

### D1: CPU indexed解凍 + スケジューラ土台再利用
- 解凍タスク/ready writer を圧縮側と同型で実装
- 実際のdecodeはCPUのみ（GPU実行要求は未対応エラー）

### D2: GPU decode 実装（固定 + 動的Huffman）
- D2時点で固定Huffman・動的Huffmanの両方をGPU decode対象にする
- `speed/balanced/ratio` すべてでGPU decode可能にする
- 未対応ケースは `CoZipDeflate` でエラー返却（自動CPUフォールバックしない）

### D3: 最適化・安定化
- D2実装の性能最適化（wait削減、batch調整、readback安定化）
- ZIP metadata 経路をデフォルト有効化
- フォールバックポリシーは `cozip` 層で一元管理

## 11. 受け入れ基準

- metadata有りZIPで Hybrid解凍が動作し、CPU+GPU双方のチャンク実績が出る。
- metadata無し/破損ZIPで従来CPU解凍が正常動作する。
- ZIP64互換を維持し、既存 roundtrip が壊れない。
- スケジューラアルゴリズム（global-local）を変更していない。
