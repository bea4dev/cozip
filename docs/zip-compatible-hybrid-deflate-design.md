# ZIP互換ハイブリッド圧縮設計（CZDF資産活用）

日付: 2026-02-26

## 目的

- 他ソフト互換（ZIP Method 8 / raw deflate）を満たす
- 既存の CZDF 系ハイブリッド実装（CPU+GPU）を可能な限り再利用する
- 速度低下を最小化する（CPU_ONLY / HYBRID の比較軸を維持）
- 大容量入力でも OOM Killer に狙われにくいストリーミング処理を維持する

## 現状と課題

- `cozip` の ZIP 経路は互換性維持のため Deflate ストリームを使う
- ただし `Hybrid` 指定時でも現状は CPU deflate ストリームにフォールバック
- 理由は、既存 GPU 圧縮経路が CZDF/CZDS フレーム前提であり、ZIP が要求する単一 raw deflate bitstream へ直接流し込めないため

## 設計方針

1. 出力形式を明確に分離する
- CZDF/CZDS（独自フレーム）
- ZIP raw-deflate（互換出力）

2. 圧縮実行基盤は共通化する
- チャンク分割
- CPU/GPU ワークスティーリング
- 結果の index 順再整列

3. 互換出力専用の Mux 層を新設する
- チャンク単位の deflate ビット列を RFC1951 として連結
- 非最終チャンクは `BFINAL=0`、最終チャンクのみ `BFINAL=1`
- ビット単位で連結し、境界ずれを吸収

## 目標アーキテクチャ

1. `ChunkScheduler`
- 入力を `chunk_size` で分割
- CPU/GPU ワーカーへ投入
- GPU が使えない場合は CPU のみ

2. `ChunkEncoderBackend`（共通 IF）
- `CpuBackend` / `GpuBackend` を同一インターフェースに統一
- 戻り値は `EncodedChunk { index, bytes, bit_len, raw_len, crc32 }`

3. `RawDeflateMuxWriter`
- `EncodedChunk` を index 順に受け取り、単一 raw-deflate に合成
- `BFINAL` 制御を一元化
- 出力 bytes、出力 CRC、出力サイズを逐次更新

4. `ZipStreamWriter`
- 既存の ZIP ヘッダ、data descriptor、central directory は維持
- 圧縮データ部分のみ `RawDeflateMuxWriter` の出力へ差し替え

## ストリーミング/OOM対策

- 全入力の一括ロードは禁止
- `read -> compress -> mux/write` を段階パイプライン化
- `inflight_chunks` を固定上限化（例: `2 * workers + 2`）
- ディレクトリ圧縮はファイル単位で順次処理
- async API は `spawn_blocking` で CPU-heavy 処理を隔離し、I/O は async のまま維持

## API 方針

1. `cozip`
- 既存 `CoZip` API を維持
- `ZipOptions` で backend 方針を選択（既定は Hybrid）

2. `cozip_deflate`
- 互換経路用 API を追加
- 例: `compress_raw_deflate_stream_hybrid(...)`
- 戻り値は `DeflateCpuStreamStats` 相当 + backend 実績（CPU/GPU チャンク数）を含める

## 互換性要件

- ZIP Method 8 の raw deflate 出力であること
- 他実装で展開可能であること（`unzip`, `7z`, `bsdtar` 等）
- CRC32 とサイズ整合の検証を従来どおり維持
- 非対応機能（暗号化/ZIP64 など）は現状方針を維持

## 性能見込み（目安）

- 絶対速度: CZDF 専用経路比で -10% から -25% 程度を想定
- speedup 比（CPU_ONLY/HYBRID）: 共通コストが両者に乗るため、絶対速度ほどは悪化しにくい
- 圧縮率: チャンク独立性により +0.5% から +3% 程度悪化する可能性

## 実装フェーズ

1. Phase 1: Mux の最小実装
- CPU backend のみで raw-deflate 連結
- 互換性テストを先行

2. Phase 2: GPU backend 接続
- 既存 GPU 圧縮結果を `EncodedChunk` に適合
- Mux 経由で ZIP 互換出力へ統合

3. Phase 3: ストリーム最適化
- bounded queue と backpressure
- chunk サイズ、batch サイズの最適化

4. Phase 4: ベンチと回帰防止
- CPU_ONLY/HYBRID の比較ベンチ
- 圧縮率、速度、メモリ上限の回帰テスト

## 受け入れ基準

- ZIP 出力を主要ツールで解凍可能
- 1GB 級入力で OOM しない
- CPU_ONLY/HYBRID の比較が継続可能
- 既存 CZDF 経路を壊さない（後方互換）
