# cozip architecture (v1 draft)

更新日: 2026-02-24 (rev2)

## 1. ゴール

- Rust製ライブラリとして ZIP 圧縮・解凍を提供する
- Deflate(Method 8)を実装する
- CPU と GPU(WebGPU)に独立タスクを割り当て、同時実行でスループットを最大化する
- まずは「cozipが生成したデータを高速に圧縮・解凍できること」を優先し、一般的なZIP互換は段階的に拡張する

## 2. 前提と制約

- GPUタスクは大きな並列性が必要で、分岐と依存が少ない処理が向く
- Deflate は可変長ビット列のため、最終ビット詰めはCPUのほうが実装容易
- ZIP互換性を守るため、最終出力は標準ZIP構造(ローカルヘッダ/セントラルディレクトリ)に従う
- WebGPU起動コストがあるため、小データはCPUのみが有利なケースがある

## 3. 全体構成

1. `ChunkPlanner`
- 入力を `host_chunk`(初期値: 2MiB)に分割
- GPU投入時は内部で `gpu_subchunk`(初期値: 128KiB)に細分化

2. `HybridScheduler`
- CPUワーカープールとGPUワーカーで共通キューを消費
- 実測スループット(EWMA)で配分比を動的調整

3. `CPUCodecWorker`
- LZ77候補探索(小サイズ・分岐が多いケース)
- Huffman生成・ビットパック

4. `GPUCodecWorker(WebGPU)`
- `host_chunk` を一括転送し、`gpu_subchunk` 並列でLZ77候補探索
- 可変長出力は2パス(長さ計算->prefix sum->emit)でトークン列生成
- 初期段階はCPU bitpack、段階的にGPU bitpackへ拡張

5. `DeflateAssembler`
- チャンクごとのトークン列からDeflateブロックを生成
- ブロック連結し、ZIPエントリの圧縮データを構築

6. `ZipContainer`
- ローカルヘッダ、データディスクリプタ、セントラルディレクトリを構築
- 並列解凍向けメタデータをZIP extra fieldへ格納

## 4. 圧縮パイプライン

1. 入力をチャンク分割
2. 各 `host_chunk` を CPU/GPU へ投げ、LZ77トークン列を生成
3. チャンク単位で Huffman 符号化情報を決定
4. ビット列へエンコード
5. チャンクを独立 Deflate member として生成し、`index` 順に連結
6. ZIPエントリとして格納

### 4.1 独立性ルール(重要)

- `host_chunk` 跨ぎの後方参照を禁止する
- 各 `host_chunk` はDeflateブロック境界から開始する
- これにより、チャンク単位で圧縮・解凍タスクを独立実行しやすくする

圧縮率はやや低下するが、GPU並列性と実装容易性を優先する。

### 4.2 実装プロファイル

- 当面は `Chunk-Member Profile (CMP)` を採用する
- `host_chunk` ごとに独立 Deflate member を作り、フレームで管理する
- CPU/GPUは共通キューからチャンクを取得して独立圧縮する

## 5. 解凍パイプライン

### A. cozip並列プロファイル(優先)

1. ZIP extra field からチャンク索引を読む
2. チャンクごとに CPU/GPU へ解凍タスク配分
3. 出力バッファへオフセット書き込みし結合

### B. 一般ZIP互換フォールバック

- 索引なし/独立性なしのDeflateはCPU逐次解凍へフォールバック
- 将来的に「チェックポイント生成 + 部分並列解凍」を追加

## 6. スケジューリング方針

- 初期: 静的比率(例 CPU 50% / GPU 50%)
- 学習: 直近Nチャンクの処理時間からEWMA配分比を更新
- しきい値:
- 入力が小さい場合(例 < 1MiB)はCPUのみ
- GPUキュー飽和時はCPUに寄せる

## 7. 公開API案(初期)

```rust
pub struct CodecOptions {
    pub chunk_size: usize,
    pub prefer_gpu: bool,
    pub parallel_profile: bool,
}

pub fn zip_compress(input: &[u8], options: &CodecOptions) -> Result<Vec<u8>, CozipError>;
pub fn zip_decompress(input: &[u8], options: &CodecOptions) -> Result<Vec<u8>, CozipError>;
```

将来的にはストリーミングAPI(`Read`/`Write`)を追加する。

## 8. 実装マイルストーン

1. M1: 純CPUでチャンク独立Deflate + ZIP入出力
2. M2: `GpuContext` 使い回し + 動的ワークキュー
3. M3: WebGPUでLZ候補探索 + トークン生成(圧縮側本体)
4. M4: 並列プロファイル解凍のGPU対応(固定Huffmanから段階導入)
5. M5: 一般ZIP向けフォールバック強化

## 9. 主要リスク

- チャンク独立化による圧縮率低下
- GPU転送コストが小データで逆効果
- Deflateビットレベル仕様の実装バグが互換性問題を招く

対策として、早期に `zlib`/`libdeflate` との相互検証テストを導入する。
