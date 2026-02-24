# Deflate Parallel Profile (cozip draft)

更新日: 2026-02-24

## 1. 目的

標準Deflate互換を維持しながら、チャンク単位でCPU/GPUに独立タスクを配分できるプロファイルを定義する。

## 2. 基本方針

- Deflate bitstream 自体は RFC 1951 準拠
- ただし cozip 圧縮時は以下の制約を追加
- チャンクを跨ぐ距離参照(distance)を禁止
- チャンク開始位置を索引化し、並列解凍可能にする
- 実装プロファイルは `Chunk-Member Profile (CMP)` を採用する

## 3. チャンク仕様(案)

- 推奨 `host_chunk` サイズ: 1〜4MiB(初期2MiB)
- GPU内部は `gpu_subchunk`: 64〜256KiB(初期128KiB)
- 各 `host_chunk` は独立した Deflate member を形成する
- 各 member 内でブロック境界を完結させる
- BFINAL は各 member 内で完結してよい(メンバー単体解凍可能を優先)

## 4. ZIPへの格納

ZIP Method 8(Deflate)で格納し、独自情報は extra field に保持する。

### 4.1 Extra Field案

- Header ID: 実装時に衝突しないIDを採番
- Data:
- バージョン
- チャンクサイズ
- チャンク数
- 各チャンクの圧縮オフセット
- 各チャンクの展開後サイズ
- フラグ(固定Huffman/動的Huffmanなど)

このextra fieldは未知フィールドとして無視可能なため、通常のZIPツールとの共存を保てる。

## 5. 圧縮ルール

1. 入力分割
2. `host_chunk` 単位でCPU/GPUへ動的配分
3. GPU経路は `gpu_subchunk` 並列でLZ77探索・トークン生成
4. Huffman符号化をチャンク単位で実施
5. 各チャンクを独立 Deflate member として出力
6. `index` 順で連結し、フレームに格納

## 6. 解凍ルール

### 6.1 cozip profileあり

- extra field索引で `host_chunk` 境界を取得
- チャンクをCPU/GPUへ分散し独立解凍
- 出力先オフセットへ直接書き込み

### 6.2 profileなし

- 互換維持のため逐次解凍
- 将来拡張でチェックポイント並列解凍を検討

## 7. 互換性戦略

- cozip生成ZIP: 高速経路(並列解凍)を利用
- 他ツール生成ZIP: フォールバック経路で解凍
- これにより「高速最適化」と「一般互換」を両立する

## 8. 未決事項

- GPU解凍を固定Huffman先行にするか、動的Huffmanまで同時実装するか
- 索引情報の圧縮方法(varintなど)
- GPU bitpack の導入時期(圧縮側でCPU bitpackをいつ置き換えるか)
- ZIP統合時にCMPメンバーをどう格納するか(Method 8 単一streamとの整合性)
