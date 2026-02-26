# ZIP互換GPU圧縮メモ

日付: 2026-02-26

## 背景

`cozip` は ZIP ラッパーとして公開APIを持つが、現在の `ZipOptions` には `deflate_mode`（`Hybrid` / `Cpu`）がある。

- 既定値は `Hybrid`
- ただし現時点の ZIP 経路では `Hybrid` 指定時も CPU deflate ストリームへフォールバックする

## 現状の仕様（重要）

1. `cozip` の ZIP 出力は ZIP Method 8（Deflate）互換である必要がある。
2. ZIP Method 8 のエントリ本体は RFC1951 の **raw deflate bitstream** でなければならない。
3. `cozip_deflate` のハイブリッド圧縮は、現在は独自フレーム系（CZDF/CZDS）であり、そのまま ZIP エントリへ格納できない。
4. そのため ZIP 経路では安全側として CPU deflate へフォールバックしている。

## 「raw-deflate要件」とは

- ZIP Method 8 の圧縮データ部分に、zlib/gzip ヘッダや独自ヘッダを含めないこと。
- 一般的な解凍ソフト（unzip/7z/Windows標準など）が展開可能な RFC1951 形式であること。

## チャンク独立方式の可否

結論: 可能。

- 1つの ZIP エントリ内で Deflate ブロックを複数連結できる。
- 各ブロックが独立した Huffman 木（dynamic Huffman）を持っても RFC1951 的に問題ない。
- したがって、チャンクごとに独立符号化して互換性を維持する設計は成立する。

## 難しさの本質

- 「仕様上できるか」ではなく、次の実装品質にある。
  - GPU で生成したトークンを RFC1951 の正しいビット列にすること
  - ブロック連結時の `BFINAL` / ヘッダ / ビット境界処理
  - 転送・同期コストを抑えて CPU 実装より速くすること
  - 圧縮率低下（チャンク境界での後方参照断絶）とのトレードオフ

## 今後の方針案

1. ZIP経路で `Hybrid` 指定時に `actual_backend_used` を返す（誤解防止）
2. もしくは、未実装期間は `Hybrid` を `Unsupported` として明示エラー化
3. 中期的には「ZIP向け raw-deflate ハイブリッド経路」を別実装として導入する

