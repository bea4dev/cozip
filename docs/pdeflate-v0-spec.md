# PDeflate v0 仕様（GPU並列解凍志向）

## 1. 目的
- LZ77由来の前方依存を弱め、GPU解凍の並列性を高める。
- チャンク単位で自己完結する形式にし、チャンク間依存を持たない。
- 既存Deflate互換は目指さず、独自形式としてまず実装可能性を優先する。

## 1.1 Deflateとの関係（重要）
- PDeflateは「Deflate系」の設計思想を継承する。
- 差し替える主対象は LZ77 の参照モデル（距離参照 -> 辞書テーブル参照）。
- エントロピー符号化は Deflate と同様に Huffman 符号化（fixed/dynamic）を使う前提とする。
- したがって、PDeflateは「DeflateのうちLZ段を置換した派生形式」である。

## 2. 設計方針
- 各チャンクに `read-only` のグローバル辞書テーブルを持つ。
- 本文は「参照命令」と「非参照リテラル命令」の列で表現する。
- チャンクを固定数セクション（デフォルト 128）に分割し、各セクションは独立して解凍可能にする。
- 参照は「過去出力位置」ではなく「辞書テーブルID」を引く（LZ77距離参照を直接使わない）。

## 3. 用語
- チャンク: 圧縮/解凍の最小独立単位（既定 4 MiB）。
- テーブルエントリ: 辞書テーブル中のバイト列。
- セクション: チャンク内の独立解凍単位（既定 128 分割）。

## 4. チャンクフォーマット（v0）

### 4.1 エンディアン
- 整数はすべて Little Endian。

### 4.2 固定ヘッダ（例）
- `magic[4] = "PDF0"`
- `version: u16 = 0`
- `flags: u16`
- `chunk_uncompressed_len: u32`
- `table_count: u16`  
  - 有効範囲: `0..=0x0FFF - 1`（最大 4095 エントリ）
- `section_count: u16`（既定 128）
- `table_index_offset: u32`
- `table_data_offset: u32`
- `section_index_offset: u32`
- `section_cmd_offset: u32`

注: 実装時は `u64` オフセット拡張を検討可（v0 は `u32` 想定）。

### 4.3 テーブルインデックス
- 配列長: `table_count`
- 各要素:
  - `entry_len: u8`（v0 上限 254、0 は無効）
- `entry_offset` は圧縮ストリームに保持しない。
- 解凍時に前処理として `entry_len` の累積和から `entry_offset` を生成する。
- 目的:
  - 圧縮データ側のメタデータを削減する。
  - 解凍実行前にランダムアクセス用オフセット表を構築し、線形探索を避ける。

### 4.4 テーブル本体
- テーブルエントリの生バイト列を連結配置。
- v0 制約:
  - エントリは「生バイト列のみ」。
  - エントリ同士の参照は禁止（再帰依存禁止）。

### 4.5 セクションインデックス
- 配列長: `section_count`
- 各要素:
  - `cmd_len: varint`（ULEB128）
- `cmd_offset` / `out_offset` は圧縮ストリームに保持しない。
- 解凍時に前処理として:
  - `cmd_offset`: `cmd_len` の累積和で生成
  - `out_offset` / `out_len`: `chunk_uncompressed_len` と `section_count` から決定的に生成
    - `out_offset(i) = floor(i * chunk_uncompressed_len / section_count)`
    - `out_len(i) = out_offset(i+1) - out_offset(i)`

この前処理で各セクションをランダムアクセスし、独立解凍できる。

## 5. 命令エンコード

### 5.1 命令ヘッダ（2バイト）
- `cmd16: u16`（Little Endian）
- ビット割り当て:
  - `ref_or_tag = cmd16 & 0x0FFF`（12 bit）
  - `len4 = (cmd16 >> 12) & 0x000F`（4 bit）

### 5.2 長さ
- `len4 != 0xF` のとき: `len = len4`
- `len4 == 0xF` のとき: 直後に `ext8: u8` を読み、`len = 0xF + ext8`

### 5.3 命令種別
- `ref_or_tag < 0x0FFF`:
  - **TABLE_REF**
  - `ref_or_tag` をテーブルIDとして参照。
  - `table[id]` の内容を繰り返し利用し、`len` バイト出力する。
  - LZ77方針に合わせ、参照命令の `len` は `>= 3` を必須とする。
- `ref_or_tag == 0x0FFF`:
  - **LITERAL_RUN**
  - 後続に `len` バイトの非圧縮リテラルをそのまま格納。

例:
- `[0xFF, 0xF1, 0x00]`
  - `cmd16=0xF1FF`
  - `ref_or_tag=0xFFF`（LITERAL_RUN）
  - `len4=1`（len=1）
  - 後続1バイト `0x00` を出力

### 5.4 Huffman符号化レイヤー
- 本節の命令表現は「論理命令列」の定義である。
- 実バイトストリーム化では、Deflate同様に Huffman（fixed/dynamic）で符号化する。
- v0ではまず命令意味論を固定し、符号語割当（シンボル設計）は別節/別版で厳密化する。

## 6. 解凍アルゴリズム（セクション単位）
1. チャンクヘッダ検証。
2. テーブルインデックス読み込み（必要ならGPU前処理バッファ化）。
3. セクションインデックス（`cmd_len` 列）から `cmd_offset` を再構築し、同時に `out_offset/out_len` を算出。
4. セクション命令列を先頭から順に実行し、`out_offset..out_offset+out_len` へ出力。
5. セクションごとに独立実行（CPU/GPUどちらでも可）。

GPU実行モデル:
- 1セクション = 1ワーカー（または1 workgroup）を基本単位。
- セクション間同期は不要（チャンク完了同期のみ）。

## 7. エンコード制約（重要）
- セクション独立性を壊す命令は禁止。
  - 他セクションの出力を参照する命令は存在しない。
- 参照命令は必ず有効なテーブルIDを指す。
- 参照命令で `len < 3` は生成しない（LZ77方針）。
- セクション出力長 `out_len` と実際の命令展開長が一致すること。

## 8. バリデーション/安全制約
- `table_count <= 4095`
- `entry_len in [1, 254]`
- `sum(entry_len)` が `table_data` 実長と一致すること
- `section_count > 0`
- `sum(section_cmd_len)` が `section_cmd` 実長と一致すること
- `section_cmd_len` の varint 列が正しく終端し、過長/オーバーフローしないこと
- 解凍時:
  - `max_expand_ratio`（例: 64x）
  - `max_ops_per_section`
  - `max_table_bytes`
  - `max_section_out_len`
- いずれか違反で即エラー（自動フォールバックはラッパー側方針に従う）。

## 9. 既定値（v0）
- chunk size: 4 MiB
- section count: 128
- table_count upper bound: 4095
- entry_len upper bound: 254
- len拡張: `len = 0xF + ext8`（`ext8` は 0..255）

## 10. 実装メモ
- 命令デコードは分岐を減らすため `TABLE_REF` と `LITERAL_RUN` の2命令に限定。
- テーブルオフセットは解凍前処理で再構築する（prefix-sum）。
- セクションの `cmd_offset` は varint `cmd_len` の prefix-sum で再構築する。
- セクションの `out_offset/out_len` は `chunk_uncompressed_len` と `section_count` から算出する。
- まず CPU 実装でフォーマット妥当性を固定し、その後 GPU 解凍へ展開する。
- 将来拡張:
  - 長さの多段拡張（varint）
  - 追加オペコード（RLE系など）
  - チャンクヘッダの 64bit オフセット化
