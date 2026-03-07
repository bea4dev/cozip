# PDeflate v0 仕様（GPU並列解凍志向）

## 1. 目的
- LZ77由来の前方依存を弱め、GPU解凍の並列性を高める。
- チャンク単位で自己完結する形式にし、チャンク間依存を持たない。
- 既存Deflate互換は目指さず、独自形式としてまず実装可能性を優先する。

## 1.1 Deflateとの関係（重要）
- PDeflateは「Deflate系」の設計思想を継承する。
- 差し替える主対象は LZ77 の参照モデル（距離参照 -> 辞書テーブル参照）。
- エントロピー符号化は任意とする。
- `flags` の `HUFFMAN` bit が立っている場合のみ Huffman を用いる。
- Huffman 使用時は、**チャンクごとに高速LUT復号テーブル（root table + subtable）を保持**し、各セクションは同一チャンク内LUTを参照して復号する。

## 2. 設計方針
- 各チャンクに `read-only` のグローバル辞書テーブル（TABLE_REF用）を持つ。
- Huffman 使用時のみ、各チャンクに `read-only` の Huffman LUT 群（root table + subtable）を持つ。
- 本文は「参照命令」と「非参照リテラル命令」の論理列で表現する。
- チャンクを固定数セクション（既定 128）に分割し、各セクションは独立して解凍可能にする。
- 参照は「過去出力位置」ではなく「辞書テーブルID」を引く（LZ77距離参照を直接使わない）。

## 3. 用語
- チャンク: 圧縮/解凍の最小独立単位（既定 4 MiB）。
- テーブルエントリ: 辞書テーブル中のバイト列。
- セクション: チャンク内の独立解凍単位（既定 128 分割）。
- Huffman LUT:
  - root table: 先頭 `R` bit（例: 9-11bit）で直接引くテーブル
  - subtable: 長符号語用の2段目テーブル

## 4. チャンクフォーマット（v0）

### 4.0 ストリームラッパー（PDS0）
- PDeflate ストリームは、固定長ヘッダ + 可変個のチャンク frame で構成する。
- 目的:
  - `zero-temp streaming` を成立させる
  - 圧縮時に全チャンク数/元サイズの事前確定を不要にする
- ストリームヘッダ:
  - `magic[4] = "PDS0"`
  - `version: u16 = 1`
  - `flags: u16`（v1 では 0 固定）
  - `chunk_size: u32`
- 本文:
  - `chunk_payload_len: u32`
  - `chunk_payload[chunk_payload_len]`
  - 上記 frame の繰り返し
- ストリームヘッダには `original_len` と `chunk_count` を保持しない。
- ストリーム終端は、最後のチャンクの `flags.FINAL_STREAM == 1` で示す。
- 空ストリームは、チャンク frame を 1 つも持たないヘッダのみのストリームとして表現してよい。

### 4.1 エンディアン
- 整数はすべて Little Endian。

### 4.2 固定ヘッダ（例）
- `magic[4] = "PDF0"`
- `version: u16 = 0`
- `flags: u16`
  - bit 0: `HUFFMAN`
    - `1`: セクション本文は Huffman 符号化済みであり、`huff_lut` を参照して復号する
    - `0`: セクション本文は Huffman 未使用であり、`section_bitstream` は論理命令列をそのまま格納する
  - bit 1: `FINAL_STREAM`
    - `1`: 当該チャンクがストリームの最終チャンク
    - `0`: 後続チャンクが存在しうる
- `chunk_uncompressed_len: u32`
- `table_count: u16`
  - 有効範囲: `0..=0x0FFF - 1`（最大 4095 エントリ）
- `section_count: u16`（既定 128）
- `table_index_offset: u32`
- `table_data_offset: u32`
- `huff_lut_offset: u32`
- `section_index_offset: u32`
- `section_bitstream_offset: u32`

注: 実装時は `u64` オフセット拡張を検討可（v0 は `u32` 想定）。

### 4.3 辞書テーブルインデックス（TABLE_REF用）
- 配列長: `table_count`
- 各要素:
  - `entry_len: u8`（v0 上限 254、0 は無効）
- `entry_offset` は圧縮ストリームに保持しない。
- 解凍時に前処理として `entry_len` の累積和から `entry_offset` を生成する。

### 4.4 辞書テーブル本体（TABLE_REF用）
- テーブルエントリの生バイト列を連結配置。
- v0 制約:
  - エントリは生バイト列のみ。
  - エントリ同士の参照は禁止（再帰依存禁止）。

### 4.5 Huffman LUT ブロック（チャンク共有）
- `huff_lut_offset..section_index_offset` に配置。
- `flags.HUFFMAN == 1` のときのみ、チャンク復号に必要な LUT 群を格納する。
- `flags.HUFFMAN == 0` のときは空でもよい。
- v0 要件:
  - 復号時に木構築を行わず、LUT参照のみで復号できること。
  - `root table + subtable` の2段参照方式を前提とする。
  - セクションは本ブロックを共有参照する（read-only）。

### 4.6 セクションインデックス
- 配列長: `section_count`
- 各要素:
  - `bit_len: varint`（ULEB128）
- `bit_offset` / `out_offset` は圧縮ストリームに保持しない。
- 解凍時に前処理として:
  - `bit_offset`: `bit_len` の累積和で生成
  - `out_offset` / `out_len`: `chunk_uncompressed_len` と `section_count` から決定的に生成
    - `raw_offset(i) = floor(i * chunk_uncompressed_len / section_count)`
    - `align_down4(x) = x & ~3`
    - `out_offset(0) = 0`
    - `out_offset(i) = align_down4(raw_offset(i))`（`1 <= i < section_count`）
    - `out_offset(section_count) = chunk_uncompressed_len`
    - `out_len(i) = out_offset(i+1) - out_offset(i)`
  - 注意:
    - 内部セクション境界（`i in [1, section_count-1]`）は必ず4-byte境界。
    - 末尾境界 `out_offset(section_count)` は `chunk_uncompressed_len` を優先し、4-byte境界でなくてもよい。
    - 端数（0..3 byte）は末尾側セクション長に集約。

### 4.7 セクションビットストリーム
- `section_bitstream_offset..chunk_end` に配置。
- 各セクションのビット列は `bit_offset/bit_len` で切り出して解釈する。
- セクション間でビット列は論理的に独立し、相互参照しない。

## 5. 論理命令エンコード

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
  - 参照命令の `len` は `>= 3` を必須とする。
- `ref_or_tag == 0x0FFF`:
  - **LITERAL_RUN**
  - 後続に `len` バイトの非圧縮リテラルを格納。

注:
- `flags.HUFFMAN == 1` のとき、上記は Huffman 符号化されたビット列として格納される。
- `flags.HUFFMAN == 0` のとき、上記は論理命令列の生バイト列として格納される。

## 6. 解凍アルゴリズム

1. ストリームラッパーがある場合は `PDS0` ヘッダを検証し、frame を順に読む。
   - `flags.FINAL_STREAM == 1` のチャンクに到達したら終端とみなす。
   - `PDS0` ヘッダだけで本文 frame が存在しない場合は空出力とみなす。
2. チャンクヘッダ検証。
3. 辞書テーブル（TABLE_REF）前処理。
4. `flags.HUFFMAN == 1` のときのみ Huffman LUT ブロックを読み込み（またはGPU用バッファへ転送）。
5. セクションインデックス（`bit_len` 列）から `bit_offset` を再構築し、同時に `out_offset/out_len` を算出。
6. 各セクションを独立復号し、`out_offset..out_offset+out_len` へ出力。

### 6.1 実行モデル（本仕様の必須方針）
- **CPU:** チャンク並列（chunk-level parallel）
- **GPU:** チャンク + セクション並列（chunk-level + section-level parallel）
- LUT はチャンク共有 read-only とし、セクション間同期は不要（チャンク完了同期のみ）。

## 7. エンコード制約（重要）
- セクション独立性を壊す命令は禁止。
  - 他セクションの出力を参照する命令は存在しない。
- 参照命令は必ず有効なテーブルIDを指す。
- 参照命令で `len < 3` は生成しない。
- セクション出力長 `out_len` と実際の命令展開長が一致すること。
- セクション境界は 4-byte ルール（内部境界）に従うこと。

## 8. バリデーション/安全制約
- `table_count <= 4095`
- `entry_len in [1, 254]`
- `sum(entry_len)` が `table_data` 実長と一致すること
- `section_count > 0`
- `sum(section_bit_len)` が `section_bitstream` 実ビット長と一致すること
- `section_bit_len` の varint 列が正しく終端し、過長/オーバーフローしないこと
- 内部セクション境界 `out_offset(i)`（`1 <= i < section_count`）が4-byte境界であること
- `flags.HUFFMAN == 1` のときの Huffman LUT 整合性:
  - root/subtable 参照が範囲外にならないこと
  - 無効シンボル/無限ループを誘発しないこと
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
- Huffman LUT: `flags.HUFFMAN == 1` のときのみ `root table + subtable`（チャンク共有）

## 10. 実装メモ
- 命令デコードは `TABLE_REF` と `LITERAL_RUN` の2命令に限定。
- 辞書テーブルオフセットは解凍前処理で再構築（prefix-sum）。
- セクション `bit_offset` は varint `bit_len` の prefix-sum で再構築。
- セクション `out_offset/out_len` は `chunk_uncompressed_len` と `section_count` から算出し、内部境界は4-byte整列。
- `flags.HUFFMAN == 1` のとき、Huffman 復号は木走査ではなく LUT 参照（root + subtable）を使う。
- `flags.HUFFMAN == 0` のとき、復号側は Huffman 復号をスキップして `section_bitstream` をそのまま論理命令列として読む。
- まず CPU 実装でフォーマット妥当性を固定し、その後 GPU 解凍へ展開する。

## 11. 互換性ポリシー（v0開発フェーズ）
- v0 は **新形式専用** とする。
- 旧PDeflateチャンク形式の読み取り互換は v0 の必須要件に含めない。
- 旧形式データを入力した場合は、サイレントフォールバックせず明示エラーを返す。
- 互換読み取りが必要になった場合は、将来の別タスク（互換モード）で `version/flags` 分岐として導入する。
