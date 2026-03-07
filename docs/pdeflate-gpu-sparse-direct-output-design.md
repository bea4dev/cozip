# PDeflate GPU Sparse Direct Output Design

Updated: 2026-03-07

この文書は、`legacy_pdeflate_cpu::gpu::compute_matches_encode_and_pack_sparse_batch` の
`sparse_lens_wait` を支配的ボトルネックから外すための大規模設計変更をまとめる。

対象:
- `src/cozip_pdeflate/src/legacy_pdeflate_cpu/gpu.rs`
- `src/cozip_pdeflate/src/legacy_pdeflate_cpu/mod.rs`
- `src/cozip_pdeflate/src/lib.rs`

前提:
- 現在の `speed` モードでは GPU table build は無効化済み
- いまの圧縮 GPU ボトルネックは主に `sparse_lens_wait` と `section_encode`
- 既存の「結果 readback -> payload readback」の 2 段 readback 構造が wait を増幅している

## Current Problem

現実装の integrated sparse path は、概ね次の流れになっている。

1. GPU が `result_buffer` に chunk ごとの最終 payload 長を書き出す
2. CPU が `result_readback_buffer` を `map_async + poll(Wait)` で待つ
3. CPU が payload 長に応じて `payload_readback_buffer` と copy plan をその場で組む
4. CPU が `out_buffer -> payload_readback_buffer` をもう一度 submit する
5. CPU が payload readback をもう一度 `map_async + poll(Wait)` で待つ

この構造の問題:

- readback が 2 段あり、`Wait` がクリティカルパスに 2 回入る
- payload 回収計画が length 判明後にしか作れない
- batch 単位で CPU が完全停止しやすい
- `sparse_lens_wait_ms` の run 間・batch 間ぶれが大きい

## Goal

`sparse_lens_wait` の原因である
「長さを読んでから payload 回収計画を作る」
という依存を壊す。

狙い:

- `result readback -> payload readback` の直列依存をなくす
- payload 本体の readback を 1 submit / 1 map に寄せる
- `device.poll(Maintain::Wait)` をクリティカルパスから可能な限り外す
- worst-case 固定長で VRAM を爆発させない

## Proposed Direction

全面 worst-case 固定スロットではなく、**サイズクラス別 direct output** を採用する。

### Core Idea

各 chunk の sparse payload 出力先を、GPU 実行前に CPU が決める。
ただし出力領域は完全固定長ではなく、サイズクラス単位で確保する。

例:
- class S: 256 KiB
- class M: 1 MiB
- class L: 4 MiB
- class XL: fallback or dedicated path

各 chunk は事前推定により class を持つ。
各 class ごとに batch 内 prefix sum で output offset を確定し、GPU はその offset に直接 payload を書く。

返すメタ:
- actual payload len
- class id
- class-local slot index
- optional overflow / fallback flag

こうすると CPU は「len を読んで copy plan を組む」必要がなくなる。
CPU が必要なのは最終的な payload の切り出しだけになる。

## Why Classed Slots Instead Of Worst-Case Slots

全面 worst-case 固定長の欠点:

- VRAM を大きく消費する
- `4 MiB * 32 chunk` のような条件ですぐ破綻する
- readback 量も悪化しやすい

サイズクラス方式の利点:

- 実 payload 分布に寄せて VRAM を抑えられる
- 回収計画を事前に固定化できる
- overflow chunk だけを別経路に逃がせる

## Data Flow (Target)

### Before

1. prepare desc/meta/index
2. sparse prepare pass
3. sparse pack pass
4. result readback wait
5. CPU builds payload copy plan
6. payload copy submit
7. payload readback wait
8. CPU materializes final payload

### After

1. CPU estimates class per chunk
2. CPU builds class-local output offsets and readback plan
3. prepare desc/meta/index + class metadata
4. sparse prepare pass
5. sparse pack pass writes directly into classed output slots
6. GPU submits only meta readback and payload readback plan that is already fixed
7. CPU reads small meta and payload body with no second planning step
8. overflow chunks only take slow path

## Required Design Changes

### 1. Per-chunk size class estimation

必要:
- `DirectState` またはその周辺に、payload 予測サイズを持たせる
- 予測は section count, command cap, table count, chunk len から行う

条件:
- underestimate は不可
- overestimate は許容
- `speed` モード優先で conservative に寄せる

### 2. Scratch layout changes

現状:
- `out_buffer` は batch 全体の単一出力領域
- payload readback は result readback 後に組み立てている

変更後:
- class ごとに slot-region を持つ
- desc に `slot_base_word`, `slot_capacity_words`, `class_id` を追加
- result/meta は slot 実使用量だけ返す

### 3. Sparse shader contract changes

必要:
- pack sparse shader が `out_base_word` だけでなく class slot 情報を扱えること
- overflow を meta に明示できること

想定:
- 正常 chunk: classed slot に直接書く
- overflow chunk: dedicated overflow buffer or CPU fallback

### 4. Readback model changes

理想:
- payload readback buffer は batch submit 前に固定済み
- result/meta readback は小さく保つ
- payload は class region ごとにまとめて readback

最低限:
- 「result を待ってから payload readback buffer を作る」段を消す

### 5. Fallback handling

必須:
- 推定を超えた chunk は overflow flag を立てる
- overflow chunk のみ slow path へ送る
- batch 全体を巻き戻さない

## Expected Impact

最も期待する改善:

- `t_sparse_lens_wait_ms` の大幅削減
- `t_wait_ms` の安定化
- batch 間ぶれの縮小

副次効果:

- `payload_readback_setup_ms` の削減
- `device.poll(Maintain::Wait)` 回数の削減
- writer HOL wait の間接的改善

## Risks

高リスク:

- slot size underestimate による payload 破損
- overflow 処理の複雑化
- shader/host 間の desc 契約変更で壊れやすい

中リスク:

- class 設計が粗すぎると VRAM を再び圧迫する
- class 設計が細かすぎると prepare が重くなる

低リスク:

- instrumentation 追加
- class 推定統計の収集

## Recommended Rollout

段階導入が必要。

1. instrumentation 追加
2. class 推定のみ導入して実分布を測る
3. classed slot scratch を hidden flag で実装
4. overflow slow path 付き direct output に切替
5. readback ring 化で wait をさらに隠す

## Success Criteria

最低成功条件:

- `t_sparse_lens_wait_ms` を現状比で 40% 以上削減
- VRAM 使用量を現状以下に維持
- roundtrip / verify で破損なし

理想:

- `t_sparse_lens_wait_ms` を 70% 以上削減
- `hybrid comp` の run 間ぶれ縮小
- `speedup_comp(cpu/hybrid)` が安定して 1.0x 超
