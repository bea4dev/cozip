# PDeflate GPU Sparse Direct Output Tasks

元設計:
- `docs/pdeflate-gpu-sparse-direct-output-design.md`

目的:
- integrated sparse path の 2 段 readback 依存を解消し、`sparse_lens_wait` を大幅削減する

## Phase 0 - Instrumentation

Goal:
- 実 payload 分布と overflow リスクを測る

Tasks:
- 各 chunk の `payload_len` 分布をログ/集計できるようにする
- `section_count`, `table_count`, `chunk_len`, `cmd cap` と `payload_len` の相関を記録する
- size class 設計用の percentile を取得する
- class 推定の under/over 量を可視化する

Done criteria:
- `bench_pdeflate` の run から class 境界設計に必要な統計が取れる

## Phase 1 - Class Estimator

Goal:
- GPU 実行前に conservative な size class を決定できるようにする

Tasks:
- host 側に `estimate_sparse_payload_class(...)` を追加する
- `DirectState` または同等構造に `payload_class` を保持する
- class table を env or constant で切替可能にする
- `speed` 用 conservative estimator を先行実装する

Done criteria:
- 推定 class と実 payload len の比較統計が取れる
- underestimation がない、または overflow flag で確実に検出できる

## Phase 2 - Classed Scratch Layout

Goal:
- class ごとの output slot 領域を scratch に持たせる

Tasks:
- `GpuSparsePackScratchCaps` に classed output capacity を追加する
- scratch の `out_buffer` レイアウトを class region ベースへ拡張する
- desc に `class_id`, `slot_base`, `slot_cap` を追加する
- payload readback 予定領域を submit 前に固定できるようにする

Done criteria:
- CPU が result readback 前に payload readback plan を構築できる

## Phase 3 - Shader Contract Update

Goal:
- sparse pack shader が classed slot に直接書けるようにする

Tasks:
- `pack_sparse_prepare` の入力 desc を拡張する
- `pack_sparse` が class slot に直接 payload を出力するように変更する
- overflow を meta/result へ返す
- overflow chunk が batch 全体を壊さないようにする

Done criteria:
- 正常 chunk は direct slot write で完結する
- overflow chunk が識別可能

## Phase 4 - Direct Output Readback

Goal:
- 「result wait 後に payload readback を組み立てる」段を廃止する

Tasks:
- payload readback buffer を submit 前に固定化する
- meta/result readback は小サイズのみに縮める
- payload readback を class region 単位で行う
- `result -> payload plan build -> second submit` を削除する

Done criteria:
- 2 段 readback が消える
- `sparse_lens_wait` が明確に減る

## Phase 5 - Overflow Slow Path

Goal:
- rare overflow を安全に処理する

Tasks:
- overflow chunk だけ旧 slow path または CPU fallback に流す
- batch 内の正常 chunk はそのまま返す
- overflow の件数と比率を統計化する

Done criteria:
- overflow があっても batch 全体の失敗にならない

## Phase 6 - Readback Ring

Goal:
- direct output 化後も残る wait を隠す

Tasks:
- classed payload readback buffer をリング化する
- ready-first 回収を導入する
- `device.poll(Maintain::Wait)` を backpressure 時のみ使う

Done criteria:
- `submit_done_wait` の run 間ぶれが縮小する

## Phase 7 - Cleanup and Tuning

Goal:
- VRAM と速度の両方を詰める

Tasks:
- class 境界の調整
- scratch pool size の再調整
- env flag を整理し、常用設定を決める
- 不要になった旧 readback 経路を整理する

Done criteria:
- `speed` モードの既定経路として成立する

## Measurement Gates

各 Phase の比較指標:
- `t_sparse_lens_wait_ms`
- `t_sparse_copy_ms`
- `payload_readback_kib`
- `gpu_call_ms_per_chunk`
- `writer_hol_wait_ms`
- `speedup_comp(cpu/hybrid)`

最低限守る条件:
- payload 破損なし
- VRAM が現状より悪化しない
- overflow が支配的にならない
