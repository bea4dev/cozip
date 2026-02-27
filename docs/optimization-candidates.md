# GPU Optimization Candidates (1.4x Target Plan)

Updated: 2026-02-27

このファイルは、`CPU+GPU` の圧縮速度を **安定して 1.4x 級**へ引き上げるための
高インパクト候補を優先順でまとめた実行計画です。

前提:
- `ratio` 探索量は下げない（速度・圧縮率悪化の既往あり）
- スケジューラーは `global-local` を基本とし、A/B 比較を維持

## Current Observations

- 改善が出る条件では `gpu_chunks` 増加と `avg_comp_ms` 低下が連動する。
- 失速条件では `write_stage_ms` と GPU readback 待ちが支配的になりやすい。
- つまり GPU カーネル単体より、**後段連結と readback パイプライン**が律速になる。

## Priority Work Items (High Impact)

### [A] GPU側で最終ストリーム連結まで実行（Writer軽量化の本命）

Goal:
- CPU writer の直列ビット連結処理を最小化し、`write_stage_ms` を大幅削減する。

Do:
- chunk 出力を GPU 側で「連結可能な bitstream 形式」に整形して返す。
- BFINAL 補正、非 byte align 時の連結調整、境界パッチ処理を GPU 側へ移す。
- CPU 側 writer は「連続バッファを順次 write するだけ」に近づける。

Expected gain:
- **+15%〜35%**（条件一致時は 1.4x 到達可能）

Success criteria:
- `write_stage_ms` / `write_pack_ms` の大幅低下
- 同等圧縮率で `avg_comp_ms` 改善
- `cpu_worker_parallelism` の維持または改善

Risk:
- ビット境界バグの混入リスクが高い。差分テストと roundtrip 検証を必須化する。

Status: **Candidate**

### [B] GPU readback パイプラインの再構成（永続リング + 非同期回収）

Goal:
- `map_async` / `poll(Wait)` 由来の停止時間を削減し、GPU稼働密度を上げる。

Do:
- readback バッファをリング化し、submit と回収を疎結合にする。
- pending を front 固定で待たず、ready 優先で回収を継続する。
- バッチ間で `Wait` を避け、一定水位を超えた時だけ backpressure をかける。

Expected gain:
- **+8%〜20%**

Success criteria:
- `t_freq_poll_wait_ms` / `t_pack_poll_wait_ms` 低下
- `gpu_worker_busy_ms` 密度向上
- `gpu_batch_avg_ms` 低下

Risk:
- 実装複雑度が高く、同期バグで不安定化しやすい。feature flag 前提で導入する。

Status: **Candidate**

### [C] Dynamic path の pass 境界をさらに融合

Goal:
- submit 回数と pass 切替 overhead を減らす。

Do:
- tokenize/finalize/freq/pack の間で不要な encoder 境界を削減。
- copy/compute/copy の再配置で GPU idle 区間を圧縮。

Expected gain:
- **+5%〜15%**

Success criteria:
- `t_freq_submit_ms` / `t_pack_submit_ms` 低下
- `t_total_ms` の再現性改善（run 間ぶれ縮小）

Risk:
- shader/バッファ依存が強く、段階導入が必要。

Status: **Candidate**

### [D] GPU取得戦略の動的化（終盤偏り抑制）

Goal:
- 終盤テール待ちを抑え、CPU/GPU の同時進行率を維持する。

Do:
- `gpu_batch_chunks` を固定値でなく残キュー量・進捗率で自動可変にする。
- tail 区間では GPU 取得サイズを段階縮小し、CPU 枯渇を抑える。

Expected gain:
- **+3%〜10%**

Success criteria:
- `writer_hol_wait_ms` 低下
- `cpu_wait_for_task_ms` / `gpu_wait_for_task_ms` バランス改善

Risk:
- 条件依存が強く、単独では 1.4x の決定打になりにくい。

Status: **Candidate**

## What Is Most Likely To Reach 1.4x

単体で 1.4x に届く可能性があるのは **[A]**。
現実的には **[A] + [B]** の組み合わせが最短ルート。

推奨実装順:
1. [A] GPU連結化（writer直列削減）
2. [B] readback再構成
3. [C] pass 融合の追い込み
4. [D] 仕上げの安定化

## Measurement Protocol (Fixed)

1. 1項目ずつ実装し、他条件は固定
2. 各条件 `runs=3` 以上で比較
3. 圧縮率悪化があれば不採用または flag 化
4. 主要指標を必ず記録

主要指標:
- `avg_comp_ms`
- `write_stage_ms`, `write_pack_ms`, `write_io_ms`
- `cpu_worker_parallelism`, `gpu_worker_parallelism`
- `gpu_chunks`, `gpu_batch_avg_ms`
- `writer_hol_wait_ms`, `cpu_wait_for_task_ms`, `gpu_wait_for_task_ms`

Suggested commands:

```bash
./bench.sh --mode ratio --runs 3 --size-mib 1024 --chunk-mib 10 --gpu-slots 6 --gpu-submit-chunks 3 --stream-pipeline-depth 3 --scheduler global-local --stream-batch-chunks 0 --stream-max-inflight-chunks 0 --stream-max-inflight-mib 0
```

```bash
./bench.sh --mode ratio --runs 3 --size-mib 4096 --chunk-mib 4 --gpu-slots 6 --gpu-submit-chunks 3 --stream-pipeline-depth 3 --scheduler global-local --stream-batch-chunks 0 --stream-max-inflight-chunks 0 --stream-max-inflight-mib 0
```

## Changelog

- 2026-02-27:
  - 旧「逐次候補リスト」を 1.4x 目標の高インパクト計画に上書き。
  - 候補を [A]〜[D] に再編。
  - 期待改善幅と優先順位を明記。
