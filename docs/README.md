# cozip docs

このディレクトリは、CPU + GPU(WebGPU)協調型の圧縮・解凍ライブラリ実装に向けた設計と引き継ぎメモを保持します。

## 目的

- 設計判断を明文化し、実装中の迷いを減らす
- セッションを跨いだコンテキスト引き継ぎを容易にする
- ZIP/Deflate互換性と並列性能のトレードオフを追跡する

## ドキュメント一覧

- `architecture.md`: 全体アーキテクチャと実装フェーズ
- `deflate-parallel-profile.md`: 並列実行前提のDeflateプロファイル仕様案
- `gpu-full-task-design.md`: GPUへ実圧縮/実解凍タスクを割り当てる詳細設計
- `gpu-deflate-chunk-pipeline.md`: 独立チャンクをCPU/GPUでDeflateし連結する実装設計
- `zip-gpu-compatibility.md`: ZIP互換を維持したGPU圧縮の制約と実装方針メモ
- `zip-compatible-hybrid-deflate-design.md`: CZDF実装を活かしつつZIP互換を満たすハイブリッド圧縮の実装設計
- `context-log.md`: 作業ログ・決定事項・未決事項

## 運用ルール

- 実装や設計変更を行ったら `context-log.md` に追記する
- 迷った判断は「採用案」「却下案」「理由」を短く残す
- 新規ドキュメントを増やしたらこの一覧に追加する
