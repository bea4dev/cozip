# CoZip（日本語）

Rustライブラリ & 圧縮解凍ソフトウェア群です。

- `cozip_deflate`: 独自フレーム形式（`CZDF`）。圧縮は CPU/GPU 補助、解凍は CPU 実装。
- `cozip`: `cozip_deflate` の CPU deflate/inflate を使う、ファイル/ディレクトリ圧縮向け ZIP ラッパー（オーケストレーター）。

英語版: [README.md](./README.md)

## ディレクトリ構成

```
cozip/
  src/
    cozip_deflate/
    cozip/
  bench.sh
  docs/
```

## ビルド

```bash
cargo check --workspace
cargo test --workspace
```

## `cozip_deflate` の基本利用

```rust
use cozip_deflate::{CoZipDeflate, HybridOptions};

let options = HybridOptions::default();
let cozip = CoZipDeflate::init(options)?;

let compressed = cozip.compress(input_bytes)?;
let decompressed = cozip.decompress_on_cpu(&compressed.bytes)?;
assert_eq!(decompressed.bytes, input_bytes);
# Ok::<(), cozip_deflate::CozipDeflateError>(())
```

主な公開API:

- `compress_hybrid(...)`
- `decompress_on_cpu(...)`
- `compress_stream(...)`
- `decompress_stream(...)`
- `CoZipDeflate::compress_file(...)`
- `CoZipDeflate::decompress_file(...)`
- `CoZipDeflate::compress_file_from_name(...)`
- `CoZipDeflate::decompress_file_from_name(...)`
- `CoZipDeflate::compress_file_async(...)`
- `CoZipDeflate::decompress_file_async(...)`
- `deflate_compress_cpu(...)`
- `deflate_decompress_on_cpu(...)`

巨大ファイル向けストリーミングAPI（全体をメモリに載せず処理）:

```rust
use cozip_deflate::{CoZipDeflate, HybridOptions, StreamOptions};
use std::fs::File;

let cozip = CoZipDeflate::init(HybridOptions::default())?;
let input = File::open("huge-input.bin")?;
let output = File::create("huge-output.czds")?;
let stats = cozip.compress_file(input, output, StreamOptions { frame_input_size: 64 * 1024 * 1024 })?;

let compressed = File::open("huge-output.czds")?;
let restored = File::create("restored.bin")?;
let _ = cozip.decompress_file(compressed, restored)?;
println!("frames={}", stats.frames);
# Ok::<(), cozip_deflate::CozipDeflateError>(())
```

## `cozip` の基本利用

```rust
use cozip::{CoZip, CoZipOptions, ZipOptions};

let cozip = CoZip::init(CoZipOptions::Zip {
    options: ZipOptions::default(),
});

// 単一ファイル（パス指定）
let _ = cozip.compress_file_from_name("input.txt", "single.zip")?;

// ディレクトリ（非同期API）
# async fn run() -> Result<(), cozip::CoZipError> {
let _ = cozip
    .compress_directory_async("assets/", "assets.zip")
    .await?;
# Ok(())
# }
# Ok::<(), cozip::CoZipError>(())
```

## ベンチマーク

リポジトリルートで実行:

```bash
./bench.sh --mode ratio --runs 5
```

注意:

- `speedup(cpu/hybrid)` は **圧縮** について表示します。
- 解凍の speedup は、解凍経路が CPU-only のため廃止（deprecated）しています。

## 補足ドキュメント

- [`docs/context-log.md`](./docs/context-log.md): 実装履歴・検証ログ
- [`docs/gpu-deflate-chunk-pipeline.md`](./docs/gpu-deflate-chunk-pipeline.md): GPU deflate パイプラインのメモ
