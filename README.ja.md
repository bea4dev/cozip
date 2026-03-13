<div align="center">
<h2>CoZip</h2>
<p>GPUを使用する圧縮解凍ソフトウェア・ライブラリ</p>
</div>

<a href="https://discord.gg/F9DfEw6fqX" data-size="large">
  <img alt="Discord" src="https://img.shields.io/discord/1481785335519117316.svg?label=Discord&logo=Discord&colorB=7289da&style=for-the-badge">
</a>

- `cozip_deflate`: 独自フレーム形式（`CZDF`）。圧縮は CPU/GPU 補助、解凍は CPU 実装。
- `cozip_pdeflate`: PDeflate 本体。単一ファイル・ストリーム・並列 read/write API を持つ低レベル実装。
- `cozip`: `cozip_deflate` の CPU deflate/inflate を使う、ファイル/ディレクトリ圧縮向け ZIP ラッパー（オーケストレーター）。
- `cozip_desktop`: GPUI ベースのデスクトップアプリ。

英語版: [README.md](./README.md)

## ディレクトリ構成

```
cozip/
  src/
    cozip_deflate/
    cozip_pdeflate/
    cozip/
    cozip_desktop/
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

PDeflate backend も選べます。

```rust
use cozip::{CoZip, CoZipOptions, PDeflateOptions};

let cozip = CoZip::init(CoZipOptions::PDeflate {
    options: PDeflateOptions::default(),
});

let _ = cozip.compress_file_from_name("input.bin", "input.cozip")?;
# Ok::<(), cozip::CoZipError>(())
```

### `cozip` で追加・更新された重要 API

- `decompress_auto(...)`
- `decompress_auto_from_name(...)`
- `decompress_file_with_progress(...)`
- `decompress_directory_with_progress(...)`
- `decompress_file_with_progress_and_expected_output_bytes(...)`
- `decompress_file_from_name_with_progress_and_expected_output_bytes(...)`
- `inspect_archive_from_name(...)`
- `inspect_archive_decode_hint_from_name(...)`
- `CoZipProgress`
- `CoZipArchiveInfo`
- `ZipOptions { parallel_read_threads, parallel_write_threads, deflate_mode, ... }`
- `PDeflateOptions { parallel_read_threads, parallel_write_threads, gpu_* , ... }`

特に次の 3 つは実用上重要です。

- `decompress_auto*`
  - ZIP / PDeflate と単一ファイル / ディレクトリを自動判別します。
- `CoZipProgress`
  - GUI や CLI から進捗率・現在ファイル・スループットを取得できます。
- `inspect_archive_*`
  - アーカイブ形式・種別・並列解凍可否のヒントを事前に調べられます。

### `cozip_pdeflate` の主な API

- `compress_stream_with_options(...)`
- `decompress_stream_with_options(...)`
- `compress_file_with_options(...)`
- `compress_file_parallel_read_with_options(...)`
- `decompress_file_parallel_write_with_options(...)`
- `pdeflate_stream_suggested_name(...)`
- `pdeflate_stream_uncompressed_size(...)`

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
- [`docs/pdeflate-v0-spec.md`](./docs/pdeflate-v0-spec.md): 現行 PDeflate v0 フォーマットの唯一の仕様ソース
- [`docs/pdeflate-v0-baseline.md`](./docs/pdeflate-v0-baseline.md): 比較用ベンチコマンドと実装前ベースライン
