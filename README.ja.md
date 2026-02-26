# CoZip（日本語）

Rustライブラリ & 圧縮解凍ソフトウェア群です。

- `cozip_deflate`: 独自フレーム形式（`CZDF`）。圧縮は CPU/GPU 補助、解凍は CPU 実装。
- `cozip_zip`: `cozip_deflate` の CPU deflate/inflate を使う最小 ZIP 単一エントリ実装。

英語版: [README.md](./README.md)

## ディレクトリ構成

```
cozip/
  src/
    cozip_deflate/
    cozip_zip/
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
- `deflate_compress_cpu(...)`
- `deflate_decompress_on_cpu(...)`

## `cozip_zip` の基本利用

```rust
use cozip_zip::{ZipOptions, zip_compress_single, zip_decompress_single};

let zip = zip_compress_single("hello.txt", b"hello", &ZipOptions::default())?;
let entry = zip_decompress_single(&zip)?;
assert_eq!(entry.name, "hello.txt");
assert_eq!(entry.data, b"hello");
# Ok::<(), cozip_zip::CozipZipError>(())
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
