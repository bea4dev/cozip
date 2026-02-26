# CoZip

A set of Rust libraries and compression/decompression software tools.

- `cozip_deflate`: custom frame format (`CZDF`) with CPU/GPU-assisted **compression** and CPU **decompression**.
- `cozip_zip`: minimal ZIP single-entry helper built on top of `cozip_deflate` CPU deflate/inflate.

Japanese version: [README.ja.md](./README.ja.md)

## Workspace Layout

```
cozip/
  src/
    cozip_deflate/
    cozip_zip/
  bench.sh
  docs/
```

## Build

```bash
cargo check --workspace
cargo test --workspace
```

## `cozip_deflate` Quick Use

```rust
use cozip_deflate::{CoZipDeflate, HybridOptions};

let options = HybridOptions::default();
let cozip = CoZipDeflate::init(options)?;

let compressed = cozip.compress(input_bytes)?;
let decompressed = cozip.decompress_on_cpu(&compressed.bytes)?;
assert_eq!(decompressed.bytes, input_bytes);
# Ok::<(), cozip_deflate::CozipDeflateError>(())
```

Main public helpers:

- `compress_hybrid(...)`
- `decompress_on_cpu(...)`
- `deflate_compress_cpu(...)`
- `deflate_decompress_on_cpu(...)`

## `cozip_zip` Quick Use

```rust
use cozip_zip::{ZipOptions, zip_compress_single, zip_decompress_single};

let zip = zip_compress_single("hello.txt", b"hello", &ZipOptions::default())?;
let entry = zip_decompress_single(&zip)?;
assert_eq!(entry.name, "hello.txt");
assert_eq!(entry.data, b"hello");
# Ok::<(), cozip_zip::CozipZipError>(())
```

## Benchmark

Run process-restart benchmark from repository root:

```bash
./bench.sh --mode ratio --runs 5
```

Notes:

- `speedup(cpu/hybrid)` is reported for **compression**.
- Decompression speedup is intentionally omitted/deprecated because decompression is CPU-only now.

## Additional Docs

- [`docs/context-log.md`](./docs/context-log.md): implementation history and experiment notes.
- [`docs/gpu-deflate-chunk-pipeline.md`](./docs/gpu-deflate-chunk-pipeline.md): GPU deflate pipeline notes.
