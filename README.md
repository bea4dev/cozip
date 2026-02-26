# CoZip

A set of Rust libraries and compression/decompression software tools.

- `cozip_deflate`: custom frame format (`CZDF`) with CPU/GPU-assisted **compression** and CPU **decompression**.
- `cozip`: ZIP wrapper/orchestrator for file and directory compression APIs, built on top of `cozip_deflate` CPU deflate/inflate.

日本語: [README.ja.md](./README.ja.md)

## Workspace Layout

```
cozip/
  src/
    cozip_deflate/
    cozip/
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

Streaming API for large files (bounded memory, avoids reading full file into RAM):

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

## `cozip` Quick Use

```rust
use cozip::{CoZip, CoZipOptions, ZipOptions};

let cozip = CoZip::init(CoZipOptions::Zip {
    options: ZipOptions::default(),
});

// Single file (path-based)
let _ = cozip.compress_file_from_name("input.txt", "single.zip")?;

// Directory (async API)
# async fn run() -> Result<(), cozip::CoZipError> {
let _ = cozip
    .compress_directory_async("assets/", "assets.zip")
    .await?;
# Ok(())
# }
# Ok::<(), cozip::CoZipError>(())
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
