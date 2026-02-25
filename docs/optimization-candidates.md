# Optimization Candidates (Latest)

Updated: 2026-02-25

This file is the current optimization candidate list. If a previous list exists elsewhere, this file should be treated as the overwritten latest version.

Legend: `~~strikethrough~~` = Completed, `Open` = not started.

## Priority 1

1. ~~Reduce or gate GPU output full CPU re-validation.~~
- Current behavior validates GPU chunks by CPU inflate+compare in non-`Speed` modes.
- Status: **Completed**
- Implemented: `COZIP_GPU_VALIDATION_MODE=always|sample|off`, `COZIP_GPU_DYNAMIC_SELF_CHECK`.
- Implemented: `bench.sh` defaults validation/self-check to off for perf runs (env override still possible).

2. Reuse dynamic-path temporary GPU buffers instead of per-chunk creation. (Open)
- Current behavior creates readback/staging buffers repeatedly in dynamic flow.
- Candidate: make these buffers persistent per slot and recycle them.

3. Reduce hard `Maintain::Wait` barriers in GPU dynamic pipeline. (Open)
- Current behavior has multiple phase waits (`freq`, `pack bits`, `payload`).
- Candidate: increase overlap (`Poll` + in-flight queue) and collapse synchronization points.

## Priority 2

4. Prevent CPU from over-consuming GPU-eligible pending tasks. (Open)
- Current CPU claim path can take generic `Pending` tasks aggressively.
- Candidate: keep a minimum GPU-eligible reserve unless GPU is clearly stalled.

5. Tune CPU worker count for GPU-enabled mode. (Open)
- Current policy is fixed (`available_parallelism - 1`).
- Candidate: mode-aware or adaptive worker count to reduce contention.

## Priority 3

6. Reduce host-side chunk copy cost. (Open)
- Current task build copies each chunk into `Vec<u8>`.
- Candidate: move to shared backing/slice-based representation where safe.

## Recently Completed (Stability)

1. ~~Fix intermittent GPU dynamic corruption (bitpack shift-edge case).~~
- Status: **Completed**
- Notes: fixed `bitpack` path edge condition to avoid implementation-dependent shift behavior; intermittent `length_mismatch` fallback warnings stopped in follow-up runs.

## Notes

- The delay after `[cozip][timing][scheduler]` is expected in `bench_1gb`: benchmark then measures decompression in the same iteration.
- `decode_descriptor_gpu` currently performs CPU decode path internally.
