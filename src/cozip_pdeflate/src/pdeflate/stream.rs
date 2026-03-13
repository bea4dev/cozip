use super::*;
use std::fs::File as StdFile;
use cozip_util::{ParallelFileWriter, ParallelFileWriterOptions};

#[derive(Debug)]
struct StreamCompressTask {
    index: usize,
    chunk: Vec<u8>,
}

#[derive(Debug, Default)]
struct StreamCompressState {
    queue: VecDeque<StreamCompressTask>,
    cpu_queue: VecDeque<StreamCompressTask>,
    gpu_queue: VecDeque<StreamCompressTask>,
    buffered_bytes: usize,
    produced_count: usize,
    producer_eof: bool,
    stopped: bool,
    results: Vec<Option<(ChunkCompressed, usize)>>,
}

#[derive(Debug)]
struct StreamDecodeTask {
    index: usize,
    output_offset: usize,
    payload: Vec<u8>,
    table_count: usize,
    chunk_uncompressed_len: usize,
    section_meta: Vec<gpu::GpuDecodeSectionMeta>,
}

#[derive(Debug, Default)]
struct StreamDecodeState {
    queue: VecDeque<StreamDecodeTask>,
    cpu_queue: VecDeque<StreamDecodeTask>,
    gpu_queue: VecDeque<StreamDecodeTask>,
    buffered_result_bytes: usize,
    next_write_index: usize,
    produced_count: usize,
    producer_eof: bool,
    stopped: bool,
    input_bytes: u64,
    results: Vec<Option<(Vec<u8>, ChunkDecoded, bool)>>,
}

#[derive(Debug, Default)]
struct ParallelDecodeMetrics {
    chunk_count: usize,
    table_entries_total: usize,
    section_count_total: usize,
}

fn decode_result_limit_bytes(
    _options: &PDeflateOptions,
    _cpu_workers: usize,
    _gpu_batch_limit: usize,
) -> usize {
    2 * 1024 * 1024 * 1024
}

pub(crate) fn compress_reader_with_stats<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &PDeflateOptions,
    uncompressed_size_hint: Option<u64>,
    file_name_hint: Option<&str>,
    target_buffered_bytes: usize,
    low_watermark_bytes: usize,
) -> Result<PDeflateStats, PDeflateError> {
    validate_options(options)?;

    let chunk_size = options.chunk_size.max(1);
    let target_buffered_bytes = target_buffered_bytes.max(chunk_size);
    let low_watermark_bytes = low_watermark_bytes.max(chunk_size).min(target_buffered_bytes);

    let mut input_bytes = 0u64;
    let header = encode_stream_header(options.chunk_size, uncompressed_size_hint, file_name_hint)?;
    let mut output_bytes =
        u64::try_from(header.len()).map_err(|_| PDeflateError::NumericOverflow)?;
    let mut chunk_count = 0usize;
    let mut table_entries_total = 0usize;
    let mut section_count_total = 0usize;

    let state = Arc::new((Mutex::new(StreamCompressState::default()), Condvar::new()));
    let err_slot = Arc::new(Mutex::new(None::<PDeflateError>));
    let err_flag = Arc::new(AtomicBool::new(false));

    writer.write_all(&header)?;

    thread::scope(|scope| -> Result<(), PDeflateError> {
        let state_ref = Arc::clone(&state);
        let err_slot_ref = Arc::clone(&err_slot);
        let err_flag_ref = Arc::clone(&err_flag);
        scope.spawn(move || {
            let mut next_index = 0usize;
            loop {
                let mut buffer = vec![0u8; chunk_size];
                let mut filled = 0usize;
                while filled < chunk_size {
                    match reader.read(&mut buffer[filled..]) {
                        Ok(0) => break,
                        Ok(read) => filled = filled.saturating_add(read),
                        Err(err) => {
                            set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                            let (lock, cv) = &*state_ref;
                            if let Ok(mut state) = lock.lock() {
                                state.stopped = true;
                                cv.notify_all();
                            }
                            return;
                        }
                    }
                }

                let (lock, cv) = &*state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                while state.buffered_bytes >= target_buffered_bytes
                    && !state.stopped
                    && !err_flag_ref.load(Ordering::Relaxed)
                {
                    state = match cv.wait(state) {
                        Ok(guard) => guard,
                        Err(_) => return,
                    };
                }
                if state.stopped || err_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                if filled == 0 {
                    state.producer_eof = true;
                    cv.notify_all();
                    return;
                }
                buffer.truncate(filled);
                let task = StreamCompressTask {
                    index: next_index,
                    chunk: buffer,
                };
                next_index = next_index.saturating_add(1);
                state.buffered_bytes = state.buffered_bytes.saturating_add(task.chunk.len());
                state.produced_count = state.produced_count.saturating_add(1);
                state.results.push(None);
                let gpu_eligible = options.gpu_compress_enabled
                    && gpu::is_runtime_available()
                    && task.chunk.len() >= options.gpu_min_chunk_size;
                match options.hybrid_scheduler_policy {
                    PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.push_back(task),
                    PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                        if gpu_eligible {
                            state.gpu_queue.push_back(task);
                        } else {
                            state.cpu_queue.push_back(task);
                        }
                    }
                }
                cv.notify_all();
            }
        });

        let gpu_enabled = options.gpu_compress_enabled && gpu::is_runtime_available();
        let cpu_workers = cpu_worker_count().max(1);
        let gpu_workers = if gpu_enabled {
            options.gpu_workers.max(1)
        } else {
            0
        };
        let gpu_batch_limit = options
            .gpu_slot_count
            .max(options.gpu_submit_chunks)
            .max(1);

        for _ in 0..cpu_workers {
            let state_ref = Arc::clone(&state);
            let err_slot_ref = Arc::clone(&err_slot);
            let err_flag_ref = Arc::clone(&err_flag);
            scope.spawn(move || loop {
                if err_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                let task = pop_stream_compress_task(
                    &state_ref,
                    &err_slot_ref,
                    &err_flag_ref,
                    options,
                    false,
                    low_watermark_bytes,
                );
                let Some(task) = task else {
                    return;
                };
                let encoded = match compress_chunk_cpu_only(&task.chunk, options) {
                    Ok(encoded) => encoded,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                };
                let (lock, cv) = &*state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                if let Some(slot) = state.results.get_mut(task.index) {
                    *slot = Some((encoded, task.chunk.len()));
                }
                cv.notify_all();
            });
        }

        for _ in 0..gpu_workers {
            let state_ref = Arc::clone(&state);
            let err_slot_ref = Arc::clone(&err_slot);
            let err_flag_ref = Arc::clone(&err_flag);
            scope.spawn(move || loop {
                if err_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                let tasks = pop_stream_compress_gpu_batch(
                    &state_ref,
                    &err_slot_ref,
                    &err_flag_ref,
                    options,
                    gpu_batch_limit,
                    low_watermark_bytes,
                );
                let Some(tasks) = tasks else {
                    append_gpu_compress_trace("[gpu_worker] exit_no_tasks");
                    return;
                };
                let claimed_count = tasks.len();
                let first_index = tasks.first().map(|task| task.index).unwrap_or(0);
                let last_index = tasks.last().map(|task| task.index).unwrap_or(0);
                append_gpu_compress_trace(&format!(
                    "[gpu_worker] batch_claimed count={} first_index={} last_index={}",
                    claimed_count,
                    first_index,
                    last_index
                ));
                let batch = match compress_stream_gpu_batch(&tasks, options) {
                    Ok(batch) => batch,
                    Err(err) => {
                        append_gpu_compress_trace(&format!(
                            "[gpu_worker] batch_error count={} first_index={} last_index={} err={}",
                            tasks.len(),
                            first_index,
                            last_index,
                            err
                        ));
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                };
                append_gpu_compress_trace(&format!(
                    "[gpu_worker] batch_returned count={} first_index={} last_index={}",
                    batch.len(),
                    first_index,
                    last_index
                ));
                let (lock, cv) = &*state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                for (task, encoded) in tasks.into_iter().zip(batch.into_iter()) {
                    if let Some(slot) = state.results.get_mut(task.index) {
                        *slot = Some((encoded, task.chunk.len()));
                    }
                }
                append_gpu_compress_trace(&format!(
                    "[gpu_worker] batch_published count={} first_index={} last_index={}",
                    claimed_count,
                    first_index,
                    last_index
                ));
                cv.notify_all();
            });
        }

        let mut pending_encoded: Option<ChunkCompressed> = None;
        let mut next_index = 0usize;
        loop {
            let ready = {
                let (lock, cv) = &*state;
                let mut state = lock.lock().map_err(|_| {
                    PDeflateError::InvalidStream("stream compression state poisoned")
                })?;
                loop {
                    if err_flag.load(Ordering::Relaxed) {
                        state.stopped = true;
                        cv.notify_all();
                        break None;
                    }
                    if next_index < state.results.len() {
                        if let Some((encoded, input_len)) = state.results[next_index].take() {
                            let is_final =
                                state.producer_eof && next_index + 1 == state.produced_count;
                            break Some((encoded, input_len, is_final));
                        }
                    }
                    if state.producer_eof && next_index == state.produced_count {
                        state.stopped = true;
                        cv.notify_all();
                        break None;
                    }
                    state = cv.wait(state).map_err(|_| {
                        PDeflateError::InvalidStream("stream compression state poisoned")
                    })?;
                }
            };
            if let Err(err) = take_worker_error(&err_slot) {
                return Err(err);
            }
            let Some((mut encoded, encoded_input_len, is_final)) = ready else {
                break;
            };

            input_bytes = input_bytes
                .checked_add(
                    u64::try_from(encoded_input_len).map_err(|_| PDeflateError::NumericOverflow)?,
                )
                .ok_or(PDeflateError::NumericOverflow)?;
            chunk_count = chunk_count.saturating_add(1);
            table_entries_total = table_entries_total.saturating_add(encoded.table_entries);
            section_count_total = section_count_total.saturating_add(encoded.section_count);

            if let Some(prev) = pending_encoded.take() {
                write_stream_chunk_frame(writer, &prev.payload)?;
                output_bytes = output_bytes
                    .checked_add(4)
                    .and_then(|n| {
                        n.checked_add(
                            u64::try_from(prev.payload.len())
                                .map_err(|_| PDeflateError::NumericOverflow)
                                .ok()?,
                        )
                    })
                    .ok_or(PDeflateError::NumericOverflow)?;
            }
            next_index = next_index.saturating_add(1);
            if is_final {
                set_chunk_final_stream_flag(&mut encoded.payload, true)?;
                write_stream_chunk_frame(writer, &encoded.payload)?;
                output_bytes = output_bytes
                    .checked_add(4)
                    .and_then(|n| {
                        n.checked_add(
                            u64::try_from(encoded.payload.len())
                                .map_err(|_| PDeflateError::NumericOverflow)
                                .ok()?,
                        )
                    })
                    .ok_or(PDeflateError::NumericOverflow)?;
            } else {
                pending_encoded = Some(encoded);
            }
        }

        take_worker_error(&err_slot)?;
        Ok(())
    })?;

    Ok(PDeflateStats {
        input_bytes,
        output_bytes,
        chunk_count,
        table_entries_total,
        section_count_total,
    })
}

pub(crate) fn decompress_reader_with_stats<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &PDeflateOptions,
    decode_backlog_reporter: Option<crate::DecodeBacklogReporter>,
    output_write_reporter: Option<crate::OutputWriteReporter>,
) -> Result<PDeflateStats, PDeflateError> {
    validate_options(options)?;
    let header = read_stream_header(reader)?;

    let state = Arc::new((Mutex::new(StreamDecodeState {
        input_bytes: u64::try_from(header.encoded_len)
            .map_err(|_| PDeflateError::NumericOverflow)?,
        ..StreamDecodeState::default()
    }), Condvar::new()));
    report_decode_backlog(&decode_backlog_reporter, 0);
    let err_slot = Arc::new(Mutex::new(None::<PDeflateError>));
    let err_flag = Arc::new(AtomicBool::new(false));

    thread::scope(|scope| -> Result<PDeflateStats, PDeflateError> {
        let state_ref = Arc::clone(&state);
        let err_slot_ref = Arc::clone(&err_slot);
        let err_flag_ref = Arc::clone(&err_flag);
        scope.spawn(move || {
            let mut chunk_index = 0usize;
            let mut output_offset = 0usize;
            loop {
                let mut len_buf = [0u8; 4];
                match reader.read_exact(&mut len_buf) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            if chunk_index == 0 {
                                state.producer_eof = true;
                            } else {
                                set_worker_error_once(
                                    &err_slot_ref,
                                    &err_flag_ref,
                                    PDeflateError::InvalidStream("missing final chunk"),
                                );
                                state.stopped = true;
                            }
                            cv.notify_all();
                        }
                        return;
                    }
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                }
                let mut len_cursor = 0usize;
                let payload_len = match read_u32_le(&len_buf, &mut len_cursor) {
                    Ok(v) => v as usize,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        return;
                    }
                };
                let mut payload = vec![0u8; payload_len];
                if let Err(err) = reader.read_exact(&mut payload) {
                    set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                    let (lock, cv) = &*state_ref;
                    if let Ok(mut state) = lock.lock() {
                        state.stopped = true;
                        cv.notify_all();
                    }
                    return;
                }
                let preprocess = match preprocess_chunk_for_gpu_decode(&payload) {
                    Ok(v) => v,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                };
                let final_chunk = match read_chunk_flags(&payload) {
                    Ok(flags) => chunk_final_stream_enabled(flags),
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        return;
                    }
                };

                let task = StreamDecodeTask {
                    index: chunk_index,
                    output_offset,
                    payload,
                    table_count: preprocess.table_count,
                    chunk_uncompressed_len: preprocess.chunk_uncompressed_len,
                    section_meta: preprocess.section_meta,
                };
                output_offset = output_offset.saturating_add(task.chunk_uncompressed_len);
                chunk_index = chunk_index.saturating_add(1);

                let (lock, cv) = &*state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                state.input_bytes = state
                    .input_bytes
                    .saturating_add(4)
                    .saturating_add(u64::try_from(task.payload.len()).unwrap_or(u64::MAX));
                state.produced_count = state.produced_count.saturating_add(1);
                state.results.push(None);
                let gpu_eligible = (options.gpu_decompress_enabled || options.gpu_decompress_force_gpu)
                    && gpu::is_runtime_available()
                    && task.chunk_uncompressed_len >= options.gpu_min_chunk_size;
                match options.hybrid_scheduler_policy {
                    PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.push_back(task),
                    PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                        if gpu_eligible {
                            state.gpu_queue.push_back(task);
                        } else {
                            state.cpu_queue.push_back(task);
                        }
                    }
                }
                if final_chunk {
                    let mut trailing = [0u8; 1];
                    match reader.read(&mut trailing) {
                        Ok(0) => {
                            state.producer_eof = true;
                            cv.notify_all();
                            return;
                        }
                        Ok(_) => {
                            set_worker_error_once(
                                &err_slot_ref,
                                &err_flag_ref,
                                PDeflateError::InvalidStream("trailing bytes in stream"),
                            );
                            state.stopped = true;
                            cv.notify_all();
                            return;
                        }
                        Err(err) => {
                            set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                            state.stopped = true;
                            cv.notify_all();
                            return;
                        }
                    }
                }
                cv.notify_all();
            }
        });

        let gpu_enabled = (options.gpu_decompress_enabled || options.gpu_decompress_force_gpu)
            && gpu::is_runtime_available();
        let cpu_workers = if options.gpu_decompress_force_gpu {
            0
        } else {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .max(1)
        };
        let gpu_batch_limit = gpu_decode_batch_limit(options, usize::MAX / 2);
        let result_limit_bytes =
            decode_result_limit_bytes(options, cpu_workers, gpu_batch_limit);

        for _ in 0..cpu_workers {
            let state_ref = Arc::clone(&state);
            let err_slot_ref = Arc::clone(&err_slot);
            let err_flag_ref = Arc::clone(&err_flag);
            let decode_backlog_reporter = decode_backlog_reporter.clone();
            scope.spawn(move || loop {
                if err_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                let task = pop_stream_decode_task(
                    &state_ref,
                    &err_slot_ref,
                    &err_flag_ref,
                    options,
                    false,
                    result_limit_bytes,
                );
                let Some(task) = task else {
                    return;
                };
                let mut restored = vec![0u8; task.chunk_uncompressed_len];
                let decoded = match decompress_chunk_into(&task.payload, &mut restored) {
                    Ok(decoded) => decoded,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                };
                let (lock, cv) = &*state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                state.buffered_result_bytes = state
                    .buffered_result_bytes
                    .saturating_add(restored.len());
                report_decode_backlog(&decode_backlog_reporter, state.buffered_result_bytes);
                if let Some(slot) = state.results.get_mut(task.index) {
                    *slot = Some((restored, decoded, false));
                }
                cv.notify_all();
            });
        }

        if gpu_enabled {
            let state_ref = Arc::clone(&state);
            let err_slot_ref = Arc::clone(&err_slot);
            let err_flag_ref = Arc::clone(&err_flag);
            let decode_backlog_reporter = decode_backlog_reporter.clone();
            scope.spawn(move || loop {
                if err_flag_ref.load(Ordering::Relaxed) {
                    return;
                }
                let tasks = pop_stream_decode_gpu_batch(
                    &state_ref,
                    &err_slot_ref,
                    &err_flag_ref,
                    options,
                    gpu_batch_limit,
                    result_limit_bytes,
                );
                let Some(tasks) = tasks else {
                    return;
                };
                let jobs: Vec<_> = tasks
                    .iter()
                    .map(|task| gpu::GpuDecodeJob {
                        chunk_index: task.index,
                        payload: task.payload.as_slice(),
                        table_count: task.table_count,
                        chunk_uncompressed_len: task.chunk_uncompressed_len,
                        out_offset: 0,
                        out_len: task.chunk_uncompressed_len,
                        section_meta: task.section_meta.clone(),
                        preferred_slot: None,
                    })
                    .collect();
                let gpu_results = match gpu::decode_chunks_gpu_v2(&jobs, options) {
                    Ok(v) => v,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        let (lock, cv) = &*state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                };
                let mut completed = Vec::with_capacity(tasks.len());
                for (task, result) in tasks.into_iter().zip(gpu_results.into_iter()) {
                    match result.disposition {
                        gpu::GpuDecodeDisposition::SubmittedGpu => {
                            let Some(decoded_chunk) = result.decoded_chunk else {
                                set_worker_error_once(
                                    &err_slot_ref,
                                    &err_flag_ref,
                                    PDeflateError::Gpu(
                                        "gpu decode result missing payload".to_string(),
                                    ),
                                );
                                return;
                            };
                            completed.push((
                                task.index,
                                decoded_chunk,
                                ChunkDecoded {
                                    table_entries: task.table_count,
                                    section_count: task.section_meta.len(),
                                    profile: ChunkDecodeProfile::default(),
                                },
                                true,
                            ));
                        }
                        gpu::GpuDecodeDisposition::CpuFallback => {
                            let mut restored = vec![0u8; task.chunk_uncompressed_len];
                            let decoded =
                                match decompress_chunk_into(&task.payload, &mut restored) {
                                    Ok(decoded) => decoded,
                                    Err(err) => {
                                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                                        return;
                                    }
                                };
                            completed.push((task.index, restored, decoded, false));
                        }
                    }
                }
                let (lock, cv) = &*state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                for (index, restored, decoded, from_gpu) in completed {
                    state.buffered_result_bytes = state
                        .buffered_result_bytes
                        .saturating_add(restored.len());
                    report_decode_backlog(&decode_backlog_reporter, state.buffered_result_bytes);
                    if let Some(slot) = state.results.get_mut(index) {
                        *slot = Some((restored, decoded, from_gpu));
                    }
                }
                cv.notify_all();
            });
        }

        let mut output_bytes = 0u64;
        let mut chunk_count = 0usize;
        let mut table_entries_total = 0usize;
        let mut section_count_total = 0usize;
        let mut next_index = 0usize;
        loop {
            let ready = {
                let (lock, cv) = &*state;
                let mut state = lock
                    .lock()
                    .map_err(|_| PDeflateError::InvalidStream("stream decode state poisoned"))?;
                loop {
                    if err_flag.load(Ordering::Relaxed) {
                        state.stopped = true;
                        cv.notify_all();
                        break None;
                    }
                    if next_index < state.results.len() {
                        if let Some(item) = state.results[next_index].take() {
                            state.buffered_result_bytes =
                                state.buffered_result_bytes.saturating_sub(item.0.len());
                            report_decode_backlog(
                                &decode_backlog_reporter,
                                state.buffered_result_bytes,
                            );
                            state.next_write_index = next_index.saturating_add(1);
                            cv.notify_all();
                            break Some(item);
                        }
                    }
                    if state.producer_eof && next_index == state.produced_count {
                        state.stopped = true;
                        cv.notify_all();
                        break None;
                    }
                    state = cv.wait(state).map_err(|_| {
                        PDeflateError::InvalidStream("stream decode state poisoned")
                    })?;
                }
            };
            if let Err(err) = take_worker_error(&err_slot) {
                return Err(err);
            }
            let Some((restored, decoded, _from_gpu)) = ready else {
                break;
            };
            writer.write_all(&restored)?;
            report_output_write(&output_write_reporter, restored.len());
            next_index = next_index.saturating_add(1);
            chunk_count = chunk_count.saturating_add(1);
            output_bytes = output_bytes
                .checked_add(
                    u64::try_from(restored.len()).map_err(|_| PDeflateError::NumericOverflow)?,
                )
                .ok_or(PDeflateError::NumericOverflow)?;
            table_entries_total = table_entries_total.saturating_add(decoded.table_entries);
            section_count_total = section_count_total.saturating_add(decoded.section_count);
        }

        let (lock, _) = &*state;
        let state = lock
            .lock()
            .map_err(|_| PDeflateError::InvalidStream("stream decode state poisoned"))?;
        take_worker_error(&err_slot)?;
        report_decode_backlog(&decode_backlog_reporter, 0);
        Ok(PDeflateStats {
            input_bytes: state.input_bytes,
            output_bytes,
            chunk_count,
            table_entries_total,
            section_count_total,
        })
    })
}

pub(crate) fn decompress_file_parallel_write_with_stats(
    input_file: StdFile,
    output_file: StdFile,
    options: &PDeflateOptions,
    decode_backlog_reporter: Option<crate::DecodeBacklogReporter>,
    output_write_reporter: Option<crate::OutputWriteReporter>,
) -> Result<PDeflateStats, PDeflateError> {
    validate_options(options)?;

    let mut reader = std::io::BufReader::new(input_file);
    let header = read_stream_header(&mut reader)?;
    let total_output_bytes = header.uncompressed_size.ok_or(PDeflateError::InvalidStream(
        "parallel file write requires stream uncompressed size metadata",
    ))?;
    output_file.set_len(total_output_bytes)?;

    let decode_state = Arc::new((
        Mutex::new(StreamDecodeState {
            input_bytes: u64::try_from(header.encoded_len)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            ..StreamDecodeState::default()
        }),
        Condvar::new(),
    ));
    let gpu_enabled = (options.gpu_decompress_enabled || options.gpu_decompress_force_gpu)
        && gpu::is_runtime_available();
    let cpu_workers = if options.gpu_decompress_force_gpu {
        0
    } else {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    };
    let gpu_batch_limit = gpu_decode_batch_limit(options, usize::MAX / 2);
    let result_limit_bytes = decode_result_limit_bytes(options, cpu_workers, gpu_batch_limit);
    let writer = Arc::new(
        ParallelFileWriter::new(
            output_file,
            ParallelFileWriterOptions {
                worker_threads: options.parallel_write_threads,
                max_backlog_bytes: result_limit_bytes,
                backlog_reporter: decode_backlog_reporter.clone(),
                write_reporter: output_write_reporter.clone(),
            },
        )
        .map_err(map_parallel_file_writer_error)?,
    );
    let metrics = Arc::new(Mutex::new(ParallelDecodeMetrics::default()));
    let err_slot = Arc::new(Mutex::new(None::<PDeflateError>));
    let err_flag = Arc::new(AtomicBool::new(false));
    report_decode_backlog(&decode_backlog_reporter, 0);

    thread::scope(|scope| -> Result<(), PDeflateError> {
        let decode_state_ref = Arc::clone(&decode_state);
        let err_slot_ref = Arc::clone(&err_slot);
        let err_flag_ref = Arc::clone(&err_flag);
        scope.spawn(move || {
            let mut chunk_index = 0usize;
            let mut output_offset = 0usize;
            loop {
                let mut len_buf = [0u8; 4];
                match reader.read_exact(&mut len_buf) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                        let (lock, cv) = &*decode_state_ref;
                        if let Ok(mut state) = lock.lock() {
                            if chunk_index == 0 {
                                state.producer_eof = true;
                            } else {
                                set_worker_error_once(
                                    &err_slot_ref,
                                    &err_flag_ref,
                                    PDeflateError::InvalidStream("missing final chunk"),
                                );
                                state.stopped = true;
                            }
                            cv.notify_all();
                        }
                        return;
                    }
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                        let (lock, cv) = &*decode_state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                }
                let mut len_cursor = 0usize;
                let payload_len = match read_u32_le(&len_buf, &mut len_cursor) {
                    Ok(v) => v as usize,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        return;
                    }
                };
                let mut payload = vec![0u8; payload_len];
                if let Err(err) = reader.read_exact(&mut payload) {
                    set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                    let (lock, cv) = &*decode_state_ref;
                    if let Ok(mut state) = lock.lock() {
                        state.stopped = true;
                        cv.notify_all();
                    }
                    return;
                }
                let preprocess = match preprocess_chunk_for_gpu_decode(&payload) {
                    Ok(v) => v,
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        let (lock, cv) = &*decode_state_ref;
                        if let Ok(mut state) = lock.lock() {
                            state.stopped = true;
                            cv.notify_all();
                        }
                        return;
                    }
                };
                let final_chunk = match read_chunk_flags(&payload) {
                    Ok(flags) => chunk_final_stream_enabled(flags),
                    Err(err) => {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        return;
                    }
                };
                let task = StreamDecodeTask {
                    index: chunk_index,
                    output_offset,
                    payload,
                    table_count: preprocess.table_count,
                    chunk_uncompressed_len: preprocess.chunk_uncompressed_len,
                    section_meta: preprocess.section_meta,
                };
                output_offset = output_offset.saturating_add(task.chunk_uncompressed_len);
                chunk_index = chunk_index.saturating_add(1);

                let (lock, cv) = &*decode_state_ref;
                let mut state = match lock.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                state.input_bytes = state
                    .input_bytes
                    .saturating_add(4)
                    .saturating_add(u64::try_from(task.payload.len()).unwrap_or(u64::MAX));
                state.produced_count = state.produced_count.saturating_add(1);
                let gpu_eligible = (options.gpu_decompress_enabled || options.gpu_decompress_force_gpu)
                    && gpu::is_runtime_available()
                    && task.chunk_uncompressed_len >= options.gpu_min_chunk_size;
                match options.hybrid_scheduler_policy {
                    PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.push_back(task),
                    PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                        if gpu_eligible {
                            state.gpu_queue.push_back(task);
                        } else {
                            state.cpu_queue.push_back(task);
                        }
                    }
                }
                if final_chunk {
                    let mut trailing = [0u8; 1];
                    match reader.read(&mut trailing) {
                        Ok(0) => {
                            state.producer_eof = true;
                            cv.notify_all();
                            return;
                        }
                        Ok(_) => {
                            set_worker_error_once(
                                &err_slot_ref,
                                &err_flag_ref,
                                PDeflateError::InvalidStream("trailing bytes in stream"),
                            );
                            state.stopped = true;
                            cv.notify_all();
                            return;
                        }
                        Err(err) => {
                            set_worker_error_once(&err_slot_ref, &err_flag_ref, err.into());
                            state.stopped = true;
                            cv.notify_all();
                            return;
                        }
                    }
                }
                cv.notify_all();
            }
        });

        for _ in 0..cpu_workers {
            let decode_state_ref = Arc::clone(&decode_state);
            let writer_ref = Arc::clone(&writer);
            let metrics_ref = Arc::clone(&metrics);
            let err_slot_ref = Arc::clone(&err_slot);
            let err_flag_ref = Arc::clone(&err_flag);
            scope.spawn(move || {
                loop {
                    if err_flag_ref.load(Ordering::Relaxed) {
                        return;
                    }
                    let task = pop_parallel_decode_task(
                        &decode_state_ref,
                        &err_slot_ref,
                        &err_flag_ref,
                        options,
                        false,
                    );
                    let Some(task) = task else {
                        return;
                    };
                    let mut restored = vec![0u8; task.chunk_uncompressed_len];
                    let decoded = match decompress_chunk_into(&task.payload, &mut restored) {
                        Ok(decoded) => decoded,
                        Err(err) => {
                            set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                            stop_decode_state(&decode_state_ref);
                            return;
                        }
                    };
                    {
                        let mut stats = match metrics_ref.lock() {
                            Ok(guard) => guard,
                            Err(_) => {
                                stop_decode_state(&decode_state_ref);
                                return;
                            }
                        };
                        stats.chunk_count = stats.chunk_count.saturating_add(1);
                        stats.table_entries_total =
                            stats.table_entries_total.saturating_add(decoded.table_entries);
                        stats.section_count_total =
                            stats.section_count_total.saturating_add(decoded.section_count);
                    }
                    if let Err(err) = writer_ref
                        .submit(u64::try_from(task.output_offset).unwrap_or(u64::MAX), restored)
                        .map_err(map_parallel_file_writer_error)
                    {
                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                        stop_decode_state(&decode_state_ref);
                        return;
                    }
                }
            });
        }

        if gpu_enabled {
            let decode_state_ref = Arc::clone(&decode_state);
            let writer_ref = Arc::clone(&writer);
            let metrics_ref = Arc::clone(&metrics);
            let err_slot_ref = Arc::clone(&err_slot);
            let err_flag_ref = Arc::clone(&err_flag);
            scope.spawn(move || {
                loop {
                    if err_flag_ref.load(Ordering::Relaxed) {
                        return;
                    }
                    let tasks = pop_parallel_decode_gpu_batch(
                        &decode_state_ref,
                        &err_slot_ref,
                        &err_flag_ref,
                        options,
                        gpu_batch_limit,
                    );
                    let Some(tasks) = tasks else {
                        return;
                    };
                    let jobs: Vec<_> = tasks
                        .iter()
                        .map(|task| gpu::GpuDecodeJob {
                            chunk_index: task.index,
                            payload: task.payload.as_slice(),
                            table_count: task.table_count,
                            chunk_uncompressed_len: task.chunk_uncompressed_len,
                            out_offset: 0,
                            out_len: task.chunk_uncompressed_len,
                            section_meta: task.section_meta.clone(),
                            preferred_slot: None,
                        })
                        .collect();
                    let gpu_results = match gpu::decode_chunks_gpu_v2(&jobs, options) {
                        Ok(v) => v,
                        Err(err) => {
                            set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                            stop_decode_state(&decode_state_ref);
                            return;
                        }
                    };
                    for (task, result) in tasks.into_iter().zip(gpu_results.into_iter()) {
                        let (restored, decoded) = match result.disposition {
                            gpu::GpuDecodeDisposition::SubmittedGpu => {
                                let Some(decoded_chunk) = result.decoded_chunk else {
                                    set_worker_error_once(
                                        &err_slot_ref,
                                        &err_flag_ref,
                                        PDeflateError::Gpu(
                                            "gpu decode result missing payload".to_string(),
                                        ),
                                    );
                                    stop_decode_state(&decode_state_ref);
                                    return;
                                };
                                (
                                    decoded_chunk,
                                    ChunkDecoded {
                                        table_entries: task.table_count,
                                        section_count: task.section_meta.len(),
                                        profile: ChunkDecodeProfile::default(),
                                    },
                                )
                            }
                            gpu::GpuDecodeDisposition::CpuFallback => {
                                let mut restored = vec![0u8; task.chunk_uncompressed_len];
                                let decoded = match decompress_chunk_into(&task.payload, &mut restored)
                                {
                                    Ok(decoded) => decoded,
                                    Err(err) => {
                                        set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                                        stop_decode_state(&decode_state_ref);
                                        return;
                                    }
                                };
                                (restored, decoded)
                            }
                        };
                        {
                            let mut stats = match metrics_ref.lock() {
                                Ok(guard) => guard,
                                Err(_) => {
                                    stop_decode_state(&decode_state_ref);
                                    return;
                                }
                            };
                            stats.chunk_count = stats.chunk_count.saturating_add(1);
                            stats.table_entries_total =
                                stats.table_entries_total.saturating_add(decoded.table_entries);
                            stats.section_count_total =
                                stats.section_count_total.saturating_add(decoded.section_count);
                        }
                        if let Err(err) = writer_ref
                            .submit(u64::try_from(task.output_offset).unwrap_or(u64::MAX), restored)
                            .map_err(map_parallel_file_writer_error)
                        {
                            set_worker_error_once(&err_slot_ref, &err_flag_ref, err);
                            stop_decode_state(&decode_state_ref);
                            return;
                        }
                    }
                }
            });
        }

        Ok(())
    })?;

    writer.drain().map_err(map_parallel_file_writer_error)?;
    take_worker_error(&err_slot)?;
    report_decode_backlog(&decode_backlog_reporter, 0);
    let input_bytes = {
        let (lock, _) = &*decode_state;
        lock.lock()
            .map_err(|_| PDeflateError::InvalidStream("stream decode state poisoned"))?
            .input_bytes
    };
    let metrics = metrics
        .lock()
        .map_err(|_| PDeflateError::InvalidStream("parallel decode metrics poisoned"))?;
    Ok(PDeflateStats {
        input_bytes,
        output_bytes: total_output_bytes,
        chunk_count: metrics.chunk_count,
        table_entries_total: metrics.table_entries_total,
        section_count_total: metrics.section_count_total,
    })
}

fn report_decode_backlog(reporter: &Option<crate::DecodeBacklogReporter>, bytes: usize) {
    if let Some(reporter) = reporter {
        reporter(u64::try_from(bytes).unwrap_or(u64::MAX));
    }
}

fn report_output_write(reporter: &Option<crate::OutputWriteReporter>, bytes: usize) {
    if let Some(reporter) = reporter {
        reporter(u64::try_from(bytes).unwrap_or(u64::MAX));
    }
}

fn stop_decode_state(decode_state_ref: &Arc<(Mutex<StreamDecodeState>, Condvar)>) {
    let (decode_lock, decode_cv) = &**decode_state_ref;
    if let Ok(mut state) = decode_lock.lock() {
        state.stopped = true;
        decode_cv.notify_all();
    }
}

fn map_parallel_file_writer_error(err: cozip_util::ParallelFileWriterError) -> PDeflateError {
    match err {
        cozip_util::ParallelFileWriterError::Io(err) => PDeflateError::Io(err),
        cozip_util::ParallelFileWriterError::NumericOverflow => PDeflateError::NumericOverflow,
        cozip_util::ParallelFileWriterError::Closed => {
            PDeflateError::InvalidStream("parallel writer closed")
        }
    }
}

fn pop_parallel_decode_task(
    state_ref: &Arc<(Mutex<StreamDecodeState>, Condvar)>,
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &Arc<AtomicBool>,
    options: &PDeflateOptions,
    gpu_only: bool,
) -> Option<StreamDecodeTask> {
    let (lock, cv) = &**state_ref;
    let mut state = match lock.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_worker_error_once(
                err_slot,
                err_flag,
                PDeflateError::Gpu("stream decode state mutex poisoned".to_string()),
            );
            return None;
        }
    };
    loop {
        let task = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                if gpu_only {
                    None
                } else {
                    state.queue.pop_front()
                }
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_only {
                    state.gpu_queue.pop_front()
                } else {
                    state.cpu_queue.pop_front()
                }
            }
        };
        if let Some(task) = task {
            return Some(task);
        }
        let queue_empty = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.is_empty(),
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_only {
                    state.gpu_queue.is_empty()
                } else {
                    state.cpu_queue.is_empty()
                }
            }
        };
        if state.producer_eof && queue_empty {
            return None;
        }
        if state.stopped || err_flag.load(Ordering::Relaxed) {
            return None;
        }
        state = match cv.wait(state) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
    }
}

fn pop_parallel_decode_gpu_batch(
    state_ref: &Arc<(Mutex<StreamDecodeState>, Condvar)>,
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &Arc<AtomicBool>,
    options: &PDeflateOptions,
    batch_limit: usize,
) -> Option<Vec<StreamDecodeTask>> {
    let (lock, cv) = &**state_ref;
    let mut state = match lock.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_worker_error_once(
                err_slot,
                err_flag,
                PDeflateError::Gpu("stream decode state mutex poisoned".to_string()),
            );
            return None;
        }
    };
    loop {
        let mut tasks = Vec::with_capacity(batch_limit);
        match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                while tasks.len() < batch_limit {
                    let pos = state
                        .queue
                        .iter()
                        .position(|task| task.chunk_uncompressed_len >= options.gpu_min_chunk_size);
                    let Some(pos) = pos else {
                        break;
                    };
                    if let Some(task) = state.queue.remove(pos) {
                        tasks.push(task);
                    }
                }
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                while tasks.len() < batch_limit {
                    let Some(task) = state.gpu_queue.pop_front() else {
                        break;
                    };
                    tasks.push(task);
                }
            }
        }
        if !tasks.is_empty() {
            return Some(tasks);
        }
        let queue_empty = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => state
                .queue
                .iter()
                .all(|task| task.chunk_uncompressed_len < options.gpu_min_chunk_size),
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => state.gpu_queue.is_empty(),
        };
        if state.producer_eof && queue_empty {
            return None;
        }
        if state.stopped || err_flag.load(Ordering::Relaxed) {
            return None;
        }
        state = match cv.wait(state) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
    }
}

fn pop_stream_compress_task(
    state_ref: &Arc<(Mutex<StreamCompressState>, Condvar)>,
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &Arc<AtomicBool>,
    options: &PDeflateOptions,
    gpu_only: bool,
    low_watermark_bytes: usize,
) -> Option<StreamCompressTask> {
    let (lock, cv) = &**state_ref;
    let mut state = match lock.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_worker_error_once(
                err_slot,
                err_flag,
                PDeflateError::Gpu("stream compression state mutex poisoned".to_string()),
            );
            return None;
        }
    };
    loop {
        let task = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                if gpu_only {
                    None
                } else {
                    state.queue.pop_front()
                }
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_only {
                    state.gpu_queue.pop_front()
                } else {
                    state.cpu_queue.pop_front()
                }
            }
        };
        if let Some(task) = task {
            state.buffered_bytes = state.buffered_bytes.saturating_sub(task.chunk.len());
            if state.buffered_bytes < low_watermark_bytes {
                cv.notify_all();
            }
            return Some(task);
        }
        let queue_empty = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.is_empty(),
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_only {
                    state.gpu_queue.is_empty()
                } else {
                    state.cpu_queue.is_empty()
                }
            }
        };
        if state.producer_eof && queue_empty {
            return None;
        }
        if state.stopped || err_flag.load(Ordering::Relaxed) {
            return None;
        }
        state = match cv.wait(state) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
    }
}

fn pop_stream_compress_gpu_batch(
    state_ref: &Arc<(Mutex<StreamCompressState>, Condvar)>,
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &Arc<AtomicBool>,
    options: &PDeflateOptions,
    batch_limit: usize,
    low_watermark_bytes: usize,
) -> Option<Vec<StreamCompressTask>> {
    let (lock, cv) = &**state_ref;
    let mut state = match lock.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_worker_error_once(
                err_slot,
                err_flag,
                PDeflateError::Gpu("stream compression state mutex poisoned".to_string()),
            );
            return None;
        }
    };
    loop {
        let mut tasks = Vec::with_capacity(batch_limit);
        match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                while tasks.len() < batch_limit {
                    let pos = state
                        .queue
                        .iter()
                        .position(|task| task.chunk.len() >= options.gpu_min_chunk_size);
                    let Some(pos) = pos else {
                        break;
                    };
                    if let Some(task) = state.queue.remove(pos) {
                        state.buffered_bytes = state.buffered_bytes.saturating_sub(task.chunk.len());
                        tasks.push(task);
                    }
                }
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                while tasks.len() < batch_limit {
                    let Some(task) = state.gpu_queue.pop_front() else {
                        break;
                    };
                    state.buffered_bytes = state.buffered_bytes.saturating_sub(task.chunk.len());
                    tasks.push(task);
                }
            }
        }
        if !tasks.is_empty() {
            if state.buffered_bytes < low_watermark_bytes {
                cv.notify_all();
            }
            return Some(tasks);
        }
        let queue_empty = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.is_empty(),
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => state.gpu_queue.is_empty(),
        };
        if state.producer_eof && queue_empty {
            return None;
        }
        if state.stopped || err_flag.load(Ordering::Relaxed) {
            return None;
        }
        state = match cv.wait(state) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
    }
}

fn compress_stream_gpu_batch(
    tasks: &[StreamCompressTask],
    options: &PDeflateOptions,
) -> Result<Vec<ChunkCompressed>, PDeflateError> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = options.chunk_size.max(1);
    let mut flat_input =
        Vec::with_capacity(tasks.iter().map(|task| task.chunk.len()).sum::<usize>().max(chunk_size));
    let mut indices = Vec::with_capacity(tasks.len());
    for (idx, task) in tasks.iter().enumerate() {
        if idx + 1 != tasks.len() && task.chunk.len() != chunk_size {
            return Err(PDeflateError::InvalidStream(
                "non-final stream chunk shorter than chunk size",
            ));
        }
        flat_input.extend_from_slice(&task.chunk);
        indices.push(idx);
    }
    let (mut batch, _) = compress_chunk_gpu_batch(&flat_input, chunk_size, &indices, options)?;
    batch.sort_unstable_by_key(|(idx, _)| *idx);
    Ok(batch.into_iter().map(|(_, chunk)| chunk).collect())
}

fn pop_stream_decode_task(
    state_ref: &Arc<(Mutex<StreamDecodeState>, Condvar)>,
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &Arc<AtomicBool>,
    options: &PDeflateOptions,
    gpu_only: bool,
    result_limit_bytes: usize,
) -> Option<StreamDecodeTask> {
    let (lock, cv) = &**state_ref;
    let mut state = match lock.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_worker_error_once(
                err_slot,
                err_flag,
                PDeflateError::Gpu("stream decode state mutex poisoned".to_string()),
            );
            return None;
        }
    };
    loop {
        let throttle = state.buffered_result_bytes >= result_limit_bytes;
        let task = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                if gpu_only {
                    None
                } else if throttle {
                    match state.queue.front() {
                        Some(task) if task.index == state.next_write_index => state.queue.pop_front(),
                        _ => None,
                    }
                } else {
                    state.queue.pop_front()
                }
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_only {
                    if throttle {
                        match state.gpu_queue.front() {
                            Some(task) if task.index == state.next_write_index => {
                                state.gpu_queue.pop_front()
                            }
                            _ => None,
                        }
                    } else {
                        state.gpu_queue.pop_front()
                    }
                } else {
                    if throttle {
                        match state.cpu_queue.front() {
                            Some(task) if task.index == state.next_write_index => {
                                state.cpu_queue.pop_front()
                            }
                            _ => None,
                        }
                    } else {
                        state.cpu_queue.pop_front()
                    }
                }
            }
        };
        if let Some(task) = task {
            return Some(task);
        }
        let queue_empty = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.is_empty(),
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_only {
                    state.gpu_queue.is_empty()
                } else {
                    state.cpu_queue.is_empty()
                }
            }
        };
        if state.producer_eof && queue_empty {
            return None;
        }
        if state.stopped || err_flag.load(Ordering::Relaxed) {
            return None;
        }
        state = match cv.wait(state) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
    }
}

fn pop_stream_decode_gpu_batch(
    state_ref: &Arc<(Mutex<StreamDecodeState>, Condvar)>,
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &Arc<AtomicBool>,
    options: &PDeflateOptions,
    batch_limit: usize,
    result_limit_bytes: usize,
) -> Option<Vec<StreamDecodeTask>> {
    let (lock, cv) = &**state_ref;
    let mut state = match lock.lock() {
        Ok(guard) => guard,
        Err(_) => {
            set_worker_error_once(
                err_slot,
                err_flag,
                PDeflateError::Gpu("stream decode state mutex poisoned".to_string()),
            );
            return None;
        }
    };
    loop {
        let mut tasks = Vec::with_capacity(batch_limit);
        let throttle = state.buffered_result_bytes >= result_limit_bytes;
        match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                if throttle {
                    let pos = state.queue.iter().position(|task| {
                        task.index == state.next_write_index
                            && task.chunk_uncompressed_len >= options.gpu_min_chunk_size
                    });
                    if let Some(pos) = pos {
                        if let Some(task) = state.queue.remove(pos) {
                            tasks.push(task);
                        }
                    }
                } else {
                    while tasks.len() < batch_limit {
                        let pos = state
                            .queue
                            .iter()
                            .position(|task| task.chunk_uncompressed_len >= options.gpu_min_chunk_size);
                        let Some(pos) = pos else {
                            break;
                        };
                        if let Some(task) = state.queue.remove(pos) {
                            tasks.push(task);
                        }
                    }
                }
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if throttle {
                    if matches!(state.gpu_queue.front(), Some(task) if task.index == state.next_write_index)
                    {
                        if let Some(task) = state.gpu_queue.pop_front() {
                            tasks.push(task);
                        }
                    }
                } else {
                    while tasks.len() < batch_limit {
                        let Some(task) = state.gpu_queue.pop_front() else {
                            break;
                        };
                        tasks.push(task);
                    }
                }
            }
        }
        if !tasks.is_empty() {
            return Some(tasks);
        }
        let queue_empty = match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => state.queue.is_empty(),
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => state.gpu_queue.is_empty(),
        };
        if state.producer_eof && queue_empty {
            return None;
        }
        if state.stopped || err_flag.load(Ordering::Relaxed) {
            return None;
        }
        state = match cv.wait(state) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
    }
}
