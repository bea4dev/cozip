use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use cozip::{CoZip, CoZipOptions, CoZipProgress, PDeflateOptions, ZipOptions};

use crate::launch::{ArchiveFormat, CompressMode, CompressPlan, DesktopCommand, ExtractPlan};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobStatus {
    Idle,
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug)]
pub struct JobSnapshot {
    pub status: JobStatus,
    pub progress: Option<CoZipProgress>,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub current_task_label: Option<String>,
    pub note: Option<String>,
    pub error: Option<String>,
}

impl Default for JobSnapshot {
    fn default() -> Self {
        Self {
            status: JobStatus::Idle,
            progress: None,
            total_tasks: 0,
            completed_tasks: 0,
            current_task_label: None,
            note: None,
            error: None,
        }
    }
}

pub type SharedJobSnapshot = Arc<Mutex<JobSnapshot>>;

pub fn spawn_job(command: DesktopCommand) -> SharedJobSnapshot {
    let shared = Arc::new(Mutex::new(initial_snapshot(&command)));
    let worker_shared = Arc::clone(&shared);
    thread::spawn(move || {
        let result = match command {
            DesktopCommand::Compress(plan) => run_compress(plan, &worker_shared),
            DesktopCommand::Extract(plan) => run_extract(plan, &worker_shared),
        };

        let mut state = worker_shared.lock().expect("job snapshot poisoned");
        match result {
            Ok(note) => {
                state.status = JobStatus::Succeeded;
                state.note = Some(note);
                state.error = None;
            }
            Err(error) => {
                state.status = JobStatus::Failed;
                state.error = Some(error);
            }
        }
    });
    shared
}

fn initial_snapshot(command: &DesktopCommand) -> JobSnapshot {
    let total_tasks = match command {
        DesktopCommand::Compress(_) => 1,
        DesktopCommand::Extract(plan) => plan.tasks.len(),
    };
    JobSnapshot {
        status: JobStatus::Running,
        progress: None,
        total_tasks,
        completed_tasks: 0,
        current_task_label: None,
        note: None,
        error: None,
    }
}

fn run_compress(plan: CompressPlan, shared: &SharedJobSnapshot) -> Result<String, String> {
    let progress = CoZipProgress::new();
    set_current_task(
        shared,
        display_path(plan.sources.first().map(PathBuf::as_path)),
        progress.clone(),
    );

    if let Some(parent) = plan.output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }

    let cozip = init_compress_cozip(&plan)?;
    let stats = match plan.mode {
        CompressMode::SingleFile => cozip
            .compress_file_from_name_with_progress(
                &plan.sources[0],
                &plan.output_path,
                Some(progress.clone()),
            )
            .map_err(|error| error.to_string())?,
        CompressMode::SingleDirectory => cozip
            .compress_directory_with_progress(
                &plan.sources[0],
                &plan.output_path,
                Some(progress.clone()),
            )
            .map_err(|error| error.to_string())?,
        CompressMode::MultiSelection => {
            let staging_dir = build_staging_directory(&plan.sources)?;
            let result = cozip
                .compress_directory_with_progress(
                    &staging_dir,
                    &plan.output_path,
                    Some(progress.clone()),
                )
                .map_err(|error| error.to_string());
            let _ = fs::remove_dir_all(&staging_dir);
            result?
        }
    };

    let mut state = shared.lock().expect("job snapshot poisoned");
    state.completed_tasks = 1;
    Ok(format!(
        "Output: {} ({:.2} MiB -> {:.2} MiB)",
        plan.output_path.display(),
        stats.input_bytes as f64 / (1024.0 * 1024.0),
        stats.output_bytes as f64 / (1024.0 * 1024.0)
    ))
}

fn run_extract(plan: ExtractPlan, shared: &SharedJobSnapshot) -> Result<String, String> {
    for (index, task) in plan.tasks.iter().enumerate() {
        let progress = CoZipProgress::new();
        let output_path = plan.output_path_for(task);
        set_current_task(
            shared,
            display_path(Some(&task.archive_path)),
            progress.clone(),
        );

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }

        let cozip =
            init_extract_cozip(task.archive_format, &plan.zip_options, &plan.pdeflate_options)?;
        let result = cozip
            .decompress_auto_from_name_with_progress(
                &task.archive_path,
                &output_path,
                Some(progress.clone()),
            )
            .map_err(|error| error.to_string());
        match result {
            Ok(_) => {
                let mut state = shared.lock().expect("job snapshot poisoned");
                state.completed_tasks = index + 1;
                state.progress = Some(progress.clone());
            }
            Err(error) => {
                return Err(format!("{}: {error}", task.archive_path.display()));
            }
        }
    }

    Ok(format!(
        "{} archive(s) extracted",
        plan.tasks.len()
    ))
}

fn init_compress_cozip(plan: &CompressPlan) -> Result<CoZip, String> {
    let options = match plan.format {
        ArchiveFormat::Zip => CoZipOptions::Zip {
            options: plan.zip_options.clone(),
        },
        ArchiveFormat::Cozip => CoZipOptions::PDeflate {
            options: plan.pdeflate_options.clone(),
        },
    };
    CoZip::init(options).map_err(|error| error.to_string())
}

fn init_extract_cozip(
    format: ArchiveFormat,
    zip_options: &ZipOptions,
    pdeflate_options: &PDeflateOptions,
) -> Result<CoZip, String> {
    let options = match format {
        ArchiveFormat::Zip => CoZipOptions::Zip {
            options: zip_options.clone(),
        },
        ArchiveFormat::Cozip => CoZipOptions::PDeflate {
            options: pdeflate_options.clone(),
        },
    };
    CoZip::init(options).map_err(|error| error.to_string())
}

fn set_current_task(shared: &SharedJobSnapshot, label: String, progress: CoZipProgress) {
    let mut state = shared.lock().expect("job snapshot poisoned");
    state.current_task_label = Some(label);
    state.progress = Some(progress);
}

fn build_staging_directory(sources: &[PathBuf]) -> Result<PathBuf, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    let root = std::env::temp_dir().join(format!(
        "cozip-desktop-stage-{}-{timestamp}",
        std::process::id()
    ));
    fs::create_dir_all(&root)
        .map_err(|error| format!("failed to create {}: {error}", root.display()))?;

    for source in sources {
        let file_name = source
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("item");
        let target = unique_child_path(&root, file_name);
        copy_path(source, &target)?;
    }

    Ok(root)
}

fn unique_child_path(parent: &Path, base_name: &str) -> PathBuf {
    let candidate = parent.join(base_name);
    if !candidate.exists() {
        return candidate;
    }

    let base = Path::new(base_name);
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("item");
    let extension = base.extension().and_then(|value| value.to_str());

    for suffix in 2..=9999 {
        let name = match extension {
            Some(ext) => format!("{stem} ({suffix}).{ext}"),
            None => format!("{stem} ({suffix})"),
        };
        let candidate = parent.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }

    parent.join(base_name)
}

fn copy_path(source: &Path, target: &Path) -> Result<(), String> {
    if source.is_dir() {
        fs::create_dir_all(target)
            .map_err(|error| format!("failed to create {}: {error}", target.display()))?;
        let entries = fs::read_dir(source)
            .map_err(|error| format!("failed to read {}: {error}", source.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!("failed to read entry in {}: {error}", source.display())
            })?;
            let child = entry.path();
            copy_path(&child, &target.join(entry.file_name()))?;
        }
        return Ok(());
    }

    fs::copy(source, target).map_err(|error| {
        format!(
            "failed to copy {} -> {}: {error}",
            source.display(),
            target.display()
        )
    })?;
    Ok(())
}

fn display_path(path: Option<&Path>) -> String {
    path.map(|value| value.display().to_string())
        .unwrap_or_else(|| "operation".to_string())
}
