use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use cozip::{CoZipArchiveFormat, CoZipArchiveInfo, CoZipArchiveKind, inspect_archive_from_name};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialScreen {
    Compress,
    Decompress,
    CompressSettings,
    DecompressSettings,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveFormat {
    Zip,
    Cozip,
}

#[derive(Clone, Debug)]
pub enum DesktopCommand {
    Compress(CompressPlan),
    Extract(ExtractPlan),
}

#[derive(Clone, Debug)]
pub struct LaunchRequest {
    pub initial_screen: InitialScreen,
    pub command: Option<DesktopCommand>,
    pub auto_start: bool,
    pub startup_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CompressPlan {
    pub format: ArchiveFormat,
    pub hybrid: bool,
    pub sources: Vec<PathBuf>,
    pub output_path: PathBuf,
    pub mode: CompressMode,
}

#[derive(Clone, Debug)]
pub enum CompressMode {
    SingleFile,
    SingleDirectory,
    MultiSelection,
}

#[derive(Clone, Debug)]
pub struct ExtractPlan {
    pub tasks: Vec<ExtractTask>,
    pub ignored_inputs: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct ExtractTask {
    pub archive_path: PathBuf,
    pub archive_format: ArchiveFormat,
    pub archive_kind: ExtractArchiveKind,
    pub output_path: PathBuf,
}

#[derive(Clone, Debug)]
pub enum ExtractArchiveKind {
    SingleFile,
    Directory,
}

impl LaunchRequest {
    pub fn from_env() -> Self {
        let mut args = env::args_os();
        let _ = args.next();
        match parse_args(args.collect()) {
            Ok(request) => request,
            Err(message) => Self {
                initial_screen: InitialScreen::Compress,
                command: None,
                auto_start: false,
                startup_error: Some(message),
            },
        }
    }
}

fn parse_args(args: Vec<OsString>) -> Result<LaunchRequest, String> {
    if args.is_empty() {
        return Ok(LaunchRequest {
            initial_screen: InitialScreen::Compress,
            command: None,
            auto_start: false,
            startup_error: None,
        });
    }

    let command = os_to_string(&args[0])?;
    match command.as_str() {
        "compress" => parse_compress_args(&args[1..], true),
        "extract" => parse_extract_args(&args[1..], true),
        "ui" => parse_ui_args(&args[1..]),
        other => Err(format!("unsupported command: {other}")),
    }
}

fn parse_ui_args(args: &[OsString]) -> Result<LaunchRequest, String> {
    let Some(subcommand) = args.first() else {
        return Err("ui subcommand is missing".to_string());
    };
    let subcommand = os_to_string(subcommand)?;
    match subcommand.as_str() {
        "compress-details" => parse_compress_args(&args[1..], false),
        "extract-details" => parse_extract_args(&args[1..], false),
        other => Err(format!("unsupported ui subcommand: {other}")),
    }
}

fn parse_compress_args(args: &[OsString], auto_start: bool) -> Result<LaunchRequest, String> {
    let mut format = ArchiveFormat::Zip;
    let mut hybrid = false;
    let mut paths = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match os_to_string(&args[index])?.as_str() {
            "--format" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--format requires a value".to_string())?;
                format = match os_to_string(value)?.as_str() {
                    "zip" => ArchiveFormat::Zip,
                    "cozip" => ArchiveFormat::Cozip,
                    other => return Err(format!("unsupported archive format: {other}")),
                };
            }
            "--hybrid" => {
                hybrid = true;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unsupported compress option: {flag}"));
            }
            _ => paths.push(PathBuf::from(&args[index])),
        }
        index += 1;
    }

    let plan = build_compress_plan(paths, format, hybrid)?;
    Ok(LaunchRequest {
        initial_screen: if auto_start {
            InitialScreen::Compress
        } else {
            InitialScreen::CompressSettings
        },
        command: Some(DesktopCommand::Compress(plan)),
        auto_start,
        startup_error: None,
    })
}

fn parse_extract_args(args: &[OsString], auto_start: bool) -> Result<LaunchRequest, String> {
    let mut here = false;
    let mut paths = Vec::new();

    for arg in args {
        let value = os_to_string(arg)?;
        if value == "--here" {
            here = true;
        } else if value.starts_with('-') {
            return Err(format!("unsupported extract option: {value}"));
        } else {
            paths.push(PathBuf::from(arg));
        }
    }

    if !here {
        return Err("extract currently requires --here".to_string());
    }

    let plan = build_extract_plan(paths)?;
    Ok(LaunchRequest {
        initial_screen: if auto_start {
            InitialScreen::Decompress
        } else {
            InitialScreen::DecompressSettings
        },
        command: Some(DesktopCommand::Extract(plan)),
        auto_start,
        startup_error: None,
    })
}

fn build_compress_plan(
    mut sources: Vec<PathBuf>,
    format: ArchiveFormat,
    hybrid: bool,
) -> Result<CompressPlan, String> {
    if sources.is_empty() {
        return Err("no input files or folders were provided".to_string());
    }

    for path in &sources {
        if !path.exists() {
            return Err(format!("input path does not exist: {}", path.display()));
        }
    }

    sources.sort();
    let extension = match format {
        ArchiveFormat::Zip => "zip",
        ArchiveFormat::Cozip => "pdz",
    };

    let mode = if sources.len() == 1 {
        let path = &sources[0];
        if path.is_dir() {
            CompressMode::SingleDirectory
        } else if path.is_file() {
            CompressMode::SingleFile
        } else {
            return Err(format!("unsupported input path: {}", path.display()));
        }
    } else {
        CompressMode::MultiSelection
    };

    let output_path = match mode {
        CompressMode::SingleFile => {
            let source = &sources[0];
            let parent = source.parent().unwrap_or_else(|| Path::new("."));
            let stem = source
                .file_stem()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("archive");
            unique_path(parent.join(format!("{stem}.{extension}")))
        }
        CompressMode::SingleDirectory => {
            let source = &sources[0];
            let parent = source.parent().unwrap_or_else(|| Path::new("."));
            let name = source
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("archive");
            unique_path(parent.join(format!("{name}.{extension}")))
        }
        CompressMode::MultiSelection => {
            let parent = common_parent(&sources)
                .or_else(|| sources[0].parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("."));
            unique_path(parent.join(format!("Archive.{extension}")))
        }
    };

    Ok(CompressPlan {
        format,
        hybrid,
        sources,
        output_path,
        mode,
    })
}

fn build_extract_plan(inputs: Vec<PathBuf>) -> Result<ExtractPlan, String> {
    if inputs.is_empty() {
        return Err("no archives were provided".to_string());
    }

    let mut tasks = Vec::new();
    let mut ignored = Vec::new();
    let mut seen = BTreeSet::new();

    for input in inputs {
        if !input.exists() {
            ignored.push(input);
            continue;
        }

        if input.is_file() {
            match inspect_archive_from_name(&input) {
                Ok(info) => {
                    if seen.insert(input.clone()) {
                        tasks.push(build_extract_task(input, info));
                    }
                }
                Err(_) => ignored.push(input),
            }
            continue;
        }

        if input.is_dir() {
            let read_dir = std::fs::read_dir(&input)
                .map_err(|error| format!("failed to read {}: {error}", input.display()))?;
            for entry in read_dir {
                let entry = entry.map_err(|error| {
                    format!("failed to read entry in {}: {error}", input.display())
                })?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                match inspect_archive_from_name(&path) {
                    Ok(info) => {
                        if seen.insert(path.clone()) {
                            tasks.push(build_extract_task(path, info));
                        }
                    }
                    Err(_) => {}
                }
            }
            continue;
        }

        ignored.push(input);
    }

    if tasks.is_empty() {
        return Err("no supported archives were found in the selection".to_string());
    }

    Ok(ExtractPlan {
        tasks,
        ignored_inputs: ignored,
    })
}

fn build_extract_task(path: PathBuf, info: CoZipArchiveInfo) -> ExtractTask {
    let archive_format = match info.format {
        CoZipArchiveFormat::Zip => ArchiveFormat::Zip,
        CoZipArchiveFormat::PDeflate => ArchiveFormat::Cozip,
    };
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("archive");
    let (archive_kind, output_path) = match info.kind {
        CoZipArchiveKind::SingleFile { suggested_name } => (
            ExtractArchiveKind::SingleFile,
            unique_path(parent.join(suggested_name)),
        ),
        CoZipArchiveKind::Directory => (
            ExtractArchiveKind::Directory,
            unique_path(parent.join(stem)),
        ),
    };

    ExtractTask {
        archive_path: path,
        archive_format,
        archive_kind,
        output_path,
    }
}

fn os_to_string(value: &OsString) -> Result<String, String> {
    value
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| "non-utf8 arguments are not supported".to_string())
}

fn common_parent(paths: &[PathBuf]) -> Option<PathBuf> {
    let first_parent = paths.first()?.parent()?.to_path_buf();
    if paths
        .iter()
        .all(|path| path.parent().map(Path::to_path_buf) == Some(first_parent.clone()))
    {
        Some(first_parent)
    } else {
        None
    }
}

fn unique_path(candidate: PathBuf) -> PathBuf {
    if !candidate.exists() {
        return candidate;
    }

    let parent = candidate
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = candidate
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("output");
    let extension = candidate.extension().and_then(|value| value.to_str());

    for suffix in 2..=9999 {
        let name = match extension {
            Some(ext) => format!("{stem} ({suffix}).{ext}"),
            None => format!("{stem} ({suffix})"),
        };
        let next = parent.join(name);
        if !next.exists() {
            return next;
        }
    }

    candidate
}
