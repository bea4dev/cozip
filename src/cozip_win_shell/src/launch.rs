use std::path::PathBuf;
use std::process::Command;

use thiserror::Error;

const COZIP_DESKTOP_EXE_PATH: &str = r"C:\Program Files\CoZip\cozip_desktop.exe";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellCommandKind {
    CompressZip,
    CompressCozip,
    CompressDetails,
    ExtractHere,
    ExtractDetails,
}

#[derive(Debug, Error)]
pub enum LaunchError {
    #[error("no items were selected")]
    NoItems,
    #[error("failed to start cozip_desktop.exe: {0}")]
    Spawn(std::io::Error),
}

pub fn launch_command(kind: ShellCommandKind, selected_paths: &[PathBuf]) -> Result<(), LaunchError> {
    if selected_paths.is_empty() {
        return Err(LaunchError::NoItems);
    }

    let mut command = Command::new(COZIP_DESKTOP_EXE_PATH);
    match kind {
        ShellCommandKind::CompressZip => {
            command.args(["compress", "--format", "zip", "--hybrid"]);
        }
        ShellCommandKind::CompressCozip => {
            command.args(["compress", "--format", "cozip", "--hybrid"]);
        }
        ShellCommandKind::CompressDetails => {
            command.args(["ui", "compress-details"]);
        }
        ShellCommandKind::ExtractHere => {
            command.args(["extract", "--here"]);
        }
        ShellCommandKind::ExtractDetails => {
            command.args(["ui", "extract-details"]);
        }
    }
    command.args(selected_paths);
    command.spawn().map(|_| ()).map_err(LaunchError::Spawn)
}
