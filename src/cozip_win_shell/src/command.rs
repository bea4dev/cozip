use std::cell::RefCell;
use std::path::PathBuf;

use windows::Win32::Foundation::{BOOL, E_FAIL, E_NOTIMPL, S_FALSE};
use windows::Win32::System::Com::CoTaskMemAlloc;
use windows::Win32::UI::Shell::{
    IEnumExplorerCommand, IEnumExplorerCommand_Impl, IExplorerCommand, IExplorerCommand_Impl,
    IShellItemArray, SIGDN_FILESYSPATH,
};
use windows::core::{GUID, HRESULT, PWSTR, Result, implement};

use crate::i18n::I18n;
use crate::launch::{ShellCommandKind, launch_command};

const ECF_HASSUBCOMMANDS: u32 = 0x001;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExplorerVerb {
    CompressRoot,
    CompressZip,
    CompressCozip,
    CompressDetails,
    ExtractRoot,
    ExtractHere,
    ExtractDetails,
}

impl ExplorerVerb {
    pub fn title_key(self) -> &'static str {
        match self {
            Self::CompressRoot => "menu.compress_root",
            Self::CompressZip => "menu.compress_zip",
            Self::CompressCozip => "menu.compress_cozip",
            Self::CompressDetails => "menu.compress_details",
            Self::ExtractRoot => "menu.extract_root",
            Self::ExtractHere => "menu.extract_here",
            Self::ExtractDetails => "menu.extract_details",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Self::CompressRoot | Self::CompressZip | Self::CompressCozip | Self::CompressDetails => {
                r"C:\Program Files\CoZip\icons\comp.ico"
            }
            Self::ExtractRoot | Self::ExtractHere | Self::ExtractDetails => {
                r"C:\Program Files\CoZip\icons\decomp.ico"
            }
        }
    }

    pub fn children(self) -> &'static [ExplorerVerb] {
        match self {
            Self::CompressRoot => &[
                Self::CompressZip,
                Self::CompressCozip,
                Self::CompressDetails,
            ],
            Self::ExtractRoot => &[Self::ExtractHere, Self::ExtractDetails],
            _ => &[],
        }
    }

    pub fn launch_kind(self) -> Option<ShellCommandKind> {
        match self {
            Self::CompressZip => Some(ShellCommandKind::CompressZip),
            Self::CompressCozip => Some(ShellCommandKind::CompressCozip),
            Self::CompressDetails => Some(ShellCommandKind::CompressDetails),
            Self::ExtractHere => Some(ShellCommandKind::ExtractHere),
            Self::ExtractDetails => Some(ShellCommandKind::ExtractDetails),
            Self::CompressRoot | Self::ExtractRoot => None,
        }
    }
}

#[implement(IExplorerCommand)]
pub struct ExplorerCommand {
    verb: ExplorerVerb,
}

impl ExplorerCommand {
    pub fn new(verb: ExplorerVerb) -> Self {
        Self { verb }
    }
}

impl IExplorerCommand_Impl for ExplorerCommand {
    fn GetTitle(&self, _items: Option<&IShellItemArray>) -> Result<PWSTR> {
        alloc_pwstr(I18n::shared().text(self.verb.title_key()))
    }

    fn GetIcon(&self, _items: Option<&IShellItemArray>) -> Result<PWSTR> {
        alloc_pwstr(self.verb.icon())
    }

    fn GetToolTip(&self, _items: Option<&IShellItemArray>) -> Result<PWSTR> {
        Err(E_NOTIMPL.into())
    }

    fn GetCanonicalName(&self) -> Result<GUID> {
        Err(E_NOTIMPL.into())
    }

    fn GetState(&self, _items: Option<&IShellItemArray>, _ok_to_be_slow: BOOL) -> Result<u32> {
        Ok(0)
    }

    fn Invoke(
        &self,
        items: Option<&IShellItemArray>,
        _bind_ctx: Option<&windows::Win32::System::Com::IBindCtx>,
    ) -> Result<()> {
        let Some(kind) = self.verb.launch_kind() else {
            return Ok(());
        };
        let Some(items) = items else {
            return Err(E_FAIL.into());
        };
        let selected = collect_paths(items)?;
        launch_command(kind, &selected).map_err(|_| E_FAIL.into())
    }

    fn GetFlags(&self) -> Result<u32> {
        if self.verb.children().is_empty() {
            Ok(0)
        } else {
            Ok(ECF_HASSUBCOMMANDS)
        }
    }

    fn EnumSubCommands(&self) -> Result<IEnumExplorerCommand> {
        if self.verb.children().is_empty() {
            Err(E_NOTIMPL.into())
        } else {
            Ok(ExplorerCommandEnum::new(self.verb.children()).into())
        }
    }
}

#[implement(IEnumExplorerCommand)]
struct ExplorerCommandEnum {
    commands: Vec<IExplorerCommand>,
    cursor: RefCell<usize>,
}

impl ExplorerCommandEnum {
    fn new(verbs: &[ExplorerVerb]) -> Self {
        let commands = verbs
            .iter()
            .copied()
            .map(|verb| IExplorerCommand::from(ExplorerCommand::new(verb)))
            .collect();
        Self {
            commands,
            cursor: RefCell::new(0),
        }
    }
}

impl IEnumExplorerCommand_Impl for ExplorerCommandEnum {
    fn Next(
        &self,
        count: u32,
        result: *mut Option<IExplorerCommand>,
        fetched: *mut u32,
    ) -> HRESULT {
        let mut cursor = self.cursor.borrow_mut();
        let mut written = 0_u32;
        for index in 0..count as usize {
            if *cursor >= self.commands.len() {
                break;
            }
            unsafe {
                result.add(index).write(Some(self.commands[*cursor].clone()));
            }
            *cursor += 1;
            written += 1;
        }
        if !fetched.is_null() {
            unsafe {
                fetched.write(written);
            }
        }
        if written == count {
            HRESULT(0)
        } else {
            S_FALSE
        }
    }

    fn Skip(&self, count: u32) -> Result<()> {
        let mut cursor = self.cursor.borrow_mut();
        *cursor = (*cursor + count as usize).min(self.commands.len());
        Ok(())
    }

    fn Reset(&self) -> Result<()> {
        *self.cursor.borrow_mut() = 0;
        Ok(())
    }

    fn Clone(&self) -> Result<IEnumExplorerCommand> {
        Ok(IEnumExplorerCommand::from(ExplorerCommandEnum {
            commands: self.commands.clone(),
            cursor: RefCell::new(*self.cursor.borrow()),
        }))
    }
}

fn collect_paths(items: &IShellItemArray) -> Result<Vec<PathBuf>> {
    let count = unsafe { items.GetCount()? };
    let mut paths = Vec::with_capacity(count as usize);
    for index in 0..count {
        let item = unsafe { items.GetItemAt(index)? };
        let display_name = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH)? };
        let path = unsafe { display_name.to_string() }.map(PathBuf::from)?;
        paths.push(path);
    }
    Ok(paths)
}

fn alloc_pwstr(value: &str) -> Result<PWSTR> {
    let mut wide = value.encode_utf16().collect::<Vec<_>>();
    wide.push(0);
    let bytes = wide.len() * std::mem::size_of::<u16>();
    let raw = unsafe { CoTaskMemAlloc(bytes) } as *mut u16;
    if raw.is_null() {
        return Err(E_FAIL.into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(wide.as_ptr(), raw, wide.len());
    }
    Ok(PWSTR(raw as *mut _))
}
