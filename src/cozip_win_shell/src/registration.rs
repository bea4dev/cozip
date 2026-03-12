use winreg::RegKey;
use winreg::enums::HKEY_CURRENT_USER;
use windows::Win32::Foundation::E_FAIL;
use windows::core::{Error, GUID, Result};

use crate::class_ids::{CLSID_COMPRESS_ROOT, CLSID_EXTRACT_ROOT};
use crate::i18n::I18n;

const COMP_ICON_PATH: &str = r"C:\Program Files\CoZip\icons\comp.ico";
const DECOMP_ICON_PATH: &str = r"C:\Program Files\CoZip\icons\decomp.ico";
const COZIP_WIN_SHELL_DLL_PATH: &str = r"C:\Program Files\CoZip\cozip_win_shell.dll";

pub fn register_server() -> Result<()> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let classes = hkcu
        .create_subkey("Software\\Classes")
        .map_err(io_to_winerr)?
        .0;
    let i18n = I18n::shared();

    register_clsid(&classes, CLSID_COMPRESS_ROOT)?;
    register_clsid(&classes, CLSID_EXTRACT_ROOT)?;
    register_menu_root(
        &classes,
        "AllFilesystemObjects\\shell\\CozipCompress",
        i18n.text("menu.compress_root"),
        COMP_ICON_PATH,
        CLSID_COMPRESS_ROOT,
    )?;
    register_menu_root(
        &classes,
        "AllFilesystemObjects\\shell\\CozipExtract",
        i18n.text("menu.extract_root"),
        DECOMP_ICON_PATH,
        CLSID_EXTRACT_ROOT,
    )?;
    Ok(())
}

pub fn unregister_server() -> Result<()> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let classes = hkcu
        .create_subkey("Software\\Classes")
        .map_err(io_to_winerr)?
        .0;

    let _ = classes.delete_subkey_all(format!("CLSID\\{{{}}}", clsid_string(CLSID_COMPRESS_ROOT)));
    let _ = classes.delete_subkey_all(format!("CLSID\\{{{}}}", clsid_string(CLSID_EXTRACT_ROOT)));
    let _ = classes.delete_subkey_all("AllFilesystemObjects\\shell\\CozipCompress");
    let _ = classes.delete_subkey_all("AllFilesystemObjects\\shell\\CozipExtract");
    Ok(())
}

fn register_clsid(classes: &RegKey, clsid: GUID) -> Result<()> {
    let clsid_path = format!("CLSID\\{{{}}}", clsid_string(clsid));
    let (key, _) = classes.create_subkey(&clsid_path).map_err(io_to_winerr)?;
    key.set_value("", &I18n::shared().text("registration.command_name"))
        .map_err(io_to_winerr)?;

    let (inproc, _) = key.create_subkey("InprocServer32").map_err(io_to_winerr)?;
    inproc
        .set_value("", &COZIP_WIN_SHELL_DLL_PATH)
        .map_err(io_to_winerr)?;
    inproc
        .set_value("ThreadingModel", &"Apartment")
        .map_err(io_to_winerr)?;
    Ok(())
}

fn register_menu_root(
    classes: &RegKey,
    subkey: &str,
    title: &str,
    icon_path: &str,
    clsid: GUID,
) -> Result<()> {
    let (key, _) = classes.create_subkey(subkey).map_err(io_to_winerr)?;
    key.set_value("MUIVerb", &title).map_err(io_to_winerr)?;
    key.set_value("Icon", &icon_path).map_err(io_to_winerr)?;
    key.set_value("MultiSelectModel", &"Player")
        .map_err(io_to_winerr)?;
    key.set_value("ExplorerCommandHandler", &format!("{{{}}}", clsid_string(clsid)))
        .map_err(io_to_winerr)?;
    let (command_key, _) = key.create_subkey("command").map_err(io_to_winerr)?;
    command_key
        .set_value("DelegateExecute", &format!("{{{}}}", clsid_string(clsid)))
        .map_err(io_to_winerr)?;
    command_key
        .set_value("", &"")
        .map_err(io_to_winerr)?;
    Ok(())
}

fn io_to_winerr(error: std::io::Error) -> Error {
    Error::new(E_FAIL.into(), error.to_string())
}

fn clsid_string(clsid: GUID) -> String {
    let d4 = clsid.data4;
    format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        clsid.data1,
        clsid.data2,
        clsid.data3,
        d4[0],
        d4[1],
        d4[2],
        d4[3],
        d4[4],
        d4[5],
        d4[6],
        d4[7]
    )
}
