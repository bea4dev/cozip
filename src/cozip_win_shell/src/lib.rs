#![cfg(target_os = "windows")]

mod class_ids;
mod command;
mod factory;
mod i18n;
mod launch;
mod registration;

use std::ffi::c_void;

use factory::ShellClassFactory;
use windows::Win32::Foundation::{BOOL, HINSTANCE, S_FALSE, S_OK};
use windows::core::{GUID, HRESULT, Interface};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(
    _hinst_dll: HINSTANCE,
    _fdw_reason: u32,
    _reserved: *mut c_void,
) -> BOOL {
    BOOL(1)
}

#[unsafe(no_mangle)]
pub extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

#[unsafe(no_mangle)]
pub extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    if rclsid.is_null() || ppv.is_null() {
        return windows::Win32::Foundation::E_POINTER;
    }

    let clsid = unsafe { *rclsid };
    match ShellClassFactory::new(clsid) {
        Ok(factory) => unsafe { factory.query(riid, ppv) },
        Err(error) => error.code(),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn DllRegisterServer() -> HRESULT {
    match registration::register_server() {
        Ok(()) => S_OK,
        Err(error) => error.code(),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn DllUnregisterServer() -> HRESULT {
    match registration::unregister_server() {
        Ok(()) => S_OK,
        Err(error) => error.code(),
    }
}
