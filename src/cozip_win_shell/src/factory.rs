use windows::Win32::Foundation::{CLASS_E_CLASSNOTAVAILABLE, E_NOINTERFACE};
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl};
use windows::core::{GUID, IUnknown, Interface, Result, implement};

use crate::class_ids::{CLSID_COMPRESS_ROOT, CLSID_EXTRACT_ROOT};
use crate::command::{ExplorerCommand, ExplorerVerb};

#[implement(IClassFactory)]
pub struct ShellClassFactory {
    clsid: GUID,
}

impl ShellClassFactory {
    pub fn new(clsid: GUID) -> Result<IClassFactory> {
        if clsid == CLSID_COMPRESS_ROOT || clsid == CLSID_EXTRACT_ROOT {
            Ok(Self { clsid }.into())
        } else {
            Err(CLASS_E_CLASSNOTAVAILABLE.into())
        }
    }
}

impl IClassFactory_Impl for ShellClassFactory {
    fn CreateInstance(
        &self,
        outer: Option<&IUnknown>,
        iid: *const GUID,
        object: *mut *mut core::ffi::c_void,
    ) -> Result<()> {
        if outer.is_some() {
            return Err(windows::Win32::Foundation::CLASS_E_NOAGGREGATION.into());
        }

        let command = match self.clsid {
            CLSID_COMPRESS_ROOT => ExplorerCommand::new(ExplorerVerb::CompressRoot),
            CLSID_EXTRACT_ROOT => ExplorerCommand::new(ExplorerVerb::ExtractRoot),
            _ => return Err(CLASS_E_CLASSNOTAVAILABLE.into()),
        };

        let unknown: IUnknown = command.into();
        let hr = unsafe { unknown.query(iid, object) };
        if hr.is_ok() {
            Ok(())
        } else {
            Err(E_NOINTERFACE.into())
        }
    }

    fn LockServer(&self, _increment: BOOL) -> Result<()> {
        Ok(())
    }
}

use windows::Win32::Foundation::BOOL;
