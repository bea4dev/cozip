#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use cozip_win_installer::i18n::I18n;
use gpui::{
    App, AppContext, Application, Bounds, Context, FontWeight, IntoElement, ParentElement,
    Render, SharedString, Styled, Timer, Window, WindowBackgroundAppearance, WindowBounds,
    WindowOptions, div, prelude::*, px, rgb, size,
};
use windows_sys::Win32::Foundation::{CloseHandle, FreeLibrary, HANDLE, HWND};
use windows_sys::Win32::Security::{
    GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation,
};
use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_DELAY_UNTIL_REBOOT, MoveFileExW};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows_sys::Win32::System::Registry::{
    HKEY_CURRENT_USER, RegDeleteTreeW,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
use windows_sys::Win32::UI::Shell::ShellExecuteW;

const INSTALL_DIR: &str = r"C:\Program Files\CoZip";
const COZIP_WIN_SHELL_DLL_PATH: &str = r"C:\Program Files\CoZip\cozip_win_shell.dll";
const COZIP_UNINSTALL_REG_KEY: &str =
    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CoZip";
const UNREGISTER_PROGRESS_END: u8 = 34;
const REMOVE_PROGRESS_END: u8 = 100;

const CLSID_COMPRESS_ROOT: &str = "{7f742be7-3d6d-4fa6-8d7c-68d7c6297141}";
const CLSID_EXTRACT_ROOT: &str = "{fcd0419f-b0e2-4308-b3d2-0eaf643354a3}";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UninstallerStep {
    Confirm,
    Uninstalling,
    Complete,
}

enum UninstallWorkerMessage {
    Progress(u8),
    Finished(Result<Option<String>, String>),
}

struct UninstallerApp {
    i18n: I18n,
    step: UninstallerStep,
    uninstall_running: bool,
    uninstall_progress: u8,
    uninstall_note: Option<String>,
}

impl UninstallerApp {
    fn new(i18n: I18n) -> Self {
        Self {
            i18n,
            step: UninstallerStep::Confirm,
            uninstall_running: false,
            uninstall_progress: 0,
            uninstall_note: None,
        }
    }

    fn t(&self, key: &str) -> SharedString {
        self.i18n.text(key).to_owned().into()
    }

    fn continue_label(&self) -> SharedString {
        match self.step {
            UninstallerStep::Confirm => self.t("buttons.uninstall"),
            UninstallerStep::Complete => self.t("buttons.finish"),
            UninstallerStep::Uninstalling => self.t("buttons.uninstall"),
        }
    }

    fn back_label(&self) -> SharedString {
        match self.step {
            UninstallerStep::Confirm => self.t("buttons.cancel"),
            UninstallerStep::Uninstalling | UninstallerStep::Complete => self.t("buttons.back"),
        }
    }

    fn can_go_back(&self) -> bool {
        self.step == UninstallerStep::Confirm && !self.uninstall_running
    }

    fn can_continue(&self) -> bool {
        match self.step {
            UninstallerStep::Confirm => !self.uninstall_running,
            UninstallerStep::Complete => true,
            UninstallerStep::Uninstalling => false,
        }
    }

    fn footer_text(&self) -> SharedString {
        match self.step {
            UninstallerStep::Confirm => self.t("footer.uninstall_confirm"),
            UninstallerStep::Uninstalling => self.t("footer.uninstalling"),
            UninstallerStep::Complete => self.t("footer.uninstall_complete"),
        }
    }

    fn next(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.step {
            UninstallerStep::Confirm => self.start_uninstall(window, cx),
            UninstallerStep::Complete => cx.quit(),
            UninstallerStep::Uninstalling => {}
        }
    }

    fn back(&mut self, cx: &mut Context<Self>) {
        if self.step == UninstallerStep::Confirm {
            cx.quit();
        }
    }

    fn start_uninstall(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.uninstall_running {
            return;
        }

        self.step = UninstallerStep::Uninstalling;
        self.uninstall_running = true;
        self.uninstall_progress = 0;
        self.uninstall_note = None;

        let (sender, receiver) = mpsc::channel::<UninstallWorkerMessage>();
        thread::spawn(move || {
            let result = perform_uninstall(|progress| {
                let _ = sender.send(UninstallWorkerMessage::Progress(progress));
            });
            let _ = sender.send(UninstallWorkerMessage::Finished(result));
        });

        let entity = cx.entity().clone();
        window
            .spawn(cx, async move |cx| {
                loop {
                    let mut latest_progress = None;
                    let mut finished = None;
                    while let Ok(message) = receiver.try_recv() {
                        match message {
                            UninstallWorkerMessage::Progress(progress) => {
                                latest_progress = Some(progress);
                            }
                            UninstallWorkerMessage::Finished(result) => {
                                finished = Some(result);
                                break;
                            }
                        }
                    }

                    if let Some(progress) = latest_progress {
                        let _ = entity.update(cx, |this, cx| {
                            this.uninstall_progress = progress;
                            cx.notify();
                        });
                    }

                    if let Some(result) = finished {
                        let _ = entity.update(cx, |this, cx| {
                            this.uninstall_progress = 100;
                            this.uninstall_running = false;
                            this.step = UninstallerStep::Complete;
                            this.uninstall_note = match result {
                                Ok(note) => note,
                                Err(error) => Some(error),
                            };
                            cx.notify();
                        });
                        break;
                    }

                    Timer::after(Duration::from_millis(120)).await;
                }
            })
            .detach();
    }

    fn uninstall_status(&self) -> SharedString {
        let key = match self.uninstall_progress {
            0..=9 => "uninstall.status_preparing",
            10..=34 => "uninstall.status_unregistering",
            35..=99 => "uninstall.status_removing_files",
            _ => "uninstall.status_completed",
        };
        self.t(key)
    }

    fn header(&self) -> impl IntoElement {
        div()
            .px_6()
            .py_5()
            .bg(rgb(0xffffff))
            .rounded_t_xl()
            .border_b_1()
            .border_color(rgb(0xd4d4d8))
            .gap_1()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0x111827))
                    .child(self.t("header.uninstall_title")),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x4b5563))
                    .child(self.t("header.uninstall_subtitle")),
            )
    }

    fn content_header(&self, title: SharedString, description: SharedString) -> impl IntoElement {
        div()
            .gap_1()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_lg()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(0x111827))
                    .child(title),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x4b5563))
                    .child(description),
            )
    }

    fn confirm_screen(&self) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(
                self.t("uninstall.confirm_title"),
                self.t("uninstall.confirm_description"),
            ))
            .child(value_row(self.t("common.install_dir"), INSTALL_DIR.into()))
            .child(
                div()
                    .w_full()
                    .bg(rgb(0xfffbeb))
                    .border_1()
                    .rounded_lg()
                    .border_color(rgb(0xf59e0b))
                    .px_4()
                    .py_3()
                    .text_sm()
                    .text_color(rgb(0x92400e))
                    .child(self.t("uninstall.confirm_warning")),
            )
    }

    fn uninstall_screen(&self) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(
                self.t("uninstall.progress_title"),
                self.t("uninstall.progress_description"),
            ))
            .child(
                div()
                    .w_full()
                    .bg(rgb(0xffffff))
                    .border_1()
                    .rounded_lg()
                    .border_color(rgb(0xd4d4d8))
                    .p_5()
                    .gap_4()
                    .flex()
                    .flex_col()
                    .child(simple_progress_bar(self.uninstall_progress))
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x111827))
                                    .child(self.uninstall_status()),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(0x111827))
                                    .child(format!("{}%", self.uninstall_progress)),
                            ),
                    ),
            )
    }

    fn complete_screen(&self) -> impl IntoElement {
        div()
            .w_full()
            .h_full()
            .min_h(px(0.0))
            .gap_4()
            .flex()
            .flex_col()
            .child(self.content_header(
                self.t("uninstall.complete_title"),
                self.t("uninstall.complete_description"),
            ))
            .child(value_row(self.t("common.install_dir"), INSTALL_DIR.into()))
            .child(value_row(
                self.t("complete.status_label"),
                self.t("uninstall.complete_state"),
            ))
            .when_some(self.uninstall_note.as_ref(), |this, note| {
                this.child(value_row(self.t("complete.note"), note.clone().into()))
            })
    }

    fn button(
        &self,
        id: &'static str,
        label: SharedString,
        enabled: bool,
        primary: bool,
        on_click: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let mut button = div()
            .id(SharedString::from(id))
            .min_w(px(92.0))
            .px_4()
            .py_2()
            .border_1()
            .rounded_lg()
            .border_color(if enabled {
                rgb(0xa1a1aa)
            } else {
                rgb(0xd4d4d8)
            })
            .bg(if primary && enabled {
                rgb(0xf3f4f6)
            } else {
                rgb(0xffffff)
            })
            .text_sm()
            .text_color(if enabled {
                rgb(0x111827)
            } else {
                rgb(0x9ca3af)
            })
            .font_weight(FontWeight::MEDIUM)
            .child(label);

        if enabled {
            button = button
                .cursor_pointer()
                .hover(|style| style.bg(rgb(0xf3f4f6)))
                .on_click(cx.listener(move |this, _, window, cx| {
                    on_click(this, window, cx);
                }));
        }

        button
    }
}

impl Render for UninstallerApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let content = match self.step {
            UninstallerStep::Confirm => self.confirm_screen().into_any_element(),
            UninstallerStep::Uninstalling => self.uninstall_screen().into_any_element(),
            UninstallerStep::Complete => self.complete_screen().into_any_element(),
        };

        div()
            .size_full()
            .bg(rgb(0xffffff))
            .child(
                div()
                    .size_full()
                    .bg(rgb(0xffffff))
                    .flex()
                    .flex_col()
                    .child(self.header())
                    .child(
                        div()
                            .flex_grow()
                            .min_h(px(0.0))
                            .px_6()
                            .pt_5()
                            .pb_8()
                            .bg(rgb(0xffffff))
                            .child(div().w_full().h_full().min_h(px(0.0)).pb_2().child(content)),
                    )
                    .child(
                        div()
                            .px_6()
                            .py_5()
                            .border_t_1()
                            .border_color(rgb(0xd4d4d8))
                            .bg(rgb(0xf9fafb))
                            .flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div()
                                    .max_w(px(360.0))
                                    .pr_4()
                                    .text_sm()
                                    .text_color(rgb(0x6b7280))
                                    .child(self.footer_text()),
                            )
                            .child(
                                div()
                                    .flex()
                                    .gap_2()
                                    .when(self.can_go_back(), |this| {
                                        this.child(self.button(
                                            "back-button",
                                            self.back_label(),
                                            true,
                                            false,
                                            |this, _, cx| this.back(cx),
                                            cx,
                                        ))
                                    })
                                    .child(self.button(
                                        "next-button",
                                        self.continue_label(),
                                        self.can_continue(),
                                        true,
                                        |this, window, cx| this.next(window, cx),
                                        cx,
                                    )),
                            ),
                    ),
            )
    }
}

fn value_row(label: SharedString, value: SharedString) -> impl IntoElement {
    div()
        .w_full()
        .bg(rgb(0xffffff))
        .border_1()
        .rounded_lg()
        .border_color(rgb(0xd4d4d8))
        .px_4()
        .py_3()
        .flex()
        .justify_between()
        .gap_4()
        .child(div().text_sm().text_color(rgb(0x4b5563)).child(label))
        .child(div().text_sm().text_color(rgb(0x111827)).child(value))
}

fn simple_progress_bar(progress: u8) -> impl IntoElement {
    let width = (f32::from(progress) / 100.0) * 560.0;
    div()
        .w_full()
        .max_w(px(560.0))
        .h(px(18.0))
        .border_1()
        .rounded_full()
        .overflow_hidden()
        .border_color(rgb(0xa1a1aa))
        .bg(rgb(0xffffff))
        .child(
            div()
                .h_full()
                .w(px(width))
                .bg(rgb(0x2563eb))
                .rounded_full(),
        )
}

fn perform_uninstall(mut on_progress: impl FnMut(u8)) -> Result<Option<String>, String> {
    let mut last_progress = 0_u8;
    let mut report_progress = |progress: u8| {
        if progress > last_progress {
            last_progress = progress;
            on_progress(progress);
        }
    };

    report_progress(1);
    uninstall_explorer_menu_entries()?;
    report_progress(UNREGISTER_PROGRESS_END / 2);

    uninstall_archive_file_associations()?;
    report_progress(UNREGISTER_PROGRESS_END);

    let note = remove_installation_files(|done, total| {
        report_progress(scale_progress(
            UNREGISTER_PROGRESS_END + 1,
            REMOVE_PROGRESS_END,
            done,
            total,
        ));
    })?;

    report_progress(REMOVE_PROGRESS_END);
    Ok(note)
}

fn remove_installation_files(mut on_progress: impl FnMut(u64, u64)) -> Result<Option<String>, String> {
    let install_dir = Path::new(INSTALL_DIR);
    if !install_dir.exists() {
        on_progress(1, 1);
        return Ok(None);
    }

    let current_exe = std::env::current_exe().ok();
    let current_exe_in_install_dir = current_exe
        .as_ref()
        .is_some_and(|path| path.starts_with(install_dir));

    let mut files = Vec::new();
    let mut dirs = Vec::new();
    collect_install_entries(install_dir, &mut files, &mut dirs)?;

    let total = (files.len() + dirs.len()).max(1) as u64;
    let mut done = 0_u64;
    let mut restart_required = false;

    for file in &files {
        let should_defer = current_exe
            .as_ref()
            .is_some_and(|exe| paths_equal(file, exe));
        if should_defer {
            schedule_delete_on_reboot(file)?;
            restart_required = true;
        } else if let Err(error) = fs::remove_file(file) {
            if file.exists() {
                schedule_delete_on_reboot(file).map_err(|schedule_error| {
                    format!(
                        "failed to remove {} ({error}) and failed to schedule deletion: {schedule_error}",
                        file.display()
                    )
                })?;
                restart_required = true;
            }
        }
        done = done.saturating_add(1);
        on_progress(done, total);
    }

    dirs.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for dir in &dirs {
        if let Err(error) = fs::remove_dir(dir) {
            if dir.exists() {
                schedule_delete_on_reboot(dir).map_err(|schedule_error| {
                    format!(
                        "failed to remove directory {} ({error}) and failed to schedule deletion: {schedule_error}",
                        dir.display()
                    )
                })?;
                restart_required = true;
            }
        }
        done = done.saturating_add(1);
        on_progress(done, total);
    }

    if current_exe_in_install_dir && install_dir.exists() {
        let _ = schedule_delete_on_reboot(install_dir).map(|_| {
            restart_required = true;
        });
    }

    Ok(if restart_required {
        Some(
            I18n::load()
                .text("uninstall.restart_required_note")
                .to_string(),
        )
    } else {
        None
    })
}

fn collect_install_entries(
    dir: &Path,
    files: &mut Vec<PathBuf>,
    dirs: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir)
        .map_err(|error| format!("failed to read {}: {error}", dir.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to inspect {}: {error}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
        if file_type.is_dir() {
            collect_install_entries(&path, files, dirs)?;
            dirs.push(path);
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn scale_progress(start: u8, end: u8, completed: u64, total: u64) -> u8 {
    if end <= start || total == 0 {
        return end;
    }
    let span = u64::from(end - start);
    let bounded = completed.min(total);
    let offset = (bounded.saturating_mul(span) / total) as u8;
    start.saturating_add(offset).min(end)
}

fn uninstall_explorer_menu_entries() -> Result<(), String> {
    let _ = call_shell_extension_registration(COZIP_WIN_SHELL_DLL_PATH, b"DllUnregisterServer\0");
    uninstall_shell_extension_registry_entries()?;
    uninstall_legacy_static_explorer_menu_entries()?;
    Ok(())
}

fn uninstall_archive_file_associations() -> Result<(), String> {
    reg_delete_tree(r"HKCU\Software\Classes\.cozip")?;
    reg_delete_tree(r"HKCU\Software\Classes\CoZip.Archive")?;
    reg_delete_tree(COZIP_UNINSTALL_REG_KEY)?;
    Ok(())
}

fn uninstall_shell_extension_registry_entries() -> Result<(), String> {
    let keys = [
        r"HKCU\Software\Classes\AllFilesystemObjects\shell\CozipCompress",
        r"HKCU\Software\Classes\AllFilesystemObjects\shell\CozipExtract",
    ];
    for key in keys {
        reg_delete_tree(key)?;
    }
    reg_delete_tree(&format!(r"HKCU\Software\Classes\CLSID\{CLSID_COMPRESS_ROOT}"))?;
    reg_delete_tree(&format!(r"HKCU\Software\Classes\CLSID\{CLSID_EXTRACT_ROOT}"))?;
    Ok(())
}

fn uninstall_legacy_static_explorer_menu_entries() -> Result<(), String> {
    let keys = [
        r"HKCU\Software\Classes\AllFilesystemObjects\shell\CozipCompress",
        r"HKCU\Software\Classes\AllFilesystemObjects\shell\CozipExtract",
        r"HKCU\Software\Classes\CoZip.ContextMenus\Compress",
        r"HKCU\Software\Classes\CoZip.ContextMenus\Extract",
    ];
    for key in keys {
        reg_delete_tree(key)?;
    }
    Ok(())
}

fn schedule_delete_on_reboot(path: &Path) -> Result<(), String> {
    let path_wide = to_wide(path.as_os_str());
    let ok = unsafe {
        MoveFileExW(
            path_wide.as_ptr(),
            std::ptr::null(),
            MOVEFILE_DELAY_UNTIL_REBOOT,
        )
    };
    if ok == 0 {
        Err(format!(
            "failed to schedule deletion on reboot: {}",
            path.display()
        ))
    } else {
        Ok(())
    }
}

fn reg_delete_tree(key: &str) -> Result<(), String> {
    let subkey = strip_hkcu_prefix(key)?;
    let subkey_wide = to_wide(OsStr::new(subkey));
    let status = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, subkey_wide.as_ptr()) };
    match status {
        0 | 2 | 3 => Ok(()),
        code => Err(format!("registry delete failed for {key}: win32 error {code}")),
    }
}

fn call_shell_extension_registration(dll_path: &str, proc_name: &[u8]) -> Result<(), String> {
    if !Path::new(dll_path).exists() {
        return Ok(());
    }

    let dll_wide = to_wide(OsStr::new(dll_path));
    let module = unsafe { LoadLibraryW(dll_wide.as_ptr()) };
    if module.is_null() {
        return Err(format!("failed to load shell extension dll: {dll_path}"));
    }

    let result = (|| {
        let proc = unsafe { GetProcAddress(module, proc_name.as_ptr()) };
        if proc.is_none() {
            return Err("shell extension registration entry point was not found".to_string());
        }
        let callback: unsafe extern "system" fn() -> i32 = unsafe { std::mem::transmute(proc) };
        let hr = unsafe { callback() };
        if hr >= 0 {
            Ok(())
        } else {
            Err(format!("shell extension registration failed: HRESULT 0x{hr:08X}"))
        }
    })();

    unsafe {
        FreeLibrary(module);
    }
    result
}

fn ensure_elevated() -> Result<(), String> {
    if is_process_elevated()? {
        return Ok(());
    }

    let exe_path =
        std::env::current_exe().map_err(|error| format!("current_exe failed: {error}"))?;
    let current_dir =
        std::env::current_dir().map_err(|error| format!("current_dir failed: {error}"))?;
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| quote_windows_arg(&arg))
        .collect::<Vec<_>>()
        .join(" ");

    let exe_wide = to_wide(exe_path.as_os_str());
    let verb_wide = to_wide(OsStr::new("runas"));
    let dir_wide = to_wide(current_dir.as_os_str());
    let args_wide = if args.is_empty() {
        Vec::new()
    } else {
        to_wide(OsStr::new(&args))
    };

    let result = unsafe {
        ShellExecuteW(
            0 as HWND,
            verb_wide.as_ptr(),
            exe_wide.as_ptr(),
            if args_wide.is_empty() {
                std::ptr::null()
            } else {
                args_wide.as_ptr()
            },
            dir_wide.as_ptr(),
            1,
        )
    } as isize;

    if result <= 32 {
        return Err(format!("ShellExecuteW failed with code {result}"));
    }

    std::process::exit(0);
}

fn is_process_elevated() -> Result<bool, String> {
    let mut token: HANDLE = std::ptr::null_mut();
    let opened = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
    if opened == 0 {
        return Err("OpenProcessToken failed".to_string());
    }

    let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
    let mut returned_len = 0_u32;
    let ok = unsafe {
        GetTokenInformation(
            token,
            TokenElevation,
            &mut elevation as *mut _ as *mut _,
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut returned_len,
        )
    };
    unsafe {
        CloseHandle(token);
    }

    if ok == 0 {
        return Err("GetTokenInformation(TokenElevation) failed".to_string());
    }

    Ok(elevation.TokenIsElevated != 0)
}

fn strip_hkcu_prefix(key: &str) -> Result<&str, String> {
    key.strip_prefix(r"HKCU\")
        .ok_or_else(|| format!("unsupported registry hive path: {key}"))
}

fn to_wide(value: &OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    value.encode_wide().chain(std::iter::once(0)).collect()
}

fn quote_windows_arg(arg: &OsStr) -> String {
    let value = arg.to_string_lossy();
    if !value.contains([' ', '\t', '"']) {
        return value.into_owned();
    }

    let mut quoted = String::from("\"");
    let mut backslashes = 0_usize;
    for ch in value.chars() {
        match ch {
            '\\' => backslashes += 1,
            '"' => {
                quoted.push_str(&"\\".repeat((backslashes * 2) + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                if backslashes > 0 {
                    quoted.push_str(&"\\".repeat(backslashes));
                    backslashes = 0;
                }
                quoted.push(ch);
            }
        }
    }
    if backslashes > 0 {
        quoted.push_str(&"\\".repeat(backslashes * 2));
    }
    quoted.push('"');
    quoted
}

fn paths_equal(left: &Path, right: &Path) -> bool {
    left.as_os_str()
        .to_string_lossy()
        .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
}

fn main() {
    if let Err(error) = ensure_elevated() {
        eprintln!("failed to elevate uninstaller: {error}");
        std::process::exit(1);
    }

    let i18n = I18n::load();

    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(720.0), px(520.0)), cx);
        let i18n = i18n.clone();
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(Default::default()),
                window_background: WindowBackgroundAppearance::Opaque,
                window_min_size: Some(size(px(680.0), px(480.0))),
                app_id: Some("cozip-uninstaller".to_string()),
                ..Default::default()
            },
            move |_, cx| cx.new(|_| UninstallerApp::new(i18n.clone())),
        )
        .expect("failed to open uninstaller window");
        cx.activate(true);
    });
}
