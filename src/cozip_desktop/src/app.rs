use std::path::{Path, PathBuf};
use std::time::Duration;

use cozip::ZipDeflateMode;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Timer, Window, div, px, rgb,
};
use rfd::{FileDialog, MessageButtons, MessageDialog, MessageDialogResult, MessageLevel};

use crate::i18n::I18n;
use crate::jobs::{JobSnapshot, JobStatus, SharedJobSnapshot, spawn_job};
use crate::launch::{
    ArchiveFormat, CompressMode, CompressPlan, DesktopCommand, ExtractDecodeHint, ExtractPlan,
    InitialScreen, LaunchRequest, cycle_archive_format,
};
use crate::screens::widgets::{action_button, labeled_value, panel, progress_bar, separator};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScreenKind {
    Compress,
    Decompress,
    CompressSettings,
    DecompressSettings,
}

pub struct CozipDesktopApp {
    i18n: I18n,
    launch: LaunchRequest,
    command: Option<DesktopCommand>,
    active_screen: ScreenKind,
    job: Option<SharedJobSnapshot>,
    job_snapshot: JobSnapshot,
    displayed_progress: f32,
    displayed_throughput: f64,
    poll_started: bool,
    auto_start_consumed: bool,
}

impl CozipDesktopApp {
    pub fn new(launch: LaunchRequest) -> Self {
        let i18n = I18n::load();
        Self {
            i18n,
            command: launch.command.clone(),
            active_screen: match launch.initial_screen {
                InitialScreen::Compress => ScreenKind::Compress,
                InitialScreen::Decompress => ScreenKind::Decompress,
                InitialScreen::CompressSettings => ScreenKind::CompressSettings,
                InitialScreen::DecompressSettings => ScreenKind::DecompressSettings,
            },
            launch,
            job: None,
            job_snapshot: JobSnapshot::default(),
            displayed_progress: 0.0,
            displayed_throughput: 0.0,
            poll_started: false,
            auto_start_consumed: false,
        }
    }

    fn t(&self, key: &str) -> SharedString {
        self.i18n.text(key).to_owned().into()
    }

    fn start_polling_if_needed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.poll_started {
            return;
        }

        self.poll_started = true;
        let entity = cx.entity().clone();
        window.spawn(cx, async move |cx| loop {
            Timer::after(Duration::from_millis(33)).await;
            if entity
                .update(cx, |this, cx| {
                    this.refresh_job_snapshot();
                    this.step_displayed_progress();
                    cx.notify();
                })
                .is_err()
            {
                break;
            }
        })
        .detach();
    }

    fn maybe_start_from_launch(&mut self) {
        if self.auto_start_consumed || !self.launch.auto_start {
            return;
        }
        self.auto_start_consumed = true;
        self.start_command();
    }

    fn refresh_job_snapshot(&mut self) {
        let Some(shared) = &self.job else {
            return;
        };
        self.job_snapshot = shared.lock().expect("job snapshot poisoned").clone();
        self.update_displayed_throughput();
    }

    fn update_displayed_throughput(&mut self) {
        let current = self
            .job_snapshot
            .progress
            .as_ref()
            .map(|progress| progress.snapshot().throughput_bytes_per_sec)
            .unwrap_or(0.0);
        let finished = matches!(
            self.job_snapshot.status,
            JobStatus::Succeeded | JobStatus::Failed
        );

        if matches!(self.job_snapshot.status, JobStatus::Running) && !finished {
            self.displayed_throughput = current;
        } else if self.displayed_throughput <= 0.0 {
            self.displayed_throughput = current;
        }
    }

    fn step_displayed_progress(&mut self) {
        let target = self.target_progress_fraction();
        if matches!(self.job_snapshot.status, JobStatus::Succeeded) {
            self.displayed_progress = 1.0;
            return;
        }
        if matches!(self.job_snapshot.status, JobStatus::Failed) {
            self.displayed_progress = target;
            return;
        }

        if target <= self.displayed_progress {
            self.displayed_progress = target;
            return;
        }

        let delta = target - self.displayed_progress;
        let next = self.displayed_progress + delta * 0.22 + 0.002;
        self.displayed_progress = next.min(target).min(0.999);
    }

    fn start_command(&mut self) {
        if matches!(self.job_snapshot.status, JobStatus::Running) {
            return;
        }
        let Some(command) = self.command.clone() else {
            return;
        };
        self.displayed_progress = 0.0;
        self.displayed_throughput = 0.0;
        self.job = Some(spawn_job(command));
        self.refresh_job_snapshot();
        self.step_displayed_progress();
    }

    fn nav_item(
        &self,
        label_key: &'static str,
        target: ScreenKind,
        active_screen: ScreenKind,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let active = active_screen == target;
        div()
            .id(SharedString::from(format!("nav-{target:?}")))
            .px_3()
            .py_2()
            .rounded_md()
            .cursor_pointer()
            .bg(if active { rgb(0xdbeafe) } else { rgb(0xf8fafc) })
            .border_1()
            .border_color(if active { rgb(0x93c5fd) } else { rgb(0xe2e8f0) })
            .text_color(if active { rgb(0x0f172a) } else { rgb(0x475569) })
            .text_sm()
            .font_weight(FontWeight::MEDIUM)
            .on_click(cx.listener(move |this, _, _, _| {
                this.active_screen = target;
            }))
            .child(self.t(label_key))
    }

    fn shell(&self, content: AnyElement, cx: &mut Context<Self>) -> impl IntoElement {
        if self.command.is_some() {
            return div()
                .size_full()
                .bg(rgb(0xf3f4f6))
                .text_color(rgb(0x0f172a))
                .child(
                    div()
                        .size_full()
                        .p_6()
                        .child(div().max_w(px(720.0)).mx_auto().child(content)),
                );
        }

        let active = self.active_screen;
        div()
            .size_full()
            .bg(rgb(0xf3f4f6))
            .text_color(rgb(0x0f172a))
            .child(
                div()
                    .size_full()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .px_6()
                            .py_5()
                            .bg(rgb(0xffffff))
                            .border_b_1()
                            .border_color(rgb(0xe5e7eb))
                            .gap_4()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .gap_1()
                                    .flex()
                                    .flex_col()
                                    .child(
                                        div()
                                            .text_xl()
                                            .font_weight(FontWeight::BOLD)
                                            .child(self.t("app.title")),
                                    )
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(rgb(0x64748b))
                                            .child(self.t("app.subtitle")),
                                    ),
                            )
                            .child(
                                div()
                                    .gap_2()
                                    .flex()
                                    .flex_row()
                                    .flex_wrap()
                                    .child(self.nav_item("nav.compress", ScreenKind::Compress, active, cx))
                                    .child(self.nav_item("nav.decompress", ScreenKind::Decompress, active, cx))
                                    .child(self.nav_item(
                                        "nav.compress_settings",
                                        ScreenKind::CompressSettings,
                                        active,
                                        cx,
                                    ))
                                    .child(self.nav_item(
                                        "nav.decompress_settings",
                                        ScreenKind::DecompressSettings,
                                        active,
                                        cx,
                                    )),
                            ),
                    )
                    .child(
                        div()
                            .id("content-scroll")
                            .flex_grow()
                            .p_6()
                            .overflow_y_scroll()
                            .child(div().max_w(px(980.0)).mx_auto().child(content)),
                    ),
            )
    }

    fn command(&self) -> Option<&DesktopCommand> {
        self.command.as_ref()
    }

    fn compress_plan(&self) -> Option<&CompressPlan> {
        match self.command() {
            Some(DesktopCommand::Compress(plan)) => Some(plan),
            _ => None,
        }
    }

    fn extract_plan(&self) -> Option<&ExtractPlan> {
        match self.command() {
            Some(DesktopCommand::Extract(plan)) => Some(plan),
            _ => None,
        }
    }

    fn compress_plan_mut(&mut self) -> Option<&mut CompressPlan> {
        match self.command.as_mut() {
            Some(DesktopCommand::Compress(plan)) => Some(plan),
            _ => None,
        }
    }

    fn extract_plan_mut(&mut self) -> Option<&mut ExtractPlan> {
        match self.command.as_mut() {
            Some(DesktopCommand::Extract(plan)) => Some(plan),
            _ => None,
        }
    }

    fn compress_output_warning(&self) -> Option<String> {
        let plan = self.compress_plan()?;
        if plan.output_path.exists() {
            Some(format!(
                "{} {}",
                self.i18n.text("warning.output_exists"),
                plan.output_path.display()
            ))
        } else {
            None
        }
    }

    fn extract_output_warning(&self) -> Option<String> {
        let plan = self.extract_plan()?;
        let conflicts = plan
            .tasks
            .iter()
            .map(|task| plan.output_path_for(task))
            .filter(|path| path.exists())
            .collect::<Vec<_>>();

        if conflicts.is_empty() {
            None
        } else {
            Some(format!(
                "{} ({})",
                self.i18n.text("warning.extract_exists"),
                conflicts.len()
            ))
        }
    }

    fn browse_compress_output(&mut self) {
        let (format, current_output, title) = match self.compress_plan() {
            Some(plan) => (
                plan.format,
                plan.output_path.clone(),
                self.i18n.text("dialog.compress_output").to_string(),
            ),
            None => return,
        };
        let extension = match format {
            ArchiveFormat::Zip => "zip",
            ArchiveFormat::Cozip => "pdz",
        };
        let file_name = current_output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("archive");

        let dialog = FileDialog::new()
            .set_title(&title)
            .set_file_name(file_name)
            .add_filter(extension, &[extension]);

        let selected = if let Some(parent) = current_output.parent() {
            dialog.set_directory(parent).save_file()
        } else {
            dialog.save_file()
        };
        let Some(path) = selected else {
            return;
        };
        if self.confirm_overwrite(&path) {
            if let Some(plan) = self.compress_plan_mut() {
                plan.output_path = path;
            }
        }
    }

    fn browse_extract_output(&mut self) {
        let (current_output, title) = match self.extract_plan() {
            Some(plan) => (
                plan.output_dir.clone(),
                self.i18n.text("dialog.extract_output").to_string(),
            ),
            None => return,
        };
        let dialog = FileDialog::new()
            .set_title(&title)
            .set_directory(&current_output);
        if let Some(path) = dialog.pick_folder() {
            if let Some(plan) = self.extract_plan_mut() {
                plan.output_dir = path;
            }
        }
    }

    fn confirm_overwrite(&self, path: &Path) -> bool {
        if !path.exists() {
            return true;
        }

        MessageDialog::new()
            .set_title(self.i18n.text("dialog.overwrite_title"))
            .set_description(format!(
                "{}\n{}",
                self.i18n.text("dialog.overwrite_body"),
                path.display()
            ))
            .set_level(MessageLevel::Warning)
            .set_buttons(MessageButtons::OkCancel)
            .show()
            == MessageDialogResult::Ok
    }

    fn confirm_extract_conflicts(&self) -> bool {
        let Some(plan) = self.extract_plan() else {
            return true;
        };

        let conflicts = plan
            .tasks
            .iter()
            .map(|task| plan.output_path_for(task))
            .filter(|path| path.exists())
            .collect::<Vec<_>>();
        if conflicts.is_empty() {
            return true;
        }

        MessageDialog::new()
            .set_title(self.i18n.text("dialog.overwrite_title"))
            .set_description(format!(
                "{} ({})",
                self.i18n.text("dialog.extract_overwrite_body"),
                conflicts.len()
            ))
            .set_level(MessageLevel::Warning)
            .set_buttons(MessageButtons::OkCancel)
            .show()
            == MessageDialogResult::Ok
    }

    fn banner_panel(&self, text: SharedString, error: bool) -> impl IntoElement {
        panel(
            if error {
                self.t("status.error")
            } else {
                self.t("status.notice")
            },
            div()
                .text_sm()
                .text_color(if error { rgb(0x991b1b) } else { rgb(0x475569) })
                .child(text),
        )
    }

    fn render_compress_screen(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.command.is_some() {
            let mut content = div().gap_4().flex().flex_col();
            if let Some(error) = &self.launch.startup_error {
                content = content.child(self.banner_panel(error.clone().into(), true));
            }
            if self.compress_plan().is_some() {
                content = content.child(self.operation_panel(window, cx, true));
            } else {
                content = content.child(self.empty_state_panel("compress.empty"));
            }
            return content;
        }

        let mut content = div().gap_4().flex().flex_col();

        if let Some(error) = &self.launch.startup_error {
            content = content.child(self.banner_panel(error.clone().into(), true));
        }

        if let Some(plan) = self.compress_plan() {
            content = content.child(self.compress_summary_panel(plan));
            content = content.child(self.operation_panel(window, cx, true));
        } else {
            content = content.child(self.empty_state_panel("compress.empty"));
        }

        content
    }

    fn render_decompress_screen(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        if self.command.is_some() {
            let mut content = div().gap_4().flex().flex_col();
            if let Some(error) = &self.launch.startup_error {
                content = content.child(self.banner_panel(error.clone().into(), true));
            }
            if self.extract_plan().is_some() {
                content = content.child(self.operation_panel(window, cx, false));
            } else {
                content = content.child(self.empty_state_panel("decompress.empty"));
            }
            return content;
        }

        let mut content = div().gap_4().flex().flex_col();

        if let Some(error) = &self.launch.startup_error {
            content = content.child(self.banner_panel(error.clone().into(), true));
        }

        if let Some(plan) = self.extract_plan() {
            if !plan.ignored_inputs.is_empty() {
                let ignored = plan
                    .ignored_inputs
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                content = content.child(self.banner_panel(ignored.into(), false));
            }
            content = content.child(self.extract_summary_panel(plan));
            content = content.child(self.operation_panel(window, cx, false));
        } else {
            content = content.child(self.empty_state_panel("decompress.empty"));
        }

        content
    }

    fn render_compress_settings_screen(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        if self.command.is_some() {
            let mut content = div().gap_4().flex().flex_col();
            if let Some(error) = &self.launch.startup_error {
                content = content.child(self.banner_panel(error.clone().into(), true));
            }
            if self.compress_plan().is_some() {
                content = content.child(self.compress_settings_form(cx));
                content = content.child(self.settings_action_row(true, cx));
            } else {
                content = content.child(self.empty_state_panel("compress.empty"));
            }
            return content;
        }

        let mut content = div().gap_4().flex().flex_col();

        if let Some(error) = &self.launch.startup_error {
            content = content.child(self.banner_panel(error.clone().into(), true));
        }

        if let Some(plan) = self.compress_plan() {
            content = content.child(self.compress_summary_panel(plan));
            content = content.child(self.compress_settings_form(cx));
            content = content.child(self.settings_action_row(true, cx));
        } else {
            content = content.child(self.empty_state_panel("compress.empty"));
        }

        content
    }

    fn render_decompress_settings_screen(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        if self.command.is_some() {
            let mut content = div().gap_4().flex().flex_col();
            if let Some(error) = &self.launch.startup_error {
                content = content.child(self.banner_panel(error.clone().into(), true));
            }
            if self.extract_plan().is_some() {
                content = content.child(self.decompress_settings_form(cx));
                content = content.child(self.settings_action_row(false, cx));
            } else {
                content = content.child(self.empty_state_panel("decompress.empty"));
            }
            return content;
        }

        let mut content = div().gap_4().flex().flex_col();

        if let Some(error) = &self.launch.startup_error {
            content = content.child(self.banner_panel(error.clone().into(), true));
        }

        if let Some(plan) = self.extract_plan() {
            content = content.child(self.extract_summary_panel(plan));
            content = content.child(self.decompress_settings_form(cx));
            content = content.child(self.settings_action_row(false, cx));
        } else {
            content = content.child(self.empty_state_panel("decompress.empty"));
        }

        content
    }

    fn compress_settings_form(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(plan) = self.compress_plan() else {
            return div();
        };

        let format_label = match plan.format {
            ArchiveFormat::Zip => "ZIP",
            ArchiveFormat::Cozip => "PDeflate",
        };

        let mut form = div()
            .gap_4()
            .flex()
            .flex_col()
            .child(self.path_picker_row(
                self.t("settings.output_file"),
                &plan.output_path,
                "browse-compress-output",
                self.t("settings.browse"),
                |this, _, _| {
                    this.browse_compress_output();
                },
                self.compress_output_warning(),
                cx,
            ))
            .child(self.settings_row(
                self.t("settings.archive_format"),
                self.control_button(
                    "cycle-compress-format",
                    format_label.into(),
                    false,
                    |this, _, _| {
                        if let Some(plan) = this.compress_plan_mut() {
                            plan.format = cycle_archive_format(plan.format);
                            plan.output_path = plan.default_output_path(plan.format);
                        }
                    },
                    cx,
                ),
            ));

        match plan.format {
            ArchiveFormat::Zip => {
                form = form
                    .child(self.settings_row(
                        self.t("settings.compression_level"),
                        self.stepper_control(
                            "zip-level-dec",
                            "zip-level-inc",
                            plan.zip_options.compression_level.to_string(),
                            |this, _, _| {
                                if let Some(plan) = this.compress_plan_mut() {
                                    plan.zip_options.compression_level =
                                        plan.zip_options.compression_level.saturating_sub(1);
                                }
                            },
                            |this, _, _| {
                                if let Some(plan) = this.compress_plan_mut() {
                                    plan.zip_options.compression_level =
                                        (plan.zip_options.compression_level + 1).min(9);
                                }
                            },
                            cx,
                        ),
                    ))
                    .child(self.settings_row(
                        self.t("settings.deflate_mode"),
                        self.control_button(
                            "cycle-zip-mode",
                            match plan.zip_options.deflate_mode {
                                ZipDeflateMode::Hybrid => "Hybrid(CPU + GPU)".into(),
                                ZipDeflateMode::Cpu => "CPU".into(),
                            },
                            false,
                            |this, _, _| {
                                if let Some(plan) = this.compress_plan_mut() {
                                    plan.zip_options.deflate_mode = match plan.zip_options.deflate_mode {
                                        ZipDeflateMode::Hybrid => ZipDeflateMode::Cpu,
                                        ZipDeflateMode::Cpu => ZipDeflateMode::Hybrid,
                                    };
                                }
                            },
                            cx,
                        ),
                    ));
            }
            ArchiveFormat::Cozip => {
                let opts = &plan.pdeflate_options;
                form = form
                    .child(self.settings_row(
                        self.t("settings.huffman"),
                        self.toggle_control(
                            "toggle-huffman",
                            opts.huffman_encode_enabled,
                            |this, _, _| {
                                if let Some(plan) = this.compress_plan_mut() {
                                    plan.pdeflate_options.huffman_encode_enabled =
                                        !plan.pdeflate_options.huffman_encode_enabled;
                                }
                            },
                            cx,
                        ),
                    ))
                    .child(self.settings_row(
                        self.t("settings.gpu_compress"),
                        self.toggle_control(
                            "toggle-gpu-compress",
                            opts.gpu_compress_enabled,
                            |this, _, _| {
                                if let Some(plan) = this.compress_plan_mut() {
                                    plan.pdeflate_options.gpu_compress_enabled =
                                        !plan.pdeflate_options.gpu_compress_enabled;
                                }
                            },
                            cx,
                        ),
                    ));
            }
        }

        form
    }

    fn decompress_settings_form(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(plan) = self.extract_plan() else {
            return div();
        };

        let mut form = div()
            .gap_4()
            .flex()
            .flex_col()
            .child(self.path_picker_row(
                self.t("settings.output_dir"),
                &plan.output_dir,
                "browse-extract-output",
                self.t("settings.browse"),
                |this, _, _| {
                    this.browse_extract_output();
                },
                self.extract_output_warning(),
                cx,
            ));

        if plan.has_zip_tasks() {
            let zip_opts = &plan.zip_options;
            form = form.child(self.settings_row(
                self.t("settings.parallel_write_threads"),
                self.stepper_control(
                    "dec-zip-parallel-write-threads",
                    "inc-zip-parallel-write-threads",
                    zip_opts.parallel_write_threads.to_string(),
                    |this, _, _| {
                        if let Some(plan) = this.extract_plan_mut() {
                            plan.zip_options.parallel_write_threads =
                                plan.zip_options.parallel_write_threads.saturating_sub(1).max(1);
                        }
                    },
                    |this, _, _| {
                        if let Some(plan) = this.extract_plan_mut() {
                            plan.zip_options.parallel_write_threads =
                                plan.zip_options.parallel_write_threads.saturating_add(1).min(128);
                        }
                    },
                    cx,
                ),
            ));
        }

        if !plan.has_pdeflate_tasks() {
            return form;
        }

        let opts = &plan.pdeflate_options;
        form = form.child(self.settings_row(
            self.t("settings.gpu_decode"),
            self.toggle_control(
                "toggle-gpu-decode",
                opts.gpu_decompress_enabled,
                |this, _, _| {
                    if let Some(plan) = this.extract_plan_mut() {
                        plan.pdeflate_options.gpu_decompress_enabled =
                            !plan.pdeflate_options.gpu_decompress_enabled;
                    }
                },
                cx,
            ),
        ));
        form = form.child(self.settings_row(
            self.t("settings.parallel_write_threads"),
            self.stepper_control(
                "dec-parallel-write-threads",
                "inc-parallel-write-threads",
                opts.parallel_write_threads.to_string(),
                |this, _, _| {
                    if let Some(plan) = this.extract_plan_mut() {
                        plan.pdeflate_options.parallel_write_threads =
                            plan.pdeflate_options.parallel_write_threads.saturating_sub(1).max(1);
                    }
                },
                |this, _, _| {
                    if let Some(plan) = this.extract_plan_mut() {
                        plan.pdeflate_options.parallel_write_threads =
                            plan.pdeflate_options.parallel_write_threads.saturating_add(1).min(128);
                    }
                },
                cx,
            ),
        ));

        form
    }

    fn compress_summary_panel(&self, plan: &CompressPlan) -> impl IntoElement {
        let mode_text = match plan.mode {
            CompressMode::SingleFile => self.t("summary.single_file"),
            CompressMode::SingleDirectory => self.t("summary.directory"),
            CompressMode::MultiSelection => self.t("summary.multi_selection"),
        };
        let format_text: SharedString = match plan.format {
            ArchiveFormat::Zip => "zip".into(),
            ArchiveFormat::Cozip => "cozip".into(),
        };
        panel(
            self.t("summary.compress"),
            div()
                .gap_3()
                .flex()
                .flex_col()
                .child(labeled_value(self.t("summary.mode"), mode_text))
                .child(labeled_value(self.t("summary.format"), format_text))
                .child(labeled_value(
                    self.t("summary.input_count"),
                    plan.sources.len().to_string(),
                ))
                .child(labeled_value(
                    self.t("summary.output"),
                    plan.output_path.display().to_string(),
                ))
                .child(separator())
                .child(self.path_list(plan.sources.iter().map(PathBuf::as_path))),
        )
    }

    fn extract_summary_panel(&self, plan: &ExtractPlan) -> impl IntoElement {
        let first_output = plan
            .tasks
            .first()
            .map(|task| plan.output_path_for(task).display().to_string())
            .unwrap_or_default();
        panel(
            self.t("summary.extract"),
            div()
                .gap_3()
                .flex()
                .flex_col()
                .child(labeled_value(
                    self.t("summary.archive_count"),
                    plan.tasks.len().to_string(),
                ))
                .child(labeled_value(self.t("summary.first_output"), first_output))
            .child(labeled_value(
                self.t("summary.ignored"),
                plan.ignored_inputs.len().to_string(),
            ))
            .child(labeled_value(
                self.t("summary.output"),
                plan.output_dir.display().to_string(),
            ))
            .child(separator())
                .child(self.path_list(plan.tasks.iter().map(|task| task.archive_path.as_path()))),
        )
    }

    fn path_list<'a>(
        &self,
        paths: impl Iterator<Item = &'a Path>,
    ) -> impl IntoElement {
        let mut column = div().gap_2().flex().flex_col();
        for path in paths.take(5) {
            column = column.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(path.display().to_string()),
            );
        }
        column
    }

    fn operation_panel(&self, _window: &mut Window, cx: &mut Context<Self>, compress: bool) -> impl IntoElement {
        let snapshot = &self.job_snapshot;
        let progress = self.progress_fraction();
        let status_text = self.status_line(compress);
        let current = self.current_item_line();
        let throughput = self.throughput_line();
        let backlog_warning = self.backlog_warning_text();
        let runtime = self.runtime_line();
        let mut throughput_row = div()
            .text_sm()
            .text_color(rgb(0x475569))
            .flex()
            .gap_2()
            .child(throughput);
        if let Some(backlog_warning) = backlog_warning {
            throughput_row = throughput_row.child(
                div()
                    .text_color(rgb(0xb45309))
                    .child(backlog_warning),
            );
        }
        let mut body = div()
            .gap_3()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_base()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(0x111827))
                    .child(status_text),
            )
            .child(progress_bar(progress, if compress { rgb(0x4ea1ff) } else { rgb(0xf0a84b) }))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(current),
            )
            .child(
                throughput_row,
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(runtime),
            );

        if let Some(note) = &snapshot.note {
            body = body.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(note.clone()),
            );
        }

        if let Some(error) = &snapshot.error {
            body = body.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x991b1b))
                    .child(error.clone()),
            );
        }

        if !matches!(snapshot.status, JobStatus::Running) {
            body = body.child(separator()).child(
                div()
                    .flex()
                    .justify_end()
                    .gap_2()
                    .child(self.window_button("close-job", "common.close", true, move |_, window, _| {
                        window.remove_window();
                    }, cx)),
            );
        }

        if self.command.is_some() {
            body.into_any_element()
        } else {
            panel(
                if compress {
                    self.t("compress.title")
                } else {
                    self.t("decompress.title")
                },
                body,
            )
            .into_any_element()
        }
    }

    fn settings_action_row(&self, compress: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let body = div()
            .gap_3()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(if compress {
                        self.t("summary.ready_compress")
                    } else {
                        self.t("summary.ready_extract")
                    }),
            )
            .child(
                div()
                    .flex()
                    .justify_end()
                    .gap_2()
                    .child(self.window_button(
                        if compress { "start-compress" } else { "start-extract" },
                        if compress {
                            "summary.start_compress"
                        } else {
                            "summary.start_extract"
                        },
                        true,
                        move |this, _, _| {
                            let allowed = if compress {
                                this.compress_output_warning().is_none()
                                    || this
                                        .compress_plan()
                                        .map(|plan| this.confirm_overwrite(&plan.output_path))
                                        .unwrap_or(true)
                            } else {
                                this.confirm_extract_conflicts()
                            };
                            if !allowed {
                                return;
                            }
                            this.active_screen = if compress {
                                ScreenKind::Compress
                            } else {
                                ScreenKind::Decompress
                            };
                            this.start_command();
                        },
                        cx,
                    )),
            );

        if self.command.is_some() {
            body.into_any_element()
        } else {
            panel(
            self.t("summary.ready"),
            body,
        )
            .into_any_element()
        }
    }

    fn control_button(
        &self,
        id: &'static str,
        label: SharedString,
        primary: bool,
        on_click: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .id(id)
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, window, cx| {
                on_click(this, window, cx);
            }))
            .child(action_button(label, primary))
    }

    fn settings_row(
        &self,
        label: SharedString,
        control: impl IntoElement,
    ) -> impl IntoElement {
        div()
            .gap_3()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(0x111827))
                    .child(label),
            )
            .child(control)
    }

    fn value_chip(&self, value: String) -> impl IntoElement {
        div()
            .px_3()
            .py_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0xe5e7eb))
            .bg(rgb(0xffffff))
            .text_sm()
            .text_color(rgb(0x111827))
            .child(value)
    }

    fn toggle_control(
        &self,
        id: &'static str,
        checked: bool,
        on_click: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        self.control_button(
            id,
            if checked {
                self.t("common.enabled")
            } else {
                self.t("common.disabled")
            },
            checked,
            on_click,
            cx,
        )
    }

    fn stepper_control(
        &self,
        dec_id: &'static str,
        inc_id: &'static str,
        value: String,
        on_dec: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        on_inc: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .gap_2()
            .flex()
            .flex_row()
            .items_center()
            .child(self.control_button(dec_id, "-".into(), false, on_dec, cx))
            .child(self.value_chip(value))
            .child(self.control_button(inc_id, "+".into(), false, on_inc, cx))
    }

    fn path_picker_row(
        &self,
        label: SharedString,
        path: &Path,
        browse_id: &'static str,
        browse_label: SharedString,
        on_browse: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        warning: Option<String>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let mut control = div()
            .gap_2()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x475569))
                    .child(path.display().to_string()),
            )
            .child(self.control_button(browse_id, browse_label, false, on_browse, cx));

        if let Some(warning) = warning {
            control = control.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x9a3412))
                    .child(warning),
            );
        }

        self.settings_row(label, control)
    }

    fn window_button(
        &self,
        id: &'static str,
        label_key: &'static str,
        primary: bool,
        on_click: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .id(id)
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, window, cx| {
                on_click(this, window, cx);
            }))
            .child(action_button(self.t(label_key), primary))
    }

    fn empty_state_panel(&self, key: &'static str) -> impl IntoElement {
        panel(
            self.t("summary.no_selection"),
            div()
                .text_sm()
                .text_color(rgb(0x475569))
                .child(self.t(key)),
        )
    }

    fn progress_fraction(&self) -> f32 {
        self.displayed_progress
    }

    fn target_progress_fraction(&self) -> f32 {
        let snapshot = &self.job_snapshot;
        let progress_snapshot = self.progress_snapshot();
        if snapshot.total_tasks == 0 {
            return 0.0;
        }

        let current_fraction = progress_snapshot
            .as_ref()
            .and_then(|progress| {
                progress.total_bytes.and_then(|total| {
                    if total == 0 {
                        None
                    } else {
                        Some((progress.processed_bytes as f32 / total as f32).clamp(0.0, 1.0))
                    }
                })
            })
            .unwrap_or(0.0);

        if snapshot.total_tasks == 1 {
            return current_fraction.max(match snapshot.status {
                JobStatus::Succeeded => 1.0,
                JobStatus::Failed => current_fraction,
                JobStatus::Idle => 0.0,
                JobStatus::Running => current_fraction,
            });
        }

        ((snapshot.completed_tasks as f32 + current_fraction) / snapshot.total_tasks as f32)
            .clamp(0.0, 1.0)
    }

    fn status_line(&self, compress: bool) -> String {
        let snapshot = &self.job_snapshot;
        let total = snapshot.total_tasks;
        let done = snapshot.completed_tasks;
        let progress_pct = match snapshot.status {
            JobStatus::Succeeded => 100,
            JobStatus::Running => (self.progress_fraction() * 100.0).floor() as i32,
            _ => (self.progress_fraction() * 100.0).round() as i32,
        };

        let action = match snapshot.status {
            JobStatus::Idle => {
                if compress {
                    self.i18n.text("summary.waiting_compress")
                } else {
                    self.i18n.text("summary.waiting_extract")
                }
            }
            JobStatus::Running => {
                if compress {
                    self.i18n.text("compress.inline_status")
                } else {
                    self.i18n.text("decompress.inline_status")
                }
            }
            JobStatus::Succeeded => self.i18n.text("summary.completed"),
            JobStatus::Failed => self.i18n.text("summary.failed"),
        };

        if total > 0 {
            format!("{action} {progress_pct}% ({done} / {total})")
        } else {
            action.to_string()
        }
    }

    fn current_item_line(&self) -> String {
        let snapshot = &self.job_snapshot;
        if let Some(progress) = self.progress_snapshot() {
            if let Some(current) = &progress.current_entry {
                return current.clone();
            }
        }
        snapshot
            .current_task_label
            .clone()
            .unwrap_or_else(|| self.i18n.text("summary.no_active_item").to_string())
    }

    fn throughput_line(&self) -> String {
        format!(
            "{} {} / s",
            self.i18n.text("summary.throughput"),
            human_bytes(self.displayed_throughput)
        )
    }

    fn backlog_warning_text(&self) -> Option<String> {
        const BACKLOG_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
        const WARN_THRESHOLD: u64 = BACKLOG_LIMIT * 9 / 10;

        let backlog = self
            .progress_snapshot()
            .and_then(|progress| progress.pending_output_backlog_bytes)?;
        if backlog < WARN_THRESHOLD {
            return None;
        }

        Some(format!(
            "{} {} / {}",
            self.i18n.text("warning.decode_backlog_high"),
            human_bytes(backlog as f64),
            human_bytes(BACKLOG_LIMIT as f64)
        ))
    }

    fn runtime_line(&self) -> String {
        let cpu_enabled = self.cpu_runtime_text();
        let gpu_enabled = if self.gpu_enabled() {
            self.t("common.enabled")
        } else {
            self.t("common.disabled")
        };
        format!("CPU: {cpu_enabled} | GPU: {gpu_enabled}")
    }

    fn cpu_runtime_text(&self) -> SharedString {
        match self.command() {
            Some(DesktopCommand::Extract(plan)) => match self.current_extract_task(plan) {
                Some(task) if task.decode_hint == ExtractDecodeHint::SingleThread => {
                    self.t("common.enabled_single_thread")
                }
                _ => self.t("common.enabled"),
            },
            _ => self.t("common.enabled"),
        }
    }

    fn current_extract_task<'a>(
        &self,
        plan: &'a ExtractPlan,
    ) -> Option<&'a crate::launch::ExtractTask> {
        if plan.tasks.is_empty() {
            return None;
        }
        let index = self
            .job_snapshot
            .completed_tasks
            .min(plan.tasks.len().saturating_sub(1));
        plan.tasks.get(index)
    }

    fn gpu_enabled(&self) -> bool {
        match self.command() {
            Some(DesktopCommand::Compress(plan)) => match plan.format {
                ArchiveFormat::Zip => matches!(plan.zip_options.deflate_mode, ZipDeflateMode::Hybrid),
                ArchiveFormat::Cozip => plan.pdeflate_options.gpu_compress_enabled,
            },
            Some(DesktopCommand::Extract(plan)) => {
                plan.has_pdeflate_tasks() && plan.pdeflate_options.gpu_decompress_enabled
            }
            None => false,
        }
    }

    fn progress_snapshot(&self) -> Option<cozip::CoZipProgressSnapshot> {
        self.job_snapshot
            .progress
            .as_ref()
            .map(|progress| progress.snapshot())
    }
}

impl Render for CozipDesktopApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.start_polling_if_needed(window, cx);
        self.maybe_start_from_launch();

        let content: AnyElement = match self.active_screen {
            ScreenKind::Compress => self.render_compress_screen(window, cx).into_any_element(),
            ScreenKind::Decompress => self.render_decompress_screen(window, cx).into_any_element(),
            ScreenKind::CompressSettings => self
                .render_compress_settings_screen(cx)
                .into_any_element(),
            ScreenKind::DecompressSettings => self
                .render_decompress_settings_screen(cx)
                .into_any_element(),
        };

        self.shell(content, cx)
    }
}

fn human_bytes(value: f64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit_index = 0;
    let mut current = value.max(0.0);
    while current >= 1024.0 && unit_index < units.len() - 1 {
        current /= 1024.0;
        unit_index += 1;
    }
    if unit_index == 0 {
        format!("{current:.0} {}", units[unit_index])
    } else {
        format!("{current:.1} {}", units[unit_index])
    }
}
