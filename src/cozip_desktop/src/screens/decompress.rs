use gpui::{div, rgb, FontWeight, IntoElement, ParentElement, Styled};

use crate::i18n::I18n;

use super::widgets::{action_button, panel, progress_bar, separator};

pub struct DecompressScreen {
    total_progress: f32,
    completed_files: usize,
    total_files: usize,
    current_file: String,
}

impl DecompressScreen {
    pub fn mock() -> Self {
        Self {
            total_progress: 0.78,
            completed_files: 78,
            total_files: 100,
            current_file: "...example.txt".into(),
        }
    }

    pub fn render(&self, i18n: &I18n) -> impl IntoElement {
        let t = |key: &str| i18n.text(key).to_owned();
        div()
            .gap_3()
            .flex()
            .flex_col()
            .child(
                panel(
                    t("decompress.title"),
                    div()
                        .gap_3()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .text_base()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(rgb(0x111827))
                                .child(format!(
                                    "{} {}% ( {} / {}{} )",
                                    t("decompress.inline_status"),
                                    (self.total_progress * 100.0).round() as i32,
                                    self.completed_files,
                                    self.total_files,
                                    t("decompress.files_suffix")
                                )),
                        )
                        .child(progress_bar(self.total_progress, rgb(0xf0a84b)))
                        .child(
                            div()
                                .text_sm()
                                .text_color(rgb(0x475569))
                                .child(format!(
                                    "{} {} {}",
                                    t("decompress.file_prefix"),
                                    self.current_file,
                                    t("decompress.file_suffix")
                                )),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(rgb(0x475569))
                                .child(format!(
                                    "{} 2.8 GiB/s | {} {} | {} {}",
                                    t("decompress.throughput_prefix"),
                                    t("decompress.cpu_prefix"),
                                    t("common.enabled"),
                                    t("decompress.gpu_prefix"),
                                    t("common.enabled")
                                )),
                        )
                        .child(separator())
                        .child(
                            div()
                                .flex()
                                .justify_end()
                                .gap_2()
                                .child(action_button(t("common.cancel"), false))
                                .child(action_button(t("common.close"), true)),
                        ),
                ),
            )
    }
}
