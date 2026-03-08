use gpui::{div, rgb, FontWeight, IntoElement, ParentElement, Styled};

use crate::i18n::I18n;

use super::widgets::{labeled_value, panel};

pub struct CompressSettingsScreen;

impl CompressSettingsScreen {
    pub fn mock() -> Self {
        Self
    }

    pub fn render(&self, i18n: &I18n) -> impl IntoElement {
        let t = |key: &str| i18n.text(key).to_owned();
        div()
            .gap_5()
            .flex()
            .flex_col()
            .child(
                div()
                    .gap_2()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_3xl()
                            .font_weight(FontWeight::BOLD)
                            .child(t("compress_settings.title")),
                    )
                    .child(
                        div()
                            .text_base()
                            .text_color(rgb(0x90a5bd))
                            .child(t("compress_settings.subtitle")),
                    ),
            )
            .child(panel(
                t("compress_settings.panel.profile"),
                div()
                    .gap_3()
                    .flex()
                    .flex_col()
                    .child(labeled_value(t("compress_settings.mode"), "Speed"))
                    .child(labeled_value(t("compress_settings.codec"), "PDeflate"))
                    .child(labeled_value(t("compress_settings.chunk_mib"), "4 MiB"))
                    .child(labeled_value(t("compress_settings.sections"), "128"))
                    .child(labeled_value(t("compress_settings.huffman"), t("common.disabled"))),
            ))
            .child(panel(
                t("compress_settings.panel.hybrid"),
                div()
                    .gap_3()
                    .flex()
                    .flex_col()
                    .child(labeled_value(t("compress_settings.gpu_compress"), t("common.enabled")))
                    .child(labeled_value(t("compress_settings.gpu_batch_chunks"), "32"))
                    .child(labeled_value(t("compress_settings.gpu_submit_chunks"), "32"))
                    .child(labeled_value(t("compress_settings.gpu_slot_count"), "6"))
                    .child(labeled_value(t("compress_settings.scheduler"), "GlobalQueueLocalBuffers")),
            ))
    }
}
