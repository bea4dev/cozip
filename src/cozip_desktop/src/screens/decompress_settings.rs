use gpui::{div, rgb, FontWeight, IntoElement, ParentElement, Styled};

use crate::i18n::I18n;

use super::widgets::{labeled_value, panel};

pub struct DecompressSettingsScreen;

impl DecompressSettingsScreen {
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
                            .child(t("decompress_settings.title")),
                    )
                    .child(
                        div()
                            .text_base()
                            .text_color(rgb(0x90a5bd))
                            .child(t("decompress_settings.subtitle")),
                    ),
            )
            .child(panel(
                t("decompress_settings.panel.runtime"),
                div()
                    .gap_3()
                    .flex()
                    .flex_col()
                    .child(labeled_value(t("decompress_settings.gpu_decode"), t("common.enabled")))
                    .child(labeled_value(t("decompress_settings.force_gpu"), t("common.disabled")))
                    .child(labeled_value(t("decompress_settings.batch_chunks"), "96"))
                    .child(labeled_value(t("decompress_settings.target_inflight"), "3"))
                    .child(labeled_value(t("decompress_settings.wait_high_watermark"), "4")),
            ))
            .child(panel(
                t("decompress_settings.panel.batch"),
                div()
                    .gap_3()
                    .flex()
                    .flex_col()
                    .child(labeled_value(t("decompress_settings.true_batch_jobs"), "16"))
                    .child(labeled_value(t("decompress_settings.copy_jobs"), "32"))
                    .child(labeled_value(t("decompress_settings.queue_workers"), "48"))
                    .child(labeled_value(t("decompress_settings.readback_ring"), "8"))
                    .child(labeled_value(t("decompress_settings.wait_probe"), t("common.enabled"))),
            ))
    }
}
