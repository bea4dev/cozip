use gpui::{
    div, prelude::*, px, rgb, AnyElement, Context, FontWeight, IntoElement, ParentElement, Render,
    SharedString, Styled, Window,
};

use crate::i18n::I18n;
use crate::screens::{
    compress::CompressScreen, compress_settings::CompressSettingsScreen,
    decompress::DecompressScreen, decompress_settings::DecompressSettingsScreen,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScreenKind {
    Compress,
    Decompress,
    CompressSettings,
    DecompressSettings,
}

pub struct CozipDesktopApp {
    i18n: I18n,
    active_screen: ScreenKind,
    compress: CompressScreen,
    decompress: DecompressScreen,
    compress_settings: CompressSettingsScreen,
    decompress_settings: DecompressSettingsScreen,
}

impl CozipDesktopApp {
    pub fn new() -> Self {
        let i18n = I18n::load();
        Self {
            i18n,
            active_screen: ScreenKind::Compress,
            compress: CompressScreen::mock(),
            decompress: DecompressScreen::mock(),
            compress_settings: CompressSettingsScreen::mock(),
            decompress_settings: DecompressSettingsScreen::mock(),
        }
    }

    fn t(&self, key: &str) -> SharedString {
        self.i18n.text(key).to_owned().into()
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
            .bg(if active {
                rgb(0xdbeafe)
            } else {
                rgb(0xf8fafc)
            })
            .border_1()
            .border_color(if active {
                rgb(0x93c5fd)
            } else {
                rgb(0xe2e8f0)
            })
            .text_color(if active {
                rgb(0x0f172a)
            } else {
                rgb(0x475569)
            })
            .text_sm()
            .font_weight(FontWeight::MEDIUM)
            .on_click(cx.listener(move |this, _, _, _| {
                this.active_screen = target;
            }))
            .child(self.t(label_key))
    }

    fn shell(&self, content: AnyElement, cx: &mut Context<Self>) -> impl IntoElement {
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
                            .child(
                                div()
                                    .max_w(px(980.0))
                                    .mx_auto()
                                    .child(content),
                            ),
                    ),
            )
    }
}

impl Render for CozipDesktopApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let content: AnyElement = match self.active_screen {
            ScreenKind::Compress => self.compress.render(&self.i18n).into_any_element(),
            ScreenKind::Decompress => self.decompress.render(&self.i18n).into_any_element(),
            ScreenKind::CompressSettings => self
                .compress_settings
                .render(&self.i18n)
                .into_any_element(),
            ScreenKind::DecompressSettings => self
                .decompress_settings
                .render(&self.i18n)
                .into_any_element(),
        };

        self.shell(content, cx)
    }
}
