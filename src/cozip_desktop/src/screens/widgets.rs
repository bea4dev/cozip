use gpui::{
    div, px, rgb, FontWeight, IntoElement, ParentElement, SharedString, Styled,
};

pub fn panel(title: impl Into<SharedString>, body: impl IntoElement) -> impl IntoElement {
    div()
        .rounded_lg()
        .bg(rgb(0xffffff))
        .border_1()
        .border_color(rgb(0xe5e7eb))
        .p_4()
        .gap_3()
        .flex()
        .flex_col()
        .child(
            div()
                .text_base()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(0x111827))
                .child(title.into()),
        )
        .child(body)
}

pub fn stat_card(
    title: impl Into<SharedString>,
    value: impl Into<SharedString>,
    note: impl Into<SharedString>,
) -> impl IntoElement {
    div()
        .min_w(px(180.0))
        .rounded_lg()
        .bg(rgb(0xffffff))
        .border_1()
        .border_color(rgb(0xe5e7eb))
        .p_4()
        .gap_1()
        .flex()
        .flex_col()
        .child(div().text_sm().text_color(rgb(0x6b7280)).child(title.into()))
        .child(
            div()
                .text_lg()
                .font_weight(FontWeight::BOLD)
                .text_color(rgb(0x111827))
                .child(value.into()),
        )
        .child(div().text_sm().text_color(rgb(0x94a3b8)).child(note.into()))
}

pub fn labeled_value(
    label: impl Into<SharedString>,
    value: impl Into<SharedString>,
) -> impl IntoElement {
    div()
        .flex()
        .justify_between()
        .gap_4()
        .child(div().text_sm().text_color(rgb(0x6b7280)).child(label.into()))
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::MEDIUM)
                .text_color(rgb(0x111827))
                .child(value.into()),
        )
}

pub fn progress_bar(progress: f32, accent: gpui::Rgba) -> impl IntoElement {
    let width = progress.clamp(0.0, 1.0) * 480.0;
    div()
        .w_full()
        .h(px(12.0))
        .rounded_full()
        .bg(rgb(0xe5e7eb))
        .child(
            div()
                .h_full()
                .w(px(width))
                .rounded_full()
                .bg(accent),
        )
}

pub fn separator() -> impl IntoElement {
    div().w_full().h(px(1.0)).bg(rgb(0xe5e7eb))
}

pub fn action_button(
    label: impl Into<SharedString>,
    primary: bool,
) -> impl IntoElement {
    div()
        .px_3()
        .py_2()
        .rounded_md()
        .border_1()
        .border_color(if primary { rgb(0xcbd5e1) } else { rgb(0xe5e7eb) })
        .bg(if primary { rgb(0xf8fafc) } else { rgb(0xffffff) })
        .text_sm()
        .font_weight(FontWeight::MEDIUM)
        .text_color(rgb(0x111827))
        .child(label.into())
}
