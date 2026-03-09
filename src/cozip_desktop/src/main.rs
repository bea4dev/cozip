#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;
mod i18n;
mod jobs;
mod launch;
mod screens;

use app::CozipDesktopApp;
use gpui::{App, AppContext, Application, Bounds, WindowBounds, WindowOptions, px, size};
use launch::{InitialScreen, LaunchRequest};

fn main() {
    let launch = LaunchRequest::from_env();
    Application::new().run(move |cx: &mut App| {
        let window_size = match launch.command.as_ref().map(|_| launch.initial_screen) {
            Some(InitialScreen::Compress) | Some(InitialScreen::Decompress) => {
                size(px(620.0), px(300.0))
            }
            Some(InitialScreen::CompressSettings) | Some(InitialScreen::DecompressSettings) => {
                size(px(760.0), px(520.0))
            }
            None => size(px(1360.0), px(920.0)),
        };
        let bounds = Bounds::centered(None, window_size, cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                let launch = launch.clone();
                cx.new(move |_| CozipDesktopApp::new(launch.clone()))
            },
        )
        .expect("failed to open cozip desktop window");
    });
}
