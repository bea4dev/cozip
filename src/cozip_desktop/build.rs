fn main() {
    #[cfg(target_os = "windows")]
    {
        embed_resource::compile("desktop.rc", embed_resource::NONE)
            .manifest_optional()
            .expect("failed to compile cozip_desktop resources");
    }
}
