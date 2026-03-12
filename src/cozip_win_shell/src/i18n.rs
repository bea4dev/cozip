use std::{collections::HashMap, env, sync::OnceLock};

const EN_US: &str = include_str!("../locales/en_US.toml");
const JA_JP: &str = include_str!("../locales/ja_JP.toml");

#[derive(Debug)]
pub struct I18n {
    active: HashMap<String, String>,
    fallback: HashMap<String, String>,
}

impl I18n {
    pub fn shared() -> &'static Self {
        static INSTANCE: OnceLock<I18n> = OnceLock::new();
        INSTANCE.get_or_init(Self::load)
    }

    fn load() -> Self {
        let fallback = parse_locale(EN_US);
        let active = match current_locale().as_deref() {
            Some("ja_JP") => parse_locale(JA_JP),
            Some("en_US") => fallback.clone(),
            _ => fallback.clone(),
        };
        Self { active, fallback }
    }

    pub fn text<'a>(&'a self, key: &'a str) -> &'a str {
        self.active
            .get(key)
            .or_else(|| self.fallback.get(key))
            .map(String::as_str)
            .unwrap_or(key)
    }
}

fn parse_locale(source: &str) -> HashMap<String, String> {
    let mut flat = HashMap::new();
    let Ok(value) = source.parse::<toml::Value>() else {
        return flat;
    };
    flatten_value("", &value, &mut flat);
    flat
}

fn flatten_value(prefix: &str, value: &toml::Value, flat: &mut HashMap<String, String>) {
    match value {
        toml::Value::String(text) => {
            if !prefix.is_empty() {
                flat.insert(prefix.to_owned(), text.clone());
            }
        }
        toml::Value::Table(table) => {
            for (key, child) in table {
                let next = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_value(&next, child, flat);
            }
        }
        _ => {}
    }
}

fn current_locale() -> Option<String> {
    windows_locale()
        .or_else(env_locale)
        .map(|raw| normalize_locale(&raw))
}

#[cfg(target_os = "windows")]
fn windows_locale() -> Option<String> {
    use windows::Win32::Globalization::GetUserDefaultLocaleName;

    const LOCALE_NAME_MAX_LENGTH: i32 = 85;

    let mut buffer = [0u16; LOCALE_NAME_MAX_LENGTH as usize];
    let len = unsafe { GetUserDefaultLocaleName(&mut buffer) };
    if len <= 1 {
        return None;
    }
    Some(String::from_utf16_lossy(&buffer[..len as usize - 1]))
}

#[cfg(not(target_os = "windows"))]
fn windows_locale() -> Option<String> {
    None
}

fn env_locale() -> Option<String> {
    env::var("LC_ALL")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| env::var("LANG").ok().filter(|value| !value.is_empty()))
}

fn normalize_locale(raw: &str) -> String {
    let normalized = raw
        .split('.')
        .next()
        .unwrap_or_default()
        .replace('-', "_")
        .to_lowercase();

    if normalized.starts_with("ja") {
        "ja_JP".to_string()
    } else {
        "en_US".to_string()
    }
}
