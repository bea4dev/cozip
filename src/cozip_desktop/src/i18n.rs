use std::{collections::HashMap, env};

const EN_US: &str = include_str!("../locales/en_US.toml");
const JA_JP: &str = include_str!("../locales/ja_JP.toml");

#[derive(Clone, Debug)]
pub struct I18n {
    active: HashMap<String, String>,
    fallback: HashMap<String, String>,
}

impl I18n {
    pub fn load() -> Self {
        let fallback = parse_locale(EN_US);
        let current = current_locale();
        let active = match current.as_deref() {
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
    flatten_toml_value("", &value, &mut flat);
    flat
}

fn flatten_toml_value(prefix: &str, value: &toml::Value, flat: &mut HashMap<String, String>) {
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
                flatten_toml_value(&next, child, flat);
            }
        }
        _ => {}
    }
}

fn current_locale() -> Option<String> {
    #[cfg(target_os = "windows")]
    if let Some(locale) = current_windows_locale() {
        return Some(locale);
    }

    let raw = env::var("LC_ALL")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| env::var("LANG").ok().filter(|value| !value.is_empty()))?;

    let normalized = raw.split('.').next().unwrap_or_default().replace('-', "_");
    match normalized.as_str() {
        "ja_JP" | "en_US" => Some(normalized),
        _ => None,
    }
}

#[cfg(target_os = "windows")]
fn current_windows_locale() -> Option<String> {
    use windows_sys::Win32::Globalization::GetUserDefaultLocaleName;

    let mut buffer = [0_u16; 85];
    let len = unsafe { GetUserDefaultLocaleName(buffer.as_mut_ptr(), buffer.len() as i32) };
    if len <= 1 {
        return None;
    }

    let locale = String::from_utf16_lossy(&buffer[..(len as usize - 1)]).replace('-', "_");
    match locale.as_str() {
        "ja_JP" | "en_US" => Some(locale),
        _ => None,
    }
}
