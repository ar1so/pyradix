//! Researcher-facing errors. `code` is stable. `message` names the field it can name.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixError {
    code: &'static str,
    message: String,
}

impl RadixError {
    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn setting(name: &str, detail: impl Into<String>) -> Self {
        Self {
            code: "setting",
            message: format!("setting {name}: {}", detail.into()),
        }
    }

    pub fn image_type(name: &str, detail: impl Into<String>) -> Self {
        Self {
            code: "image_type",
            message: format!("image type {name}: {}", detail.into()),
        }
    }

    pub fn feature(class: &str, feature: &str, detail: impl Into<String>) -> Self {
        Self {
            code: "feature",
            message: format!("feature {class} {feature}: {}", detail.into()),
        }
    }

    pub fn step(step: &str, detail: impl Into<String>) -> Self {
        Self {
            code: "case",
            message: format!("{step}: {}", detail.into()),
        }
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self {
            code: "internal",
            message: detail.into(),
        }
    }
}

impl fmt::Display for RadixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RadixError {}
