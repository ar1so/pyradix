//! IBSI-2 Mean, Laws, and Gabor image types. `ibsi` mode only.

use crate::error::RadixError;
use crate::filters::{self, Padding};
use crate::registry::ImageType;
use crate::Mode;

const IBSI: &[Mode] = &[Mode::Ibsi];

pub struct Mean;
pub struct Laws;
pub struct Gabor;

fn setting<'a>(settings: &'a [(&str, &str)], key: &str) -> Option<&'a str> {
    settings
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, value)| *value)
}

fn check_len(image: &[f64], dim: [usize; 3], name: &str) -> Result<(), RadixError> {
    if image.len() != dim[0] * dim[1] * dim[2] {
        return Err(RadixError::image_type(
            name,
            "image length does not match shape",
        ));
    }
    Ok(())
}

fn padding_of(name: &str, settings: &[(&str, &str)]) -> Result<Padding, RadixError> {
    let raw = setting(settings, "padding").unwrap_or("mirror");
    Padding::parse(raw).map_err(|detail| RadixError::image_type(name, detail))
}

fn tokens<'a>(name: &str, key: &str, raw: &'a str) -> Result<Vec<&'a str>, RadixError> {
    let tokens: Vec<&str> = raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err(RadixError::image_type(
            name,
            format!("{key} must list at least one value"),
        ));
    }
    Ok(tokens)
}

fn positive_f64(name: &str, key: &str, raw: &str) -> Result<f64, RadixError> {
    let value: f64 = raw
        .parse()
        .map_err(|_| RadixError::image_type(name, format!("{key} {raw} is not a number")))?;
    if value <= 0.0 {
        return Err(RadixError::image_type(name, format!("{key} must be > 0")));
    }
    Ok(value)
}

impl ImageType for Mean {
    fn name(&self) -> &'static str {
        "Mean"
    }

    fn modes(&self) -> &'static [Mode] {
        IBSI
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        &["support", "padding", "binWidth", "binCount"]
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "Mean")?;
        let support = match setting(settings, "support") {
            Some(raw) => {
                let value: i64 = raw.parse().map_err(|_| {
                    RadixError::image_type("Mean", format!("support {raw} is not an integer"))
                })?;
                if value < 1 || value % 2 == 0 {
                    return Err(RadixError::image_type(
                        "Mean",
                        "support must be an odd integer >= 1",
                    ));
                }
                value as usize
            }
            None => 3,
        };
        let padding = padding_of("Mean", settings)?;
        let volume = filters::mean(image, dim, support, padding, 3)
            .map_err(|detail| RadixError::image_type("Mean", detail))?;
        Ok(vec![("mean".to_string(), volume)])
    }
}

impl ImageType for Laws {
    fn name(&self) -> &'static str {
        "Laws"
    }

    fn modes(&self) -> &'static [Mode] {
        IBSI
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        &["kernel", "padding", "binWidth", "binCount"]
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "Laws")?;
        let padding = padding_of("Laws", settings)?;
        let raw = setting(settings, "kernel").unwrap_or("E5L5S5");
        let mut out = Vec::new();
        for token in tokens("Laws", "kernel", raw)? {
            let volume = filters::laws(image, dim, token, padding, false, None)
                .map_err(|detail| RadixError::image_type("Laws", detail))?;
            let canonical: String = token
                .chars()
                .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
                .collect::<String>()
                .to_ascii_uppercase();
            out.push((format!("laws-{canonical}"), volume));
        }
        Ok(out)
    }
}

impl ImageType for Gabor {
    fn name(&self) -> &'static str {
        "Gabor"
    }

    fn modes(&self) -> &'static [Mode] {
        IBSI
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        &[
            "sigma", "lambda", "gamma", "theta", "padding", "binWidth", "binCount",
        ]
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "Gabor")?;
        let padding = padding_of("Gabor", settings)?;
        let sigma = match setting(settings, "sigma") {
            Some(raw) => positive_f64("Gabor", "sigma", raw)?,
            None => 1.0,
        };
        let lambda = match setting(settings, "lambda") {
            Some(raw) => positive_f64("Gabor", "lambda", raw)?,
            None => 2.0,
        };
        let gamma = match setting(settings, "gamma") {
            Some(raw) => positive_f64("Gabor", "gamma", raw)?,
            None => 1.0,
        };
        let raw = setting(settings, "theta").unwrap_or("0");
        let mut out = Vec::new();
        for token in tokens("Gabor", "theta", raw)? {
            let theta: f64 = token.parse().map_err(|_| {
                RadixError::image_type("Gabor", format!("theta {token} is not a number"))
            })?;
            let volume = filters::gabor(
                image, dim, spacing, sigma, lambda, gamma, theta, padding, false, None, false,
            )
            .map_err(|detail| RadixError::image_type("Gabor", detail))?;
            out.push((format!("gabor-{token}"), volume));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Padding;

    #[test]
    fn mean_matches_the_ibsi_filter() {
        let image: Vec<f64> = (0..27).map(|v| v as f64).collect();
        let dim = [3, 3, 3];
        let out = Mean
            .apply(
                &image,
                dim,
                [1.0; 3],
                &[("support", "3"), ("padding", "zero")],
            )
            .unwrap();
        assert_eq!(out[0].0, "mean");
        let expected = filters::mean(&image, dim, 3, Padding::Zero, 3).unwrap();
        assert_eq!(out[0].1, expected);
    }

    #[test]
    fn laws_and_gabor_prefixes_follow_each_response() {
        let image: Vec<f64> = (0..64).map(|v| v as f64).collect();
        let dim = [4, 4, 4];
        let laws = Laws
            .apply(&image, dim, [1.0; 3], &[("kernel", "e5-l5-s5,L3E3")])
            .unwrap();
        let names: Vec<&str> = laws.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["laws-E5L5S5", "laws-L3E3"]);
        let gabor = Gabor
            .apply(
                &image,
                dim,
                [1.0; 3],
                &[
                    ("sigma", "1"),
                    ("lambda", "2"),
                    ("gamma", "1"),
                    ("theta", "0,0.5"),
                ],
            )
            .unwrap();
        let names: Vec<&str> = gabor.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["gabor-0", "gabor-0.5"]);
        assert!(gabor
            .iter()
            .all(|(_, volume)| volume.iter().all(|v| v.is_finite())));
    }
}
