//! LoG and wavelet image types. `pyradiomics` mode follows 3.1.0.
//! `ibsi` mode calls the IBSI-2 filters in `filters`.

use crate::error::RadixError;
use crate::filters::{self, Padding};
use crate::itk_log;
use crate::registry::ImageType;
use crate::swt;
use crate::Mode;

const IBSI: &[Mode] = &[Mode::Ibsi];
const PYRAD: &[Mode] = &[Mode::Pyradiomics];
const LOG_KEYS: &[&str] = &["sigma", "binWidth", "binCount"];
const WAVELET_KEYS: &[&str] = &["wavelet", "start_level", "level", "binWidth", "binCount"];

pub struct PyradLoG;
pub struct IbsiLoG;
pub struct PyradWavelet;
pub struct IbsiWavelet;

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

fn sigma_tokens(raw: &str) -> Result<Vec<&str>, RadixError> {
    let tokens: Vec<&str> = raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err(RadixError::image_type(
            "LoG",
            "sigma must be a list of numbers > 0",
        ));
    }
    for token in &tokens {
        let value: f64 = token
            .parse()
            .map_err(|_| RadixError::image_type("LoG", format!("sigma {token} is not a number")))?;
        if value <= 0.0 {
            return Err(RadixError::image_type(
                "LoG",
                format!("sigma {token} must be > 0"),
            ));
        }
    }
    Ok(tokens)
}

fn log_name(token: &str) -> String {
    format!("log-sigma-{}-mm-3D", token.replace('.', "-"))
}

/// PyRadiomics skips a sigma when the image is smaller than `ceil(sigma / spacing) + 1`.
fn sigma_fits(dim: [usize; 3], spacing_zyx: [f64; 3], sigma: f64) -> bool {
    let size = [dim[2], dim[1], dim[0]];
    let spacing = [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]];
    size.iter()
        .zip(spacing)
        .all(|(length, space)| *length as f64 >= (sigma / space).ceil() + 1.0)
}

impl ImageType for PyradLoG {
    fn name(&self) -> &'static str {
        "LoG"
    }

    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        LOG_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "LoG")?;
        let Some(raw) = setting(settings, "sigma") else {
            return Ok(Vec::new());
        };
        if dim.iter().any(|axis| *axis < 4) {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for token in sigma_tokens(raw)? {
            let sigma: f64 = token.parse().unwrap_or(0.0);
            if !sigma_fits(dim, spacing, sigma) {
                continue;
            }
            out.push((
                log_name(token),
                itk_log::laplacian(image, dim, spacing, sigma),
            ));
        }
        Ok(out)
    }
}

impl ImageType for IbsiLoG {
    fn name(&self) -> &'static str {
        "LoG"
    }

    fn modes(&self) -> &'static [Mode] {
        IBSI
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        LOG_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "LoG")?;
        let Some(raw) = setting(settings, "sigma") else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for token in sigma_tokens(raw)? {
            let sigma: f64 = token.parse().unwrap_or(0.0);
            let volume = filters::log(image, dim, spacing, sigma, Padding::Mirror, 3, 4.0)
                .map_err(|detail| RadixError::image_type("LoG", detail))?;
            out.push((format!("log-sigma-{token}"), volume));
        }
        Ok(out)
    }
}

fn positive_int(name: &str, raw: &str, setting_name: &str) -> Result<usize, RadixError> {
    let value: i64 = raw.parse().map_err(|_| {
        RadixError::image_type(name, format!("{setting_name} {raw} is not an integer"))
    })?;
    if value < 0 {
        return Err(RadixError::image_type(
            name,
            format!("{setting_name} must be >= 0"),
        ));
    }
    Ok(value as usize)
}

fn wavelet_names(level_index: usize, n_levels: usize, letters: &str) -> String {
    if n_levels == 1 || level_index == 1 {
        format!("wavelet-{letters}")
    } else {
        format!("wavelet{level_index}-{letters}")
    }
}

fn pyrad_wavelet_volumes(
    image: &[f64],
    dim: [usize; 3],
    family: &str,
    start_level: usize,
    level: usize,
    force_2d: bool,
    force_axis: usize,
) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
    if level == 0 {
        return Err(RadixError::image_type("Wavelet", "level must be >= 1"));
    }
    let (lo, hi) = swt::analysis_filters(family).ok_or_else(|| {
        RadixError::image_type("Wavelet", format!("unknown wavelet family {family}"))
    })?;
    let mut axes = vec![2usize, 1, 0];
    if force_2d {
        if force_axis > 2 {
            return Err(RadixError::image_type(
                "Wavelet",
                "force2Ddimension must be 0, 1, or 2",
            ));
        }
        axes.retain(|axis| *axis != force_axis);
    }
    let (mut data, current_dim) = swt::pad_odd(image, dim);
    let approx_key = "a".repeat(axes.len());
    for _ in 0..start_level {
        let bands = swt::swtn_level(&data, current_dim, &axes, lo, hi);
        data = bands
            .into_iter()
            .find(|(name, _)| name == &approx_key)
            .map(|(_, volume)| volume)
            .unwrap_or_default();
    }
    type LevelBands = Vec<(String, Vec<f64>)>;
    let mut emitted: Vec<(usize, LevelBands)> = Vec::new();
    for step in 0..level {
        let bands = swt::swtn_level(&data, current_dim, &axes, lo, hi);
        let mut details = Vec::new();
        for (name, volume) in bands {
            if name == approx_key {
                data = volume;
            } else {
                let letters: String = name
                    .chars()
                    .map(|letter| if letter == 'a' { 'L' } else { 'H' })
                    .collect();
                details.push((letters, swt::crop(&volume, current_dim, dim)));
            }
        }
        emitted.push((step + 1, details));
    }
    let final_approx = swt::crop(&data, current_dim, dim);
    let n_levels = level;
    let mut out = Vec::new();
    for (index, details) in emitted {
        for (letters, volume) in details {
            let shown = if n_levels == 1 { 1 } else { index };
            out.push((wavelet_names(shown, n_levels, &letters), volume));
        }
    }
    let low = "L".repeat(axes.len());
    let approx_index = if n_levels == 1 { 1 } else { n_levels };
    out.push((wavelet_names(approx_index, n_levels, &low), final_approx));
    Ok(out)
}

impl ImageType for PyradWavelet {
    fn name(&self) -> &'static str {
        "Wavelet"
    }

    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        WAVELET_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "Wavelet")?;
        let family = setting(settings, "wavelet").unwrap_or("coif1");
        let start_level = match setting(settings, "start_level") {
            Some(raw) => positive_int("Wavelet", raw, "start_level")?,
            None => 0,
        };
        let level = match setting(settings, "level") {
            Some(raw) => positive_int("Wavelet", raw, "level")?,
            None => 1,
        };
        let force_2d = setting(settings, "force2D").is_some_and(|value| value == "true");
        let force_axis = match setting(settings, "force2Ddimension") {
            Some(raw) => positive_int("Wavelet", raw, "force2Ddimension")?,
            None => 0,
        };
        pyrad_wavelet_volumes(image, dim, family, start_level, level, force_2d, force_axis)
    }
}

impl ImageType for IbsiWavelet {
    fn name(&self) -> &'static str {
        "Wavelet"
    }

    fn modes(&self) -> &'static [Mode] {
        IBSI
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        WAVELET_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "Wavelet")?;
        let family = setting(settings, "wavelet").unwrap_or("coif1");
        let start_level = match setting(settings, "start_level") {
            Some(raw) => positive_int("Wavelet", raw, "start_level")?,
            None => 0,
        };
        let level = match setting(settings, "level") {
            Some(raw) => positive_int("Wavelet", raw, "level")?,
            None => 1,
        };
        if level == 0 {
            return Err(RadixError::image_type("Wavelet", "level must be >= 1"));
        }
        let mut out = Vec::new();
        for step in 1..=level {
            let decomposition = start_level + step;
            for combo in ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"] {
                let volume = filters::wavelet(
                    image,
                    dim,
                    family,
                    combo,
                    decomposition as u32,
                    Padding::Mirror,
                    false,
                )
                .map_err(|detail| RadixError::image_type("Wavelet", detail))?;
                let prefix = if level == 1 && start_level == 0 {
                    format!("wavelet-{combo}")
                } else {
                    format!("wavelet{decomposition}-{combo}")
                };
                out.push((prefix, volume));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyradiomics_wavelet_defaults_to_eight_names() {
        let image = vec![1.0; 64];
        let out = PyradWavelet
            .apply(&image, [4, 4, 4], [1.0, 1.0, 1.0], &[])
            .unwrap();
        let names: Vec<&str> = out.iter().map(|(name, _)| name.as_str()).collect();
        for expected in [
            "wavelet-LLL",
            "wavelet-LLH",
            "wavelet-LHL",
            "wavelet-LHH",
            "wavelet-HLL",
            "wavelet-HLH",
            "wavelet-HHL",
            "wavelet-HHH",
        ] {
            assert!(
                names.contains(&expected),
                "{expected} missing from {names:?}"
            );
        }
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn ibsi_log_uses_the_ibsi_kernel() {
        let dim = [4, 4, 4];
        let image: Vec<f64> = (0..64).map(|v| v as f64).collect();
        let out = IbsiLoG
            .apply(&image, dim, [1.0, 1.0, 1.0], &[("sigma", "2")])
            .unwrap();
        assert_eq!(out[0].0, "log-sigma-2");
        let reference =
            filters::log(&image, dim, [1.0, 1.0, 1.0], 2.0, Padding::Mirror, 3, 4.0).unwrap();
        assert_eq!(out[0].1, reference);
    }

    #[test]
    fn ibsi_wavelet_matches_the_undecimated_filter() {
        let dim = [4, 4, 4];
        let image: Vec<f64> = (0..64).map(|value| value as f64 - 8.0).collect();
        let out = IbsiWavelet
            .apply(&image, dim, [1.0, 1.0, 1.0], &[("wavelet", "haar")])
            .unwrap();
        let lll = out.iter().find(|(name, _)| name == "wavelet-LLL").unwrap();
        let reference =
            filters::wavelet(&image, dim, "haar", "LLL", 1, Padding::Mirror, false).unwrap();
        assert_eq!(lll.1, reference);
    }

    #[test]
    fn missing_sigma_emits_no_log_volume() {
        let image = vec![1.0; 64];
        let out = PyradLoG
            .apply(&image, [4, 4, 4], [1.0, 1.0, 1.0], &[])
            .unwrap();
        assert!(out.is_empty());
    }
}
