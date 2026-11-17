//! PyRadiomics 3.1.0 intensity filters.
//! Bodies of `getSquareImage`, `getSquareRootImage`, `getLogarithmImage`,
//! and `getExponentialImage` in `imageoperations.py`.

use crate::error::RadixError;
use crate::registry::ImageType;
use crate::Mode;

const PYRAD: &[Mode] = &[Mode::Pyradiomics];
const BIN_KEYS: &[&str] = &["binWidth", "binCount"];

pub struct Square;
pub struct SquareRoot;
pub struct Logarithm;
pub struct Exponential;

fn abs_max(image: &[f64]) -> f64 {
    let mut peak = 0.0;
    for value in image {
        let magnitude = value.abs();
        if magnitude > peak {
            peak = magnitude;
        }
    }
    peak
}

fn volume(
    name: &str,
    prefix: &str,
    image: &[f64],
    dim: [usize; 3],
    values: Vec<f64>,
) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
    let n = dim[0] * dim[1] * dim[2];
    if image.len() != n {
        return Err(RadixError::image_type(
            name,
            "image length does not match shape",
        ));
    }
    Ok(vec![(prefix.to_string(), values)])
}

/// `(c x)^2` with `c = 1 / sqrt(max |x|)`. The square stays non-negative.
pub fn square(image: &[f64]) -> Vec<f64> {
    let coeff = 1.0 / abs_max(image).sqrt();
    image.iter().map(|value| (coeff * value).powi(2)).collect()
}

/// `sqrt(|x| * max|x|)` with the original sign restored.
pub fn square_root(image: &[f64]) -> Vec<f64> {
    let peak = abs_max(image);
    image
        .iter()
        .map(|value| {
            if *value > 0.0 {
                (*value * peak).sqrt()
            } else if *value < 0.0 {
                -(-value * peak).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// Signed `ln(|x| + 1)`, scaled so the peak returns to `max |x|`.
pub fn logarithm(image: &[f64]) -> Vec<f64> {
    let peak = abs_max(image);
    let mut filtered: Vec<f64> = image
        .iter()
        .map(|value| {
            if *value > 0.0 {
                (value + 1.0).ln()
            } else if *value < 0.0 {
                -(-(value - 1.0)).ln()
            } else {
                0.0
            }
        })
        .collect();
    let scale = peak / abs_max(&filtered);
    for value in &mut filtered {
        *value *= scale;
    }
    filtered
}

/// `exp(c x)` with `c = ln(max |x|) / max |x|`. The result stays positive.
pub fn exponential(image: &[f64]) -> Vec<f64> {
    let peak = abs_max(image);
    let coeff = peak.ln() / peak;
    image.iter().map(|value| (coeff * value).exp()).collect()
}

impl ImageType for Square {
    fn name(&self) -> &'static str {
        "Square"
    }

    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        BIN_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        _settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        volume("Square", "square", image, dim, square(image))
    }
}

impl ImageType for SquareRoot {
    fn name(&self) -> &'static str {
        "SquareRoot"
    }

    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        BIN_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        _settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        volume("SquareRoot", "squareroot", image, dim, square_root(image))
    }
}

impl ImageType for Logarithm {
    fn name(&self) -> &'static str {
        "Logarithm"
    }

    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        BIN_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        _settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        volume("Logarithm", "logarithm", image, dim, logarithm(image))
    }
}

impl ImageType for Exponential {
    fn name(&self) -> &'static str {
        "Exponential"
    }

    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        BIN_KEYS
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        _settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        volume("Exponential", "exponential", image, dim, exponential(image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: [f64; 4] = [-4.0, 0.0, 2.0, 9.0];

    fn close(got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len());
        for (index, (left, right)) in got.iter().zip(expected).enumerate() {
            assert!(
                (left - right).abs() <= 1e-12,
                "index {index}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn formulas_match_pyradiomics_on_a_signed_line() {
        close(&square(&IMAGE), &[16.0 / 9.0, 0.0, 4.0 / 9.0, 9.0]);
        close(&square_root(&IMAGE), &[-6.0, 0.0, 4.242640687119285, 9.0]);
        close(
            &logarithm(&IMAGE),
            &[-6.290730039024168, 0.0, 4.294091292476962, 9.0],
        );
        close(
            &exponential(&IMAGE),
            &[0.376610321303251, 1.0, 1.6294982222188463, 9.0],
        );
        assert!(square(&IMAGE)[0] > 0.0);
        assert!(square_root(&IMAGE)[0] < 0.0);
        assert!(logarithm(&IMAGE)[0] < 0.0);
        assert!(exponential(&IMAGE)[0] > 0.0);
    }

    #[test]
    fn apply_uses_the_pyradiomics_prefix() {
        let out = Square
            .apply(&IMAGE, [1, 1, 4], [1.0, 1.0, 1.0], &[])
            .unwrap();
        assert_eq!(out[0].0, "square");
        assert!(SquareRoot.modes() == PYRAD);
        assert!(!Square.modes().contains(&Mode::Ibsi));
    }
}
