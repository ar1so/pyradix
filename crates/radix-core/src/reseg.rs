//! Intensity-mask re-segmentation (IBSI range + outlier filter).

use crate::error::RadixError;

/// Build the intensity mask from the morphological mask.
///
/// Range re-segmentation is a closed interval `[a, b]`. Outlier filtering then
/// drops voxels outside `[\mu - k\sigma, \mu + k\sigma]` of the *current*
/// intensity mask (range applied first when both are set). \(\sigma\) is the
/// sample standard deviation (\(N-1\)).
pub fn intensity_mask(
    image: &[f64],
    morph: &[bool],
    range: Option<(f64, f64)>,
    outlier_sigma: Option<f64>,
) -> Result<Vec<bool>, RadixError> {
    if image.len() != morph.len() {
        return Err(RadixError::step(
            "resegment",
            "image length does not match the mask",
        ));
    }
    let mut mask = morph.to_vec();
    if let Some((lo, hi)) = range {
        if lo > hi {
            return Err(RadixError::setting(
                "resegmentRange",
                "resegmentRange minimum must be <= maximum",
            ));
        }
        for i in 0..mask.len() {
            if mask[i] && (image[i] < lo || image[i] > hi) {
                mask[i] = false;
            }
        }
    }
    if let Some(k) = outlier_sigma {
        if k <= 0.0 {
            return Err(RadixError::setting(
                "removeOutliers",
                "removeOutliers must be > 0",
            ));
        }
        let mut n = 0.0;
        let mut sum = 0.0;
        for i in 0..mask.len() {
            if mask[i] {
                n += 1.0;
                sum += image[i];
            }
        }
        if n >= 2.0 {
            let mean = sum / n;
            let mut sq = 0.0;
            for i in 0..mask.len() {
                if mask[i] {
                    let d = image[i] - mean;
                    sq += d * d;
                }
            }
            let sigma = (sq / (n - 1.0)).sqrt();
            let lo = mean - k * sigma;
            let hi = mean + k * sigma;
            for i in 0..mask.len() {
                if mask[i] && (image[i] < lo || image[i] > hi) {
                    mask[i] = false;
                }
            }
        }
    }
    if !mask.iter().any(|m| *m) {
        return Err(RadixError::step(
            "resegment",
            "the intensity mask is empty after resegmentation",
        ));
    }
    Ok(mask)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResegmentMode {
    Absolute,
    Relative,
    Sigma,
}

impl ResegmentMode {
    pub fn parse(s: &str) -> Result<Self, RadixError> {
        match s {
            "absolute" => Ok(Self::Absolute),
            "relative" => Ok(Self::Relative),
            "sigma" => Ok(Self::Sigma),
            other => Err(RadixError::setting(
                "resegmentMode",
                format!("unknown resegmentMode: {other}"),
            )),
        }
    }
}

/// Thresholds for `resegmentMode`, then the same closed interval as [`intensity_mask`].
pub fn resegment_mask(
    image: &[f64],
    morph: &[bool],
    range: Option<(f64, f64)>,
    mode: ResegmentMode,
    outlier_sigma: Option<f64>,
) -> Result<Vec<bool>, RadixError> {
    let range = match range {
        Some((lo, hi)) => Some(thresholds(image, morph, lo, hi, mode)?),
        None => None,
    };
    intensity_mask(image, morph, range, outlier_sigma)
}

fn thresholds(
    image: &[f64],
    morph: &[bool],
    lo: f64,
    hi: f64,
    mode: ResegmentMode,
) -> Result<(f64, f64), RadixError> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let values: Vec<f64> = image
        .iter()
        .zip(morph.iter())
        .filter_map(|(v, m)| if *m { Some(*v) } else { None })
        .collect();
    if values.is_empty() {
        return Err(RadixError::step("resegment", "the ROI has no voxels"));
    }
    let (lo, hi) = match mode {
        ResegmentMode::Absolute => (lo, hi),
        ResegmentMode::Relative => {
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (max * lo, max * hi)
        }
        ResegmentMode::Sigma => {
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            let std = var.sqrt();
            (mean + std * lo, mean + std * hi)
        }
    };
    Ok(if lo <= hi { (lo, hi) } else { (hi, lo) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_drops_outside_closed_interval() {
        let image = vec![0.0, 1.0, 6.0, 7.0];
        let morph = vec![true; 4];
        let m = intensity_mask(&image, &morph, Some((1.0, 6.0)), None).unwrap();
        assert_eq!(m, vec![false, true, true, false]);
    }

    #[test]
    fn outlier_filter_drops_far_voxel() {
        let mut image = vec![10.0; 20];
        image[0] = 1000.0;
        let morph = vec![true; 20];
        let m = intensity_mask(&image, &morph, None, Some(3.0)).unwrap();
        assert!(!m[0]);
        assert!(m.iter().skip(1).all(|v| *v));
    }

    #[test]
    fn range_then_outlier_is_intersection() {
        let image = vec![-2000.0, 0.0, 10.0, 20.0, 30.0, 4000.0];
        let morph = vec![true; 6];
        let m = intensity_mask(&image, &morph, Some((-1000.0, 400.0)), Some(3.0)).unwrap();
        assert!(!m[0]);
        assert!(!m[5]);
        assert!(m[1] && m[2] && m[3] && m[4]);
    }

    #[test]
    fn relative_thresholds_use_the_roi_maximum() {
        let image = vec![0.0, 10.0];
        let morph = vec![true, true];
        let m = resegment_mask(
            &image,
            &morph,
            Some((0.5, 1.0)),
            ResegmentMode::Relative,
            None,
        )
        .unwrap();
        assert_eq!(m, vec![false, true]);
    }

    #[test]
    fn sigma_thresholds_drop_the_far_voxel() {
        let image = vec![0.0, 0.0, 0.0, 10.0];
        let morph = vec![true; 4];
        let m = resegment_mask(
            &image,
            &morph,
            Some((-1.0, 1.0)),
            ResegmentMode::Sigma,
            None,
        )
        .unwrap();
        assert_eq!(m, vec![true, true, true, false]);
    }
}
