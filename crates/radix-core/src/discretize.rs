//! Gray-level discretization (IBSI vs PyRadiomics fixed-bin-size).

use crate::error::RadixError;
use crate::Mode;

/// Returns per-voxel gray levels (0 = outside mask) and Ng.
///
/// `bin_min` is the IBSI FBS origin \(X_{gl,\min}\). When set (typically the
/// lower bound of the re-segmentation range), edges start there instead of the
/// ROI minimum.
pub fn discretize(
    image: &[f64],
    mask: &[bool],
    mode: Mode,
    bin_width: Option<f64>,
    bin_count: Option<u32>,
    bin_min: Option<f64>,
) -> Result<(Vec<u32>, u32), RadixError> {
    let roi: Vec<f64> = image
        .iter()
        .zip(mask.iter())
        .filter_map(|(v, m)| if *m { Some(*v) } else { None })
        .collect();
    if roi.is_empty() {
        return Err(RadixError::step("discretize", "the ROI has no voxels"));
    }
    let min_v = roi.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = roi.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut out = vec![0u32; image.len()];
    if let Some(ng) = bin_count {
        if ng == 0 {
            return Err(RadixError::setting("binCount", "binCount must be > 0"));
        }
        if (max_v - min_v).abs() < f64::EPSILON {
            for (o, m) in out.iter_mut().zip(mask.iter()) {
                if *m {
                    *o = 1;
                }
            }
            return Ok((out, 1));
        }
        let mut used_max = 1u32;
        for i in 0..image.len() {
            if !mask[i] {
                continue;
            }
            let mut g = ((f64::from(ng) * (image[i] - min_v) / (max_v - min_v)).floor() as i64) + 1;
            if g < 1 {
                g = 1;
            }
            if g > ng as i64 {
                g = ng as i64;
            }
            out[i] = g as u32;
            used_max = used_max.max(out[i]);
        }
        return Ok((out, used_max));
    }

    let bw =
        bin_width.ok_or_else(|| RadixError::setting("binWidth", "set binWidth or binCount"))?;
    if bw <= 0.0 {
        return Err(RadixError::setting("binWidth", "binWidth must be > 0"));
    }
    let ibsi_min = bin_min.unwrap_or(min_v);
    let mut used_max = 1u32;
    for i in 0..image.len() {
        if !mask[i] {
            continue;
        }
        let g = match mode {
            Mode::Pyradiomics => {
                // FAQ: edges equally spaced from 0; lowest intensity → bin 1.
                ((image[i] / bw).floor() - (min_v / bw).floor()) as i64 + 1
            }
            Mode::Ibsi => ((image[i] - ibsi_min) / bw).floor() as i64 + 1,
        };
        let g = g.max(1) as u32;
        out[i] = g;
        used_max = used_max.max(g);
    }
    Ok((out, used_max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibsi_fbs_uses_reseg_lower_bound() {
        let image = vec![-939.0, 393.0];
        let mask = vec![true, true];
        let (gray, ng) =
            discretize(&image, &mask, Mode::Ibsi, Some(25.0), None, Some(-1000.0)).unwrap();
        // floor((-939+1000)/25)+1 = 3; floor((393+1000)/25)+1 = 56
        assert_eq!(gray, vec![3, 56]);
        assert_eq!(ng, 56);
    }
}
