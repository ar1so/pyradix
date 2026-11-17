//! First-order statistics — PyRadiomics firstorder class (19 features).

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{percentile, EPS};
use crate::Mode;

pub fn firstorder(
    values: &[f64],
    gray: &[u32],
    ng: u32,
    voxel_volume: f64,
    shift: f64,
    mode: Mode,
) -> BTreeMap<String, f64> {
    let n = values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return BTreeMap::new();
    }
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = values.iter().sum::<f64>() / n;
    let energy: f64 = values.iter().map(|v| (v + shift).powi(2)).sum();
    let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let mu3: f64 = values.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
    let mu4: f64 = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let skew = if std < EPS { 0.0 } else { mu3 / std.powi(3) };
    let kurt_raw = if var < EPS { 0.0 } else { mu4 / var.powi(2) };
    let kurt = match mode {
        Mode::Ibsi => kurt_raw - 3.0,
        Mode::Pyradiomics => kurt_raw,
    };
    let mad: f64 = values.iter().map(|v| (v - mean).abs()).sum::<f64>() / n;
    let p10 = percentile(&sorted, 10.0);
    let p90 = percentile(&sorted, 90.0);
    let band: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| *v >= p10 && *v <= p90)
        .collect();
    let rmad = if band.is_empty() {
        0.0
    } else {
        let m = band.iter().sum::<f64>() / band.len() as f64;
        band.iter().map(|v| (v - m).abs()).sum::<f64>() / band.len() as f64
    };
    let mut hist = vec![0.0; ng as usize];
    for g in gray {
        if *g >= 1 {
            hist[(*g as usize) - 1] += 1.0;
        }
    }
    let mut entropy = 0.0;
    let mut uniformity = 0.0;
    for h in hist {
        if h <= 0.0 {
            continue;
        }
        let p = h / n;
        entropy -= p * (p + EPS).log2();
        uniformity += p * p;
    }

    let mut m = BTreeMap::new();
    m.insert("Energy".into(), energy);
    m.insert("TotalEnergy".into(), energy * voxel_volume);
    m.insert("Entropy".into(), entropy);
    m.insert("Minimum".into(), min);
    m.insert("10Percentile".into(), p10);
    m.insert("90Percentile".into(), p90);
    m.insert("Maximum".into(), max);
    m.insert("Mean".into(), mean);
    m.insert("Median".into(), percentile(&sorted, 50.0));
    m.insert(
        "InterquartileRange".into(),
        percentile(&sorted, 75.0) - percentile(&sorted, 25.0),
    );
    m.insert("Range".into(), max - min);
    m.insert("MeanAbsoluteDeviation".into(), mad);
    m.insert("RobustMeanAbsoluteDeviation".into(), rmad);
    m.insert("RootMeanSquared".into(), (energy / n).sqrt());
    m.insert("StandardDeviation".into(), std);
    m.insert("Skewness".into(), skew);
    m.insert("Kurtosis".into(), kurt);
    m.insert("Variance".into(), var);
    m.insert("Uniformity".into(), uniformity);
    m
}

const FIRSTORDER_NAMES: &[&str] = &[
    "10Percentile",
    "90Percentile",
    "Energy",
    "Entropy",
    "InterquartileRange",
    "Kurtosis",
    "Maximum",
    "Mean",
    "MeanAbsoluteDeviation",
    "Median",
    "Minimum",
    "Range",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "Skewness",
    "StandardDeviation",
    "TotalEnergy",
    "Uniformity",
    "Variance",
];

pub struct FirstOrderClass;

impl FeatureClass for FirstOrderClass {
    fn name(&self) -> &'static str {
        "firstorder"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        FIRSTORDER_NAMES
    }

    fn deprecated(&self) -> &'static [&'static str] {
        &["StandardDeviation"]
    }

    fn on_original_only(&self) -> bool {
        false
    }

    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError> {
        let mut values = Vec::new();
        let mut gray = Vec::new();
        for ((v, g), m) in ctx
            .image
            .iter()
            .zip(ctx.gray.iter())
            .zip(ctx.intensity_mask.iter())
        {
            if *m {
                values.push(*v);
                gray.push(*g);
            }
        }
        let voxel_vol = ctx.spacing[0] * ctx.spacing[1] * ctx.spacing[2];
        let out = firstorder(
            &values,
            &gray,
            ctx.ng,
            voxel_vol,
            ctx.voxel_array_shift,
            ctx.mode,
        );
        if out.is_empty() {
            return Err(RadixError::step("discretize", "the ROI has no voxels"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mode;

    #[test]
    fn mean_min_max() {
        let v = [1.0, 2.0, 3.0];
        let g = [1, 2, 3];
        let m = firstorder(&v, &g, 3, 1.0, 0.0, Mode::Ibsi);
        assert!((m["Mean"] - 2.0).abs() < 1e-12);
        assert_eq!(m["Minimum"], 1.0);
        assert_eq!(m["Maximum"], 3.0);
        assert!((m["Kurtosis"] + 1.5).abs() < 1e-9); // excess of uniform 3-sample
    }
}
