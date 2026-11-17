//! GLRLM features.

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{idx, in_bounds, log2_eps, mean_maps, texture_angles};

fn runs_one(
    gray: &[u32],
    mask: &[bool],
    dim: [usize; 3],
    dir: [i32; 3],
) -> (Vec<(u32, u32)>, usize) {
    let mut runs = Vec::new();
    let mut nv = 0usize;
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let i0 = idx(z, y, x, dim);
                if !mask[i0] {
                    continue;
                }
                nv += 1;
                let pz = z as i32 - dir[0];
                let py = y as i32 - dir[1];
                let px = x as i32 - dir[2];
                if in_bounds(pz, py, px, dim) {
                    let ip = idx(pz as usize, py as usize, px as usize, dim);
                    if mask[ip] && gray[ip] == gray[i0] {
                        continue;
                    }
                }
                let g = gray[i0];
                let mut len = 0u32;
                let mut cz = z as i32;
                let mut cy = y as i32;
                let mut cx = x as i32;
                while in_bounds(cz, cy, cx, dim) {
                    let ic = idx(cz as usize, cy as usize, cx as usize, dim);
                    if !mask[ic] || gray[ic] != g {
                        break;
                    }
                    len += 1;
                    cz += dir[0];
                    cy += dir[1];
                    cx += dir[2];
                }
                runs.push((g, len));
            }
        }
    }
    (runs, nv)
}

fn features_from_runs(runs: &[(u32, u32)], nv: usize, ng: u32) -> BTreeMap<String, f64> {
    let nr = runs.len() as f64;
    let mut m = BTreeMap::new();
    if nr == 0.0 {
        return m;
    }
    let mut sre = 0.0;
    let mut lre = 0.0;
    let mut lgre = 0.0;
    let mut hgre = 0.0;
    let mut srlge = 0.0;
    let mut srhge = 0.0;
    let mut lrlge = 0.0;
    let mut lrhge = 0.0;
    let mut gln_map: BTreeMap<u32, f64> = BTreeMap::new();
    let mut rln_map: BTreeMap<u32, f64> = BTreeMap::new();
    let mut mu_i = 0.0;
    let mut mu_j = 0.0;
    for (g, r) in runs {
        let i = *g as f64;
        let j = *r as f64;
        sre += 1.0 / (j * j);
        lre += j * j;
        lgre += 1.0 / (i * i);
        hgre += i * i;
        srlge += 1.0 / (i * i * j * j);
        srhge += i * i / (j * j);
        lrlge += j * j / (i * i);
        lrhge += i * i * j * j;
        *gln_map.entry(*g).or_insert(0.0) += 1.0;
        *rln_map.entry(*r).or_insert(0.0) += 1.0;
        mu_i += i;
        mu_j += j;
    }
    mu_i /= nr;
    mu_j /= nr;
    let gln: f64 = gln_map.values().map(|v| v * v).sum();
    let rln: f64 = rln_map.values().map(|v| v * v).sum();
    let mut glv = 0.0;
    let mut rv = 0.0;
    let mut rent = 0.0;
    let mut p_map: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for (g, r) in runs {
        *p_map.entry((*g, *r)).or_insert(0.0) += 1.0;
        glv += (*g as f64 - mu_i).powi(2);
        rv += (*r as f64 - mu_j).powi(2);
    }
    for c in p_map.values() {
        let p = c / nr;
        rent -= p * log2_eps(p);
    }
    let _ = ng;
    m.insert("ShortRunEmphasis".into(), sre / nr);
    m.insert("LongRunEmphasis".into(), lre / nr);
    m.insert("GrayLevelNonUniformity".into(), gln / nr);
    m.insert("GrayLevelNonUniformityNormalized".into(), gln / (nr * nr));
    m.insert("RunLengthNonUniformity".into(), rln / nr);
    m.insert("RunLengthNonUniformityNormalized".into(), rln / (nr * nr));
    m.insert("RunPercentage".into(), nr / nv as f64);
    m.insert("GrayLevelVariance".into(), glv / nr);
    m.insert("RunVariance".into(), rv / nr);
    m.insert("RunEntropy".into(), rent);
    m.insert("LowGrayLevelRunEmphasis".into(), lgre / nr);
    m.insert("HighGrayLevelRunEmphasis".into(), hgre / nr);
    m.insert("ShortRunLowGrayLevelEmphasis".into(), srlge / nr);
    m.insert("ShortRunHighGrayLevelEmphasis".into(), srhge / nr);
    m.insert("LongRunLowGrayLevelEmphasis".into(), lrlge / nr);
    m.insert("LongRunHighGrayLevelEmphasis".into(), lrhge / nr);
    m
}

fn glrlm_weight(norm: &str, angle: [i32; 3], spacing: [f64; 3]) -> f64 {
    let scaled = [
        (angle[0] as f64).abs() * spacing[0],
        (angle[1] as f64).abs() * spacing[1],
        (angle[2] as f64).abs() * spacing[2],
    ];
    match norm {
        "infinity" => scaled.iter().copied().fold(0.0_f64, f64::max),
        "euclidean" => {
            (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2]).sqrt()
        }
        "manhattan" => scaled[0] + scaled[1] + scaled[2],
        _ => 1.0,
    }
}

pub fn glrlm(
    gray: &[u32],
    mask: &[bool],
    dim: [usize; 3],
    ng: u32,
    distances: &[u32],
    weighting: Option<&str>,
    spacing: [f64; 3],
) -> BTreeMap<String, f64> {
    let angles = texture_angles(dim, distances, false, 0);
    let maps: Vec<_> = angles
        .iter()
        .map(|d| {
            let (runs, nv) = runs_one(gray, mask, dim, *d);
            features_from_runs(&runs, nv, ng)
        })
        .collect();
    if let Some(norm) = weighting {
        let weights: Vec<f64> = angles
            .iter()
            .map(|angle| glrlm_weight(norm, *angle, spacing))
            .collect();
        return weighted_mean(&maps, &weights);
    }
    mean_maps(&maps)
}

fn weighted_mean(maps: &[BTreeMap<String, f64>], weights: &[f64]) -> BTreeMap<String, f64> {
    let mut acc: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for (map, weight) in maps.iter().zip(weights) {
        if *weight == 0.0 {
            continue;
        }
        for (key, value) in map {
            if value.is_nan() {
                continue;
            }
            let entry = acc.entry(key.clone()).or_insert((0.0, 0.0));
            entry.0 += *value * *weight;
            entry.1 += *weight;
        }
    }
    acc.into_iter()
        .map(|(key, (sum, weight))| {
            (
                key,
                if weight == 0.0 {
                    f64::NAN
                } else {
                    sum / weight
                },
            )
        })
        .collect()
}

const GLRLM_NAMES: &[&str] = &[
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "GrayLevelVariance",
    "HighGrayLevelRunEmphasis",
    "LongRunEmphasis",
    "LongRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelEmphasis",
    "LowGrayLevelRunEmphasis",
    "RunEntropy",
    "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized",
    "RunPercentage",
    "RunVariance",
    "ShortRunEmphasis",
    "ShortRunHighGrayLevelEmphasis",
    "ShortRunLowGrayLevelEmphasis",
];

pub struct GlrlmClass;

impl FeatureClass for GlrlmClass {
    fn name(&self) -> &'static str {
        "glrlm"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        GLRLM_NAMES
    }

    fn deprecated(&self) -> &'static [&'static str] {
        &[]
    }

    fn on_original_only(&self) -> bool {
        false
    }

    fn averages_by_slice(&self) -> bool {
        true
    }

    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError> {
        Ok(glrlm(
            ctx.gray,
            ctx.intensity_mask,
            ctx.dim,
            ctx.ng,
            ctx.distances,
            ctx.weighting_norm,
            ctx.spacing,
        ))
    }
}
