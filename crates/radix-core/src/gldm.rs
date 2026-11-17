//! GLDM features. alpha = 0 (equal gray levels), 26-neighbour.

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{chebyshev_neighbors, idx, in_bounds, log2_eps};

pub fn gldm(
    gray: &[u32],
    mask: &[bool],
    dim: [usize; 3],
    _ng: u32,
    alpha: u32,
    distances: &[u32],
) -> BTreeMap<String, f64> {
    let neigh = chebyshev_neighbors(dim, distances);
    let mut deps: Vec<(u32, u32)> = Vec::new();
    let mut nv = 0usize;
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let i0 = idx(z, y, x, dim);
                if !mask[i0] {
                    continue;
                }
                nv += 1;
                let mut dep = 0u32;
                for d in &neigh {
                    let zz = z as i32 + d[0];
                    let yy = y as i32 + d[1];
                    let xx = x as i32 + d[2];
                    if !in_bounds(zz, yy, xx, dim) {
                        continue;
                    }
                    let ic = idx(zz as usize, yy as usize, xx as usize, dim);
                    if !mask[ic] {
                        continue;
                    }
                    let di = (gray[i0] as i32 - gray[ic] as i32).unsigned_abs();
                    if di <= alpha {
                        dep += 1;
                    }
                }
                deps.push((gray[i0], dep));
            }
        }
    }
    features(&deps, nv)
}

fn features(deps: &[(u32, u32)], nv: usize) -> BTreeMap<String, f64> {
    let n = deps.len() as f64;
    let mut m = BTreeMap::new();
    if n == 0.0 {
        return m;
    }
    let mut sde = 0.0;
    let mut lde = 0.0;
    let mut lgle = 0.0;
    let mut hgle = 0.0;
    let mut sdlgle = 0.0;
    let mut sdhgle = 0.0;
    let mut ldlgle = 0.0;
    let mut ldhgle = 0.0;
    let mut gln_map: BTreeMap<u32, f64> = BTreeMap::new();
    let mut dn_map: BTreeMap<u32, f64> = BTreeMap::new();
    let mut mu_i = 0.0;
    let mut mu_j = 0.0;
    for (g, d) in deps {
        let i = *g as f64;
        let j = (*d as f64) + 1.0; // dependence size including a +1 so j>=1
        sde += 1.0 / (j * j);
        lde += j * j;
        lgle += 1.0 / (i * i);
        hgle += i * i;
        sdlgle += 1.0 / (i * i * j * j);
        sdhgle += i * i / (j * j);
        ldlgle += j * j / (i * i);
        ldhgle += i * i * j * j;
        *gln_map.entry(*g).or_insert(0.0) += 1.0;
        *dn_map.entry(*d).or_insert(0.0) += 1.0;
        mu_i += i;
        mu_j += j;
    }
    mu_i /= n;
    mu_j /= n;
    let gln: f64 = gln_map.values().map(|v| v * v).sum();
    let dn: f64 = dn_map.values().map(|v| v * v).sum();
    let mut glv = 0.0;
    let mut dv = 0.0;
    let mut p_map: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for (g, d) in deps {
        *p_map.entry((*g, *d)).or_insert(0.0) += 1.0;
        let j = (*d as f64) + 1.0;
        glv += (*g as f64 - mu_i).powi(2);
        dv += (j - mu_j).powi(2);
    }
    let mut dent = 0.0;
    for c in p_map.values() {
        let p = c / n;
        dent -= p * log2_eps(p);
    }
    m.insert("SmallDependenceEmphasis".into(), sde / n);
    m.insert("LargeDependenceEmphasis".into(), lde / n);
    m.insert("GrayLevelNonUniformity".into(), gln / n);
    m.insert("GrayLevelNonUniformityNormalized".into(), gln / (n * n));
    m.insert("DependenceNonUniformity".into(), dn / n);
    m.insert("DependenceNonUniformityNormalized".into(), dn / (n * n));
    m.insert("GrayLevelVariance".into(), glv / n);
    m.insert("DependenceVariance".into(), dv / n);
    m.insert("DependenceEntropy".into(), dent);
    m.insert("DependencePercentage".into(), n / nv as f64);
    m.insert("LowGrayLevelEmphasis".into(), lgle / n);
    m.insert("HighGrayLevelEmphasis".into(), hgle / n);
    m.insert("SmallDependenceLowGrayLevelEmphasis".into(), sdlgle / n);
    m.insert("SmallDependenceHighGrayLevelEmphasis".into(), sdhgle / n);
    m.insert("LargeDependenceLowGrayLevelEmphasis".into(), ldlgle / n);
    m.insert("LargeDependenceHighGrayLevelEmphasis".into(), ldhgle / n);
    m
}

const GLDM_NAMES: &[&str] = &[
    "DependenceEntropy",
    "DependenceNonUniformity",
    "DependenceNonUniformityNormalized",
    "DependencePercentage",
    "DependenceVariance",
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "GrayLevelVariance",
    "HighGrayLevelEmphasis",
    "LargeDependenceEmphasis",
    "LargeDependenceHighGrayLevelEmphasis",
    "LargeDependenceLowGrayLevelEmphasis",
    "LowGrayLevelEmphasis",
    "SmallDependenceEmphasis",
    "SmallDependenceHighGrayLevelEmphasis",
    "SmallDependenceLowGrayLevelEmphasis",
];

pub struct GldmClass;

impl FeatureClass for GldmClass {
    fn name(&self) -> &'static str {
        "gldm"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        GLDM_NAMES
    }

    fn deprecated(&self) -> &'static [&'static str] {
        &["DependencePercentage"]
    }

    fn on_original_only(&self) -> bool {
        false
    }

    fn averages_by_slice(&self) -> bool {
        true
    }

    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError> {
        Ok(gldm(
            ctx.gray,
            ctx.intensity_mask,
            ctx.dim,
            ctx.ng,
            ctx.gldm_a,
            ctx.distances,
        ))
    }
}
