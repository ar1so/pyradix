//! GLSZM features — 26-connected (or 8-connected in 2D).

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{idx, in_bounds, log2_eps};

fn neighbors_zone(dim: [usize; 3]) -> Vec<[i32; 3]> {
    let mut v = Vec::new();
    let zvals: Vec<i32> = if dim[0] == 1 { vec![0] } else { vec![-1, 0, 1] };
    for dz in zvals {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dz == 0 && dy == 0 && dx == 0 {
                    continue;
                }
                v.push([dz, dy, dx]);
            }
        }
    }
    v
}

pub fn glszm(gray: &[u32], mask: &[bool], dim: [usize; 3], _ng: u32) -> BTreeMap<String, f64> {
    let n = dim[0] * dim[1] * dim[2];
    let mut seen = vec![false; n];
    let neigh = neighbors_zone(dim);
    let mut zones: Vec<(u32, u32)> = Vec::new();
    let mut nv = 0usize;
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let i0 = idx(z, y, x, dim);
                if !mask[i0] {
                    continue;
                }
                nv += 1;
                if seen[i0] {
                    continue;
                }
                let g = gray[i0];
                let mut stack = vec![(z as i32, y as i32, x as i32)];
                seen[i0] = true;
                let mut size = 0u32;
                while let Some((cz, cy, cx)) = stack.pop() {
                    size += 1;
                    for d in &neigh {
                        let zz = cz + d[0];
                        let yy = cy + d[1];
                        let xx = cx + d[2];
                        if !in_bounds(zz, yy, xx, dim) {
                            continue;
                        }
                        let ic = idx(zz as usize, yy as usize, xx as usize, dim);
                        if seen[ic] || !mask[ic] || gray[ic] != g {
                            continue;
                        }
                        seen[ic] = true;
                        stack.push((zz, yy, xx));
                    }
                }
                zones.push((g, size));
            }
        }
    }
    features(&zones, nv)
}

fn features(zones: &[(u32, u32)], nv: usize) -> BTreeMap<String, f64> {
    let nz = zones.len() as f64;
    let mut m = BTreeMap::new();
    if nz == 0.0 {
        return m;
    }
    let mut sae = 0.0;
    let mut lae = 0.0;
    let mut lglze = 0.0;
    let mut hglze = 0.0;
    let mut salgle = 0.0;
    let mut sahgle = 0.0;
    let mut lalgle = 0.0;
    let mut lahgle = 0.0;
    let mut gln_map: BTreeMap<u32, f64> = BTreeMap::new();
    let mut szn_map: BTreeMap<u32, f64> = BTreeMap::new();
    let mut mu_i = 0.0;
    let mut mu_j = 0.0;
    for (g, s) in zones {
        let i = *g as f64;
        let j = *s as f64;
        sae += 1.0 / (j * j);
        lae += j * j;
        lglze += 1.0 / (i * i);
        hglze += i * i;
        salgle += 1.0 / (i * i * j * j);
        sahgle += i * i / (j * j);
        lalgle += j * j / (i * i);
        lahgle += i * i * j * j;
        *gln_map.entry(*g).or_insert(0.0) += 1.0;
        *szn_map.entry(*s).or_insert(0.0) += 1.0;
        mu_i += i;
        mu_j += j;
    }
    mu_i /= nz;
    mu_j /= nz;
    let gln: f64 = gln_map.values().map(|v| v * v).sum();
    let szn: f64 = szn_map.values().map(|v| v * v).sum();
    let mut glv = 0.0;
    let mut zv = 0.0;
    let mut p_map: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    for (g, s) in zones {
        *p_map.entry((*g, *s)).or_insert(0.0) += 1.0;
        glv += (*g as f64 - mu_i).powi(2);
        zv += (*s as f64 - mu_j).powi(2);
    }
    let mut zent = 0.0;
    for c in p_map.values() {
        let p = c / nz;
        zent -= p * log2_eps(p);
    }
    m.insert("SmallAreaEmphasis".into(), sae / nz);
    m.insert("LargeAreaEmphasis".into(), lae / nz);
    m.insert("GrayLevelNonUniformity".into(), gln / nz);
    m.insert("GrayLevelNonUniformityNormalized".into(), gln / (nz * nz));
    m.insert("SizeZoneNonUniformity".into(), szn / nz);
    m.insert("SizeZoneNonUniformityNormalized".into(), szn / (nz * nz));
    m.insert("ZonePercentage".into(), nz / nv as f64);
    m.insert("GrayLevelVariance".into(), glv / nz);
    m.insert("ZoneVariance".into(), zv / nz);
    m.insert("ZoneEntropy".into(), zent);
    m.insert("LowGrayLevelZoneEmphasis".into(), lglze / nz);
    m.insert("HighGrayLevelZoneEmphasis".into(), hglze / nz);
    m.insert("SmallAreaLowGrayLevelEmphasis".into(), salgle / nz);
    m.insert("SmallAreaHighGrayLevelEmphasis".into(), sahgle / nz);
    m.insert("LargeAreaLowGrayLevelEmphasis".into(), lalgle / nz);
    m.insert("LargeAreaHighGrayLevelEmphasis".into(), lahgle / nz);
    m
}

const GLSZM_NAMES: &[&str] = &[
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "GrayLevelVariance",
    "HighGrayLevelZoneEmphasis",
    "LargeAreaEmphasis",
    "LargeAreaHighGrayLevelEmphasis",
    "LargeAreaLowGrayLevelEmphasis",
    "LowGrayLevelZoneEmphasis",
    "SizeZoneNonUniformity",
    "SizeZoneNonUniformityNormalized",
    "SmallAreaEmphasis",
    "SmallAreaHighGrayLevelEmphasis",
    "SmallAreaLowGrayLevelEmphasis",
    "ZoneEntropy",
    "ZonePercentage",
    "ZoneVariance",
];

pub struct GlszmClass;

impl FeatureClass for GlszmClass {
    fn name(&self) -> &'static str {
        "glszm"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        GLSZM_NAMES
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
        Ok(glszm(ctx.gray, ctx.intensity_mask, ctx.dim, ctx.ng))
    }
}
