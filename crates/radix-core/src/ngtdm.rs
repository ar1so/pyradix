//! NGTDM features (26-neighbour, masked).

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{chebyshev_neighbors, idx, in_bounds, EPS};
use crate::Mode;

pub fn ngtdm(
    gray: &[u32],
    mask: &[bool],
    dim: [usize; 3],
    ng: u32,
    mode: Mode,
    distances: &[u32],
) -> BTreeMap<String, f64> {
    let n = ng as usize;
    let mut count = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut n_valid = 0.0;
    let neigh = chebyshev_neighbors(dim, distances);
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let i0 = idx(z, y, x, dim);
                if !mask[i0] {
                    continue;
                }
                let mut acc = 0.0;
                let mut nn = 0.0;
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
                    acc += gray[ic] as f64;
                    nn += 1.0;
                }
                // PyRadiomics still counts a voxel that has no ROI neighbour and adds a
                // difference of 0. IBSI leaves that voxel out of the matrix.
                if nn == 0.0 && mode != Mode::Pyradiomics {
                    continue;
                }
                n_valid += 1.0;
                let gi = gray[i0] as usize - 1;
                count[gi] += 1.0;
                if nn > 0.0 {
                    s[gi] += (gray[i0] as f64 - acc / nn).abs();
                }
            }
        }
    }
    let mut m = BTreeMap::new();
    if n_valid == 0.0 {
        return m;
    }
    let mut coarseness_den = 0.0;
    let mut contrast_w = 0.0;
    let mut busyness_num = 0.0;
    let mut complexity = 0.0;
    let mut strength_num = 0.0;
    let mut busy_den = 0.0;
    let ng_nz = count.iter().filter(|c| **c > 0.0).count() as f64;
    let sum_s: f64 = s.iter().sum();
    for i in 0..n {
        if count[i] == 0.0 {
            continue;
        }
        let pi = count[i] / n_valid;
        let ii = (i + 1) as f64;
        coarseness_den += pi * s[i];
        busyness_num += pi * s[i];
        for j in 0..n {
            if count[j] == 0.0 {
                continue;
            }
            let pj = count[j] / n_valid;
            let jj = (j + 1) as f64;
            let di = (ii - jj).abs();
            contrast_w += pi * pj * di * di;
            complexity += di * (pi * s[i] + pj * s[j]) / (n_valid * (pi + pj));
            strength_num += (pi + pj) * di * di;
            busy_den += (ii * pi - jj * pj).abs();
        }
    }
    // IBSI QCDE: when ∑ p_i s_i is 0, coarseness is capped at 10^6.
    // PyRadiomics ngtdm.py uses the same cap.
    let coarseness = (1.0 / coarseness_den.max(EPS)).min(1e6);
    let contrast = if ng_nz > 1.0 {
        (1.0 / (ng_nz * (ng_nz - 1.0))) * contrast_w * (sum_s / n_valid)
    } else {
        0.0
    };
    let busyness = if busy_den > EPS && ng_nz > 1.0 {
        busyness_num / busy_den
    } else {
        0.0
    };
    let strength = if sum_s > EPS {
        strength_num / sum_s
    } else {
        0.0
    };
    m.insert("Coarseness".into(), coarseness);
    m.insert("Contrast".into(), contrast);
    m.insert("Busyness".into(), busyness);
    m.insert("Complexity".into(), complexity);
    m.insert("Strength".into(), strength);
    m
}

const NGTDM_NAMES: &[&str] = &[
    "Busyness",
    "Coarseness",
    "Complexity",
    "Contrast",
    "Strength",
];

pub struct NgtdmClass;

impl FeatureClass for NgtdmClass {
    fn name(&self) -> &'static str {
        "ngtdm"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        NGTDM_NAMES
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
        Ok(ngtdm(
            ctx.gray,
            ctx.intensity_mask,
            ctx.dim,
            ctx.ng,
            ctx.mode,
            ctx.distances,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyradiomics_counts_voxels_with_no_neighbour() {
        let gray = vec![1, 2, 0, 0, 3];
        let mask = vec![true, true, false, false, true];
        let dim = [1, 1, 5];
        let pyr = ngtdm(&gray, &mask, dim, 3, Mode::Pyradiomics, &[1]);
        let ibsi = ngtdm(&gray, &mask, dim, 3, Mode::Ibsi, &[1]);
        assert!((pyr["Contrast"] - 4.0 / 27.0).abs() < 1e-12);
        assert!((ibsi["Contrast"] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn zero_neighbourhood_sum_caps_coarseness_at_one_million() {
        let gray = vec![1; 8];
        let mask = vec![true; 8];
        let dim = [2, 2, 2];
        for mode in [Mode::Ibsi, Mode::Pyradiomics] {
            let got = ngtdm(&gray, &mask, dim, 1, mode, &[1]);
            assert!(
                (got["Coarseness"] - 1.0e6).abs() < 1e-6,
                "{mode:?} coarseness {}",
                got["Coarseness"]
            );
        }
    }
}
