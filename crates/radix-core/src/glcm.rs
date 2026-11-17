//! GLCM features — PyRadiomics glcm class.

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{idx, in_bounds, log2_eps, mean_maps, texture_angles, EPS};

fn glcm_one(
    gray: &[u32],
    mask: &[bool],
    dim: [usize; 3],
    ng: u32,
    dir: [i32; 3],
    symmetrical: bool,
) -> Vec<f64> {
    let n = ng as usize;
    let mut p = vec![0.0; n * n];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let i0 = idx(z, y, x, dim);
                if !mask[i0] {
                    continue;
                }
                let z2 = z as i32 + dir[0];
                let y2 = y as i32 + dir[1];
                let x2 = x as i32 + dir[2];
                if !in_bounds(z2, y2, x2, dim) {
                    continue;
                }
                let i1 = idx(z2 as usize, y2 as usize, x2 as usize, dim);
                if !mask[i1] {
                    continue;
                }
                let a = gray[i0] as usize - 1;
                let b = gray[i1] as usize - 1;
                p[a * n + b] += 1.0;
                if symmetrical {
                    p[b * n + a] += 1.0;
                }
            }
        }
    }
    p
}

fn features_from_p(p: &[f64], n: usize) -> Result<BTreeMap<String, f64>, RadixError> {
    let sum: f64 = p.iter().sum();
    let mut m = BTreeMap::new();
    if sum <= 0.0 {
        return Ok(m);
    }
    let p: Vec<f64> = p.iter().map(|v| v / sum).collect();
    let mut px = vec![0.0; n];
    let mut py = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            px[i] += p[i * n + j];
            py[j] += p[i * n + j];
        }
    }
    let mut ux = 0.0;
    let mut uy = 0.0;
    for i in 0..n {
        ux += (i + 1) as f64 * px[i];
        uy += (i + 1) as f64 * py[i];
    }
    let mut sx2 = 0.0;
    let mut sy2 = 0.0;
    for i in 0..n {
        sx2 += px[i] * ((i + 1) as f64 - ux).powi(2);
        sy2 += py[i] * ((i + 1) as f64 - uy).powi(2);
    }
    let sx = sx2.sqrt();
    let sy = sy2.sqrt();

    let mut pxy = vec![0.0; 2 * n]; // k = i+j, range 2..=2n → index k-2
    let mut px_y = vec![0.0; n]; // k = |i-j|
    for i in 0..n {
        for j in 0..n {
            let pv = p[i * n + j];
            pxy[i + j] += pv;
            px_y[(i as i32 - j as i32).unsigned_abs() as usize] += pv;
        }
    }

    let mut auto = 0.0;
    let mut cprom = 0.0;
    let mut cshade = 0.0;
    let mut ctend = 0.0;
    let mut contrast = 0.0;
    let mut corr_num = 0.0;
    let mut energy = 0.0;
    let mut entropy = 0.0;
    let mut idm = 0.0;
    let mut idmn = 0.0;
    let mut id = 0.0;
    let mut idn = 0.0;
    let mut inv_var = 0.0;
    let mut maxp: f64 = 0.0;
    let mut sum_sq = 0.0;
    let mut homo1 = 0.0;
    let mut diss = 0.0;
    for i in 0..n {
        for j in 0..n {
            let pv = p[i * n + j];
            let ii = (i + 1) as f64;
            let jj = (j + 1) as f64;
            auto += pv * ii * jj;
            let t = ii + jj - ux - uy;
            cprom += t.powi(4) * pv;
            cshade += t.powi(3) * pv;
            ctend += t.powi(2) * pv;
            contrast += (ii - jj).powi(2) * pv;
            corr_num += pv * ii * jj;
            energy += pv * pv;
            entropy -= pv * log2_eps(pv);
            let d = (ii - jj).abs();
            idm += pv / (1.0 + d * d);
            idmn += pv / (1.0 + (d * d) / (n as f64).powi(2));
            id += pv / (1.0 + d);
            idn += pv / (1.0 + d / n as f64);
            if i != j {
                inv_var += pv / (ii - jj).powi(2);
            }
            maxp = maxp.max(pv);
            sum_sq += (ii - ux).powi(2) * pv;
            homo1 += pv / (1.0 + d);
            diss += pv * d;
        }
    }
    let corr = if sx < EPS || sy < EPS {
        1.0
    } else {
        (corr_num - ux * uy) / (sx * sy)
    };

    let mut da = 0.0;
    let mut dent = 0.0;
    for (k, pk) in px_y.iter().enumerate() {
        da += k as f64 * pk;
        if *pk > 0.0 {
            dent -= pk * log2_eps(*pk);
        }
    }
    let mut dv = 0.0;
    for (k, pk) in px_y.iter().enumerate() {
        dv += (k as f64 - da).powi(2) * pk;
    }

    let mut sa = 0.0;
    let mut sent = 0.0;
    for (k, pk) in pxy.iter().enumerate() {
        let kk = (k + 2) as f64;
        sa += kk * pk;
        if *pk > 0.0 {
            sent -= pk * log2_eps(*pk);
        }
    }
    let mut sv = 0.0;
    for (k, pk) in pxy.iter().enumerate() {
        let kk = (k + 2) as f64;
        sv += (kk - sa).powi(2) * pk;
    }

    // IMC
    let mut hx = 0.0;
    let mut hy = 0.0;
    for i in 0..n {
        if px[i] > 0.0 {
            hx -= px[i] * log2_eps(px[i]);
        }
        if py[i] > 0.0 {
            hy -= py[i] * log2_eps(py[i]);
        }
    }
    let mut hxy1 = 0.0;
    let mut hxy2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let pv = p[i * n + j];
            let pxpy = px[i] * py[j];
            if pxpy > 0.0 {
                hxy1 -= pv * log2_eps(pxpy);
                hxy2 -= pxpy * log2_eps(pxpy);
            }
        }
    }
    let imc1 = if hx.max(hy) > 0.0 {
        (entropy - hxy1) / hx.max(hy)
    } else {
        0.0
    };
    // PyRadiomics: IMC2 = sqrt(1 - exp(-2*(HXY2-HXY)))
    let imc2 = {
        let inner = 1.0 - (-2.0 * (hxy2 - entropy)).exp();
        if inner < 0.0 {
            0.0
        } else {
            inner.sqrt()
        }
    };

    // MCC: second largest eigenvalue of Q, Q_ij = sum_k P_ik P_jk / (px_i py_k)
    m.insert("Autocorrelation".into(), auto);
    m.insert("JointAverage".into(), ux);
    m.insert("ClusterProminence".into(), cprom);
    m.insert("ClusterShade".into(), cshade);
    m.insert("ClusterTendency".into(), ctend);
    m.insert("Contrast".into(), contrast);
    m.insert("Correlation".into(), corr);
    m.insert("DifferenceAverage".into(), da);
    m.insert("DifferenceEntropy".into(), dent);
    m.insert("DifferenceVariance".into(), dv);
    m.insert("Dissimilarity".into(), diss);
    m.insert("JointEnergy".into(), energy);
    m.insert("JointEntropy".into(), entropy);
    m.insert("Homogeneity1".into(), homo1);
    m.insert("Homogeneity2".into(), idm);
    m.insert("Imc1".into(), imc1);
    m.insert("Imc2".into(), imc2);
    m.insert("Idm".into(), idm);
    m.insert("MCC".into(), mcc_value(&p, &px, &py, n)?);
    m.insert("Idmn".into(), idmn);
    m.insert("Id".into(), id);
    m.insert("Idn".into(), idn);
    m.insert("InverseVariance".into(), inv_var);
    m.insert("MaximumProbability".into(), maxp);
    m.insert("SumAverage".into(), sa);
    m.insert("SumVariance".into(), sv);
    m.insert("SumEntropy".into(), sent);
    m.insert("SumSquares".into(), sum_sq);
    Ok(m)
}

fn mcc_value(p: &[f64], px: &[f64], py: &[f64], n: usize) -> Result<f64, RadixError> {
    if n == 1 {
        return Ok(1.0);
    }
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                // PyRadiomics: divide by (px[i]*py[k] + eps); do not symmetrize Q.
                let den = px[i] * py[k] + crate::util::EPS;
                s += p[i * n + k] * p[j * n + k] / den;
            }
            q[i * n + j] = s;
        }
    }
    let mut eigs = crate::eig::eigvals_real(&q, n)?;
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if eigs.len() < 2 {
        return Ok(1.0);
    }
    let lam = eigs[eigs.len() - 2];
    if lam < 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(lam.sqrt())
    }
}

fn glcm_weight(norm: &str, angle: [i32; 3], spacing: [f64; 3]) -> f64 {
    let scaled = [
        (angle[0] as f64).abs() * spacing[0],
        (angle[1] as f64).abs() * spacing[1],
        (angle[2] as f64).abs() * spacing[2],
    ];
    match norm {
        "infinity" => (-scaled.iter().copied().fold(0.0_f64, f64::max).powi(2)).exp(),
        "euclidean" => {
            (-(scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2])).exp()
        }
        "manhattan" => (-(scaled[0] + scaled[1] + scaled[2]).powi(2)).exp(),
        _ => 1.0,
    }
}

pub fn glcm(
    gray: &[u32],
    mask: &[bool],
    dim: [usize; 3],
    ng: u32,
    distances: &[u32],
    symmetrical: bool,
    weighting: Option<&str>,
    spacing: [f64; 3],
) -> Result<BTreeMap<String, f64>, RadixError> {
    let angles = texture_angles(dim, distances, false, 0);
    let n = ng as usize;
    if weighting.is_none() {
        let mut maps = Vec::new();
        for angle in &angles {
            maps.push(features_from_p(
                &glcm_one(gray, mask, dim, ng, *angle, symmetrical),
                n,
            )?);
        }
        return Ok(mean_maps(&maps));
    }
    let norm = weighting.unwrap_or("no_weighting");
    let mut acc = vec![0.0; n * n];
    for angle in &angles {
        let weight = glcm_weight(norm, *angle, spacing);
        let matrix = glcm_one(gray, mask, dim, ng, *angle, symmetrical);
        for (slot, count) in acc.iter_mut().zip(matrix) {
            *slot += count * weight;
        }
    }
    features_from_p(&acc, n)
}

const GLCM_NAMES: &[&str] = &[
    "Autocorrelation",
    "ClusterProminence",
    "ClusterShade",
    "ClusterTendency",
    "Contrast",
    "Correlation",
    "DifferenceAverage",
    "DifferenceEntropy",
    "DifferenceVariance",
    "Dissimilarity",
    "Homogeneity1",
    "Homogeneity2",
    "Id",
    "Idm",
    "Idmn",
    "Idn",
    "Imc1",
    "Imc2",
    "InverseVariance",
    "JointAverage",
    "JointEnergy",
    "JointEntropy",
    "MCC",
    "MaximumProbability",
    "SumAverage",
    "SumEntropy",
    "SumSquares",
    "SumVariance",
];

pub struct GlcmClass;

impl FeatureClass for GlcmClass {
    fn name(&self) -> &'static str {
        "glcm"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        GLCM_NAMES
    }

    fn deprecated(&self) -> &'static [&'static str] {
        &[
            "Dissimilarity",
            "Homogeneity1",
            "Homogeneity2",
            "SumVariance",
        ]
    }

    fn on_original_only(&self) -> bool {
        false
    }

    fn averages_by_slice(&self) -> bool {
        true
    }

    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError> {
        glcm(
            ctx.gray,
            ctx.intensity_mask,
            ctx.dim,
            ctx.ng,
            ctx.distances,
            ctx.symmetrical_glcm,
            ctx.weighting_norm,
            ctx.spacing,
        )
    }
}
