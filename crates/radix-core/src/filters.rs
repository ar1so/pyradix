//! IBSI-2 convolutional filters.
//! Mean S60F: \(g[\boldsymbol{k}]=1/M^D\) on odd support \(M\) (YNOF).

use crate::util::idx;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// Z3VE, \(C=0\).
    Zero,
    /// SIJG (replicate / nearest boundary).
    Nearest,
    /// Z7YO (wrap / periodic).
    Periodic,
    /// ZDTV (mirror, boundary pixels included).
    Mirror,
}

impl Padding {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "zero" | "constant" => Ok(Self::Zero),
            "nearest" | "edge" | "replicate" => Ok(Self::Nearest),
            "periodic" | "wrap" | "circular" => Ok(Self::Periodic),
            "mirror" | "symmetric" => Ok(Self::Mirror),
            other => Err(format!("unknown padding: {other}")),
        }
    }
}

fn mirror_index(i: i32, n: i32) -> usize {
    // IBSI: k' = k mod N if floor(k/N) even, else N-(k mod N + 1).
    let tile = i.div_euclid(n);
    let rem = i.rem_euclid(n);
    if tile.rem_euclid(2) == 0 {
        rem as usize
    } else {
        (n - (rem + 1)) as usize
    }
}

fn sample_axis(i: i32, n: usize, pad: Padding) -> Option<usize> {
    let n = n as i32;
    match pad {
        Padding::Zero => {
            if i >= 0 && i < n {
                Some(i as usize)
            } else {
                None
            }
        }
        Padding::Nearest => Some(i.clamp(0, n - 1) as usize),
        Padding::Periodic => Some(i.rem_euclid(n) as usize),
        Padding::Mirror => Some(mirror_index(i, n)),
    }
}

/// IBSI-2 mean filter (S60F). `ndims` is 2 (axial) or 3. `support` \(M\) must be odd.
pub fn mean(
    image: &[f64],
    dim: [usize; 3],
    support: usize,
    padding: Padding,
    ndims: u8,
) -> Result<Vec<f64>, String> {
    if support < 1 || support % 2 == 0 {
        return Err("mean support M must be odd and ≥ 1".into());
    }
    if ndims != 2 && ndims != 3 {
        return Err("mean ndims must be 2 (axial) or 3".into());
    }
    let n = dim[0] * dim[1] * dim[2];
    if image.len() != n {
        return Err("image length does not match shape".into());
    }
    let r = (support / 2) as i32;
    let axes: &[usize] = if ndims == 3 { &[2, 1, 0] } else { &[2, 1] };
    let mut src = image.to_vec();
    let mut dst = vec![0.0; n];
    let m = support as f64;
    for &axis in axes {
        box_1d(&src, &mut dst, dim, axis, r, m, padding);
        std::mem::swap(&mut src, &mut dst);
    }
    Ok(src)
}

fn box_1d(
    src: &[f64],
    dst: &mut [f64],
    dim: [usize; 3],
    axis: usize,
    r: i32,
    m: f64,
    pad: Padding,
) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut s = 0.0;
                for t in -r..=r {
                    let (zz, yy, xx) = match axis {
                        0 => (z as i32 + t, y as i32, x as i32),
                        1 => (z as i32, y as i32 + t, x as i32),
                        _ => (z as i32, y as i32, x as i32 + t),
                    };
                    let iz = match sample_axis(zz, nz, pad) {
                        Some(v) => v,
                        None => continue,
                    };
                    let iy = match sample_axis(yy, ny, pad) {
                        Some(v) => v,
                        None => continue,
                    };
                    let ix = match sample_axis(xx, nx, pad) {
                        Some(v) => v,
                        None => continue,
                    };
                    s += src[idx(iz, iy, ix, dim)];
                }
                dst[idx(z, y, x, dim)] = s / m;
            }
        }
    }
}

/// IBSI-2 LoG (L6PA, Eq. 4.2). `sigma_mm` is \(\sigma^*\) (physical).
/// Voxel scale \(\sigma=\sigma^*/s\). Support \(M=1+2\lfloor d\sigma+0.5\rfloor\), default \(d=4\).
pub fn log(
    image: &[f64],
    dim: [usize; 3],
    spacing: [f64; 3],
    sigma_mm: f64,
    padding: Padding,
    ndims: u8,
    cutoff: f64,
) -> Result<Vec<f64>, String> {
    if ndims != 2 && ndims != 3 {
        return Err("LoG ndims must be 2 (axial) or 3".into());
    }
    if sigma_mm <= 0.0 || cutoff <= 0.0 {
        return Err("LoG sigma and cutoff must be > 0".into());
    }
    if spacing.iter().any(|s| *s <= 0.0) {
        return Err("spacing must be positive".into());
    }
    let n = dim[0] * dim[1] * dim[2];
    if image.len() != n {
        return Err("image length does not match shape".into());
    }
    // Voxel σ = σ*/s. In-plane (x) spacing; D3 phantoms are 2 mm isotropic.
    let sigma = sigma_mm / spacing[2];
    let r = (cutoff * sigma + 0.5).floor() as i32;
    let d = ndims as i32;
    let norm = (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())).powi(d);
    let s2 = sigma * sigma;
    // (D − ||k||²/σ²) exp(−||k||²/(2σ²)) is a sum of separable products, so the
    // same discrete kernel is three or four 1-D passes instead of one dense box.
    let len = (2 * r + 1) as usize;
    let mut gaussian = vec![0.0; len];
    let mut radial = vec![0.0; len];
    for t in -r..=r {
        let along = (t * t) as f64 / s2;
        let decay = (-along / 2.0).exp();
        let slot = (t + r) as usize;
        gaussian[slot] = decay;
        radial[slot] = along * decay;
    }
    let z_kernel = if ndims == 3 {
        Some(gaussian.as_slice())
    } else {
        None
    };
    let mut acc = laws_separable(image, dim, &gaussian, &gaussian, z_kernel, padding);
    let dims = ndims as f64;
    for value in &mut acc {
        *value *= dims;
    }
    let along_x = laws_separable(image, dim, &radial, &gaussian, z_kernel, padding);
    let along_y = laws_separable(image, dim, &gaussian, &radial, z_kernel, padding);
    for i in 0..n {
        acc[i] -= along_x[i] + along_y[i];
    }
    if ndims == 3 {
        let along_z = laws_separable(
            image,
            dim,
            &gaussian,
            &gaussian,
            Some(radial.as_slice()),
            padding,
        );
        for i in 0..n {
            acc[i] -= along_z[i];
        }
    }
    let scale = -norm / s2;
    for value in &mut acc {
        *value *= scale;
    }
    Ok(acc)
}

/// IBSI-2 Laws 1-D kernels (JTXT), already \(L^2\)-normalised.
fn laws_kernel(name: &str) -> Result<Vec<f64>, String> {
    let (scale, vals): (f64, &[f64]) = match name {
        "L3" => (6.0_f64.sqrt(), &[1.0, 2.0, 1.0]),
        "L5" => (70.0_f64.sqrt(), &[1.0, 4.0, 6.0, 4.0, 1.0]),
        "E3" => (2.0_f64.sqrt(), &[-1.0, 0.0, 1.0]),
        "E5" => (10.0_f64.sqrt(), &[-1.0, -2.0, 0.0, 2.0, 1.0]),
        "S3" => (6.0_f64.sqrt(), &[-1.0, 2.0, -1.0]),
        "S5" => (6.0_f64.sqrt(), &[-1.0, 0.0, 2.0, 0.0, -1.0]),
        "W5" => (10.0_f64.sqrt(), &[-1.0, 2.0, 0.0, -2.0, 1.0]),
        "R5" => (70.0_f64.sqrt(), &[1.0, -4.0, 6.0, -4.0, 1.0]),
        other => return Err(format!("unknown Laws kernel: {other}")),
    };
    Ok(vals.iter().map(|v| v / scale).collect())
}

fn parse_laws_names(s: &str) -> Result<Vec<String>, String> {
    let s: String = s
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .collect::<String>()
        .to_ascii_uppercase();
    if s.len() < 4 || s.len() % 2 != 0 {
        return Err("Laws kernel string must be 2 or 3 tokens (e.g. E5L5S5)".into());
    }
    let mut names = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let token = s[i..i + 2].to_string();
        laws_kernel(&token)?;
        names.push(token);
        i += 2;
    }
    if names.len() != 2 && names.len() != 3 {
        return Err("Laws needs 2 (axial) or 3 (3D) kernels".into());
    }
    Ok(names)
}

fn flip_kernel(k: &[f64]) -> Vec<f64> {
    k.iter().rev().copied().collect()
}

fn pick_kernel(bank: &[Vec<f64>], src: usize, flipped: bool) -> Vec<f64> {
    if flipped {
        flip_kernel(&bank[src])
    } else {
        bank[src].clone()
    }
}

/// Appendix A.1: 2-D right-angle rotations of \(g_1\otimes g_2\) as (x, y).
const ROTS_2D: [[(usize, bool); 2]; 4] = [
    [(0, false), (1, false)], // g1 ⊗ g2
    [(1, true), (0, false)],  // Jg2 ⊗ g1
    [(0, true), (1, true)],   // Jg1 ⊗ Jg2
    [(1, false), (0, true)],  // g2 ⊗ Jg1
];

/// Appendix A.2: 24 cube symmetries of \(g_1\otimes g_2\otimes g_3\) as (x, y, z).
const ROTS_3D: [[(usize, bool); 3]; 24] = [
    [(0, false), (1, false), (2, false)], // (0,0,0)
    [(2, true), (1, false), (0, false)],  // (0, π/2, 0)
    [(0, true), (1, false), (2, true)],   // (0, π, 0)
    [(2, false), (1, false), (0, true)],  // (0, 3π/2, 0)
    [(1, false), (2, false), (0, false)], // (π/2, 0, π/2)
    [(1, false), (2, true), (0, true)],   // (π/2, 0, 3π/2)
    [(1, false), (0, true), (2, false)],  // (π/2, 0, 0)
    [(0, true), (1, true), (2, false)],   // (π, 0, 0)
    [(1, true), (0, false), (2, false)],  // (3π/2, 0, 0)
    [(2, true), (0, true), (1, false)],   // (0, π/2, 3π/2)
    [(2, true), (1, true), (0, true)],    // (0, π/2, π)
    [(2, true), (0, false), (1, true)],   // (0, π/2, π/2)
    [(1, true), (0, true), (2, true)],    // (π/2, π, 0)
    [(0, false), (1, true), (2, true)],   // (π, π, 0)
    [(1, false), (0, false), (2, true)],  // (3π/2, π, 0)
    [(2, false), (0, true), (1, true)],   // (0, 3π/2, π/2)
    [(2, false), (1, true), (0, false)],  // (0, 3π/2, π)
    [(2, false), (0, false), (1, false)], // (0, 3π/2, 3π/2)
    [(0, true), (2, false), (1, false)],  // (π, 0, π/2)
    [(1, true), (2, false), (0, true)],   // (3π/2, 0, π/2)
    [(0, false), (2, false), (1, true)],  // (0, 0, π/2)
    [(0, true), (2, true), (1, true)],    // (π, 0, 3π/2)
    [(1, true), (2, true), (0, false)],   // (3π/2, 0, 3π/2)
    [(0, false), (2, true), (1, false)],  // (0, 0, 3π/2)
];

/// Discrete 1-D convolution along `axis`: \(h[x]=\sum_t g[t]f[x-t]\).
fn conv_1d(
    src: &[f64],
    dst: &mut [f64],
    dim: [usize; 3],
    axis: usize,
    kernel: &[f64],
    pad: Padding,
) {
    let radius = kernel.len() / 2;
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let mut extended = vec![0.0; dim[axis] + 2 * radius];
    match axis {
        2 => {
            for z in 0..nz {
                for y in 0..ny {
                    fill_axis(&mut extended, nx, radius, pad, |x| src[idx(z, y, x, dim)]);
                    let row = z * ny * nx + y * nx;
                    for x in 0..nx {
                        dst[row + x] = dot_extended(&extended, kernel, x, radius);
                    }
                }
            }
        }
        1 => {
            for z in 0..nz {
                for x in 0..nx {
                    fill_axis(&mut extended, ny, radius, pad, |y| src[idx(z, y, x, dim)]);
                    for y in 0..ny {
                        dst[idx(z, y, x, dim)] = dot_extended(&extended, kernel, y, radius);
                    }
                }
            }
        }
        _ => {
            for y in 0..ny {
                for x in 0..nx {
                    fill_axis(&mut extended, nz, radius, pad, |z| src[idx(z, y, x, dim)]);
                    for z in 0..nz {
                        dst[idx(z, y, x, dim)] = dot_extended(&extended, kernel, z, radius);
                    }
                }
            }
        }
    }
}

fn fill_axis(
    extended: &mut [f64],
    n: usize,
    radius: usize,
    pad: Padding,
    sample: impl Fn(usize) -> f64,
) {
    let r = radius as i32;
    for i in 0..extended.len() {
        extended[i] = match sample_axis(i as i32 - r, n, pad) {
            Some(index) => sample(index),
            None => 0.0,
        };
    }
}

fn dot_extended(extended: &[f64], kernel: &[f64], at: usize, radius: usize) -> f64 {
    let mut acc = 0.0;
    for (t, weight) in kernel.iter().enumerate() {
        acc += *weight * extended[at + 2 * radius - t];
    }
    acc
}

fn laws_separable(
    image: &[f64],
    dim: [usize; 3],
    kx: &[f64],
    ky: &[f64],
    kz: Option<&[f64]>,
    pad: Padding,
) -> Vec<f64> {
    let n = image.len();
    let mut src = image.to_vec();
    let mut dst = vec![0.0; n];
    conv_1d(&src, &mut dst, dim, 2, kx, pad);
    std::mem::swap(&mut src, &mut dst);
    conv_1d(&src, &mut dst, dim, 1, ky, pad);
    std::mem::swap(&mut src, &mut dst);
    if let Some(kz) = kz {
        conv_1d(&src, &mut dst, dim, 0, kz, pad);
        std::mem::swap(&mut src, &mut dst);
    }
    src
}

/// IBSI-2 Laws (JTXT / PQSD). `kernels` is e.g. `E5L5S5` (\(g_1\) along \(k_1=x\)).
/// Energy, if requested, is the mean of \(|h|\) after optional max pooling (Appendix A).
pub fn laws(
    image: &[f64],
    dim: [usize; 3],
    kernels: &str,
    padding: Padding,
    rotation_invariant: bool,
    energy_delta: Option<usize>,
) -> Result<Vec<f64>, String> {
    let names = parse_laws_names(kernels)?;
    let n = dim[0] * dim[1] * dim[2];
    if image.len() != n {
        return Err("image length does not match shape".into());
    }
    let bank: Vec<Vec<f64>> = names
        .iter()
        .map(|name| laws_kernel(name))
        .collect::<Result<_, _>>()?;
    let ndims = names.len();
    let mut response = if ndims == 2 {
        let rots: &[[(usize, bool); 2]] = if rotation_invariant {
            &ROTS_2D
        } else {
            &ROTS_2D[..1]
        };
        let mut acc: Option<Vec<f64>> = None;
        for rot in rots {
            let kx = pick_kernel(&bank, rot[0].0, rot[0].1);
            let ky = pick_kernel(&bank, rot[1].0, rot[1].1);
            let h = laws_separable(image, dim, &kx, &ky, None, padding);
            match acc {
                None => acc = Some(h),
                Some(ref mut a) => {
                    for (ai, hi) in a.iter_mut().zip(h.iter()) {
                        if *hi > *ai {
                            *ai = *hi;
                        }
                    }
                }
            }
        }
        acc.unwrap()
    } else {
        let rots: &[[(usize, bool); 3]] = if rotation_invariant {
            &ROTS_3D
        } else {
            &ROTS_3D[..1]
        };
        let mut acc: Option<Vec<f64>> = None;
        for rot in rots {
            let kx = pick_kernel(&bank, rot[0].0, rot[0].1);
            let ky = pick_kernel(&bank, rot[1].0, rot[1].1);
            let kz = pick_kernel(&bank, rot[2].0, rot[2].1);
            let h = laws_separable(image, dim, &kx, &ky, Some(&kz), padding);
            match acc {
                None => acc = Some(h),
                Some(ref mut a) => {
                    for (ai, hi) in a.iter_mut().zip(h.iter()) {
                        if *hi > *ai {
                            *ai = *hi;
                        }
                    }
                }
            }
        }
        acc.unwrap()
    };
    if let Some(delta) = energy_delta {
        let support = 2 * delta + 1;
        for v in response.iter_mut() {
            *v = v.abs();
        }
        response = mean(&response, dim, support, padding, ndims as u8)?;
    }
    Ok(response)
}

fn gabor_support(n: usize) -> usize {
    // IBSI-2 Eq. 4.5: M = N if odd, N+1 if even.
    if n % 2 == 0 {
        n + 1
    } else {
        n
    }
}

fn gabor_complex(k1: i32, k2: i32, sigma: f64, lambda: f64, gamma: f64, theta: f64) -> (f64, f64) {
    let (c, s) = (theta.cos(), theta.sin());
    let t1 = k1 as f64 * c + k2 as f64 * s;
    let t2 = k1 as f64 * s - k2 as f64 * c;
    let env = (-(t1 * t1 + gamma * gamma * t2 * t2) / (2.0 * sigma * sigma)).exp();
    let phase = 2.0 * std::f64::consts::PI * t1 / lambda;
    (env * phase.cos(), env * phase.sin())
}

fn shift_axis(z: i32, y: i32, x: i32, axis: usize, t: i32) -> (i32, i32, i32) {
    match axis {
        0 => (z - t, y, x),
        1 => (z, y - t, x),
        _ => (z, y, x - t),
    }
}

/// 2-D Gabor modulus in one orthogonal plane. `axis1` is \(k_1\), `axis2` is \(k_2\).
fn gabor_plane(
    image: &[f64],
    dim: [usize; 3],
    axis1: usize,
    axis2: usize,
    sigma: f64,
    lambda: f64,
    gamma: f64,
    theta: f64,
    pad: Padding,
) -> Vec<f64> {
    let n1 = dim[axis1];
    let n2 = dim[axis2];
    let m1 = gabor_support(n1);
    let m2 = gabor_support(n2);
    let r1 = (m1 / 2) as i32;
    let r2 = (m2 / 2) as i32;
    let mut kernel = Vec::with_capacity(m1 * m2);
    for k2 in -r2..=r2 {
        for k1 in -r1..=r1 {
            let (gr, gi) = gabor_complex(k1, k2, sigma, lambda, gamma, theta);
            if gr.abs() + gi.abs() < 1e-18 {
                continue;
            }
            kernel.push((k1, k2, (gr, gi)));
        }
    }
    let n = image.len();
    let mut out = vec![0.0; n];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let mut re = 0.0;
                let mut im = 0.0;
                for &(k1, k2, (gr, gi)) in &kernel {
                    let (zz, yy, xx) = shift_axis(z as i32, y as i32, x as i32, axis1, k1);
                    let (zz, yy, xx) = shift_axis(zz, yy, xx, axis2, k2);
                    let Some(iz) = sample_axis(zz, dim[0], pad) else {
                        continue;
                    };
                    let Some(iy) = sample_axis(yy, dim[1], pad) else {
                        continue;
                    };
                    let Some(ix) = sample_axis(xx, dim[2], pad) else {
                        continue;
                    };
                    let v = image[idx(iz, iy, ix, dim)];
                    re += gr * v;
                    im += gi * v;
                }
                out[idx(z, y, x, dim)] = (re * re + im * im).sqrt();
            }
        }
    }
    out
}

fn gabor_thetas(
    rotation_invariant: bool,
    theta: f64,
    delta_theta: Option<f64>,
) -> Result<Vec<f64>, String> {
    if !rotation_invariant {
        return Ok(vec![theta]);
    }
    let step =
        delta_theta.ok_or_else(|| "Gabor rotation invariance requires delta_theta".to_string())?;
    if step <= 0.0 || step > std::f64::consts::PI {
        return Err("Gabor delta_theta must be in (0, π]".into());
    }
    let mut out = Vec::new();
    let mut t = 0.0;
    while t < std::f64::consts::PI - 1e-12 {
        out.push(t);
        t += step;
    }
    if out.is_empty() {
        return Err("Gabor orientation set is empty".into());
    }
    Ok(out)
}

/// IBSI-2 Gabor (Q88H, Eq. 4.3). Returns the modulus response.
/// `theta` is unused when `rotation_invariant` (orientations \(0,\Delta\theta,\dots\) on \([0,\pi)\)).
/// `orthogonal_planes` averages axial \((x,y)\), coronal \((x,z)\), sagittal \((y,z)\).
pub fn gabor(
    image: &[f64],
    dim: [usize; 3],
    spacing: [f64; 3],
    sigma_mm: f64,
    lambda_mm: f64,
    gamma: f64,
    theta: f64,
    padding: Padding,
    rotation_invariant: bool,
    delta_theta: Option<f64>,
    orthogonal_planes: bool,
) -> Result<Vec<f64>, String> {
    if sigma_mm <= 0.0 || lambda_mm <= 0.0 || gamma <= 0.0 {
        return Err("Gabor sigma, lambda and gamma must be > 0".into());
    }
    if spacing.iter().any(|s| *s <= 0.0) {
        return Err("spacing must be positive".into());
    }
    let n = dim[0] * dim[1] * dim[2];
    if image.len() != n {
        return Err("image length does not match shape".into());
    }
    let thetas = gabor_thetas(rotation_invariant, theta, delta_theta)?;
    // (k1 axis, k2 axis): axial, coronal, sagittal.
    let planes: &[(usize, usize)] = if orthogonal_planes {
        &[(2, 1), (2, 0), (1, 0)]
    } else {
        &[(2, 1)]
    };
    let mut acc = vec![0.0; n];
    let n_maps = (planes.len() * thetas.len()) as f64;
    for &(axis1, axis2) in planes {
        let s = spacing[axis1];
        let sigma = sigma_mm / s;
        let lambda = lambda_mm / s;
        for &th in &thetas {
            let h = gabor_plane(image, dim, axis1, axis2, sigma, lambda, gamma, th, padding);
            for (a, v) in acc.iter_mut().zip(h.iter()) {
                *a += *v;
            }
        }
    }
    for a in acc.iter_mut() {
        *a /= n_maps;
    }
    Ok(acc)
}

/// PyWavelets analysis kernels (IBSI: “from the PyWavelets website”).
fn wavelet_pair(family: &str) -> Result<(Vec<f64>, Vec<f64>), String> {
    match family.to_ascii_lowercase().as_str() {
        "haar" | "db1" => {
            let s = std::f64::consts::SQRT_2;
            Ok((vec![1.0 / s, 1.0 / s], vec![-1.0 / s, 1.0 / s]))
        }
        "db2" | "daubechies2" => Ok((
            vec![
                -0.12940952255126037,
                0.2241438680420134,
                0.8365163037378079,
                0.48296291314453416,
            ],
            vec![
                -0.48296291314453416,
                0.8365163037378079,
                -0.2241438680420134,
                -0.12940952255126037,
            ],
        )),
        "coif1" | "coiflet1" => Ok((
            vec![
                -0.01565572813546454,
                -0.0727326195128539,
                0.38486484686420286,
                0.8525720202123255,
                0.3378976624578092,
                -0.0727326195128539,
            ],
            vec![
                0.0727326195128539,
                0.3378976624578092,
                -0.8525720202123255,
                0.38486484686420286,
                0.0727326195128539,
                -0.01565572813546454,
            ],
        )),
        other => Err(format!("unknown wavelet family: {other}")),
    }
}

/// À trous, IBSI “second alternative” (MATLAB / PyWavelets): zeros after every tap.
fn atrous(base: &[f64], level: u32) -> Vec<f64> {
    if level <= 1 {
        return base.to_vec();
    }
    let holes = (1usize << (level - 1)) - 1;
    let mut out = Vec::with_capacity(base.len() * (holes + 1));
    for &c in base {
        out.push(c);
        out.resize(out.len() + holes, 0.0);
    }
    out
}

fn wavelet_odd(base: &[f64], level: u32) -> Vec<f64> {
    let mut k = atrous(base, level);
    if k.len() % 2 == 0 {
        k.push(0.0);
    }
    k
}

fn parse_wavelet_combo(combo: &str) -> Result<Vec<bool>, String> {
    let s: String = combo
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .collect::<String>()
        .to_ascii_uppercase();
    if s.len() != 2 && s.len() != 3 {
        return Err("wavelet combo must be 2 (axial) or 3 (3D) letters, e.g. LHL".into());
    }
    let mut out = Vec::new();
    for c in s.chars() {
        match c {
            'L' => out.push(false),
            'H' => out.push(true),
            _ => return Err(format!("wavelet combo letters must be L or H, got {c}")),
        }
    }
    Ok(out)
}

fn combo_bank(lo: &[f64], hi: &[f64], high: &[bool]) -> Vec<Vec<f64>> {
    high.iter()
        .map(|h| if *h { hi.to_vec() } else { lo.to_vec() })
        .collect()
}

fn wavelet_orient(
    image: &[f64],
    dim: [usize; 3],
    bank: &[Vec<f64>],
    pad: Padding,
    rot: &[(usize, bool)],
) -> Vec<f64> {
    let kx = pick_kernel(bank, rot[0].0, rot[0].1);
    let ky = pick_kernel(bank, rot[1].0, rot[1].1);
    if bank.len() == 2 {
        laws_separable(image, dim, &kx, &ky, None, pad)
    } else {
        let kz = pick_kernel(bank, rot[2].0, rot[2].1);
        laws_separable(image, dim, &kx, &ky, Some(&kz), pad)
    }
}

/// IBSI-2 undecimated separable wavelet (25BO / CVCQ).
/// `combo` is e.g. `LHL` (\(g_L\) along \(k_1=x\), \(g_H\) along \(k_2=y\), \(g_L\) along \(k_3=z\)).
/// Level \(>1\) filters the previous LLL approximation with à-trous kernels.
/// Rotation invariance: Appendix A, average pooling (Table 6.1).
pub fn wavelet(
    image: &[f64],
    dim: [usize; 3],
    family: &str,
    combo: &str,
    level: u32,
    padding: Padding,
    rotation_invariant: bool,
) -> Result<Vec<f64>, String> {
    if level < 1 {
        return Err("wavelet level must be ≥ 1".into());
    }
    let n = dim[0] * dim[1] * dim[2];
    if image.len() != n {
        return Err("image length does not match shape".into());
    }
    let high = parse_wavelet_combo(combo)?;
    let (lo0, hi0) = wavelet_pair(family)?;
    let ndims = high.len();
    let identity_2 = [[(0, false), (1, false)]];
    let identity_3 = [[(0, false), (1, false), (2, false)]];

    let apply_level = |src: &[f64], lvl: u32, highpass: &[bool], rot_inv: bool| -> Vec<f64> {
        let lo = wavelet_odd(&lo0, lvl);
        let hi = wavelet_odd(&hi0, lvl);
        let bank = combo_bank(&lo, &hi, highpass);
        if ndims == 2 {
            let rots: &[[(usize, bool); 2]] = if rot_inv { &ROTS_2D } else { &identity_2 };
            let mut acc: Option<Vec<f64>> = None;
            let nrot = rots.len() as f64;
            for rot in rots {
                let h = wavelet_orient(src, dim, &bank, padding, rot);
                match acc {
                    None => acc = Some(h),
                    Some(ref mut a) => {
                        for (ai, hi) in a.iter_mut().zip(h.iter()) {
                            *ai += *hi;
                        }
                    }
                }
            }
            let mut out = acc.unwrap();
            if rot_inv {
                for v in out.iter_mut() {
                    *v /= nrot;
                }
            }
            out
        } else {
            let rots: &[[(usize, bool); 3]] = if rot_inv { &ROTS_3D } else { &identity_3 };
            let mut acc: Option<Vec<f64>> = None;
            let nrot = rots.len() as f64;
            for rot in rots {
                let h = wavelet_orient(src, dim, &bank, padding, rot);
                match acc {
                    None => acc = Some(h),
                    Some(ref mut a) => {
                        for (ai, hi) in a.iter_mut().zip(h.iter()) {
                            *ai += *hi;
                        }
                    }
                }
            }
            let mut out = acc.unwrap();
            if rot_inv {
                for v in out.iter_mut() {
                    *v /= nrot;
                }
            }
            out
        }
    };

    // IBSI troubleshooting: corresponding orientations through the cascade.
    // Equivalent when we average at each level (linear).
    let approx: Vec<bool> = vec![false; ndims];
    let mut current = image.to_vec();
    for j in 1..level {
        current = apply_level(&current, j, &approx, rotation_invariant);
    }
    Ok(apply_level(&current, level, &high, rotation_invariant))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulse_3d_m3_zero() {
        let dim = [5, 5, 5];
        let mut img = vec![0.0; 125];
        img[idx(2, 2, 2, dim)] = 1.0;
        let out = mean(&img, dim, 3, Padding::Zero, 3).unwrap();
        assert!((out[idx(2, 2, 2, dim)] - 1.0 / 27.0).abs() < 1e-12);
        assert!((out[idx(1, 1, 1, dim)] - 1.0 / 27.0).abs() < 1e-12);
        assert!(out[idx(0, 0, 0, dim)].abs() < 1e-15);
    }

    #[test]
    fn mirror_repeats_boundary() {
        assert_eq!(mirror_index(-1, 64), 0);
        assert_eq!(mirror_index(64, 64), 63);
        assert_eq!(mirror_index(-2, 64), 1);
    }

    #[test]
    fn log_kernel_sums_near_zero() {
        // Constant image + periodic pad → response ≈ 0 (IBSI troubleshooting note).
        let dim = [7, 7, 7];
        let img = vec![1.0; 7 * 7 * 7];
        let out = log(&img, dim, [2.0, 2.0, 2.0], 3.0, Padding::Periodic, 3, 4.0).unwrap();
        let mean: f64 = out.iter().sum::<f64>() / out.len() as f64;
        assert!(mean.abs() < 1e-3, "mean={mean}");
    }

    #[test]
    fn laws_e5l5s5_impulse_is_outer_product() {
        let dim = [7, 7, 7];
        let mut img = vec![0.0; 7 * 7 * 7];
        img[idx(3, 3, 3, dim)] = 1.0;
        let out = laws(&img, dim, "E5L5S5", Padding::Zero, false, None).unwrap();
        let e5 = laws_kernel("E5").unwrap();
        let l5 = laws_kernel("L5").unwrap();
        let s5 = laws_kernel("S5").unwrap();
        for dz in -2..=2 {
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let expected =
                        e5[(dx + 2) as usize] * l5[(dy + 2) as usize] * s5[(dz + 2) as usize];
                    let got =
                        out[idx((3 + dz) as usize, (3 + dy) as usize, (3 + dx) as usize, dim)];
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "d=({dz},{dy},{dx}) {got} vs {expected}"
                    );
                }
            }
        }
        assert!(out[idx(0, 0, 0, dim)].abs() < 1e-15);
    }

    #[test]
    fn gabor_impulse_modulus_matches_kernel() {
        let dim = [3, 5, 5];
        let mut img = vec![0.0; 75];
        img[idx(1, 2, 2, dim)] = 1.0;
        let sigma_mm = 4.0;
        let lambda_mm = 4.0;
        let spacing = [2.0, 2.0, 2.0];
        let theta = std::f64::consts::PI / 3.0;
        let out = gabor(
            &img,
            dim,
            spacing,
            sigma_mm,
            lambda_mm,
            0.5,
            theta,
            Padding::Zero,
            false,
            None,
            false,
        )
        .unwrap();
        let sigma = sigma_mm / spacing[2];
        let lambda = lambda_mm / spacing[2];
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (re, im) = gabor_complex(dx, dy, sigma, lambda, 0.5, theta);
                let expected = (re * re + im * im).sqrt();
                let got = out[idx(1, (2 + dy) as usize, (2 + dx) as usize, dim)];
                assert!(
                    (got - expected).abs() < 1e-12,
                    "d=({dy},{dx}) {got} vs {expected}"
                );
            }
        }
        assert!(out[idx(0, 2, 2, dim)].abs() < 1e-15);
    }

    #[test]
    fn haar_lhl_impulse_is_outer_product() {
        let dim = [5, 5, 5];
        let mut img = vec![0.0; 125];
        img[idx(2, 2, 2, dim)] = 1.0;
        let out = wavelet(&img, dim, "haar", "LHL", 1, Padding::Zero, false).unwrap();
        let lo = wavelet_odd(&wavelet_pair("haar").unwrap().0, 1);
        let hi = wavelet_odd(&wavelet_pair("haar").unwrap().1, 1);
        let r = (lo.len() / 2) as i32;
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let expected =
                        lo[(dx + r) as usize] * hi[(dy + r) as usize] * lo[(dz + r) as usize];
                    let got =
                        out[idx((2 + dz) as usize, (2 + dy) as usize, (2 + dx) as usize, dim)];
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "d=({dz},{dy},{dx}) {got} vs {expected}"
                    );
                }
            }
        }
    }
}
