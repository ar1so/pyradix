//! PyRadiomics 3.1.0 gradient magnitude, LBP2D, and LBP3D.

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::ImageType;
use crate::spline;
use crate::Mode;

const PYRAD: &[Mode] = &[Mode::Pyradiomics];
const PI: f64 = std::f64::consts::PI;

pub struct Gradient;
pub struct Lbp2D;
pub struct Lbp3D;

fn setting<'a>(settings: &'a [(&str, &str)], key: &str) -> Option<&'a str> {
    settings
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, value)| *value)
}

fn check_len(image: &[f64], dim: [usize; 3], name: &str) -> Result<(), RadixError> {
    if image.len() != dim[0] * dim[1] * dim[2] {
        return Err(RadixError::image_type(
            name,
            "image length does not match shape",
        ));
    }
    Ok(())
}

fn parse_f64(name: &str, key: &str, raw: &str) -> Result<f64, RadixError> {
    raw.parse()
        .map_err(|_| RadixError::image_type(name, format!("{key} {raw} is not a number")))
}

fn parse_usize(name: &str, key: &str, raw: &str) -> Result<usize, RadixError> {
    let value: i64 = raw
        .parse()
        .map_err(|_| RadixError::image_type(name, format!("{key} {raw} is not an integer")))?;
    if value < 0 {
        return Err(RadixError::image_type(name, format!("{key} must be >= 0")));
    }
    Ok(value as usize)
}

fn at(image: &[f64], dim: [usize; 3], z: isize, y: isize, x: isize) -> f64 {
    let z = z.clamp(0, dim[0] as isize - 1) as usize;
    let y = y.clamp(0, dim[1] as isize - 1) as usize;
    let x = x.clamp(0, dim[2] as isize - 1) as usize;
    image[z * dim[1] * dim[2] + y * dim[2] + x]
}

/// ITK `GradientMagnitudeImageFilter`. Spacing is (z, y, x).
pub fn gradient_magnitude(
    image: &[f64],
    dim: [usize; 3],
    spacing_zyx: [f64; 3],
    use_spacing: bool,
) -> Vec<f64> {
    let (sx, sy, sz) = if use_spacing {
        (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    } else {
        (1.0, 1.0, 1.0)
    };
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let mut out = vec![0.0; image.len()];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let gx = (at(image, dim, z as isize, y as isize, x as isize + 1)
                    - at(image, dim, z as isize, y as isize, x as isize - 1))
                    / (2.0 * sx);
                let gy = (at(image, dim, z as isize, y as isize + 1, x as isize)
                    - at(image, dim, z as isize, y as isize - 1, x as isize))
                    / (2.0 * sy);
                let gz = (at(image, dim, z as isize + 1, y as isize, x as isize)
                    - at(image, dim, z as isize - 1, y as isize, x as isize))
                    / (2.0 * sz);
                out[z * ny * nx + y * nx + x] = (gx * gx + gy * gy + gz * gz).sqrt();
            }
        }
    }
    out
}

fn round5(value: f64) -> f64 {
    (value * 1e5).round() / 1e5
}

fn pixel2d(image: &[f64], rows: usize, cols: usize, r: i64, c: i64) -> f64 {
    if r < 0 || c < 0 || r >= rows as i64 || c >= cols as i64 {
        return 0.0;
    }
    image[r as usize * cols + c as usize]
}

fn bilinear(image: &[f64], rows: usize, cols: usize, r: f64, c: f64) -> f64 {
    let minr = r.floor() as i64;
    let minc = c.floor() as i64;
    let maxr = r.ceil() as i64;
    let maxc = c.ceil() as i64;
    let dr = r - minr as f64;
    let dc = c - minc as f64;
    let top = (1.0 - dc) * pixel2d(image, rows, cols, minr, minc)
        + dc * pixel2d(image, rows, cols, minr, maxc);
    let bottom = (1.0 - dc) * pixel2d(image, rows, cols, maxr, minc)
        + dc * pixel2d(image, rows, cols, maxr, maxc);
    (1.0 - dr) * top + dr * bottom
}

fn bit_rotate_right(value: u64, length: usize) -> u64 {
    let mask = if length >= 64 {
        u64::MAX
    } else {
        (1u64 << length) - 1
    };
    let value = value & mask;
    ((value >> 1) | ((value & 1) << (length - 1))) & mask
}

/// One plane, matching `skimage.feature.local_binary_pattern`.
fn lbp_plane(
    image: &[f64],
    rows: usize,
    cols: usize,
    samples: usize,
    radius: f64,
    method: &str,
) -> Vec<f64> {
    let mut rows_off = Vec::with_capacity(samples);
    let mut cols_off = Vec::with_capacity(samples);
    for i in 0..samples {
        let angle = 2.0 * PI * i as f64 / samples as f64;
        rows_off.push(round5(-radius * angle.sin()));
        cols_off.push(round5(radius * angle.cos()));
    }
    let mut out = vec![0.0; rows * cols];
    let mut texture = vec![0.0; samples];
    let mut bits = vec![0u8; samples];
    for r in 0..rows {
        for c in 0..cols {
            for i in 0..samples {
                texture[i] = bilinear(
                    image,
                    rows,
                    cols,
                    r as f64 + rows_off[i],
                    c as f64 + cols_off[i],
                );
                bits[i] = u8::from(texture[i] - image[r * cols + c] >= 0.0);
            }
            out[r * cols + c] = match method {
                "var" => {
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;
                    for value in &texture {
                        sum += value;
                        sum_sq += value * value;
                    }
                    let var = (sum_sq - (sum * sum) / samples as f64) / samples as f64;
                    if var == 0.0 {
                        f64::NAN
                    } else {
                        var
                    }
                }
                "uniform" => {
                    let mut changes = 0;
                    for i in 0..samples - 1 {
                        if bits[i] != bits[i + 1] {
                            changes += 1;
                        }
                    }
                    if changes <= 2 {
                        bits.iter().map(|bit| *bit as f64).sum()
                    } else {
                        samples as f64 + 1.0
                    }
                }
                "default" | "ror" => {
                    let mut code = 0u64;
                    for (i, bit) in bits.iter().enumerate() {
                        if *bit == 1 {
                            code += 1u64 << i;
                        }
                    }
                    if method == "ror" {
                        let mut best = code;
                        let mut current = code;
                        for _ in 1..samples {
                            current = bit_rotate_right(current, samples);
                            best = best.min(current);
                        }
                        best as f64
                    } else {
                        code as f64
                    }
                }
                _ => f64::NAN,
            };
        }
    }
    out
}

fn plane(volume: &[f64], dim: [usize; 3], axis: usize, index: usize) -> (Vec<f64>, usize, usize) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    match axis {
        0 => {
            let start = index * ny * nx;
            (volume[start..start + ny * nx].to_vec(), ny, nx)
        }
        1 => {
            let mut plane = vec![0.0; nz * nx];
            for z in 0..nz {
                for x in 0..nx {
                    plane[z * nx + x] = volume[z * ny * nx + index * nx + x];
                }
            }
            (plane, nz, nx)
        }
        _ => {
            let mut plane = vec![0.0; ny * nz];
            for y in 0..ny {
                for z in 0..nz {
                    plane[y * nz + z] = volume[z * ny * nx + y * nx + index];
                }
            }
            (plane, ny, nz)
        }
    }
}

fn write_plane(volume: &mut [f64], dim: [usize; 3], axis: usize, index: usize, plane: &[f64]) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    match axis {
        0 => {
            let start = index * ny * nx;
            volume[start..start + ny * nx].copy_from_slice(plane);
        }
        1 => {
            for z in 0..nz {
                for x in 0..nx {
                    volume[z * ny * nx + index * nx + x] = plane[z * nx + x];
                }
            }
        }
        _ => {
            for y in 0..ny {
                for z in 0..nz {
                    volume[z * ny * nx + y * nx + index] = plane[y * nz + z];
                }
            }
        }
    }
}

pub fn lbp_2d(
    image: &[f64],
    dim: [usize; 3],
    axis: usize,
    samples: usize,
    radius: f64,
    method: &str,
) -> Vec<f64> {
    let count = dim[axis];
    let mut out = image.to_vec();
    for index in 0..count {
        let (plane, rows, cols) = plane(image, dim, axis, index);
        let filtered = lbp_plane(&plane, rows, cols, samples, radius, method);
        write_plane(&mut out, dim, axis, index, &filtered);
    }
    out
}

fn factorial(n: i32) -> f64 {
    let mut value = 1.0;
    for k in 2..=n.max(0) {
        value *= k as f64;
    }
    value
}

fn assoc_legendre(n: i32, m: i32, x: f64) -> f64 {
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if n == m {
        return pmm;
    }
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if n == m + 1 {
        return pmmp1;
    }
    for ll in m + 2..=n {
        let pll = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pmmp1
}

fn lpmv(m: i32, n: i32, x: f64) -> f64 {
    if m < 0 {
        let mp = -m;
        let sign = if mp % 2 == 0 { 1.0 } else { -1.0 };
        return sign * factorial(n - mp) / factorial(n + mp) * assoc_legendre(n, mp, x);
    }
    assoc_legendre(n, m, x)
}

fn sph_harm(m: i32, n: i32, theta: f64, phi: f64) -> (f64, f64) {
    let p = lpmv(m, n, phi.cos());
    let pref = ((2.0 * n as f64 + 1.0) / (4.0 * PI) * factorial(n - m) / factorial(n + m)).sqrt();
    let phase = m as f64 * theta;
    (pref * p * phase.cos(), pref * p * phase.sin())
}

const ICO_VERTICES: [[f64; 3]; 12] = [
    [-0.5257311121191336, 0.85065080835204, 0.0],
    [0.5257311121191336, 0.85065080835204, 0.0],
    [-0.5257311121191336, -0.85065080835204, 0.0],
    [0.5257311121191336, -0.85065080835204, 0.0],
    [0.0, -0.5257311121191336, 0.85065080835204],
    [0.0, 0.5257311121191336, 0.85065080835204],
    [0.0, -0.5257311121191336, -0.85065080835204],
    [0.0, 0.5257311121191336, -0.85065080835204],
    [0.85065080835204, 0.0, -0.5257311121191336],
    [0.85065080835204, 0.0, 0.5257311121191336],
    [-0.85065080835204, 0.0, -0.5257311121191336],
    [-0.85065080835204, 0.0, 0.5257311121191336],
];

const ICO_FACES: [[usize; 3]; 20] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

fn project(vertices: &mut [[f64; 3]], radius: f64) {
    for vertex in vertices {
        let norm = (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt();
        let scale = radius / norm;
        vertex[0] *= scale;
        vertex[1] *= scale;
        vertex[2] *= scale;
    }
}

/// Trimesh `unique_rows` packs each edge into a `uint64` and sorts on that hash.
fn edge_hash(edge: (usize, usize)) -> u64 {
    let threshold = 1u64 << 31;
    let low = edge.0 as u64 + threshold;
    let high = edge.1 as u64 + threshold;
    low ^ (high << 32)
}

fn subdivide(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(faces.len() * 3);
    for face in faces {
        for pair in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            edges.push(if pair.0 <= pair.1 {
                pair
            } else {
                (pair.1, pair.0)
            });
        }
    }
    let mut unique: Vec<(usize, usize)> = edges.clone();
    unique.sort_unstable_by_key(|edge| edge_hash(*edge));
    unique.dedup();
    let mut index_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (index, edge) in unique.iter().enumerate() {
        index_of.insert(*edge, index);
    }
    let mut midpoints = Vec::with_capacity(unique.len());
    for (a, b) in &unique {
        midpoints.push([
            0.5 * (vertices[*a][0] + vertices[*b][0]),
            0.5 * (vertices[*a][1] + vertices[*b][1]),
            0.5 * (vertices[*a][2] + vertices[*b][2]),
        ]);
    }
    let base = vertices.len();
    let mut new_faces = Vec::with_capacity(faces.len() * 4);
    for (face_index, face) in faces.iter().enumerate() {
        let m0 = base + index_of[&edges[face_index * 3]];
        let m1 = base + index_of[&edges[face_index * 3 + 1]];
        let m2 = base + index_of[&edges[face_index * 3 + 2]];
        new_faces.push([face[0], m0, m2]);
        new_faces.push([m0, face[1], m1]);
        new_faces.push([m2, m1, face[2]]);
        new_faces.push([m0, m1, m2]);
    }
    let mut new_vertices = vertices.to_vec();
    new_vertices.extend(midpoints);
    (new_vertices, new_faces)
}

fn icosphere(subdivisions: usize, radius: f64) -> Vec<[f64; 3]> {
    let mut vertices: Vec<[f64; 3]> = ICO_VERTICES.to_vec();
    let mut faces: Vec<[usize; 3]> = ICO_FACES.to_vec();
    if subdivisions == 0 {
        project(&mut vertices, radius);
        return vertices;
    }
    for _ in 0..subdivisions {
        let (next_vertices, next_faces) = subdivide(&vertices, &faces);
        vertices = next_vertices;
        faces = next_faces;
        project(&mut vertices, radius);
    }
    vertices
}

fn complex_sqrt(re: f64, im: f64) -> (f64, f64) {
    if re == 0.0 && im == 0.0 {
        return (0.0, 0.0);
    }
    let radius = re.hypot(im);
    let real = ((radius + re) / 2.0).sqrt();
    let imag = im.signum() * ((radius - re) / 2.0).sqrt();
    (real, imag)
}

fn kurtosis(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for value in samples {
        let delta = value - mean;
        let delta2 = delta * delta;
        m2 += delta2;
        m4 += delta2 * delta2;
    }
    m2 /= n;
    m4 /= n;
    if m2 == 0.0 {
        f64::NAN
    } else {
        m4 / (m2 * m2) - 3.0
    }
}

/// NumPy `rint`: halfway cases go to the even integer.
fn round_half_even(value: f64) -> f64 {
    let rounded = value.round();
    let halfway = (value - value.trunc()).abs() - 0.5;
    if halfway.abs() > 1e-10 || (rounded as i64) % 2 == 0 {
        return rounded;
    }
    rounded - value.signum()
}

/// One map per spherical-harmonic level, then the kurtosis map.
pub fn lbp_3d(
    image: &[f64],
    dim: [usize; 3],
    levels: usize,
    radius: f64,
    subdivisions: usize,
    truncate_samples: bool,
    narrow_samples: bool,
    mask: Option<&[bool]>,
) -> Vec<Vec<f64>> {
    let vertices = icosphere(subdivisions, radius);
    let nv = vertices.len();
    let mut harmonics: Vec<(i32, Vec<(f64, f64)>)> = Vec::new();
    let mut n0 = Vec::with_capacity(nv);
    for vertex in &vertices {
        let theta = (vertex[2] / radius).clamp(-1.0, 1.0).acos();
        let phi = vertex[1].atan2(vertex[0]);
        n0.push(sph_harm(0, 0, theta, phi));
    }
    harmonics.push((0, n0));
    for degree in 1..levels as i32 {
        for order in -degree..=degree {
            let mut column = Vec::with_capacity(nv);
            for vertex in &vertices {
                let theta = (vertex[2] / radius).clamp(-1.0, 1.0).acos();
                let phi = vertex[1].atan2(vertex[0]);
                column.push(sph_harm(order, degree, theta, phi));
            }
            harmonics.push((degree, column));
        }
    }
    let coeff = spline::prefilter(image, dim);
    let n = image.len();
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let mut maps = vec![vec![0.0; n]; levels];
    let mut kurt = vec![0.0; n];
    let mut samples = vec![0.0; nv];
    let mut signed = vec![0.0; nv];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = z * ny * nx + y * nx + x;
                // PyRadiomics `getLBP3DImage` samples only labelled voxels.
                // The cubic prefilter above still uses the whole image.
                if let Some(roi) = mask {
                    if !roi[index] {
                        continue;
                    }
                }
                let center = image[index];
                for (vertex_index, vertex) in vertices.iter().enumerate() {
                    let mut sample = spline::sample(
                        &coeff,
                        dim,
                        z as f64 + vertex[0],
                        y as f64 + vertex[1],
                        x as f64 + vertex[2],
                    );
                    if truncate_samples {
                        // SciPy writes `map_coordinates` back into an integer image with `rint`.
                        sample = round_half_even(sample);
                    } else if narrow_samples {
                        // A float32 image keeps the interpolant at float32 precision.
                        sample = f64::from(sample as f32);
                    }
                    samples[vertex_index] = sample;
                    signed[vertex_index] = if samples[vertex_index] >= center {
                        1.0
                    } else {
                        0.0
                    };
                }
                kurt[index] = kurtosis(&samples);
                let mut coefficients = Vec::with_capacity(harmonics.len());
                for (_, column) in &harmonics {
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for (value, &(yr, yi)) in signed.iter().zip(column) {
                        re += value * yr;
                        im += value * yi;
                    }
                    coefficients.push((re, im));
                }
                for (level, plane) in maps.iter_mut().enumerate() {
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for v in 0..nv {
                        let mut tr = 0.0;
                        let mut ti = 0.0;
                        for (k, (degree, column)) in harmonics.iter().enumerate() {
                            if *degree != level as i32 {
                                continue;
                            }
                            let (cr, ci) = coefficients[k];
                            let (yr, yi) = column[v];
                            tr += cr * yr - ci * yi;
                            ti += cr * yi + ci * yr;
                        }
                        re += tr * tr - ti * ti;
                        im += 2.0 * tr * ti;
                    }
                    let (root_re, _) = complex_sqrt(re, im);
                    plane[index] = root_re;
                }
            }
        }
    }
    maps.push(kurt);
    maps
}

impl ImageType for Gradient {
    fn name(&self) -> &'static str {
        "Gradient"
    }
    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }
    fn setting_keys(&self) -> &'static [&'static str] {
        &["gradientUseSpacing", "binWidth", "binCount"]
    }
    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "Gradient")?;
        let use_spacing = match setting(settings, "gradientUseSpacing") {
            Some("false") => false,
            Some("true") | None => true,
            Some(other) => {
                return Err(RadixError::image_type(
                    "Gradient",
                    format!("gradientUseSpacing {other} is not a boolean"),
                ))
            }
        };
        Ok(vec![(
            "gradient".to_string(),
            gradient_magnitude(image, dim, spacing, use_spacing),
        )])
    }
}

impl ImageType for Lbp2D {
    fn name(&self) -> &'static str {
        "LBP2D"
    }
    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }
    fn setting_keys(&self) -> &'static [&'static str] {
        &[
            "lbp2DRadius",
            "lbp2DSamples",
            "lbp2DMethod",
            "force2Ddimension",
            "binWidth",
            "binCount",
        ]
    }
    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "LBP2D")?;
        let radius = match setting(settings, "lbp2DRadius") {
            Some(raw) => parse_f64("LBP2D", "lbp2DRadius", raw)?,
            None => 1.0,
        };
        let samples = match setting(settings, "lbp2DSamples") {
            Some(raw) => parse_usize("LBP2D", "lbp2DSamples", raw)?,
            None => 8,
        };
        let method = setting(settings, "lbp2DMethod").unwrap_or("uniform");
        let axis = match setting(settings, "force2Ddimension") {
            Some(raw) => parse_usize("LBP2D", "force2Ddimension", raw)?,
            None => 0,
        };
        if radius <= 0.0 || samples == 0 || samples > 31 || axis > 2 {
            return Err(RadixError::image_type(
                "LBP2D",
                "lbp2DRadius must be > 0, lbp2DSamples must be from 1 to 31, and force2Ddimension must be 0, 1, or 2",
            ));
        }
        if !matches!(method, "default" | "ror" | "uniform" | "var") {
            return Err(RadixError::image_type(
                "LBP2D",
                format!("unknown lbp2DMethod {method}"),
            ));
        }
        Ok(vec![(
            "lbp-2D".to_string(),
            lbp_2d(image, dim, axis, samples, radius, method),
        )])
    }
}

impl ImageType for Lbp3D {
    fn name(&self) -> &'static str {
        "LBP3D"
    }
    fn modes(&self) -> &'static [Mode] {
        PYRAD
    }
    fn setting_keys(&self) -> &'static [&'static str] {
        &[
            "lbp3DLevels",
            "lbp3DIcosphereRadius",
            "lbp3DIcosphereSubdivision",
            "binWidth",
            "binCount",
        ]
    }
    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "LBP3D")?;
        lbp3d_volumes(image, dim, settings, None)
    }

    fn apply_with_mask(
        &self,
        image: &[f64],
        mask: &[bool],
        dim: [usize; 3],
        _spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        check_len(image, dim, "LBP3D")?;
        if mask.len() != image.len() {
            return Err(RadixError::image_type(
                "LBP3D",
                "mask length does not match the image",
            ));
        }
        lbp3d_volumes(image, dim, settings, Some(mask))
    }
}

fn lbp3d_volumes(
    image: &[f64],
    dim: [usize; 3],
    settings: &[(&str, &str)],
    mask: Option<&[bool]>,
) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
    let levels = match setting(settings, "lbp3DLevels") {
        Some(raw) => parse_usize("LBP3D", "lbp3DLevels", raw)?,
        None => 2,
    };
    let radius = match setting(settings, "lbp3DIcosphereRadius") {
        Some(raw) => parse_f64("LBP3D", "lbp3DIcosphereRadius", raw)?,
        None => 1.0,
    };
    let subdivisions = match setting(settings, "lbp3DIcosphereSubdivision") {
        Some(raw) => parse_usize("LBP3D", "lbp3DIcosphereSubdivision", raw)?,
        None => 1,
    };
    if levels == 0 || radius <= 0.0 {
        return Err(RadixError::image_type(
            "LBP3D",
            "lbp3DLevels must be >= 1 and lbp3DIcosphereRadius must be > 0",
        ));
    }
    let truncate_samples = setting(settings, "integerVoxels") == Some("true");
    let narrow_samples = setting(settings, "float32Samples") == Some("true");
    let maps = lbp_3d(
        image,
        dim,
        levels,
        radius,
        subdivisions,
        truncate_samples,
        narrow_samples,
        mask,
    );
    let mut out = Vec::with_capacity(maps.len());
    for (index, volume) in maps.iter().enumerate() {
        let name = if index + 1 == maps.len() {
            "lbp-3D-k".to_string()
        } else {
            format!("lbp-3D-m{}", index + 1)
        };
        out.push((name, volume.clone()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_simpleitk_samples() {
        let dim = [4, 5, 6];
        let mut image: Vec<f64> = (0..120).map(|v| v as f64).collect();
        let at = |z: usize, y: usize, x: usize| z * dim[1] * dim[2] + y * dim[2] + x;
        image[at(1, 2, 3)] = -4.0;
        let spaced = gradient_magnitude(&image, dim, [1.5, 1.0, 2.0], true);
        assert!((spaced[0] - 10.443299293518066).abs() < 1e-4);
        assert!((spaced[at(1, 2, 3)] - 20.886598587036133).abs() < 1e-4);
        let plain = gradient_magnitude(&image, dim, [1.5, 1.0, 2.0], false);
        assert!((plain[0] - 15.305228233337402).abs() < 1e-4);
    }

    #[test]
    fn lbp3d_outside_the_mask_stays_zero_and_roi_matches() {
        let dim = [6usize, 6, 6];
        let image: Vec<f64> = (0..216).map(|v| (v % 17) as f64).collect();
        let mut mask = vec![false; 216];
        mask[1 * 36 + 2 * 6 + 3] = true;
        mask[2 * 36 + 2 * 6 + 2] = true;
        let full = lbp_3d(&image, dim, 2, 1.0, 1, false, false, None);
        let roi = lbp_3d(&image, dim, 2, 1.0, 1, false, false, Some(&mask));
        for (full_map, roi_map) in full.iter().zip(&roi) {
            for i in 0..image.len() {
                if mask[i] {
                    assert!((full_map[i] - roi_map[i]).abs() < 1e-9);
                } else {
                    assert_eq!(roi_map[i], 0.0);
                }
            }
        }
    }

    #[test]
    fn spherical_harmonic_matches_scipy() {
        let (r, i) = sph_harm(0, 0, 0.4, -1.2);
        assert!((r - 0.28209479177387814).abs() < 1e-12 && i.abs() < 1e-12);
        let (r, i) = sph_harm(1, 1, 0.4, -1.2);
        assert!((r + 0.2965945821551179).abs() < 1e-12);
        assert!((i + 0.12539817804966247).abs() < 1e-12);
        let (r, i) = sph_harm(-1, 1, 0.4, -1.2);
        assert!((r - 0.2965945821551179).abs() < 1e-12);
        assert!((i + 0.12539817804966247).abs() < 1e-12);
        let (r, i) = sph_harm(2, 2, 0.4, -1.2);
        assert!((r - 0.2337835491187458).abs() < 1e-12);
        assert!((i - 0.2407125561767383).abs() < 1e-12);
    }

    #[test]
    fn lbp_level_matches_numpy_square_root() {
        let signs: [f64; 42] = [
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let vertices = icosphere(1, 1.0);
        let mut columns: Vec<Vec<(f64, f64)>> = Vec::new();
        for order in -1..=1 {
            let mut column = Vec::new();
            for vertex in &vertices {
                let theta = (vertex[2]).clamp(-1.0, 1.0).acos();
                let phi = vertex[1].atan2(vertex[0]);
                column.push(sph_harm(order, 1, theta, phi));
            }
            columns.push(column);
        }
        let mut coefficients = Vec::new();
        for column in &columns {
            let mut re = 0.0;
            let mut im = 0.0;
            for (value, &(yr, yi)) in signs.iter().zip(column) {
                re += value * yr;
                im += value * yi;
            }
            coefficients.push((re, im));
        }
        let mut re = 0.0;
        let mut im = 0.0;
        for v in 0..vertices.len() {
            let mut tr = 0.0;
            let mut ti = 0.0;
            for (column, &(cr, ci)) in columns.iter().zip(&coefficients) {
                let (yr, yi) = column[v];
                tr += cr * yr - ci * yi;
                ti += cr * yi + ci * yr;
            }
            re += tr * tr - ti * ti;
            im += 2.0 * tr * ti;
        }
        let (root, _) = complex_sqrt(re, im);
        assert!(
            (root - 2.3735613462366674).abs() < 1e-15,
            "{root} re={re} im={im} c={coefficients:?}"
        );
    }

    #[test]
    fn lbp3d_returns_levels_and_kurtosis() {
        let image: Vec<f64> = (0..64).map(|v| v as f64).collect();
        let out = Lbp3D.apply(&image, [4, 4, 4], [1.0; 3], &[]).unwrap();
        let names: Vec<&str> = out.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["lbp-3D-m1", "lbp-3D-m2", "lbp-3D-k"]);
        assert!(out
            .iter()
            .all(|(_, volume)| volume.iter().all(|v| v.is_finite())));
    }
}
