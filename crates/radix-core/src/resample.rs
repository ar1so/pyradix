//! Image / mask resampling.
//! IBSI: centre-aligned grid, optional integer rounding, mask ≥ 0.5 after the image interpolator.
//! PyRadiomics: origin (voxel-0) preserved, no rounding, nearest-neighbour mask.

use crate::error::RadixError;
use crate::util::idx;
use crate::Mode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolator {
    Nearest,
    Linear,
    Cubic,
}

impl Interpolator {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "nearest" | "sitkNearestNeighbor" => Ok(Self::Nearest),
            "linear" | "sitkLinear" => Ok(Self::Linear),
            "bspline" | "sitkBSpline" | "cubic" => Ok(Self::Cubic),
            other => Err(format!("unknown interpolator: {other}")),
        }
    }
}

pub struct Grid {
    pub dim: [usize; 3],
    pub spacing: [f64; 3],
    pub origin: [f64; 3], // z,y,x centre of voxel 0
    /// Source continuous index of output voxel 0, per axis.
    pub start: [f64; 3],
    /// Source-index step per output voxel, per axis.
    pub step: [f64; 3],
}

/// IBSI: \( n_b = \lceil n_a s_a / s_b \rceil \), centres aligned.
/// PyRadiomics: same count, and output voxel 0 sits at
/// `origin + 0.5 * (out_spacing - in_spacing)`.
pub fn resample_grid(
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    out_spacing: [f64; 3],
    mode: Mode,
) -> Grid {
    let mut out_dim = [0usize; 3];
    let mut start = [0.0; 3];
    let mut step = [0.0; 3];
    let mut out_origin = [0.0; 3];
    for a in 0..3 {
        let n = dim[a] as f64;
        let s = spacing[a];
        let sp = out_spacing[a];
        step[a] = sp / s;
        let n_new = ((n * s / sp).ceil() as usize).max(1);
        out_dim[a] = n_new;
        match mode {
            Mode::Ibsi => {
                // \( x_{b,g} = \tfrac{1}{2}(n_a-1 - (s_b/s_a)(n_b-1)) \)
                start[a] = (n - 1.0) / 2.0 - (n_new as f64 - 1.0) / 2.0 * step[a];
                let center = origin[a] + 0.5 * (n - 1.0) * s;
                out_origin[a] = center - 0.5 * (n_new as f64 - 1.0) * sp;
            }
            Mode::Pyradiomics => {
                if s == 0.0 {
                    start[a] = 0.0;
                    out_origin[a] = origin[a];
                } else {
                    start[a] = 0.5 * (sp - s) / s;
                    out_origin[a] = origin[a] + 0.5 * (sp - s);
                }
            }
        }
    }
    Grid {
        dim: out_dim,
        spacing: out_spacing,
        origin: out_origin,
        start,
        step,
    }
}

pub struct Resampled {
    pub image: Vec<f64>,
    pub mask: Vec<bool>,
    pub dim: [usize; 3],
    pub spacing: [f64; 3],
    pub origin: [f64; 3],
}

pub fn resample(
    image: &[f64],
    mask: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    out_spacing: [f64; 3],
    mode: Mode,
    interpolator: Interpolator,
    mask_interpolator: Interpolator,
) -> Result<Resampled, RadixError> {
    resample_typed(
        image,
        mask,
        dim,
        spacing,
        origin,
        out_spacing,
        mode,
        interpolator,
        mask_interpolator,
        None,
    )
}

/// Like [`resample`], with the source image's stored pixel type when it is known.
///
/// SimpleITK writes the interpolant back into the stored pixel type, so in
/// PyRadiomics mode an integer image truncates and a float image keeps the
/// fraction, even when every float value happens to be whole (CT stored as
/// float32). `None` falls back to checking the values. IBSI mode always rounds
/// whole-valued sources, as the IBSI reference tables expect.
pub fn resample_typed(
    image: &[f64],
    mask: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    out_spacing: [f64; 3],
    mode: Mode,
    interpolator: Interpolator,
    mask_interpolator: Interpolator,
    integer_pixels: Option<bool>,
) -> Result<Resampled, RadixError> {
    let grid = resample_grid(dim, spacing, origin, out_spacing, mode);
    let n = grid.dim[0] * grid.dim[1] * grid.dim[2];
    let mut out_img = vec![0.0; n];
    let mut out_mask = vec![false; n];
    let whole_values = image.iter().all(|v| (*v - v.round()).abs() < 1e-8);
    let integer_src = match (mode, integer_pixels) {
        (Mode::Pyradiomics, Some(stored_integer)) => stored_integer,
        _ => whole_values,
    };
    let img_src: Vec<f64> = if interpolator == Interpolator::Cubic {
        bspline_prefilter(image, dim)
    } else {
        image.to_vec()
    };
    let mask_src: Vec<f64> = {
        let mask_f: Vec<f64> = mask.iter().map(|m| if *m { 1.0 } else { 0.0 }).collect();
        if mask_interpolator == Interpolator::Cubic {
            bspline_prefilter(&mask_f, dim)
        } else {
            mask_f
        }
    };
    for z in 0..grid.dim[0] {
        for y in 0..grid.dim[1] {
            for x in 0..grid.dim[2] {
                let (zs, ys, xs) = source_index(&grid, spacing, origin, mode, z, y, x);
                let i = idx(z, y, x, grid.dim);
                // SimpleITK leaves a voxel at 0 when its continuous index is outside
                // [-0.5, n - 0.5). The upper end is exclusive, so an index of exactly
                // n - 0.5 is outside and must not be clamped back onto the last voxel.
                if mode == Mode::Pyradiomics
                    && (pyradiomics_outside(zs, dim[0])
                        || pyradiomics_outside(ys, dim[1])
                        || pyradiomics_outside(xs, dim[2]))
                {
                    out_img[i] = 0.0;
                    out_mask[i] = false;
                    continue;
                }
                let mut v = sample(&img_src, dim, zs, ys, xs, interpolator, mode);
                if mode == Mode::Ibsi && integer_src {
                    v = v.round();
                }
                // SimpleITK writes the interpolant back into the source pixel type.
                // An integer image truncates toward zero; a float image keeps the fraction.
                if mode == Mode::Pyradiomics && integer_src {
                    v = v.trunc();
                }
                out_img[i] = v;
                let m = sample(&mask_src, dim, zs, ys, xs, mask_interpolator, mode);
                out_mask[i] = m >= 0.5;
            }
        }
    }
    if !out_mask.iter().any(|m| *m) {
        return Err(RadixError::step("resample", "the resampled mask is empty"));
    }
    Ok(Resampled {
        image: out_img,
        mask: out_mask,
        dim: grid.dim,
        spacing: grid.spacing,
        origin: grid.origin,
    })
}

/// Continuous source index of one output voxel.
///
/// PyRadiomics / SimpleITK convert the output voxel center to a physical point
/// and back. `start + index * step` is the same formula, but it lands one ulp
/// under an exact halfway index, and nearest-neighbor then selects the other voxel.
fn source_index(
    grid: &Grid,
    spacing: [f64; 3],
    origin: [f64; 3],
    mode: Mode,
    z: usize,
    y: usize,
    x: usize,
) -> (f64, f64, f64) {
    if mode != Mode::Pyradiomics {
        return (
            grid.start[0] + z as f64 * grid.step[0],
            grid.start[1] + y as f64 * grid.step[1],
            grid.start[2] + x as f64 * grid.step[2],
        );
    }
    let index = |axis: usize, i: usize| {
        if spacing[axis] == 0.0 {
            return 0.0;
        }
        let physical = grid.origin[axis] + i as f64 * grid.spacing[axis];
        (physical - origin[axis]) / spacing[axis]
    };
    (index(0, z), index(1, y), index(2, x))
}

fn pyradiomics_outside(index: f64, n: usize) -> bool {
    index < -0.5 || index >= n as f64 - 0.5
}

fn clamp_index(v: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    v.clamp(0.0, (n - 1) as f64)
}

fn sample(
    img: &[f64],
    dim: [usize; 3],
    z: f64,
    y: f64,
    x: f64,
    how: Interpolator,
    mode: Mode,
) -> f64 {
    match how {
        Interpolator::Nearest => {
            let zz = clamp_index(z.round(), dim[0]) as usize;
            let yy = clamp_index(y.round(), dim[1]) as usize;
            let xx = clamp_index(x.round(), dim[2]) as usize;
            img[idx(zz, yy, xx, dim)]
        }
        Interpolator::Linear => sample_linear(img, dim, z, y, x),
        Interpolator::Cubic => sample_cubic(img, dim, z, y, x, mode),
    }
}

/// ITK's cubic B-spline reflects a tap that falls off the image:
/// `j < 0` becomes `-j - 1`, and `j >= n` becomes `2n - 2 - j`.
fn mirror_index(mut j: i32, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let n_i = n as i32;
    if j < 0 {
        j = -j - 1;
    }
    if j >= n_i {
        j = 2 * n_i - 2 - j;
    }
    j.clamp(0, n_i - 1) as usize
}

fn at(img: &[f64], dim: [usize; 3], z: i32, y: i32, x: i32) -> f64 {
    let zz = z.clamp(0, dim[0] as i32 - 1) as usize;
    let yy = y.clamp(0, dim[1] as i32 - 1) as usize;
    let xx = x.clamp(0, dim[2] as i32 - 1) as usize;
    img[idx(zz, yy, xx, dim)]
}

fn sample_linear(img: &[f64], dim: [usize; 3], z: f64, y: f64, x: f64) -> f64 {
    let z = clamp_index(z, dim[0]);
    let y = clamp_index(y, dim[1]);
    let x = clamp_index(x, dim[2]);
    let z0 = z.floor();
    let y0 = y.floor();
    let x0 = x.floor();
    let zd = z - z0;
    let yd = y - y0;
    let xd = x - x0;
    let z0i = z0 as i32;
    let y0i = y0 as i32;
    let x0i = x0 as i32;
    let c000 = at(img, dim, z0i, y0i, x0i);
    let c001 = at(img, dim, z0i, y0i, x0i + 1);
    let c010 = at(img, dim, z0i, y0i + 1, x0i);
    let c011 = at(img, dim, z0i, y0i + 1, x0i + 1);
    let c100 = at(img, dim, z0i + 1, y0i, x0i);
    let c101 = at(img, dim, z0i + 1, y0i, x0i + 1);
    let c110 = at(img, dim, z0i + 1, y0i + 1, x0i);
    let c111 = at(img, dim, z0i + 1, y0i + 1, x0i + 1);
    let c00 = c000 + (c001 - c000) * xd;
    let c01 = c010 + (c011 - c010) * xd;
    let c10 = c100 + (c101 - c100) * xd;
    let c11 = c110 + (c111 - c110) * xd;
    let c0 = c00 + (c01 - c00) * yd;
    let c1 = c10 + (c11 - c10) * yd;
    c0 + (c1 - c0) * zd
}

/// Cubic B-spline pole \(z_1=\sqrt{3}-2\) (Unser / SciPy / ITK order-3).
const B_SPLINE_Z: f64 = -0.2679491924311227;

fn bspline_prefilter(image: &[f64], dim: [usize; 3]) -> Vec<f64> {
    let mut c = image.to_vec();
    let mut line_x = vec![0.0; dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                line_x[x] = c[idx(z, y, x, dim)];
            }
            apply_bspline_1d(&mut line_x);
            for x in 0..dim[2] {
                c[idx(z, y, x, dim)] = line_x[x];
            }
        }
    }
    let mut line_y = vec![0.0; dim[1]];
    for z in 0..dim[0] {
        for x in 0..dim[2] {
            for y in 0..dim[1] {
                line_y[y] = c[idx(z, y, x, dim)];
            }
            apply_bspline_1d(&mut line_y);
            for y in 0..dim[1] {
                c[idx(z, y, x, dim)] = line_y[y];
            }
        }
    }
    let mut line_z = vec![0.0; dim[0]];
    for y in 0..dim[1] {
        for x in 0..dim[2] {
            for z in 0..dim[0] {
                line_z[z] = c[idx(z, y, x, dim)];
            }
            apply_bspline_1d(&mut line_z);
            for z in 0..dim[0] {
                c[idx(z, y, x, dim)] = line_z[z];
            }
        }
    }
    c
}

fn apply_bspline_1d(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = B_SPLINE_Z;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    let z_n_1 = z.powi((n - 1) as i32);
    let c0 = c[0] + z_n_1 * c[n - 1];
    let mut acc = c0;
    let mut z_i = z;
    for i in 1..n - 1 {
        acc += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] = acc / (1.0 - z_n_1 * z_n_1);
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

fn cubic_weights(x: f64) -> [f64; 4] {
    // SciPy order-3: x is the fractional part in [0, 1).
    let y = x;
    let z = 1.0 - x;
    let w1 = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    let w2 = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    let w0 = z * z * z / 6.0;
    let w3 = 1.0 - w0 - w1 - w2;
    [w0, w1, w2, w3]
}

fn sample_cubic(img: &[f64], dim: [usize; 3], z: f64, y: f64, x: f64, mode: Mode) -> f64 {
    let (z, y, x) = if mode == Mode::Pyradiomics {
        (z, y, x)
    } else {
        (
            clamp_index(z, dim[0]),
            clamp_index(y, dim[1]),
            clamp_index(x, dim[2]),
        )
    };
    let z0 = z.floor() as i32;
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;
    let wz = cubic_weights(z - z.floor());
    let wy = cubic_weights(y - y.floor());
    let wx = cubic_weights(x - x.floor());
    let mut acc = 0.0;
    for (dz, &wz_i) in wz.iter().enumerate() {
        for (dy, &wy_i) in wy.iter().enumerate() {
            for (dx, &wx_i) in wx.iter().enumerate() {
                let zz = z0 + dz as i32 - 1;
                let yy = y0 + dy as i32 - 1;
                let xx = x0 + dx as i32 - 1;
                let value = if mode == Mode::Pyradiomics {
                    img[idx(
                        mirror_index(zz, dim[0]),
                        mirror_index(yy, dim[1]),
                        mirror_index(xx, dim[2]),
                        dim,
                    )]
                } else {
                    at(img, dim, zz, yy, xx)
                };
                acc += value * wz_i * wy_i * wx_i;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mode;

    #[test]
    fn ibsi_ceil_4x4_3mm_to_2mm() {
        let g = resample_grid(
            [4usize, 4, 4],
            [3.0, 3.0, 3.0],
            [0.0; 3],
            [2.0, 2.0, 2.0],
            Mode::Ibsi,
        );
        assert_eq!(g.dim, [6, 6, 6]);
    }

    #[test]
    fn pat1_ibsi_2mm_shape() {
        let dim = [60usize, 201, 204];
        let spacing = [3.0, 0.9769999980926514, 0.9769999980926514];
        let g = resample_grid(dim, spacing, [0.0; 3], [2.0, 2.0, 2.0], Mode::Ibsi);
        assert_eq!(g.dim, [90, 99, 100]);
        assert_eq!(g.spacing, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn ibsi_center_shifts_start_pyradiomics_keeps_origin() {
        let dim = [4usize, 4, 4];
        let spacing = [2.0, 2.0, 2.0];
        let origin = [0.0, 0.0, 0.0];
        let ibsi = resample_grid(dim, spacing, origin, [1.0, 1.0, 1.0], Mode::Ibsi);
        let pyr = resample_grid(dim, spacing, origin, [1.0, 1.0, 1.0], Mode::Pyradiomics);
        assert_eq!(ibsi.dim, [8, 8, 8]);
        assert_eq!(pyr.dim, [8, 8, 8]);
        assert!(ibsi.start[0].abs() > 1e-9);
        assert!((pyr.start[0] - (-0.25)).abs() < 1e-12);
        assert!((pyr.origin[0] - (-0.5)).abs() < 1e-12);
        assert!((ibsi.origin[0] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn pyradiomics_index_hits_exact_halfway() {
        // LUNG1-001 z axis, 3 mm -> 2 mm, output index 97. SimpleITK reports 64.5.
        let spacing = [3.0, 1.0, 1.0];
        let origin = [-681.5, 0.0, 0.0];
        let out_spacing = [2.0, 2.0, 2.0];
        let grid = resample_grid([134, 8, 8], spacing, origin, out_spacing, Mode::Pyradiomics);
        let (index, _, _) = source_index(&grid, spacing, origin, Mode::Pyradiomics, 97, 0, 0);
        assert_eq!(index, 64.5);
        let algebraic = grid.start[0] + 97.0 * grid.step[0];
        assert!(algebraic < 64.5);
    }

    #[test]
    fn pyradiomics_zero_at_exact_outer_bound() {
        // 95 slices at 3 mm become 143 slices at 2 mm. Output index 142 sits at
        // continuous index 94.5, which SimpleITK treats as outside.
        let dim = [95usize, 2, 2];
        let image = vec![50.0; dim[0] * dim[1] * dim[2]];
        let mask = vec![true; image.len()];
        let resampled = resample(
            &image,
            &mask,
            dim,
            [3.0, 1.0, 1.0],
            [0.0; 3],
            [2.0, 1.0, 1.0],
            Mode::Pyradiomics,
            Interpolator::Nearest,
            Interpolator::Nearest,
        )
        .unwrap();
        assert_eq!(resampled.dim[0], 143);
        let last = resampled.dim[0] - 1;
        for y in 0..2 {
            for x in 0..2 {
                let i = crate::util::idx(last, y, x, resampled.dim);
                assert_eq!(resampled.image[i], 0.0);
                assert!(!resampled.mask[i]);
            }
        }
        let inside = crate::util::idx(last - 1, 0, 0, resampled.dim);
        assert_eq!(resampled.image[inside], 50.0);
    }

    #[test]
    fn pyradiomics_cubic_reflects_taps_past_the_last_voxel() {
        // SimpleITK int32 B-spline of [0, 1, 4, 9, 16, 25, 36, 49] from 3 mm to 2 mm.
        // The last samples read a tap past the volume, which is reflected.
        let values = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
        let dim = [8usize, 2, 2];
        let mut image = vec![0.0; dim[0] * dim[1] * dim[2]];
        for z in 0..8 {
            for y in 0..2 {
                for x in 0..2 {
                    image[crate::util::idx(z, y, x, dim)] = values[z];
                }
            }
        }
        let resampled = resample(
            &image,
            &vec![true; image.len()],
            dim,
            [3.0, 1.0, 1.0],
            [0.0; 3],
            [2.0, 1.0, 1.0],
            Mode::Pyradiomics,
            Interpolator::Cubic,
            Interpolator::Nearest,
        )
        .unwrap();
        let expected = [0.0, 0.0, 1.0, 3.0, 6.0, 10.0, 14.0, 20.0, 26.0, 33.0, 44.0, 48.0];
        assert_eq!(resampled.dim[0], expected.len());
        for (z, value) in expected.iter().enumerate() {
            let got = resampled.image[crate::util::idx(z, 0, 0, resampled.dim)];
            assert_eq!(got, *value, "z {z}");
        }
    }

    #[test]
    fn pyradiomics_truncates_an_integer_image() {
        let dim = [2usize, 2, 2];
        let image = vec![0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mask = vec![true; 8];
        let resampled = resample(
            &image,
            &mask,
            dim,
            [1.0, 1.0, 1.0],
            [0.0; 3],
            [2.0, 2.0, 2.0],
            Mode::Pyradiomics,
            Interpolator::Cubic,
            Interpolator::Nearest,
        )
        .unwrap();
        assert!(resampled.image.iter().all(|v| v.fract() == 0.0));
        assert!(resampled.mask.iter().any(|m| *m));
    }

    #[test]
    fn ibsi_rounds_integer_image() {
        let dim = [2usize, 2, 2];
        let image = vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0];
        let mask = vec![true; 8];
        let r = resample(
            &image,
            &mask,
            dim,
            [2.0, 2.0, 2.0],
            [0.0; 3],
            [1.0, 1.0, 1.0],
            Mode::Ibsi,
            Interpolator::Linear,
            Interpolator::Linear,
        )
        .unwrap();
        assert!(r.image.iter().all(|v| (*v - v.round()).abs() < 1e-12));
    }

    #[test]
    fn cubic_bspline_1d_matches_scipy() {
        let mut c = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        apply_bspline_1d(&mut c);
        let exp = [
            -0.3282051282051282,
            0.6564102564102564,
            3.7025641025641025,
            8.533333333333333,
            16.164102564102565,
            22.81025641025641,
            42.59487179487179,
        ];
        for (g, e) in c.iter().zip(exp) {
            assert!((g - e).abs() < 1e-10, "got {g} expected {e}");
        }
    }

    #[test]
    fn cubic_bspline_reconstructs_integer_samples() {
        let dim = [4usize, 4, 4];
        let mut image = vec![0.0; 64];
        for (i, value) in image.iter_mut().enumerate() {
            *value = i as f64;
        }
        let coeff = bspline_prefilter(&image, dim);
        for z in 1..3 {
            for y in 1..3 {
                for x in 1..3 {
                    let got = sample_cubic(&coeff, dim, z as f64, y as f64, x as f64, Mode::Ibsi);
                    let exp = image[idx(z, y, x, dim)];
                    assert!(
                        (got - exp).abs() < 1e-6,
                        "({z},{y},{x}) got {got} expected {exp}"
                    );
                }
            }
        }
    }
}
