//! SciPy `map_coordinates(..., order=3, mode="constant", cval=0)`.
//! The prefilter is the cubic B-spline from `ni_splines.c`. Out-of-range
//! filter taps use mirror extension; a coordinate outside the array is 0.

const POLE: f64 = -0.2679491924311227;

fn prefilter_line(line: &mut [f64]) {
    let n = line.len();
    if n < 2 {
        return;
    }
    let z = POLE;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for value in line.iter_mut() {
        *value *= gain;
    }
    let z_n_1 = z.powi((n - 1) as i32);
    let mut z_i = z;
    line[0] += z_n_1 * line[n - 1];
    for i in 1..n - 1 {
        line[0] += z_i * (line[i] + z_n_1 * line[n - 1 - i]);
        z_i *= z;
    }
    line[0] /= 1.0 - z_n_1 * z_n_1;
    for i in 1..n {
        line[i] += z * line[i - 1];
    }
    line[n - 1] = (z * line[n - 2] + line[n - 1]) * z / (z * z - 1.0);
    for i in (0..n - 1).rev() {
        line[i] = z * (line[i + 1] - line[i]);
    }
}

fn mirror_index(idx: i64, n: i64) -> usize {
    if n <= 1 {
        return 0;
    }
    let sz2 = 2 * n - 2;
    if idx < 0 {
        let mut inn = sz2 * ((-idx) / sz2) + idx;
        if inn <= 1 - n {
            inn += sz2;
        } else {
            inn = -inn;
        }
        return inn as usize;
    }
    if idx > n - 1 {
        let mut inn = idx - sz2 * (idx / sz2);
        if inn >= n {
            inn = sz2 - inn;
        }
        return inn as usize;
    }
    idx as usize
}

fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - t).powi(3) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Spline coefficients, one separable pass per axis. `dim` is (z, y, x).
pub fn prefilter(image: &[f64], dim: [usize; 3]) -> Vec<f64> {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let mut data = image.to_vec();
    let mut line = vec![0.0; nz.max(ny).max(nx)];
    for z in 0..nz {
        for y in 0..ny {
            let start = z * ny * nx + y * nx;
            line[..nx].copy_from_slice(&data[start..start + nx]);
            prefilter_line(&mut line[..nx]);
            data[start..start + nx].copy_from_slice(&line[..nx]);
        }
    }
    for z in 0..nz {
        for x in 0..nx {
            for y in 0..ny {
                line[y] = data[z * ny * nx + y * nx + x];
            }
            prefilter_line(&mut line[..ny]);
            for y in 0..ny {
                data[z * ny * nx + y * nx + x] = line[y];
            }
        }
    }
    for y in 0..ny {
        for x in 0..nx {
            for z in 0..nz {
                line[z] = data[z * ny * nx + y * nx + x];
            }
            prefilter_line(&mut line[..nz]);
            for z in 0..nz {
                data[z * ny * nx + y * nx + x] = line[z];
            }
        }
    }
    data
}

pub fn sample(coeff: &[f64], dim: [usize; 3], z: f64, y: f64, x: f64) -> f64 {
    let (nz, ny, nx) = (dim[0] as f64, dim[1] as f64, dim[2] as f64);
    if z < 0.0 || y < 0.0 || x < 0.0 || z > nz - 1.0 || y > ny - 1.0 || x > nx - 1.0 {
        return 0.0;
    }
    let iz = z.floor() as i64;
    let iy = y.floor() as i64;
    let ix = x.floor() as i64;
    let wz = cubic_weights(z - iz as f64);
    let wy = cubic_weights(y - iy as f64);
    let wx = cubic_weights(x - ix as f64);
    let (nz_i, ny_i, nx_i) = (dim[0] as i64, dim[1] as i64, dim[2] as i64);
    let mut sum = 0.0;
    for a in 0..4 {
        let zz = mirror_index(iz - 1 + a, nz_i);
        for b in 0..4 {
            let yy = mirror_index(iy - 1 + b, ny_i);
            for d in 0..4 {
                let xx = mirror_index(ix - 1 + d, nx_i);
                let value = coeff[zz * dim[1] * dim[2] + yy * dim[2] + xx];
                sum += wz[a as usize] * wy[b as usize] * wx[d as usize] * value;
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_sample_matches_scipy() {
        let dim = [4, 5, 6];
        let mut image: Vec<f64> = (0..120).map(|v| v as f64).collect();
        image[0] = -3.0;
        let coeff = prefilter(&image, dim);
        let got = sample(&coeff, dim, 0.2, 1.4, 3.1);
        assert!((got - 13.788710197142322).abs() < 1e-9, "{got}");
    }
}
