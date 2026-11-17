//! PyWavelets 1.6 stationary wavelet, periodized, one level at a time.
//! This is the convolution `getWaveletImage` applies via `pywt.swtn`.

/// Decomposition low-pass and high-pass, copied from `pywt.Wavelet`.
pub fn analysis_filters(family: &str) -> Option<(&'static [f64], &'static [f64])> {
    const S: f64 = std::f64::consts::FRAC_1_SQRT_2;
    match family.to_ascii_lowercase().as_str() {
        "haar" | "db1" => Some((&[S, S], &[-S, S])),
        "db2" => Some((
            &[
                -0.12940952255126037,
                0.2241438680420134,
                0.8365163037378079,
                0.48296291314453416,
            ],
            &[
                -0.48296291314453416,
                0.8365163037378079,
                -0.2241438680420134,
                -0.12940952255126037,
            ],
        )),
        "coif1" => Some((
            &[
                -0.015655728135791993,
                -0.07273261951252645,
                0.3848648468648578,
                0.8525720202116004,
                0.3378976624574818,
                -0.07273261951252645,
            ],
            &[
                0.07273261951252645,
                0.3378976624574818,
                -0.8525720202116004,
                0.3848648468648578,
                0.07273261951252645,
                -0.015655728135791993,
            ],
        )),
        _ => None,
    }
}

/// `downsampling_convolution_periodization` with step 1.
fn conv_period(data: &[f64], filt: &[f64]) -> Vec<f64> {
    let n = data.len();
    let f = filt.len();
    let mut out = vec![0.0; n];
    let mut i = f / 2;
    let mut o = 0;
    while i < f && i < n {
        let mut sum = 0.0;
        let mut j = 0;
        while j <= i {
            sum += filt[j] * data[i - j];
            j += 1;
        }
        while j < f {
            let mut k = 0;
            while k < n && j < f {
                sum += filt[j] * data[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        out[o] = sum;
        i += 1;
        o += 1;
    }
    while i < n {
        let mut sum = 0.0;
        for (j, tap) in filt.iter().enumerate() {
            sum += data[i - j] * tap;
        }
        out[o] = sum;
        i += 1;
        o += 1;
    }
    while i < f && i < n + f / 2 {
        let mut sum = 0.0;
        let mut j = 0;
        while i >= n + j {
            let mut k = 0;
            while k < n && i >= n + j {
                sum += filt[i - n - j] * data[k];
                k += 1;
                j += 1;
            }
        }
        while j <= i {
            sum += filt[j] * data[i - j];
            j += 1;
        }
        while j < f {
            let mut k = 0;
            while k < n && j < f {
                sum += filt[j] * data[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        out[o] = sum;
        i += 1;
        o += 1;
    }
    while i < n + f / 2 {
        let mut sum = 0.0;
        let mut j = 0;
        while i >= n + j {
            let mut k = 0;
            while k < n && i >= n + j {
                sum += filt[i - n - j] * data[k];
                k += 1;
                j += 1;
            }
        }
        while j < f {
            sum += filt[j] * data[i - j];
            j += 1;
        }
        out[o] = sum;
        i += 1;
        o += 1;
    }
    out
}

fn filter_axis(input: &[f64], dim: [usize; 3], axis: usize, filt: &[f64]) -> Vec<f64> {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let mut out = vec![0.0; input.len()];
    match axis {
        2 => {
            let mut line = vec![0.0; nx];
            for z in 0..nz {
                for y in 0..ny {
                    let start = z * ny * nx + y * nx;
                    line.copy_from_slice(&input[start..start + nx]);
                    let filtered = conv_period(&line, filt);
                    out[start..start + nx].copy_from_slice(&filtered);
                }
            }
        }
        1 => {
            let mut line = vec![0.0; ny];
            for z in 0..nz {
                for x in 0..nx {
                    for y in 0..ny {
                        line[y] = input[z * ny * nx + y * nx + x];
                    }
                    let filtered = conv_period(&line, filt);
                    for y in 0..ny {
                        out[z * ny * nx + y * nx + x] = filtered[y];
                    }
                }
            }
        }
        _ => {
            let mut line = vec![0.0; nz];
            for y in 0..ny {
                for x in 0..nx {
                    for z in 0..nz {
                        line[z] = input[z * ny * nx + y * nx + x];
                    }
                    let filtered = conv_period(&line, filt);
                    for z in 0..nz {
                        out[z * ny * nx + y * nx + x] = filtered[z];
                    }
                }
            }
        }
    }
    out
}

/// Pad each odd axis by one wrap voxel at the end, matching `numpy.pad(..., 'wrap')`.
pub fn pad_odd(image: &[f64], dim: [usize; 3]) -> (Vec<f64>, [usize; 3]) {
    let out_dim = [
        dim[0] + dim[0] % 2,
        dim[1] + dim[1] % 2,
        dim[2] + dim[2] % 2,
    ];
    if out_dim == dim {
        return (image.to_vec(), dim);
    }
    let (nz, ny, nx) = (out_dim[0], out_dim[1], out_dim[2]);
    let mut out = vec![0.0; nz * ny * nx];
    for z in 0..nz {
        let sz = if z < dim[0] { z } else { 0 };
        for y in 0..ny {
            let sy = if y < dim[1] { y } else { 0 };
            for x in 0..nx {
                let sx = if x < dim[2] { x } else { 0 };
                out[z * ny * nx + y * nx + x] = image[sz * dim[1] * dim[2] + sy * dim[2] + sx];
            }
        }
    }
    (out, out_dim)
}

pub fn crop(image: &[f64], from: [usize; 3], to: [usize; 3]) -> Vec<f64> {
    if from == to {
        return image.to_vec();
    }
    let mut out = vec![0.0; to[0] * to[1] * to[2]];
    for z in 0..to[0] {
        for y in 0..to[1] {
            for x in 0..to[2] {
                out[z * to[1] * to[2] + y * to[2] + x] =
                    image[z * from[1] * from[2] + y * from[2] + x];
            }
        }
    }
    out
}

/// One `pywt.swtn(..., level=1)` step. Keys are `a`/`d` in `axes` order.
pub fn swtn_level(
    image: &[f64],
    dim: [usize; 3],
    axes: &[usize],
    lo: &[f64],
    hi: &[f64],
) -> Vec<(String, Vec<f64>)> {
    let mut bands: Vec<(String, Vec<f64>)> = vec![("".to_string(), image.to_vec())];
    for &axis in axes {
        let mut next = Vec::with_capacity(bands.len() * 2);
        for (name, volume) in bands {
            next.push((format!("{name}a"), filter_axis(&volume, dim, axis, lo)));
            next.push((format!("{name}d"), filter_axis(&volume, dim, axis, hi)));
        }
        bands = next;
    }
    bands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coif1_line_matches_pywavelets() {
        let (lo, _) = analysis_filters("coif1").unwrap();
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let got = conv_period(&data, lo);
        let expected = [
            2.121320343559643,
            0.8323526062728833,
            2.82842712474619,
            4.242640687119285,
            5.656854249492381,
            7.196313636951811,
            9.19238815542512,
            7.52768294287935,
        ];
        for (index, (left, right)) in got.iter().zip(expected).enumerate() {
            assert!((left - right).abs() < 1e-9, "{index}: {left} vs {right}");
        }
    }
}
