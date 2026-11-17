//! ITK 5.4 `LaplacianRecursiveGaussianImageFilter` with
//! `NormalizeAcrossScale` on. Coefficients and the causal / anti-causal
//! pass are the bodies of `itkRecursiveGaussianImageFilter` and
//! `itkRecursiveSeparableImageFilter`.

#[derive(Clone, Copy)]
struct Coeffs {
    n0: f64,
    n1: f64,
    n2: f64,
    n3: f64,
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    bn1: f64,
    bn2: f64,
    bn3: f64,
    bn4: f64,
    bm1: f64,
    bm2: f64,
    bm3: f64,
    bm4: f64,
}

#[derive(Clone, Copy)]
enum Order {
    Zero,
    Second,
}

fn emam(a1: f64, b1: f64, a2: f64, b2: f64, a3: f64, b3: f64, a4: f64, b4: f64) -> f64 {
    a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4
}

fn compute_n(
    sigmad: f64,
    a1: f64,
    b1: f64,
    w1: f64,
    l1: f64,
    a2: f64,
    b2: f64,
    w2: f64,
    l2: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let sin1 = (w1 / sigmad).sin();
    let sin2 = (w2 / sigmad).sin();
    let cos1 = (w1 / sigmad).cos();
    let cos2 = (w2 / sigmad).cos();
    let exp1 = (l1 / sigmad).exp();
    let exp2 = (l2 / sigmad).exp();
    let n0 = a1 + a2;
    let n1 =
        exp2 * (b2 * sin2 - (a2 + 2.0 * a1) * cos2) + exp1 * (b1 * sin1 - (a1 + 2.0 * a2) * cos1);
    let n2 = ((a1 + a2) * cos2 * cos1 - (b1 * cos2 * sin1 + b2 * cos1 * sin2)) * 2.0 * exp1 * exp2
        + a2 * exp1 * exp1
        + a1 * exp2 * exp2;
    let n3 =
        exp2 * exp1 * exp1 * (b2 * sin2 - a2 * cos2) + exp1 * exp2 * exp2 * (b1 * sin1 - a1 * cos1);
    let sn = n0 + n1 + n2 + n3;
    let dn = n1 + 2.0 * n2 + 3.0 * n3;
    let en = n1 + 4.0 * n2 + 9.0 * n3;
    (n0, n1, n2, n3, sn, dn, en)
}

fn compute_d(
    sigmad: f64,
    w1: f64,
    l1: f64,
    w2: f64,
    l2: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let cos1 = (w1 / sigmad).cos();
    let cos2 = (w2 / sigmad).cos();
    let exp1 = (l1 / sigmad).exp();
    let exp2 = (l2 / sigmad).exp();
    let d4 = exp1 * exp1 * exp2 * exp2;
    let d3 = -2.0 * cos1 * exp1 * exp2 * exp2 + -2.0 * cos2 * exp2 * exp1 * exp1;
    let d2 = 4.0 * cos2 * cos1 * exp1 * exp2 + exp1 * exp1 + exp2 * exp2;
    let d1 = -2.0 * (exp2 * cos2 + exp1 * cos1);
    let sd = 1.0 + d1 + d2 + d3 + d4;
    let dd = d1 + 2.0 * d2 + 3.0 * d3 + 4.0 * d4;
    let ed = d1 + 4.0 * d2 + 9.0 * d3 + 16.0 * d4;
    (d1, d2, d3, d4, sd, dd, ed)
}

fn remaining(n0: f64, n1: f64, n2: f64, n3: f64, d1: f64, d2: f64, d3: f64, d4: f64) -> Coeffs {
    let m1 = n1 - d1 * n0;
    let m2 = n2 - d2 * n0;
    let m3 = n3 - d3 * n0;
    let m4 = -d4 * n0;
    let sn = n0 + n1 + n2 + n3;
    let sm = m1 + m2 + m3 + m4;
    let sd = 1.0 + d1 + d2 + d3 + d4;
    Coeffs {
        n0,
        n1,
        n2,
        n3,
        m1,
        m2,
        m3,
        m4,
        d1,
        d2,
        d3,
        d4,
        bn1: d1 * sn / sd,
        bn2: d2 * sn / sd,
        bn3: d3 * sn / sd,
        bn4: d4 * sn / sd,
        bm1: d1 * sm / sd,
        bm2: d2 * sm / sd,
        bm3: d3 * sm / sd,
        bm4: d4 * sm / sd,
    }
}

fn setup(sigma: f64, spacing: f64, order: Order) -> Coeffs {
    let sigmad = sigma / spacing;
    const A1: [f64; 3] = [1.3530, -0.6724, -1.3563];
    const B1: [f64; 3] = [1.8151, -3.4327, 5.2318];
    const A2: [f64; 3] = [-0.3531, 0.6724, 0.3446];
    const B2: [f64; 3] = [0.0902, 0.6100, -2.2355];
    const W1: f64 = 0.6681;
    const L1: f64 = -1.3932;
    const W2: f64 = 2.0787;
    const L2: f64 = -1.3732;
    let (d1, d2, d3, d4, sd, dd, ed) = compute_d(sigmad, W1, L1, W2, L2);
    match order {
        Order::Zero => {
            let (n0, n1, n2, n3, sn, _, _) =
                compute_n(sigmad, A1[0], B1[0], W1, L1, A2[0], B2[0], W2, L2);
            let alpha0 = 2.0 * sn / sd - n0;
            remaining(
                n0 / alpha0,
                n1 / alpha0,
                n2 / alpha0,
                n3 / alpha0,
                d1,
                d2,
                d3,
                d4,
            )
        }
        Order::Second => {
            let (n0_0, n1_0, n2_0, n3_0, sn0, dn0, en0) =
                compute_n(sigmad, A1[0], B1[0], W1, L1, A2[0], B2[0], W2, L2);
            let (n0_2, n1_2, n2_2, n3_2, sn2, dn2, en2) =
                compute_n(sigmad, A1[2], B1[2], W1, L1, A2[2], B2[2], W2, L2);
            let beta = -(2.0 * sn2 - sd * n0_2) / (2.0 * sn0 - sd * n0_0);
            let n0 = n0_2 + beta * n0_0;
            let n1 = n1_2 + beta * n1_0;
            let n2 = n2_2 + beta * n2_0;
            let n3 = n3_2 + beta * n3_0;
            let sn = sn2 + beta * sn0;
            let dn = dn2 + beta * dn0;
            let en = en2 + beta * en0;
            let mut alpha2 = en * sd * sd - ed * sn * sd - 2.0 * dn * dd * sd + 2.0 * dd * dd * sn;
            alpha2 /= sd * sd * sd;
            let scale = (sigma * sigma) / alpha2;
            remaining(
                n0 * scale,
                n1 * scale,
                n2 * scale,
                n3 * scale,
                d1,
                d2,
                d3,
                d4,
            )
        }
    }
}

fn filter_line_into(data: &[f64], c: &Coeffs, causal: &mut [f64], anti: &mut [f64]) {
    let ln = data.len();
    let edge = data[0];
    causal[0] = emam(edge, c.n0, edge, c.n1, edge, c.n2, edge, c.n3);
    causal[1] = emam(data[1], c.n0, edge, c.n1, edge, c.n2, edge, c.n3);
    causal[2] = emam(data[2], c.n0, data[1], c.n1, edge, c.n2, edge, c.n3);
    causal[3] = emam(data[3], c.n0, data[2], c.n1, data[1], c.n2, edge, c.n3);
    causal[0] -= emam(edge, c.bn1, edge, c.bn2, edge, c.bn3, edge, c.bn4);
    causal[1] -= emam(causal[0], c.d1, edge, c.bn2, edge, c.bn3, edge, c.bn4);
    causal[2] -= emam(causal[1], c.d1, causal[0], c.d2, edge, c.bn3, edge, c.bn4);
    causal[3] -= emam(
        causal[2], c.d1, causal[1], c.d2, causal[0], c.d3, edge, c.bn4,
    );
    for i in 4..ln {
        causal[i] = emam(
            data[i],
            c.n0,
            data[i - 1],
            c.n1,
            data[i - 2],
            c.n2,
            data[i - 3],
            c.n3,
        );
        causal[i] -= emam(
            causal[i - 1],
            c.d1,
            causal[i - 2],
            c.d2,
            causal[i - 3],
            c.d3,
            causal[i - 4],
            c.d4,
        );
    }

    let tail = data[ln - 1];
    anti[ln - 1] = emam(tail, c.m1, tail, c.m2, tail, c.m3, tail, c.m4);
    anti[ln - 2] = emam(data[ln - 1], c.m1, tail, c.m2, tail, c.m3, tail, c.m4);
    anti[ln - 3] = emam(
        data[ln - 2],
        c.m1,
        data[ln - 1],
        c.m2,
        tail,
        c.m3,
        tail,
        c.m4,
    );
    anti[ln - 4] = emam(
        data[ln - 3],
        c.m1,
        data[ln - 2],
        c.m2,
        data[ln - 1],
        c.m3,
        tail,
        c.m4,
    );
    anti[ln - 1] -= emam(tail, c.bm1, tail, c.bm2, tail, c.bm3, tail, c.bm4);
    anti[ln - 2] -= emam(anti[ln - 1], c.d1, tail, c.bm2, tail, c.bm3, tail, c.bm4);
    anti[ln - 3] -= emam(
        anti[ln - 2],
        c.d1,
        anti[ln - 1],
        c.d2,
        tail,
        c.bm3,
        tail,
        c.bm4,
    );
    anti[ln - 4] -= emam(
        anti[ln - 3],
        c.d1,
        anti[ln - 2],
        c.d2,
        anti[ln - 1],
        c.d3,
        tail,
        c.bm4,
    );
    let mut i = ln - 4;
    while i > 0 {
        anti[i - 1] = emam(
            data[i],
            c.m1,
            data[i + 1],
            c.m2,
            data[i + 2],
            c.m3,
            data[i + 3],
            c.m4,
        );
        anti[i - 1] -= emam(
            anti[i],
            c.d1,
            anti[i + 1],
            c.d2,
            anti[i + 2],
            c.d3,
            anti[i + 3],
            c.d4,
        );
        i -= 1;
    }
    for i in 0..ln {
        causal[i] += anti[i];
    }
}

fn worker_count(voxels: usize, n_lines: usize) -> usize {
    if voxels < 262_144 || n_lines < 2 {
        return 1;
    }
    let n = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    n.clamp(1, n_lines)
}

fn filter_chunk(input: &[f64], output: &mut [f64], len: usize, coeff: &Coeffs) {
    let mut causal = vec![0.0; len];
    let mut anti = vec![0.0; len];
    let lines = input.len() / len;
    for i in 0..lines {
        let start = i * len;
        filter_line_into(&input[start..start + len], coeff, &mut causal, &mut anti);
        output[start..start + len].copy_from_slice(&causal[..len]);
    }
}

fn filter_chunk_inplace(data: &mut [f64], len: usize, coeff: &Coeffs) {
    let mut line = vec![0.0; len];
    let mut causal = vec![0.0; len];
    let mut anti = vec![0.0; len];
    let lines = data.len() / len;
    for i in 0..lines {
        let start = i * len;
        line.copy_from_slice(&data[start..start + len]);
        filter_line_into(&line, coeff, &mut causal, &mut anti);
        data[start..start + len].copy_from_slice(&causal[..len]);
    }
}

fn filter_lines_to(input: &[f64], output: &mut [f64], n_lines: usize, len: usize, coeff: &Coeffs) {
    let workers = worker_count(n_lines * len, n_lines);
    if workers == 1 {
        filter_chunk(input, output, len, coeff);
        return;
    }
    let chunk_lines = (n_lines + workers - 1) / workers;
    let step = chunk_lines * len;
    std::thread::scope(|scope| {
        let mut rest_in = input;
        let mut rest_out = output;
        while !rest_in.is_empty() {
            let take = step.min(rest_in.len());
            let (head_in, tail_in) = rest_in.split_at(take);
            let (head_out, tail_out) = rest_out.split_at_mut(take);
            rest_in = tail_in;
            rest_out = tail_out;
            let coeff = *coeff;
            scope.spawn(move || filter_chunk(head_in, head_out, len, &coeff));
        }
    });
}

fn filter_lines_inplace(data: &mut [f64], n_lines: usize, len: usize, coeff: &Coeffs) {
    let workers = worker_count(n_lines * len, n_lines);
    if workers == 1 {
        filter_chunk_inplace(data, len, coeff);
        return;
    }
    let chunk_lines = (n_lines + workers - 1) / workers;
    let step = chunk_lines * len;
    std::thread::scope(|scope| {
        let mut rest = data;
        while !rest.is_empty() {
            let take = step.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            let coeff = *coeff;
            scope.spawn(move || filter_chunk_inplace(head, len, &coeff));
        }
    });
}

/// `(z, y, x)` to y-contiguous `(z, x, y)`.
fn transpose_y_from_xyz(input: &[f64], output: &mut [f64], dim: [usize; 3]) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let workers = worker_count(input.len(), nz);
    if workers == 1 {
        for z in 0..nz {
            for x in 0..nx {
                for y in 0..ny {
                    output[(z * nx + x) * ny + y] = input[z * ny * nx + y * nx + x];
                }
            }
        }
        return;
    }
    let plane = ny * nx;
    let chunk = (nz + workers - 1) / workers;
    std::thread::scope(|scope| {
        let mut rest_in = input;
        let mut rest_out = output;
        while !rest_in.is_empty() {
            let rows = chunk.min(rest_in.len() / plane);
            let take = rows * plane;
            let (head_in, tail_in) = rest_in.split_at(take);
            let (head_out, tail_out) = rest_out.split_at_mut(take);
            rest_in = tail_in;
            rest_out = tail_out;
            scope.spawn(move || {
                for local_z in 0..rows {
                    for x in 0..nx {
                        for y in 0..ny {
                            head_out[(local_z * nx + x) * ny + y] =
                                head_in[local_z * ny * nx + y * nx + x];
                        }
                    }
                }
            });
        }
    });
}

fn transpose_y_to_xyz(input: &[f64], output: &mut [f64], dim: [usize; 3]) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let workers = worker_count(input.len(), nz);
    if workers == 1 {
        for z in 0..nz {
            for x in 0..nx {
                for y in 0..ny {
                    output[z * ny * nx + y * nx + x] = input[(z * nx + x) * ny + y];
                }
            }
        }
        return;
    }
    let plane = ny * nx;
    let chunk = (nz + workers - 1) / workers;
    std::thread::scope(|scope| {
        let mut rest_in = input;
        let mut rest_out = output;
        while !rest_in.is_empty() {
            let rows = chunk.min(rest_in.len() / plane);
            let take = rows * plane;
            let (head_in, tail_in) = rest_in.split_at(take);
            let (head_out, tail_out) = rest_out.split_at_mut(take);
            rest_in = tail_in;
            rest_out = tail_out;
            scope.spawn(move || {
                for local_z in 0..rows {
                    for x in 0..nx {
                        for y in 0..ny {
                            head_out[local_z * ny * nx + y * nx + x] =
                                head_in[(local_z * nx + x) * ny + y];
                        }
                    }
                }
            });
        }
    });
}

/// `(z, y, x)` to z-contiguous `(y, x, z)`.
fn transpose_z_from_xyz(input: &[f64], output: &mut [f64], dim: [usize; 3]) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let workers = worker_count(input.len(), ny);
    if workers == 1 {
        for y in 0..ny {
            for x in 0..nx {
                for z in 0..nz {
                    output[(y * nx + x) * nz + z] = input[z * ny * nx + y * nx + x];
                }
            }
        }
        return;
    }
    let plane = nx * nz;
    let chunk = (ny + workers - 1) / workers;
    std::thread::scope(|scope| {
        let mut rest_out = output;
        let mut y0 = 0usize;
        while y0 < ny {
            let rows = chunk.min(ny - y0);
            let (head, tail) = rest_out.split_at_mut(rows * plane);
            rest_out = tail;
            let y_base = y0;
            scope.spawn(move || {
                for local_y in 0..rows {
                    let y = y_base + local_y;
                    for x in 0..nx {
                        for z in 0..nz {
                            head[(local_y * nx + x) * nz + z] = input[z * ny * nx + y * nx + x];
                        }
                    }
                }
            });
            y0 += rows;
        }
    });
}

fn transpose_z_to_xyz(input: &[f64], output: &mut [f64], dim: [usize; 3]) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let workers = worker_count(input.len(), nz);
    if workers == 1 {
        for y in 0..ny {
            for x in 0..nx {
                for z in 0..nz {
                    output[z * ny * nx + y * nx + x] = input[(y * nx + x) * nz + z];
                }
            }
        }
        return;
    }
    let plane = ny * nx;
    let chunk = (nz + workers - 1) / workers;
    std::thread::scope(|scope| {
        let mut rest_out = output;
        let mut z0 = 0usize;
        while z0 < nz {
            let rows = chunk.min(nz - z0);
            let (head, tail) = rest_out.split_at_mut(rows * plane);
            rest_out = tail;
            let z_base = z0;
            scope.spawn(move || {
                for local_z in 0..rows {
                    let z = z_base + local_z;
                    for y in 0..ny {
                        for x in 0..nx {
                            head[local_z * ny * nx + y * nx + x] = input[(y * nx + x) * nz + z];
                        }
                    }
                }
            });
            z0 += rows;
        }
    });
}

struct Scratch {
    volume: Vec<f64>,
}

/// ITK axis 0 is x, 1 is y, 2 is z. The volume is stored (z, y, x).
/// Y and z lines are transposed so the recursive pass is contiguous.
fn filter_axis_into(
    input: &[f64],
    output: &mut [f64],
    dim: [usize; 3],
    itk_axis: usize,
    coeff: &Coeffs,
    scratch: &mut Scratch,
) {
    match itk_axis {
        0 => filter_lines_to(input, output, dim[0] * dim[1], dim[2], coeff),
        1 => {
            if scratch.volume.len() != input.len() {
                scratch.volume.resize(input.len(), 0.0);
            }
            transpose_y_from_xyz(input, &mut scratch.volume, dim);
            filter_lines_inplace(&mut scratch.volume, dim[0] * dim[2], dim[1], coeff);
            transpose_y_to_xyz(&scratch.volume, output, dim);
        }
        _ => {
            if scratch.volume.len() != input.len() {
                scratch.volume.resize(input.len(), 0.0);
            }
            transpose_z_from_xyz(input, &mut scratch.volume, dim);
            filter_lines_inplace(&mut scratch.volume, dim[1] * dim[2], dim[0], coeff);
            transpose_z_to_xyz(&scratch.volume, output, dim);
        }
    }
}

fn round_f32(values: &mut [f64]) {
    // Each RecursiveGaussian in the Laplacian writes a float image. The next
    // pass reads those values back, so the double recursion is rounded here.
    for value in values {
        *value = f64::from(*value as f32);
    }
}

/// `spacing_zyx` is (z, y, x) millimetres. `sigma` is millimetres.
pub fn laplacian(image: &[f64], dim: [usize; 3], spacing_zyx: [f64; 3], sigma: f64) -> Vec<f64> {
    let spacing_xyz = [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]];
    let n = image.len();
    let mut current = vec![0.0; n];
    let mut next = vec![0.0; n];
    let mut acc = vec![0.0f32; n];
    let mut scratch = Scratch { volume: Vec::new() };
    for dim_axis in 0..3 {
        let deriv = setup(sigma, spacing_xyz[dim_axis], Order::Second);
        filter_axis_into(image, &mut current, dim, dim_axis, &deriv, &mut scratch);
        round_f32(&mut current);
        let mut smooth_axis = 0;
        while smooth_axis < 3 {
            if smooth_axis == dim_axis {
                smooth_axis += 1;
                continue;
            }
            let coeff = setup(sigma, spacing_xyz[smooth_axis], Order::Zero);
            filter_axis_into(&current, &mut next, dim, smooth_axis, &coeff, &mut scratch);
            round_f32(&mut next);
            std::mem::swap(&mut current, &mut next);
            smooth_axis += 1;
        }
        let scale = 1.0 / (spacing_xyz[dim_axis] * spacing_xyz[dim_axis]);
        for i in 0..n {
            let sum = f64::from(acc[i]) + current[i] * scale;
            acc[i] = sum as f32;
        }
    }
    acc.iter().map(|value| f64::from(*value)).collect()
}
