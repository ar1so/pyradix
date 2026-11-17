//! Shared helpers.

pub const EPS: f64 = 2.220446049250313e-16;

pub fn log2_eps(x: f64) -> f64 {
    (x + EPS).log2()
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    // NumPy's linear quantile. The index is `(n - 1) * (p/100)`. The blend is
    // `a + (b-a)*t`, or `b - (b-a)*(1-t)` when `t >= 0.5`. A plain
    // `a*(1-t) + b*t` is one ulp off when `a == b`, and that moves the
    // 10–90 band used by robust mean absolute deviation.
    let k = (n as f64 - 1.0) * (p / 100.0);
    let f = k.floor() as usize;
    let c = k.ceil() as usize;
    if f == c {
        return sorted[f];
    }
    let t = k - f as f64;
    let left = sorted[f];
    let right = sorted[c];
    let span = right - left;
    if t >= 0.5 {
        right - span * (1.0 - t)
    } else {
        left + span * t
    }
}

pub fn idx(z: usize, y: usize, x: usize, dim: [usize; 3]) -> usize {
    z * dim[1] * dim[2] + y * dim[2] + x
}

pub fn in_bounds(z: i32, y: i32, x: i32, dim: [usize; 3]) -> bool {
    z >= 0
        && y >= 0
        && x >= 0
        && (z as usize) < dim[0]
        && (y as usize) < dim[1]
        && (x as usize) < dim[2]
}

/// 13 unique 3D directions (half-sphere). Order (dz, dy, dx).
pub const DIRS_3D: [[i32; 3]; 13] = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 1, -1],
    [1, 0, 1],
    [1, 0, -1],
    [1, 1, 0],
    [1, -1, 0],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
];

pub const DIRS_2D: [[i32; 3]; 4] = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, -1]];

pub fn dirs(dim: [usize; 3]) -> &'static [[i32; 3]] {
    if dim[0] == 1 || dim[1] == 1 || dim[2] == 1 {
        &DIRS_2D
    } else {
        &DIRS_3D
    }
}

fn plane_dirs(axis: usize) -> Vec<[i32; 3]> {
    match axis {
        0 => DIRS_2D.to_vec(),
        1 => vec![[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, -1]],
        _ => vec![[0, 1, 0], [1, 0, 0], [1, 1, 0], [1, -1, 0]],
    }
}

/// Directions for GLCM and GLRLM. Distance 1 in 3D is the 13 half-sphere angles.
pub fn texture_angles(
    dim: [usize; 3],
    distances: &[u32],
    force_2d: bool,
    axis: usize,
) -> Vec<[i32; 3]> {
    let base = if force_2d {
        plane_dirs(axis)
    } else if dim[0] == 1 || dim[1] == 1 || dim[2] == 1 {
        DIRS_2D.to_vec()
    } else {
        DIRS_3D.to_vec()
    };
    let mut dists: Vec<u32> = distances.iter().copied().filter(|d| *d > 0).collect();
    if dists.is_empty() {
        dists.push(1);
    }
    let mut out = Vec::new();
    for distance in dists {
        for dir in &base {
            out.push([
                dir[0] * distance as i32,
                dir[1] * distance as i32,
                dir[2] * distance as i32,
            ]);
        }
    }
    out
}

/// Chebyshev neighbourhood used by GLDM and NGTDM. Distance 1 in 3D is the 26-neighbour.
pub fn chebyshev_neighbors(dim: [usize; 3], distances: &[u32]) -> Vec<[i32; 3]> {
    let allowed: Vec<u32> = if distances.is_empty() {
        vec![1]
    } else {
        distances.iter().copied().filter(|d| *d > 0).collect()
    };
    let max_d = allowed.iter().copied().max().unwrap_or(1) as i32;
    let mut out = Vec::new();
    for dz in -max_d..=max_d {
        for dy in -max_d..=max_d {
            for dx in -max_d..=max_d {
                if dz == 0 && dy == 0 && dx == 0 {
                    continue;
                }
                if dim[0] == 1 && dz != 0 || dim[1] == 1 && dy != 0 || dim[2] == 1 && dx != 0 {
                    continue;
                }
                let chebyshev = dz.abs().max(dy.abs()).max(dx.abs()) as u32;
                if allowed.contains(&chebyshev) {
                    out.push([dz, dy, dx]);
                }
            }
        }
    }
    out
}

pub fn plane_slice(
    dim: [usize; 3],
    spacing: [f64; 3],
    axis: usize,
    index: usize,
) -> (Vec<usize>, [usize; 3], [f64; 3]) {
    let axis = axis.min(2);
    let (along_a, along_b) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let new_dim = [1, dim[along_a], dim[along_b]];
    let new_spacing = [spacing[axis], spacing[along_a], spacing[along_b]];
    let mut source = Vec::with_capacity(dim[along_a] * dim[along_b]);
    let at = index.min(dim[axis].saturating_sub(1));
    for i in 0..dim[along_a] {
        for j in 0..dim[along_b] {
            let mut coord = [0usize; 3];
            coord[axis] = at;
            coord[along_a] = i;
            coord[along_b] = j;
            source.push(idx(coord[0], coord[1], coord[2], dim));
        }
    }
    (source, new_dim, new_spacing)
}

pub fn mean_maps(
    maps: &[std::collections::BTreeMap<String, f64>],
) -> std::collections::BTreeMap<String, f64> {
    let mut acc: std::collections::BTreeMap<String, (f64, usize)> =
        std::collections::BTreeMap::new();
    for m in maps {
        for (k, v) in m {
            if v.is_nan() {
                continue;
            }
            let e = acc.entry(k.clone()).or_insert((0.0, 0));
            e.0 += *v;
            e.1 += 1;
        }
    }
    acc.into_iter()
        .map(|(k, (s, n))| (k, if n == 0 { f64::NAN } else { s / n as f64 }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::percentile;

    #[test]
    fn tied_order_statistics_match_numpy_percentile() {
        // 522^2 / 3070 is a repeated square-filter value. NumPy's lerp returns
        // it unchanged; weighting both copies moves the result by one ulp.
        let tied = (522.0 * 522.0) / 3070.0;
        let sorted = vec![tied; 580];
        assert_eq!(percentile(&sorted, 10.0), tied);
        assert_eq!(percentile(&sorted, 90.0), tied);
    }
}

/// Number of 26-connected foreground regions in a z-major `mask`.
/// PyRadiomics reports this as `diagnostics_Mask-*_VolumeNum`.
pub fn count_components_26(mask: &[bool], dim: [usize; 3]) -> usize {
    let [nz, ny, nx] = dim;
    assert_eq!(mask.len(), nz * ny * nx, "mask length does not match dim");
    let mut visited = vec![false; mask.len()];
    let mut stack: Vec<usize> = Vec::new();
    let mut count = 0;
    for seed in 0..mask.len() {
        if !mask[seed] || visited[seed] {
            continue;
        }
        count += 1;
        visited[seed] = true;
        stack.push(seed);
        while let Some(i) = stack.pop() {
            let z = i / (ny * nx);
            let y = (i / nx) % ny;
            let x = i % nx;
            for zz in z.saturating_sub(1)..=(z + 1).min(nz - 1) {
                for yy in y.saturating_sub(1)..=(y + 1).min(ny - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(nx - 1) {
                        let j = (zz * ny + yy) * nx + xx;
                        if mask[j] && !visited[j] {
                            visited[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod component_tests {
    use super::count_components_26;

    #[test]
    fn diagonal_voxels_join_and_gaps_split() {
        // 2x2x3: (0,0,0) and (1,1,1) touch by a corner; (0,0,2)... is apart.
        let dim = [2, 2, 3];
        let mut m = vec![false; 12];
        m[0] = true; // z0 y0 x0
        m[(1 * 2 + 1) * 3 + 1] = true; // z1 y1 x1
        assert_eq!(count_components_26(&m, dim), 1);
        let mut m2 = vec![false; 12];
        m2[0] = true; // z0 y0 x0
        m2[2] = true; // z0 y0 x2, gap at x1
        assert_eq!(count_components_26(&m2, dim), 2);
        assert_eq!(count_components_26(&vec![false; 12], dim), 0);
    }
}
