//! PyRadiomics 3.1.0 `cshape.c` mesh (quoted).
//! Pad-by-1, 128-entry folded marching cubes, vertices on edges 6/7/11.

use crate::cshape_tables::{GRID_ANGLES, TRI_TABLE, VERT_LIST};

/// Returns `(volume, surface_area, vertices)` with vertices in (z, y, x) mm.
/// The mask is cropped to its bounding box first. PyRadiomics does the same
/// before padding, and the origin-tetrahedron volume is unstable on a full
/// CT grid.
pub fn mesh(mask: &[bool], dim: [usize; 3], spacing: [f64; 3]) -> (f64, f64, Vec<[f64; 3]>) {
    let Some((cropped, cropped_dim)) = crop_mask(mask, dim) else {
        return (0.0, 0.0, Vec::new());
    };
    mesh_padded(&cropped, cropped_dim, spacing)
}

fn crop_mask(mask: &[bool], dim: [usize; 3]) -> Option<(Vec<bool>, [usize; 3])> {
    let mut min = [usize::MAX; 3];
    let mut max = [0usize; 3];
    let mut any = false;
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                if !mask[z * dim[1] * dim[2] + y * dim[2] + x] {
                    continue;
                }
                any = true;
                let coord = [z, y, x];
                for axis in 0..3 {
                    min[axis] = min[axis].min(coord[axis]);
                    max[axis] = max[axis].max(coord[axis]);
                }
            }
        }
    }
    if !any {
        return None;
    }
    let new_dim = [
        max[0] - min[0] + 1,
        max[1] - min[1] + 1,
        max[2] - min[2] + 1,
    ];
    let mut cropped = vec![false; new_dim[0] * new_dim[1] * new_dim[2]];
    for z in 0..new_dim[0] {
        for y in 0..new_dim[1] {
            for x in 0..new_dim[2] {
                let src = (min[0] + z) * dim[1] * dim[2] + (min[1] + y) * dim[2] + (min[2] + x);
                cropped[z * new_dim[1] * new_dim[2] + y * new_dim[2] + x] = mask[src];
            }
        }
    }
    Some((cropped, new_dim))
}

fn mesh_padded(mask: &[bool], dim: [usize; 3], spacing: [f64; 3]) -> (f64, f64, Vec<[f64; 3]>) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let (sz, sy, sx) = (spacing[0], spacing[1], spacing[2]);
    let pnz = nz + 2;
    let pny = ny + 2;
    let pnx = nx + 2;
    let mut padded = vec![false; pnz * pny * pnx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if mask[z * ny * nx + y * nx + x] {
                    padded[(z + 1) * pny * pnx + (y + 1) * pnx + (x + 1)] = true;
                }
            }
        }
    }

    // cshape.c: points_edges = {{6, 4, 3}, {6, 7, 11}}
    const BIT: [u8; 3] = [6, 4, 3];
    const EDGE: [usize; 3] = [6, 7, 11];
    let sp = [sz, sy, sx];

    let mut volume = 0.0;
    let mut area = 0.0;
    let mut vertices = Vec::new();

    for iz in 0..pnz - 1 {
        for iy in 0..pny - 1 {
            for ix in 0..pnx - 1 {
                let mut cube_idx: u8 = 0;
                for (a, angle) in GRID_ANGLES.iter().enumerate() {
                    let z = iz as i32 + angle[0];
                    let y = iy as i32 + angle[1];
                    let x = ix as i32 + angle[2];
                    let i = z as usize * pny * pnx + y as usize * pnx + x as usize;
                    if padded[i] {
                        cube_idx |= 1 << a;
                    }
                }
                let mut sign = 1.0;
                if cube_idx & 0x80 != 0 {
                    cube_idx ^= 0xff;
                    sign = -1.0;
                }
                for t in 0..3 {
                    if cube_idx & (1 << BIT[t]) != 0 {
                        let e = EDGE[t];
                        vertices.push([
                            (iz as f64 + VERT_LIST[e][0]) * sz,
                            (iy as f64 + VERT_LIST[e][1]) * sy,
                            (ix as f64 + VERT_LIST[e][2]) * sx,
                        ]);
                    }
                }
                if cube_idx == 0 {
                    continue;
                }
                let row = &TRI_TABLE[cube_idx as usize];
                let mut t = 0usize;
                while t * 3 < 16 && row[t * 3] >= 0 {
                    let mut a = [iz as f64, iy as f64, ix as f64];
                    let mut b = a;
                    let mut c = a;
                    let ia = row[t * 3] as usize;
                    let ib = row[t * 3 + 1] as usize;
                    let ic = row[t * 3 + 2] as usize;
                    for d in 0..3 {
                        a[d] = (a[d] + VERT_LIST[ia][d]) * sp[d];
                        b[d] = (b[d] + VERT_LIST[ib][d]) * sp[d];
                        c[d] = (c[d] + VERT_LIST[ic][d]) * sp[d];
                    }
                    let ab0 = a[1] * b[2] - b[1] * a[2];
                    let ab1 = a[2] * b[0] - b[2] * a[0];
                    let ab2 = a[0] * b[1] - b[0] * a[1];
                    volume += sign * (ab0 * c[0] + ab1 * c[1] + ab2 * c[2]);

                    let ac = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
                    let bc = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
                    let cr0 = ac[1] * bc[2] - bc[1] * ac[2];
                    let cr1 = ac[2] * bc[0] - bc[2] * ac[0];
                    let cr2 = ac[0] * bc[1] - bc[0] * ac[1];
                    area += 0.5 * (cr0 * cr0 + cr1 * cr1 + cr2 * cr2).sqrt();
                    t += 1;
                }
            }
        }
    }
    (volume / 6.0, area, vertices)
}

/// PyRadiomics `calculate_meshDiameter`: 2D diameters require equal coordinate on the dropped axis.
///
/// Same result as comparing every vertex pair, bit for bit: each candidate
/// distance is computed with the same expression, and pairs are only skipped
/// when they provably cannot exceed the running maximum.
pub fn diameters(verts: &[[f64; 3]]) -> [f64; 4] {
    [
        same_coord_max(verts, 0).sqrt(),
        same_coord_max(verts, 1).sqrt(),
        same_coord_max(verts, 2).sqrt(),
        max_dist2(verts).sqrt(),
    ]
}

#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab0 = a[0] - b[0];
    let ab1 = a[1] - b[1];
    let ab2 = a[2] - b[2];
    ab0 * ab0 + ab1 * ab1 + ab2 * ab2
}

/// Largest squared distance between two vertices that share coordinate `axis`.
pub(crate) fn same_coord_max(verts: &[[f64; 3]], axis: usize) -> f64 {
    let mut order: Vec<usize> = (0..verts.len()).collect();
    order.sort_unstable_by(|&i, &j| verts[i][axis].total_cmp(&verts[j][axis]));
    let mut best = 0.0;
    let mut start = 0;
    while start < order.len() {
        let value = verts[order[start]][axis];
        let mut end = start + 1;
        while end < order.len() && verts[order[end]][axis] == value {
            end += 1;
        }
        let group: Vec<[f64; 3]> = order[start..end].iter().map(|&i| verts[i]).collect();
        let d = max_dist2_from(&group, best);
        if d > best {
            best = d;
        }
        start = end;
    }
    best
}

/// Largest squared distance between any two vertices.
pub(crate) fn max_dist2(verts: &[[f64; 3]]) -> f64 {
    max_dist2_from(verts, 0.0)
}

/// Max squared pair distance, or `floor` if no pair beats it.
///
/// Vertices are binned on a coarse grid. A cell pair is scanned only when the
/// farthest corners of the two cell boxes could beat the best so far. The box
/// bound uses the same subtract/square/add steps as `dist2` on coordinates
/// at least as far apart, so rounding never makes it smaller than a real pair.
fn max_dist2_from(verts: &[[f64; 3]], floor: f64) -> f64 {
    let n = verts.len();
    let mut best = floor;
    if n < 2 {
        return best;
    }
    if n <= 256 {
        for i in 0..n {
            for j in 0..i {
                let d = dist2(verts[i], verts[j]);
                if d > best {
                    best = d;
                }
            }
        }
        return best;
    }

    let mut lo = verts[0];
    let mut hi = verts[0];
    for v in verts {
        for d in 0..3 {
            lo[d] = lo[d].min(v[d]);
            hi[d] = hi[d].max(v[d]);
        }
    }

    // Seed `best` with the axis-extreme vertices against everything.
    let mut extremes = [0usize; 6];
    for (i, v) in verts.iter().enumerate() {
        for d in 0..3 {
            if v[d] < verts[extremes[2 * d]][d] {
                extremes[2 * d] = i;
            }
            if v[d] > verts[extremes[2 * d + 1]][d] {
                extremes[2 * d + 1] = i;
            }
        }
    }
    for &e in &extremes {
        for v in verts {
            let d = dist2(verts[e], *v);
            if d > best {
                best = d;
            }
        }
    }

    const G: usize = 12;
    let cell_of = |v: &[f64; 3]| -> usize {
        let mut c = [0usize; 3];
        for d in 0..3 {
            let span = hi[d] - lo[d];
            c[d] = if span > 0.0 {
                (((v[d] - lo[d]) / span * G as f64) as usize).min(G - 1)
            } else {
                0
            };
        }
        (c[0] * G + c[1]) * G + c[2]
    };
    let mut cells: Vec<Vec<[f64; 3]>> = vec![Vec::new(); G * G * G];
    for v in verts {
        cells[cell_of(v)].push(*v);
    }
    struct Cell {
        pts: Vec<[f64; 3]>,
        lo: [f64; 3],
        hi: [f64; 3],
    }
    let cells: Vec<Cell> = cells
        .into_iter()
        .filter(|p| !p.is_empty())
        .map(|pts| {
            let mut lo = pts[0];
            let mut hi = pts[0];
            for v in &pts {
                for d in 0..3 {
                    lo[d] = lo[d].min(v[d]);
                    hi[d] = hi[d].max(v[d]);
                }
            }
            Cell { pts, lo, hi }
        })
        .collect();

    let bound = |a: &Cell, b: &Cell| -> f64 {
        let s: [f64; 3] =
            std::array::from_fn(|d| (a.hi[d] - b.lo[d]).abs().max((b.hi[d] - a.lo[d]).abs()));
        s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
    };
    let mut pairs: Vec<(f64, u32, u32)> = Vec::new();
    for i in 0..cells.len() {
        for j in 0..=i {
            let ub = bound(&cells[i], &cells[j]);
            if ub > best {
                pairs.push((ub, i as u32, j as u32));
            }
        }
    }
    pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    for (ub, i, j) in pairs {
        if ub <= best {
            break;
        }
        let (a, b) = (&cells[i as usize].pts, &cells[j as usize].pts);
        if i == j {
            for x in 0..a.len() {
                for y in 0..x {
                    let d = dist2(a[x], a[y]);
                    if d > best {
                        best = d;
                    }
                }
            }
        } else {
            for &p in a {
                for &q in b {
                    let d = dist2(p, q);
                    if d > best {
                        best = d;
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod diameter_tests {
    use super::{diameters, mesh};

    fn naive(verts: &[[f64; 3]]) -> [f64; 4] {
        let mut d2 = [0.0f64; 4];
        for i in 0..verts.len() {
            for j in 0..i {
                let (a, b) = (verts[i], verts[j]);
                let ab0 = a[0] - b[0];
                let ab1 = a[1] - b[1];
                let ab2 = a[2] - b[2];
                let dist = ab0 * ab0 + ab1 * ab1 + ab2 * ab2;
                for k in 0..3 {
                    if a[k] == b[k] && dist > d2[k] {
                        d2[k] = dist;
                    }
                }
                if dist > d2[3] {
                    d2[3] = dist;
                }
            }
        }
        [d2[0].sqrt(), d2[1].sqrt(), d2[2].sqrt(), d2[3].sqrt()]
    }

    // Small deterministic LCG so the test needs no rand crate.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn matches_all_pairs_on_random_masks() {
        let mut seed = 7u64;
        for (n, density) in [(9usize, 0.3), (14, 0.15), (20, 0.5), (24, 0.05)] {
            let dim = [n, n + 3, n - 2];
            let mask: Vec<bool> = (0..dim[0] * dim[1] * dim[2])
                .map(|_| lcg(&mut seed) < density)
                .collect();
            let (_, _, verts) = mesh(&mask, dim, [1.25, 0.7, 0.7]);
            assert!(verts.len() > 256, "test should take the grid path");
            let fast = diameters(&verts);
            let slow = naive(&verts);
            for k in 0..4 {
                assert_eq!(fast[k].to_bits(), slow[k].to_bits(), "k={k} n={n}");
            }
        }
    }

    #[test]
    fn matches_all_pairs_on_sphere_and_tiny_inputs() {
        let n = 26usize;
        let c = (n as f64 - 1.0) / 2.0;
        let mask: Vec<bool> = (0..n * n * n)
            .map(|i| {
                let (z, y, x) = ((i / (n * n)) as f64, ((i / n) % n) as f64, (i % n) as f64);
                ((z - c).powi(2) + (y - c).powi(2) + (x - c).powi(2)).sqrt() <= 10.0
            })
            .collect();
        let (_, _, verts) = mesh(&mask, [n, n, n], [1.0, 0.98, 0.98]);
        let fast = diameters(&verts);
        let slow = naive(&verts);
        for k in 0..4 {
            assert_eq!(fast[k].to_bits(), slow[k].to_bits(), "k={k}");
        }
        assert_eq!(diameters(&[]), [0.0; 4]);
        assert_eq!(diameters(&[[1.0, 2.0, 3.0]]), [0.0; 4]);
    }
}
