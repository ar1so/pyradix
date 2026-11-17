//! Shape 3D / 2D.
//! IBSI mesh: 0.5-iso Lorensen 256-cube. PyRadiomics mesh: `cshape.c` 128-cube.
//! PCA: voxel centres. IBSI divides covariance by N-1; PyRadiomics by N.

use std::collections::BTreeMap;

use crate::error::RadixError;
use crate::registry::{FeatureClass, FeatureContext};
use crate::util::{idx, mean_maps, plane_slice};
use crate::Mode;

fn eigen_sym3(a: [[f64; 3]; 3]) -> [f64; 3] {
    let mut m = a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..48 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = m[0][1].abs();
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                if value.abs() > max {
                    max = value.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-15 {
            break;
        }
        let app = m[p][p];
        let aqq = m[q][q];
        let apq = m[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        m[p][p] = app - t * apq;
        m[q][q] = aqq + t * apq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;
        for k in 0..3 {
            if k != p && k != q {
                let mkp = m[k][p];
                let mkq = m[k][q];
                m[k][p] = c * mkp - s * mkq;
                m[p][k] = m[k][p];
                m[k][q] = s * mkp + c * mkq;
                m[q][k] = m[k][q];
            }
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut e = [m[0][0], m[1][1], m[2][2]];
    e.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    e
}

fn pca_axes(pts: &[[f64; 3]], population: bool) -> (f64, f64, f64) {
    let n = pts.len() as f64;
    if pts.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    let mut c = [0.0; 3];
    for p in pts {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    let mut cov = [[0.0; 3]; 3];
    for p in pts {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    // PyRadiomics shape.py: physicalCoordinates /= sqrt(Np); covariance = X.T @ X (divide by N).
    // IBSI digital phantom matches sample covariance (N-1).
    let denom = if population {
        n
    } else if pts.len() > 1 {
        n - 1.0
    } else {
        n
    };
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v /= denom;
        }
    }
    let e = eigen_sym3(cov);
    let clip = |x: f64| x.max(0.0);
    (
        4.0 * clip(e[0]).sqrt(),
        4.0 * clip(e[1]).sqrt(),
        4.0 * clip(e[2]).sqrt(),
    )
}

fn max_pair(pts: &[[f64; 3]]) -> f64 {
    crate::cshape::max_dist2(&unique_points(pts)).sqrt()
}

fn unique_points(pts: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut out = pts.to_vec();
    out.sort_unstable_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    out.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1] && a[2] == b[2]);
    out
}

fn max_same_slice(pts: &[[f64; 3]], axis: usize) -> f64 {
    crate::cshape::same_coord_max(&unique_points(pts), axis).sqrt()
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn interp(p1: [f64; 3], p2: [f64; 3], v1: f64, v2: f64, iso: f64) -> [f64; 3] {
    let d = v2 - v1;
    let t = if d.abs() < 1e-12 {
        0.5
    } else {
        ((iso - v1) / d).clamp(0.0, 1.0)
    };
    add(p1, scale(sub(p2, p1), t))
}

const EDGE_VERTS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

fn mesh_from_mask(mask: &[bool], dim: [usize; 3], spacing: [f64; 3]) -> (f64, f64, Vec<[f64; 3]>) {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let (sz, sy, sx) = (spacing[0], spacing[1], spacing[2]);
    let iso = 0.5;
    let sample = |pz: usize, py: usize, px: usize| -> f64 {
        if pz == 0 || py == 0 || px == 0 || pz > nz || py > ny || px > nx {
            0.0
        } else if mask[idx(pz - 1, py - 1, px - 1, dim)] {
            1.0
        } else {
            0.0
        }
    };
    let pos = |pz: usize, py: usize, px: usize| -> [f64; 3] {
        [
            (px as f64 - 0.5) * sx,
            (py as f64 - 0.5) * sy,
            (pz as f64 - 0.5) * sz,
        ]
    };
    let mut tris = Vec::new();
    for pz in 0..=nz {
        for py in 0..=ny {
            for px in 0..=nx {
                let corners = [
                    (px, py, pz),
                    (px + 1, py, pz),
                    (px + 1, py + 1, pz),
                    (px, py + 1, pz),
                    (px, py, pz + 1),
                    (px + 1, py, pz + 1),
                    (px + 1, py + 1, pz + 1),
                    (px, py + 1, pz + 1),
                ];
                let p = corners.map(|(x, y, z)| pos(z, y, x));
                let val = corners.map(|(x, y, z)| sample(z, y, x));
                let mut cubeindex = 0u8;
                for (i, value) in val.iter().enumerate() {
                    if *value >= iso {
                        cubeindex |= 1 << i;
                    }
                }
                let edges = crate::mc_tables::EDGE_TABLE[cubeindex as usize];
                if edges == 0 {
                    continue;
                }
                let mut ev = [[0.0; 3]; 12];
                for e in 0..12 {
                    if edges & (1 << e) != 0 {
                        let [a, b] = EDGE_VERTS[e];
                        ev[e] = interp(p[a], p[b], val[a], val[b], iso);
                    }
                }
                let row = crate::mc_tables::TRI_TABLE[cubeindex as usize];
                let mut k = 0usize;
                while k + 2 < 16 && row[k] >= 0 {
                    tris.push([
                        ev[row[k] as usize],
                        ev[row[k + 1] as usize],
                        ev[row[k + 2] as usize],
                    ]);
                    k += 3;
                }
            }
        }
    }
    let mut volume = 0.0;
    let mut area = 0.0;
    let mut verts = Vec::with_capacity(tris.len() * 3);
    for t in &tris {
        let [a, b, c] = *t;
        volume += dot(a, cross(b, c)) / 6.0;
        area += 0.5 * {
            let n = cross(sub(b, a), sub(c, a));
            (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
        };
        verts.push(a);
        verts.push(b);
        verts.push(c);
    }
    (volume.abs(), area, verts)
}

pub fn shape3d(
    mask: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    mode: Mode,
) -> BTreeMap<String, f64> {
    let (nz, ny, nx) = (dim[0], dim[1], dim[2]);
    let (sz, sy, sx) = (spacing[0], spacing[1], spacing[2]);
    let vox_vol = sz * sy * sx;
    let mut nv = 0usize;
    let mut pts = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !mask[idx(z, y, x, dim)] {
                    continue;
                }
                nv += 1;
                pts.push([
                    (x as f64 + 0.5) * sx,
                    (y as f64 + 0.5) * sy,
                    (z as f64 + 0.5) * sz,
                ]);
            }
        }
    }
    let voxel_volume = nv as f64 * vox_vol;
    let (volume, area, d_slice, d_col, d_row, d3) = match mode {
        Mode::Pyradiomics => {
            let (mesh_volume, area, verts) = crate::cshape::mesh(mask, dim, spacing);
            let volume = if mesh_volume.abs() > 0.0 {
                mesh_volume.abs()
            } else {
                voxel_volume
            };
            let [slice, col, row, d3] = crate::cshape::diameters(&verts);
            (volume, area, slice, col, row, d3)
        }
        Mode::Ibsi => {
            let (mesh_volume, area, verts) = mesh_from_mask(mask, dim, spacing);
            let volume = if mesh_volume > 0.0 {
                mesh_volume
            } else {
                voxel_volume
            };
            let diam_pts = if verts.is_empty() { &pts } else { &verts };
            (
                volume,
                area,
                max_same_slice(diam_pts, 2),
                max_same_slice(diam_pts, 1),
                max_same_slice(diam_pts, 0),
                max_pair(diam_pts),
            )
        }
    };
    let (major, minor, least) = pca_axes(&pts, mode == Mode::Pyradiomics);
    let sphericity = if area > 0.0 {
        (36.0 * std::f64::consts::PI * volume * volume).cbrt() / area
    } else {
        0.0
    };
    let mut m = BTreeMap::new();
    m.insert("MeshVolume".into(), volume);
    m.insert("VoxelVolume".into(), voxel_volume);
    m.insert("SurfaceArea".into(), area);
    m.insert(
        "SurfaceVolumeRatio".into(),
        if volume > 0.0 { area / volume } else { 0.0 },
    );
    m.insert("Sphericity".into(), sphericity);
    m.insert(
        "Compactness1".into(),
        if area > 0.0 {
            volume / (std::f64::consts::PI * area.powi(3)).sqrt()
        } else {
            0.0
        },
    );
    m.insert("Compactness2".into(), sphericity.powi(3));
    m.insert(
        "SphericalDisproportion".into(),
        if sphericity > 0.0 {
            1.0 / sphericity
        } else {
            0.0
        },
    );
    m.insert("Maximum3DDiameter".into(), d3);
    m.insert("Maximum2DDiameterSlice".into(), d_slice);
    m.insert("Maximum2DDiameterColumn".into(), d_col);
    m.insert("Maximum2DDiameterRow".into(), d_row);
    m.insert("MajorAxisLength".into(), major);
    m.insert("MinorAxisLength".into(), minor);
    m.insert("LeastAxisLength".into(), least);
    m.insert(
        "Elongation".into(),
        if major > 0.0 { minor / major } else { 0.0 },
    );
    m.insert(
        "Flatness".into(),
        if major > 0.0 { least / major } else { 0.0 },
    );
    m
}

pub fn shape2d(
    mask: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    axis: usize,
) -> BTreeMap<String, f64> {
    let mut pts = Vec::new();
    let mut np = 0usize;
    let pix = match axis {
        0 => spacing[1] * spacing[2],
        1 => spacing[0] * spacing[2],
        _ => spacing[0] * spacing[1],
    };
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                if !mask[idx(z, y, x, dim)] {
                    continue;
                }
                np += 1;
                let p = [
                    (x as f64 + 0.5) * spacing[2],
                    (y as f64 + 0.5) * spacing[1],
                    (z as f64 + 0.5) * spacing[0],
                ];
                pts.push(p);
            }
        }
    }
    let area = np as f64 * pix;
    let mut peri = 0.0;
    let neigh = match axis {
        0 => [([0, 1, 0], spacing[1]), ([0, 0, 1], spacing[2])],
        1 => [([1, 0, 0], spacing[0]), ([0, 0, 1], spacing[2])],
        _ => [([1, 0, 0], spacing[0]), ([0, 1, 0], spacing[1])],
    };
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                if !mask[idx(z, y, x, dim)] {
                    continue;
                }
                for (d, len) in neigh {
                    for sign in [1i32, -1] {
                        let zz = z as i32 + sign * d[0];
                        let yy = y as i32 + sign * d[1];
                        let xx = x as i32 + sign * d[2];
                        let outside = zz < 0
                            || yy < 0
                            || xx < 0
                            || zz >= dim[0] as i32
                            || yy >= dim[1] as i32
                            || xx >= dim[2] as i32
                            || !mask[idx(zz as usize, yy as usize, xx as usize, dim)];
                        if outside {
                            peri += len;
                        }
                    }
                }
            }
        }
    }
    let (major, minor, _) = pca_axes(&pts, false);
    let sphericity = if peri > 0.0 {
        2.0 * (std::f64::consts::PI * area).sqrt() / peri
    } else {
        0.0
    };
    let mut m = BTreeMap::new();
    m.insert("MeshSurface".into(), area);
    m.insert("PixelSurface".into(), area);
    m.insert("Perimeter".into(), peri);
    m.insert(
        "PerimeterSurfaceRatio".into(),
        if area > 0.0 { peri / area } else { 0.0 },
    );
    m.insert("Sphericity".into(), sphericity);
    m.insert(
        "SphericalDisproportion".into(),
        if sphericity > 0.0 {
            1.0 / sphericity
        } else {
            0.0
        },
    );
    m.insert("MaximumDiameter".into(), max_pair(&pts));
    m.insert("MajorAxisLength".into(), major);
    m.insert("MinorAxisLength".into(), minor);
    m.insert(
        "Elongation".into(),
        if major > 0.0 { minor / major } else { 0.0 },
    );
    m
}

const SHAPE_NAMES: &[&str] = &[
    "Compactness1",
    "Compactness2",
    "Elongation",
    "Flatness",
    "LeastAxisLength",
    "MajorAxisLength",
    "Maximum2DDiameterColumn",
    "Maximum2DDiameterRow",
    "Maximum2DDiameterSlice",
    "Maximum3DDiameter",
    "MeshVolume",
    "MinorAxisLength",
    "Sphericity",
    "SphericalDisproportion",
    "SurfaceArea",
    "SurfaceVolumeRatio",
    "VoxelVolume",
];

const SHAPE2D_NAMES: &[&str] = &[
    "Elongation",
    "MajorAxisLength",
    "MaximumDiameter",
    "MeshSurface",
    "MinorAxisLength",
    "Perimeter",
    "PerimeterSurfaceRatio",
    "PixelSurface",
    "Sphericity",
    "SphericalDisproportion",
];

pub struct ShapeClass;
pub struct Shape2DClass;

impl FeatureClass for ShapeClass {
    fn name(&self) -> &'static str {
        "shape"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        SHAPE_NAMES
    }

    fn deprecated(&self) -> &'static [&'static str] {
        &["Compactness1", "Compactness2", "SphericalDisproportion"]
    }

    fn on_original_only(&self) -> bool {
        true
    }

    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError> {
        Ok(shape3d(
            ctx.morphologic_mask,
            ctx.dim,
            ctx.spacing,
            ctx.mode,
        ))
    }
}

impl FeatureClass for Shape2DClass {
    fn name(&self) -> &'static str {
        "shape2D"
    }

    fn feature_names(&self) -> &'static [&'static str] {
        SHAPE2D_NAMES
    }

    fn deprecated(&self) -> &'static [&'static str] {
        &["SphericalDisproportion"]
    }

    fn on_original_only(&self) -> bool {
        true
    }

    fn applies(&self, ctx: &FeatureContext<'_>) -> bool {
        ctx.force_2d || ctx.dim.contains(&1)
    }

    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError> {
        let axis = if ctx.force_2d {
            ctx.force_2d_dimension.min(2)
        } else if ctx.dim[0] == 1 {
            0
        } else if ctx.dim[1] == 1 {
            1
        } else {
            2
        };
        if ctx.dim[axis] <= 1 {
            return Ok(shape2d(ctx.morphologic_mask, ctx.dim, ctx.spacing, axis));
        }
        let mut maps = Vec::new();
        for index in 0..ctx.dim[axis] {
            let (source, dim, spacing) = plane_slice(ctx.dim, ctx.spacing, axis, index);
            let mask: Vec<bool> = source.iter().map(|i| ctx.morphologic_mask[*i]).collect();
            if mask.iter().any(|voxel| *voxel) {
                maps.push(shape2d(&mask, dim, spacing, 0));
            }
        }
        if maps.is_empty() {
            return Err(RadixError::step("shape2D", "the ROI has no voxels"));
        }
        Ok(mean_maps(&maps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_voxel_mesh_volume_matches_voxel_volume() {
        let dim = [3, 3, 3];
        let mut mask = vec![false; 27];
        mask[idx(1, 1, 1, dim)] = true;
        let spacing = [2.0, 2.0, 2.0];
        let m = shape3d(&mask, dim, spacing, Mode::Ibsi);
        assert!((m["VoxelVolume"] - 8.0).abs() < 1e-12);
        let mv = m["MeshVolume"];
        // 0.5-iso of one voxel is an octahedron of half-axis spacing/2 → volume 4/3.
        assert!(
            (mv - 4.0 / 3.0).abs() < 1e-6,
            "MeshVolume={mv}, expected 4/3"
        );
        let p = shape3d(&mask, dim, spacing, Mode::Pyradiomics);
        assert!(
            (p["MeshVolume"] - 4.0 / 3.0).abs() < 1e-6,
            "pyradiomics MeshVolume={}, expected 4/3",
            p["MeshVolume"]
        );
    }

    #[test]
    fn ibsi_2d_diameter_uses_one_slice() {
        // One empty slice between the voxels, so their mesh planes do not meet.
        let dim = [3, 1, 6];
        let mut mask = vec![false; 18];
        mask[idx(0, 0, 0, dim)] = true;
        mask[idx(2, 0, 5, dim)] = true;
        let m = shape3d(&mask, dim, [1.0, 1.0, 1.0], Mode::Ibsi);
        let slice = m["Maximum2DDiameterSlice"];
        let d3 = m["Maximum3DDiameter"];
        assert!(
            slice < 2.0,
            "slice diameter {slice} paired vertices from different slices"
        );
        assert!(d3 > 4.0, "3D diameter {d3}");
    }
}
