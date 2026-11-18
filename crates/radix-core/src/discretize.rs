//! Gray-level discretization (IBSI vs PyRadiomics fixed-bin-size).

use crate::error::RadixError;
use crate::Mode;

/// Returns per-voxel gray levels (0 = outside mask) and Ng.
///
/// `bin_min` is the IBSI FBS origin \(X_{gl,\min}\). When set (typically the
/// lower bound of the re-segmentation range), edges start there instead of the
/// ROI minimum.
pub fn discretize(
    image: &[f64],
    mask: &[bool],
    mode: Mode,
    bin_width: Option<f64>,
    bin_count: Option<u32>,
    bin_min: Option<f64>,
) -> Result<(Vec<u32>, u32), RadixError> {
    let roi: Vec<f64> = image
        .iter()
        .zip(mask.iter())
        .filter_map(|(v, m)| if *m { Some(*v) } else { None })
        .collect();
    if roi.is_empty() {
        return Err(RadixError::step("discretize", "the ROI has no voxels"));
    }
    let min_v = roi.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = roi.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut out = vec![0u32; image.len()];
    if mode == Mode::Pyradiomics {
        let edges = match bin_count {
            Some(0) => return Err(RadixError::setting("binCount", "binCount must be > 0")),
            Some(ng) => numpy_histogram_edges(min_v, max_v, ng),
            None => {
                let bw = bin_width
                    .ok_or_else(|| RadixError::setting("binWidth", "set binWidth or binCount"))?;
                if bw <= 0.0 {
                    return Err(RadixError::setting("binWidth", "binWidth must be > 0"));
                }
                pyradiomics_fixed_width_edges(min_v, max_v, bw)
            }
        };
        let mut used_max = 1u32;
        for i in 0..image.len() {
            if mask[i] {
                let g = numpy_digitize(image[i], &edges);
                out[i] = g;
                used_max = used_max.max(g);
            }
        }
        return Ok((out, used_max));
    }

    if let Some(ng) = bin_count {
        if ng == 0 {
            return Err(RadixError::setting("binCount", "binCount must be > 0"));
        }
        if (max_v - min_v).abs() < f64::EPSILON {
            for (o, m) in out.iter_mut().zip(mask.iter()) {
                if *m {
                    *o = 1;
                }
            }
            return Ok((out, 1));
        }
        let mut used_max = 1u32;
        for i in 0..image.len() {
            if !mask[i] {
                continue;
            }
            let mut g = ((f64::from(ng) * (image[i] - min_v) / (max_v - min_v)).floor() as i64) + 1;
            if g < 1 {
                g = 1;
            }
            if g > ng as i64 {
                g = ng as i64;
            }
            out[i] = g as u32;
            used_max = used_max.max(out[i]);
        }
        return Ok((out, used_max));
    }

    let bw =
        bin_width.ok_or_else(|| RadixError::setting("binWidth", "set binWidth or binCount"))?;
    if bw <= 0.0 {
        return Err(RadixError::setting("binWidth", "binWidth must be > 0"));
    }
    let ibsi_min = bin_min.unwrap_or(min_v);
    let mut used_max = 1u32;
    for i in 0..image.len() {
        if !mask[i] {
            continue;
        }
        let g = match mode {
            Mode::Pyradiomics => {
                // FAQ: edges equally spaced from 0; lowest intensity → bin 1.
                ((image[i] / bw).floor() - (min_v / bw).floor()) as i64 + 1
            }
            Mode::Ibsi => ((image[i] - ibsi_min) / bw).floor() as i64 + 1,
        };
        let g = g.max(1) as u32;
        out[i] = g;
        used_max = used_max.max(g);
    }
    Ok((out, used_max))
}

/// Bin edges of `numpy.histogram(values, ng)[1]` with the last edge raised by 1,
/// as PyRadiomics 3.1.0 `getBinEdges` builds them for a fixed bin count.
///
/// numpy takes the edges from `linspace(min, max, ng + 1)`, which computes edge
/// `k` as `k * step + min` and sets the last edge to `max`. Computing the bin
/// with `floor(ng * (x - min) / (max - min))` instead gives the same answer on
/// paper but not in floating point: a voxel that sits exactly on an edge (common
/// for integer images after z-scoring) can land one bin higher.
fn numpy_histogram_edges(min_v: f64, max_v: f64, ng: u32) -> Vec<f64> {
    let (first, last) = if min_v == max_v {
        (min_v - 0.5, max_v + 0.5)
    } else {
        (min_v, max_v)
    };
    let n = ng as usize;
    let step = (last - first) / ng as f64;
    let mut edges: Vec<f64> = (0..=n).map(|k| k as f64 * step + first).collect();
    edges[n] = last;
    edges[n] += 1.0;
    edges
}

/// Bin edges for a fixed bin width, as PyRadiomics 3.1.0 `getBinEdges` builds
/// them: `numpy.arange(min - min % W, max + 2 W, W)`.
///
/// numpy's float `arange` fills element `i` as `start + i * ((start + W) - start)`,
/// so that exact sequence is reproduced here.
fn pyradiomics_fixed_width_edges(min_v: f64, max_v: f64, bw: f64) -> Vec<f64> {
    let mut rem = min_v % bw;
    if rem != 0.0 && (rem < 0.0) != (bw < 0.0) {
        rem += bw;
    }
    let low = min_v - rem;
    let high = max_v + 2.0 * bw;
    let len = ((high - low) / bw).ceil().max(0.0) as usize;
    let mut edges = Vec::with_capacity(len);
    if len >= 1 {
        edges.push(low);
    }
    if len >= 2 {
        edges.push(low + bw);
        let delta = edges[1] - edges[0];
        for i in 2..len {
            edges.push(low + i as f64 * delta);
        }
    }
    if edges.len() == 1 {
        let e = edges[0];
        edges = vec![e - 0.5, e + 0.5];
    }
    edges
}

/// `numpy.digitize(x, edges)` for increasing edges: the number of edges <= x.
fn numpy_digitize(x: f64, edges: &[f64]) -> u32 {
    edges.partition_point(|e| *e <= x) as u32
}

#[cfg(test)]
mod numpy_binning_tests {
    use super::*;

    #[test]
    fn fixed_count_edges_follow_linspace() {
        // linspace(min, max, ng + 1): k * step + min, last edge = max (then + 1).
        let edges = numpy_histogram_edges(-2.0, 6.0, 4);
        assert_eq!(edges, vec![-2.0, 0.0, 2.0, 4.0, 7.0]);
        assert_eq!(numpy_digitize(-2.0, &edges), 1);
        assert_eq!(numpy_digitize(2.0, &edges), 3); // on an edge: the upper bin
        assert_eq!(numpy_digitize(6.0, &edges), 4); // the maximum stays in the top bin
    }

    #[test]
    fn fixed_count_flat_region_matches_numpy_histogram() {
        // numpy.histogram([5, 5], 32) uses edges linspace(4.5, 5.5, 33): 5 is bin 17.
        let edges = numpy_histogram_edges(5.0, 5.0, 32);
        assert_eq!(numpy_digitize(5.0, &edges), 17);
    }

    #[test]
    fn fixed_width_matches_numpy_arange_example() {
        // PyRadiomics docstring: values 54..166, W 25 -> edges 50, 75, ..., 200.
        let edges = pyradiomics_fixed_width_edges(54.0, 166.0, 25.0);
        assert_eq!(edges, vec![50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]);
        assert_eq!(numpy_digitize(54.0, &edges), 1);
        assert_eq!(numpy_digitize(166.0, &edges), 5);
        assert_eq!(numpy_digitize(150.0, &edges), 5);
    }

    #[test]
    fn fixed_width_negative_minimum_uses_floored_modulo() {
        // -1000 % 25 == 0 and -987 % 25 == 13 in Python, so low bounds -1000 and -1000.
        assert_eq!(
            pyradiomics_fixed_width_edges(-1000.0, -900.0, 25.0)[0],
            -1000.0
        );
        assert_eq!(
            pyradiomics_fixed_width_edges(-987.0, -900.0, 25.0)[0],
            -1000.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibsi_fbs_uses_reseg_lower_bound() {
        let image = vec![-939.0, 393.0];
        let mask = vec![true, true];
        let (gray, ng) =
            discretize(&image, &mask, Mode::Ibsi, Some(25.0), None, Some(-1000.0)).unwrap();
        // floor((-939+1000)/25)+1 = 3; floor((393+1000)/25)+1 = 56
        assert_eq!(gray, vec![3, 56]);
        assert_eq!(ng, 56);
    }
}
