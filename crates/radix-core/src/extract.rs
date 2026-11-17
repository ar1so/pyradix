//! Full original-image feature vector.

use std::collections::BTreeMap;

use crate::discretize::discretize;
use crate::error::RadixError;
use crate::registry::{
    catch_panic_ref, feature_classes, find_image_type, image_types as registered_image_types,
    FeatureContext,
};
use crate::resample::{resample_typed, Interpolator};
use crate::reseg::{resegment_mask, ResegmentMode};
use crate::util::{idx, mean_maps, plane_slice};
use crate::Mode;

pub struct VolumeView<'a> {
    pub image: &'a [f64],
    pub mask: &'a [bool],
    pub dim: [usize; 3], // z, y, x
    pub spacing: [f64; 3],
    pub origin: [f64; 3], // z, y, x centre of voxel 0
}

pub struct ExtractSettings {
    pub mode: Mode,
    pub bin_width: Option<f64>,
    pub bin_count: Option<u32>,
    pub voxel_array_shift: f64,
    pub resample_to: Option<[f64; 3]>,
    pub interpolator: Interpolator,
    pub mask_interpolator: Option<Interpolator>,
    pub resegment: Option<(f64, f64)>,
    pub resegment_mode: ResegmentMode,
    pub resegment_shape: bool,
    pub outlier_sigma: Option<f64>,
    pub bin_min: Option<f64>,
    pub normalize: bool,
    pub normalize_scale: f64,
    pub pad_distance: i32,
    pub pre_crop: bool,
    pub image_types: Vec<(String, Vec<(String, String)>)>,
    pub features: Option<Vec<(String, Vec<String>)>>,
    pub distances: Vec<u32>,
    pub weighting_norm: Option<String>,
    pub symmetrical_glcm: bool,
    pub gldm_a: u32,
    pub force_2d: bool,
    pub force_2d_dimension: usize,
    /// Stored pixel type of the source image is an integer type. `None` = infer from values.
    pub integer_pixels: Option<bool>,
}

impl ExtractSettings {
    fn mask_interpolator(&self) -> Interpolator {
        self.mask_interpolator.unwrap_or(match self.mode {
            Mode::Pyradiomics => Interpolator::Nearest,
            Mode::Ibsi => self.interpolator,
        })
    }

    fn fbs_bin_min(&self) -> Option<f64> {
        if self.bin_min.is_some() {
            return self.bin_min;
        }
        if self.mode == Mode::Ibsi && self.bin_count.is_none() {
            return self.resegment.map(|(lo, _)| lo);
        }
        None
    }
}

pub fn extract(
    vol: &VolumeView<'_>,
    settings: &ExtractSettings,
) -> Result<BTreeMap<String, f64>, RadixError> {
    catch_panic_ref(|| extract_inner(vol, settings))
}

fn extract_inner(
    vol: &VolumeView<'_>,
    settings: &ExtractSettings,
) -> Result<BTreeMap<String, f64>, RadixError> {
    let n = vol.dim[0] * vol.dim[1] * vol.dim[2];
    if vol.image.len() != n || vol.mask.len() != n {
        return Err(RadixError::step(
            "mask",
            "image length does not match the mask",
        ));
    }
    let bad: Vec<usize> = (0..n)
        .filter(|&i| vol.mask[i] && !vol.image[i].is_finite())
        .collect();
    if let Some(&first) = bad.first() {
        let (z, y, x) = (
            first / (vol.dim[1] * vol.dim[2]),
            (first / vol.dim[2]) % vol.dim[1],
            first % vol.dim[2],
        );
        return Err(RadixError::step(
            "image",
            format!(
                "{} voxel(s) inside the ROI are NaN or infinite (first at z={z}, y={y}, x={x}); \
                 fill or mask them before extraction",
                bad.len()
            ),
        ));
    }
    if let Some(out_sp) = settings.resample_to {
        let out_sp = resolve_spacing(out_sp, vol.spacing);
        let normalized = normalize_if_needed(vol.image, settings);
        let (src_min, src_max) = mask_bounds(vol.mask, vol.dim)?;
        let out_sp = if settings.mode == Mode::Pyradiomics {
            keep_single_slice_spacing(src_min, src_max, vol.spacing, out_sp)
        } else {
            out_sp
        };
        let r = resample_typed(
            &normalized,
            vol.mask,
            vol.dim,
            vol.spacing,
            vol.origin,
            out_sp,
            settings.mode,
            settings.interpolator,
            settings.mask_interpolator(),
            settings.integer_pixels,
        )?;
        let cropped = if settings.mode == Mode::Pyradiomics {
            let (lo, hi) = pyradiomics_window(
                src_min,
                src_max,
                vol.dim,
                vol.spacing,
                out_sp,
                settings.pad_distance,
            );
            crop_inclusive(&r.image, &r.mask, r.dim, r.spacing, r.origin, lo, hi)?
        } else {
            let padded = crop_to_roi(
                &r.image,
                &r.mask,
                r.dim,
                r.spacing,
                r.origin,
                settings.pad_distance,
            )?;
            crop_to_roi(
                &padded.image,
                &padded.mask,
                padded.dim,
                padded.spacing,
                padded.origin,
                0,
            )?
        };
        let prepared = VolumeView {
            image: &cropped.image,
            mask: &cropped.mask,
            dim: cropped.dim,
            spacing: cropped.spacing,
            origin: cropped.origin,
        };
        return extract_prepared(&prepared, settings);
    }
    let normalized = normalize_if_needed(vol.image, settings);
    if settings.pre_crop {
        let cropped = crop_to_roi(
            &normalized,
            vol.mask,
            vol.dim,
            vol.spacing,
            vol.origin,
            settings.pad_distance,
        )?;
        let prepared = VolumeView {
            image: &cropped.image,
            mask: &cropped.mask,
            dim: cropped.dim,
            spacing: cropped.spacing,
            origin: cropped.origin,
        };
        return extract_prepared(&prepared, settings);
    }
    if settings.normalize {
        let prepared = VolumeView {
            image: &normalized,
            mask: vol.mask,
            dim: vol.dim,
            spacing: vol.spacing,
            origin: vol.origin,
        };
        return extract_prepared(&prepared, settings);
    }
    extract_prepared(vol, settings)
}

fn extract_prepared(
    vol: &VolumeView<'_>,
    settings: &ExtractSettings,
) -> Result<BTreeMap<String, f64>, RadixError> {
    let morph = vol.mask;
    let mask_outlier = if settings.mode == Mode::Ibsi && !settings.normalize {
        settings.outlier_sigma
    } else {
        None
    };
    let intensity = resegment_mask(
        vol.image,
        morph,
        settings.resegment,
        settings.resegment_mode,
        mask_outlier,
    )?;
    let morphologic = if settings.resegment_shape {
        intensity.clone()
    } else {
        morph.to_vec()
    };
    let window = if settings.mode == Mode::Pyradiomics {
        feature_window(&morphologic, &intensity, vol.dim, vol.spacing, vol.origin)?
    } else {
        None
    };
    let image_types = if settings.image_types.is_empty() {
        vec![("Original".to_string(), Vec::new())]
    } else {
        settings.image_types.clone()
    };
    let mut out = BTreeMap::new();
    for (type_name, type_settings) in &image_types {
        let kind = find_image_type(type_name, settings.mode).ok_or_else(|| {
            if registered_image_types()
                .iter()
                .any(|kind| kind.name() == type_name)
            {
                RadixError::image_type(
                    type_name,
                    format!("not available in {} mode", settings.mode.as_str()),
                )
            } else {
                RadixError::image_type(type_name, "this image type is not registered")
            }
        })?;
        let refs: Vec<(&str, &str)> = type_settings
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        let volumes = kind.apply_with_mask(vol.image, &intensity, vol.dim, vol.spacing, &refs)?;
        let (bin_width, bin_count) = bins_for(settings, type_settings);
        for (prefix, image) in volumes {
            // PyRadiomics crops to the ROI box with no padding after the filter
            // and before features. The filter itself still sees the full image.
            let (feat_image, feat_morph, feat_intensity, feat_dim, feat_spacing, feat_origin) =
                match &window {
                    Some(bounds) => (
                        crop_image(&image, vol.dim, bounds.lo, bounds.dim),
                        bounds.morphologic.as_slice(),
                        bounds.intensity.as_slice(),
                        bounds.dim,
                        bounds.spacing,
                        bounds.origin,
                    ),
                    None => (
                        image,
                        morphologic.as_slice(),
                        intensity.as_slice(),
                        vol.dim,
                        vol.spacing,
                        vol.origin,
                    ),
                };
            let (gray_i, ng_i) = discretize(
                &feat_image,
                feat_intensity,
                settings.mode,
                bin_width,
                bin_count,
                settings.fbs_bin_min(),
            )?;
            let part = features_on_volume(
                settings,
                &prefix,
                &feat_image,
                feat_morph,
                feat_intensity,
                feat_dim,
                feat_spacing,
                feat_origin,
                &gray_i,
                ng_i,
            )?;
            out.extend(part);
        }
    }
    Ok(out)
}

fn features_on_volume(
    settings: &ExtractSettings,
    prefix: &str,
    image: &[f64],
    morphologic: &[bool],
    intensity: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    gray: &[u32],
    ng: u32,
) -> Result<BTreeMap<String, f64>, RadixError> {
    let distances = if settings.distances.is_empty() {
        vec![1u32]
    } else {
        settings.distances.clone()
    };
    let ctx = FeatureContext {
        image,
        morphologic_mask: morphologic,
        intensity_mask: intensity,
        dim,
        spacing,
        origin,
        gray,
        ng,
        mode: settings.mode,
        distances: &distances,
        weighting_norm: settings.weighting_norm.as_deref(),
        symmetrical_glcm: settings.symmetrical_glcm,
        gldm_a: settings.gldm_a,
        voxel_array_shift: settings.voxel_array_shift,
        force_2d: settings.force_2d,
        force_2d_dimension: settings.force_2d_dimension.min(2),
    };
    let mut out = BTreeMap::new();
    for class in feature_classes() {
        let class: &dyn crate::registry::FeatureClass = *class;
        if class.on_original_only() && prefix != "original" {
            continue;
        }
        if !class.applies(&ctx) {
            continue;
        }
        let Some(allowed) = allowed_names(settings, class) else {
            continue;
        };
        for name in &allowed {
            if !class.feature_names().contains(&name.as_str()) {
                return Err(RadixError::feature(
                    prefix,
                    name,
                    format!("class {}; unknown feature", class.name()),
                ));
            }
        }
        let computed = if settings.force_2d && class.averages_by_slice() {
            average_slices(&ctx, class)?
        } else {
            class.compute(&ctx)?
        };
        out.extend(take_features(prefix, class.name(), computed, &allowed)?);
    }
    Ok(out)
}

fn allowed_names(
    settings: &ExtractSettings,
    class: &dyn crate::registry::FeatureClass,
) -> Option<Vec<String>> {
    let non_deprecated = || {
        class
            .feature_names()
            .iter()
            .filter(|name| !class.deprecated().contains(name))
            .map(|name| (*name).to_string())
            .collect()
    };
    match &settings.features {
        None => Some(non_deprecated()),
        Some(classes) => match classes.iter().find(|(name, _)| name == class.name()) {
            None => None,
            Some((_, names)) if names.is_empty() => Some(non_deprecated()),
            Some((_, names)) => Some(names.clone()),
        },
    }
}

fn take_features(
    prefix: &str,
    class_name: &str,
    computed: BTreeMap<String, f64>,
    allowed: &[String],
) -> Result<BTreeMap<String, f64>, RadixError> {
    let mut out = BTreeMap::new();
    for name in allowed {
        let Some(value) = computed.get(name) else {
            continue;
        };
        if !value.is_finite() {
            return Err(RadixError::feature(
                prefix,
                name,
                format!("class {class_name}; this feature is undefined for this ROI"),
            ));
        }
        out.insert(format!("{prefix}_{class_name}_{name}"), *value);
    }
    Ok(out)
}

fn average_slices(
    ctx: &FeatureContext<'_>,
    class: &dyn crate::registry::FeatureClass,
) -> Result<BTreeMap<String, f64>, RadixError> {
    let axis = ctx.force_2d_dimension.min(2);
    let mut maps = Vec::new();
    for index in 0..ctx.dim[axis] {
        let (source, dim, spacing) = plane_slice(ctx.dim, ctx.spacing, axis, index);
        let intensity: Vec<bool> = source.iter().map(|i| ctx.intensity_mask[*i]).collect();
        if !intensity.iter().any(|voxel| *voxel) {
            continue;
        }
        let image: Vec<f64> = source.iter().map(|i| ctx.image[*i]).collect();
        let morphologic: Vec<bool> = source.iter().map(|i| ctx.morphologic_mask[*i]).collect();
        let gray: Vec<u32> = source.iter().map(|i| ctx.gray[*i]).collect();
        let distances = ctx.distances.to_vec();
        let slice = FeatureContext {
            image: &image,
            morphologic_mask: &morphologic,
            intensity_mask: &intensity,
            dim,
            spacing,
            origin: ctx.origin,
            gray: &gray,
            ng: ctx.ng,
            mode: ctx.mode,
            distances: &distances,
            weighting_norm: ctx.weighting_norm,
            symmetrical_glcm: ctx.symmetrical_glcm,
            gldm_a: ctx.gldm_a,
            voxel_array_shift: ctx.voxel_array_shift,
            force_2d: false,
            force_2d_dimension: 0,
        };
        maps.push(class.compute(&slice)?);
    }
    if maps.is_empty() {
        return Err(RadixError::step("mask", "the ROI has no voxels"));
    }
    Ok(mean_maps(&maps))
}

fn bins_for(
    settings: &ExtractSettings,
    type_settings: &[(String, String)],
) -> (Option<f64>, Option<u32>) {
    let mut width = settings.bin_width;
    let mut count = settings.bin_count;
    for (key, value) in type_settings {
        if key == "binWidth" {
            if let Ok(parsed) = value.parse::<f64>() {
                width = Some(parsed);
                count = None;
            }
        } else if key == "binCount" {
            if let Ok(parsed) = value.parse::<u32>() {
                count = Some(parsed);
                width = None;
            }
        }
    }
    (width, count)
}

struct Cropped {
    image: Vec<f64>,
    mask: Vec<bool>,
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
}

struct FeatureWindow {
    lo: [usize; 3],
    dim: [usize; 3],
    morphologic: Vec<bool>,
    intensity: Vec<bool>,
    spacing: [f64; 3],
    origin: [f64; 3],
}

/// Tight ROI box used for feature classes. `None` when the ROI already fills the volume.
fn feature_window(
    morphologic: &[bool],
    intensity: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
) -> Result<Option<FeatureWindow>, RadixError> {
    let (min, max) = mask_bounds(morphologic, dim)?;
    let new_dim = [
        max[0] - min[0] + 1,
        max[1] - min[1] + 1,
        max[2] - min[2] + 1,
    ];
    if new_dim == dim {
        return Ok(None);
    }
    let n = new_dim[0] * new_dim[1] * new_dim[2];
    let mut morph = vec![false; n];
    let mut inten = vec![false; n];
    for z in 0..new_dim[0] {
        for y in 0..new_dim[1] {
            for x in 0..new_dim[2] {
                let src = idx(min[0] + z, min[1] + y, min[2] + x, dim);
                let dst = idx(z, y, x, new_dim);
                morph[dst] = morphologic[src];
                inten[dst] = intensity[src];
            }
        }
    }
    let mut origin_out = origin;
    for axis in 0..3 {
        origin_out[axis] += min[axis] as f64 * spacing[axis];
    }
    Ok(Some(FeatureWindow {
        lo: min,
        dim: new_dim,
        morphologic: morph,
        intensity: inten,
        spacing,
        origin: origin_out,
    }))
}

fn crop_image(image: &[f64], dim: [usize; 3], lo: [usize; 3], new_dim: [usize; 3]) -> Vec<f64> {
    let mut out = vec![0.0; new_dim[0] * new_dim[1] * new_dim[2]];
    let row = new_dim[2];
    for z in 0..new_dim[0] {
        for y in 0..new_dim[1] {
            let src = idx(lo[0] + z, lo[1] + y, lo[2], dim);
            let dst = idx(z, y, 0, new_dim);
            out[dst..dst + row].copy_from_slice(&image[src..src + row]);
        }
    }
    out
}

fn resolve_spacing(requested: [f64; 3], original: [f64; 3]) -> [f64; 3] {
    let mut out = requested;
    for axis in 0..3 {
        if out[axis] == 0.0 {
            out[axis] = original[axis];
        }
    }
    out
}

fn normalize_if_needed(image: &[f64], settings: &ExtractSettings) -> Vec<f64> {
    if !settings.normalize {
        return image.to_vec();
    }
    let outliers = settings.outlier_sigma;
    let n = image.len() as f64;
    if n == 0.0 {
        return Vec::new();
    }
    let mean = image.iter().sum::<f64>() / n;
    let var = image.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let mut out = Vec::with_capacity(image.len());
    for value in image {
        let mut scaled = if std == 0.0 {
            0.0
        } else {
            (value - mean) / std
        };
        if let Some(limit) = outliers {
            if scaled > limit {
                scaled = limit;
            } else if scaled < -limit {
                scaled = -limit;
            }
        }
        out.push(scaled * settings.normalize_scale);
    }
    out
}

fn mask_bounds(mask: &[bool], dim: [usize; 3]) -> Result<([usize; 3], [usize; 3]), RadixError> {
    let mut min = [usize::MAX; 3];
    let mut max = [0usize; 3];
    let mut any = false;
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                if !mask[idx(z, y, x, dim)] {
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
        return Err(RadixError::step("mask", "the ROI has no voxels"));
    }
    Ok((min, max))
}

/// A one-voxel-thick axis keeps its input spacing, matching `resampleImage`.
fn keep_single_slice_spacing(
    min: [usize; 3],
    max: [usize; 3],
    spacing: [f64; 3],
    out_spacing: [f64; 3],
) -> [f64; 3] {
    let mut out = out_spacing;
    for axis in 0..3 {
        if min[axis] == max[axis] {
            out[axis] = spacing[axis];
        }
    }
    out
}

/// Inclusive output-index window of the resampled bounding box plus `padDistance`.
fn pyradiomics_window(
    min: [usize; 3],
    max: [usize; 3],
    dim: [usize; 3],
    spacing: [f64; 3],
    out_spacing: [f64; 3],
    pad: i32,
) -> ([usize; 3], [usize; 3]) {
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for axis in 0..3 {
        let ratio = if out_spacing[axis] == 0.0 {
            1.0
        } else {
            spacing[axis] / out_spacing[axis]
        };
        let bb = min[axis] as f64;
        let size = (max[axis] - min[axis] + 1) as f64;
        let mut low = ((bb - 0.5) * ratio - pad as f64).floor();
        let mut high = ((bb + size - 0.5) * ratio + pad as f64).ceil();
        let max_u = (dim[axis] as f64 * ratio).ceil() - 1.0;
        if low < 0.0 {
            low = 0.0;
        }
        if high > max_u {
            high = max_u;
        }
        if high < low {
            high = low;
        }
        lo[axis] = low as usize;
        hi[axis] = high as usize;
    }
    (lo, hi)
}

fn crop_inclusive(
    image: &[f64],
    mask: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    lo: [usize; 3],
    hi: [usize; 3],
) -> Result<Cropped, RadixError> {
    let mut new_dim = [0usize; 3];
    for axis in 0..3 {
        if hi[axis] >= dim[axis] || lo[axis] > hi[axis] {
            return Err(RadixError::step("resample", "the resampled mask is empty"));
        }
        new_dim[axis] = hi[axis] - lo[axis] + 1;
    }
    let mut out_image = vec![0.0; new_dim[0] * new_dim[1] * new_dim[2]];
    let mut out_mask = vec![false; out_image.len()];
    for z in 0..new_dim[0] {
        for y in 0..new_dim[1] {
            for x in 0..new_dim[2] {
                let src = idx(lo[0] + z, lo[1] + y, lo[2] + x, dim);
                let dst = idx(z, y, x, new_dim);
                out_image[dst] = image[src];
                out_mask[dst] = mask[src];
            }
        }
    }
    if !out_mask.iter().any(|voxel| *voxel) {
        return Err(RadixError::step("resample", "the resampled mask is empty"));
    }
    let mut out_origin = origin;
    for axis in 0..3 {
        out_origin[axis] += lo[axis] as f64 * spacing[axis];
    }
    Ok(Cropped {
        image: out_image,
        mask: out_mask,
        dim: new_dim,
        spacing,
        origin: out_origin,
    })
}

fn crop_to_roi(
    image: &[f64],
    mask: &[bool],
    dim: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    pad: i32,
) -> Result<Cropped, RadixError> {
    let mut min = [usize::MAX; 3];
    let mut max = [0usize; 3];
    let mut any = false;
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                if !mask[idx(z, y, x, dim)] {
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
        return Err(RadixError::step("mask", "the ROI has no voxels"));
    }
    let pad = pad.max(0) as usize;
    let mut lo = [0usize; 3];
    let mut new_dim = [0usize; 3];
    for axis in 0..3 {
        let start = min[axis].saturating_sub(pad);
        let end = (max[axis] + 1 + pad).min(dim[axis]);
        lo[axis] = start;
        new_dim[axis] = end - start;
    }
    let mut out_image = vec![0.0; new_dim[0] * new_dim[1] * new_dim[2]];
    let mut out_mask = vec![false; out_image.len()];
    for z in 0..new_dim[0] {
        for y in 0..new_dim[1] {
            for x in 0..new_dim[2] {
                let src = idx(lo[0] + z, lo[1] + y, lo[2] + x, dim);
                let dst = idx(z, y, x, new_dim);
                out_image[dst] = image[src];
                out_mask[dst] = mask[src];
            }
        }
    }
    let mut out_origin = origin;
    for axis in 0..3 {
        out_origin[axis] += lo[axis] as f64 * spacing[axis];
    }
    Ok(Cropped {
        image: out_image,
        mask: out_mask,
        dim: new_dim,
        spacing,
        origin: out_origin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lung1_001_resample_window_matches_pyradiomics() {
        // SimpleITK bounding box (x, y, z, sx, sy, sz) = (290, 226, 65, 97, 90, 21).
        let min = [65usize, 226, 290];
        let max = [65 + 21 - 1, 226 + 90 - 1, 290 + 97 - 1];
        let (lo, hi) = pyradiomics_window(
            min,
            max,
            [134, 512, 512],
            [3.0, 0.9765625, 0.9765625],
            [2.0, 2.0, 2.0],
            5,
        );
        assert_eq!(lo, [91, 105, 136]);
        assert_eq!(
            [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1],
            [44, 56, 59]
        );
    }

    #[test]
    fn zero_spacing_keeps_that_axis() {
        assert_eq!(
            resolve_spacing([0.0, 2.0, 2.0], [3.0, 1.0, 1.0]),
            [3.0, 2.0, 2.0]
        );
    }

    #[test]
    fn normalize_centers_then_scales() {
        let image = [0.0, 0.0, 0.0, 4.0];
        let settings = ExtractSettings {
            mode: Mode::Pyradiomics,
            bin_width: Some(25.0),
            bin_count: None,
            voxel_array_shift: 0.0,
            resample_to: None,
            interpolator: Interpolator::Linear,
            mask_interpolator: None,
            resegment: None,
            resegment_mode: ResegmentMode::Absolute,
            resegment_shape: false,
            outlier_sigma: None,
            bin_min: None,
            normalize: true,
            normalize_scale: 2.0,
            pad_distance: 0,
            pre_crop: false,
            image_types: Vec::new(),
            features: None,
            distances: vec![1],
            weighting_norm: None,
            symmetrical_glcm: true,
            gldm_a: 0,
            force_2d: false,
            force_2d_dimension: 0,
            integer_pixels: None,
        };
        let out = normalize_if_needed(&image, &settings);
        let std = 3.0_f64.sqrt();
        assert!((out[0] - (-1.0 / std * 2.0)).abs() < 1e-12);
        assert!((out[3] - (3.0 / std * 2.0)).abs() < 1e-12);
    }

    #[test]
    fn precrop_pad_stays_inside_the_image() {
        let dim = [4usize, 4, 4];
        let mut mask = vec![false; 64];
        mask[idx(1, 1, 1, dim)] = true;
        let image = vec![1.0; 64];
        let cropped = crop_to_roi(&image, &mask, dim, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(cropped.dim, [3, 3, 3]);
        let tight = crop_to_roi(&image, &mask, dim, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(tight.dim, [4, 4, 4]);
    }

    #[test]
    fn undefined_feature_names_the_image_type_and_leaves_no_table() {
        let mut computed = BTreeMap::new();
        computed.insert("MCC".into(), f64::NAN);
        let err = take_features("original", "glcm", computed, &["MCC".into()]).unwrap_err();
        assert_eq!(err.code(), "feature");
        assert!(err.message().contains("original"));
        assert!(err.message().contains("MCC"));
        assert!(err.message().contains("glcm"));
    }
}
