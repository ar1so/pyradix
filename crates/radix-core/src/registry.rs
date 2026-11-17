//! Compile-time image-type and feature-class registries.
//! A new module is one entry on the matching list.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::error::RadixError;
use crate::firstorder::FirstOrderClass;
use crate::glcm::GlcmClass;
use crate::gldm::GldmClass;
use crate::glrlm::GlrlmClass;
use crate::glszm::GlszmClass;
use crate::ngtdm::NgtdmClass;
use crate::shape::{Shape2DClass, ShapeClass};
use crate::Mode;

/// Intensities in, one or more named volumes out. No discretization and no features.
pub trait ImageType: Sync {
    fn name(&self) -> &'static str;
    fn modes(&self) -> &'static [Mode];
    fn setting_keys(&self) -> &'static [&'static str];
    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError>;

    /// Same as `apply`, with the ROI mask. The default ignores the mask.
    /// LBP3D uses it: PyRadiomics samples that filter only inside the mask.
    fn apply_with_mask(
        &self,
        image: &[f64],
        mask: &[bool],
        dim: [usize; 3],
        spacing: [f64; 3],
        settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        let _ = mask;
        self.apply(image, dim, spacing, settings)
    }
}

/// The only input a feature class receives.
pub struct FeatureContext<'a> {
    pub image: &'a [f64],
    pub morphologic_mask: &'a [bool],
    pub intensity_mask: &'a [bool],
    pub dim: [usize; 3],
    pub spacing: [f64; 3],
    pub origin: [f64; 3],
    pub gray: &'a [u32],
    pub ng: u32,
    pub mode: Mode,
    pub distances: &'a [u32],
    pub weighting_norm: Option<&'a str>,
    pub symmetrical_glcm: bool,
    pub gldm_a: u32,
    pub voxel_array_shift: f64,
    pub force_2d: bool,
    pub force_2d_dimension: usize,
}

/// Prepared context in, feature name to value out.
pub trait FeatureClass: Sync {
    fn name(&self) -> &'static str;
    fn feature_names(&self) -> &'static [&'static str];
    fn deprecated(&self) -> &'static [&'static str];
    fn on_original_only(&self) -> bool;
    /// Texture classes are calculated per slice and averaged when `force2D` is set.
    fn averages_by_slice(&self) -> bool {
        false
    }
    fn applies(&self, _ctx: &FeatureContext<'_>) -> bool {
        true
    }
    fn compute(&self, ctx: &FeatureContext<'_>) -> Result<BTreeMap<String, f64>, RadixError>;
}

struct DeclaredImage {
    name: &'static str,
    modes: &'static [Mode],
    setting_keys: &'static [&'static str],
    /// `true` for the unchanged original volume. Other types are named here
    /// so validation and the engine share one list; their filters land in
    /// later tasks and `apply` fails closed until then.
    original: bool,
}

impl ImageType for DeclaredImage {
    fn name(&self) -> &'static str {
        self.name
    }

    fn modes(&self) -> &'static [Mode] {
        self.modes
    }

    fn setting_keys(&self) -> &'static [&'static str] {
        self.setting_keys
    }

    fn apply(
        &self,
        image: &[f64],
        dim: [usize; 3],
        _spacing: [f64; 3],
        _settings: &[(&str, &str)],
    ) -> Result<Vec<(String, Vec<f64>)>, RadixError> {
        let n = dim[0] * dim[1] * dim[2];
        if image.len() != n {
            return Err(RadixError::step(
                "image",
                "image length does not match shape",
            ));
        }
        if self.original {
            return Ok(vec![("original".to_string(), image.to_vec())]);
        }
        Err(RadixError::image_type(
            self.name,
            "this image type is registered and its filter is not in the engine yet",
        ))
    }
}

const BOTH: &[Mode] = &[Mode::Ibsi, Mode::Pyradiomics];

const BIN_KEYS: &[&str] = &["binWidth", "binCount"];

fn image_registry() -> &'static [&'static dyn ImageType] {
    static REG: &[&dyn ImageType] = &[
        &DeclaredImage {
            name: "Original",
            modes: BOTH,
            setting_keys: BIN_KEYS,
            original: true,
        },
        &crate::log_wavelet::PyradLoG,
        &crate::log_wavelet::IbsiLoG,
        &crate::log_wavelet::PyradWavelet,
        &crate::log_wavelet::IbsiWavelet,
        &crate::pyrad_filters::Square,
        &crate::pyrad_filters::SquareRoot,
        &crate::pyrad_filters::Logarithm,
        &crate::pyrad_filters::Exponential,
        &crate::pyrad_local::Gradient,
        &crate::pyrad_local::Lbp2D,
        &crate::pyrad_local::Lbp3D,
        &crate::ibsi_types::Mean,
        &crate::ibsi_types::Laws,
        &crate::ibsi_types::Gabor,
    ];
    REG
}

fn class_registry() -> &'static [&'static dyn FeatureClass] {
    static REG: &[&dyn FeatureClass] = &[
        &FirstOrderClass,
        &ShapeClass,
        &Shape2DClass,
        &GlcmClass,
        &GlrlmClass,
        &GlszmClass,
        &NgtdmClass,
        &GldmClass,
    ];
    REG
}

pub fn image_types() -> &'static [&'static dyn ImageType] {
    image_registry()
}

pub fn feature_classes() -> &'static [&'static dyn FeatureClass] {
    class_registry()
}

pub fn image_type_names(mode: Mode) -> Vec<&'static str> {
    image_types()
        .iter()
        .filter(|t| t.modes().contains(&mode))
        .map(|t| t.name())
        .collect()
}

pub fn all_image_type_names() -> Vec<&'static str> {
    let mut names = Vec::new();
    for kind in image_types() {
        if !names.contains(&kind.name()) {
            names.push(kind.name());
        }
    }
    names
}

pub fn feature_class_names() -> Vec<&'static str> {
    feature_classes().iter().map(|c| c.name()).collect()
}

pub fn find_image_type(name: &str, mode: Mode) -> Option<&'static dyn ImageType> {
    image_types()
        .iter()
        .copied()
        .find(|kind| kind.name() == name && kind.modes().contains(&mode))
}

pub fn find_feature_class(name: &str) -> Option<&'static dyn FeatureClass> {
    feature_classes().iter().copied().find(|c| c.name() == name)
}

/// Run an extract body. A panic becomes [`RadixError`] code `internal`.
pub fn catch_panic<T>(
    f: impl FnOnce() -> Result<T, RadixError> + std::panic::UnwindSafe,
) -> Result<T, RadixError> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = catch_unwind(f);
    std::panic::set_hook(hook);
    match caught {
        Ok(result) => result,
        Err(_) => Err(RadixError::internal("extract panicked")),
    }
}

/// Same as [`catch_panic`] when the closure borrows local buffers.
pub fn catch_panic_ref<T>(f: impl FnOnce() -> Result<T, RadixError>) -> Result<T, RadixError> {
    catch_panic(AssertUnwindSafe(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyradiomics_types_exclude_ibsi_only() {
        let names = image_type_names(Mode::Pyradiomics);
        assert!(names.contains(&"LBP3D"));
        assert!(names.contains(&"Square"));
        assert!(!names.contains(&"Mean"));
        assert!(!names.contains(&"Laws"));
        assert!(!names.contains(&"Gabor"));
        assert!(!names.contains(&"Riesz"));
    }

    #[test]
    fn ibsi_types_exclude_pyradiomics_only() {
        let names = image_type_names(Mode::Ibsi);
        assert!(names.contains(&"Gabor"));
        assert!(names.contains(&"LoG"));
        assert!(!names.contains(&"Square"));
        assert!(!names.contains(&"Gradient"));
        assert!(!names.contains(&"LBP2D"));
    }

    #[test]
    fn classes_include_shape_and_texture() {
        let names = feature_class_names();
        for expected in [
            "firstorder",
            "shape",
            "shape2D",
            "glcm",
            "glrlm",
            "glszm",
            "ngtdm",
            "gldm",
        ] {
            assert!(names.contains(&expected), "{expected}");
        }
        assert!(find_feature_class("shape").unwrap().on_original_only());
        assert!(!find_feature_class("glcm").unwrap().on_original_only());
    }

    #[test]
    fn original_apply_returns_one_volume() {
        let kind = find_image_type("Original", Mode::Ibsi).unwrap();
        let image = vec![1.0, 2.0, 3.0, 4.0];
        let out = kind.apply(&image, [1, 2, 2], [1.0, 1.0, 1.0], &[]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "original");
        assert_eq!(out[0].1, image);
    }

    #[test]
    fn panic_becomes_internal() {
        let err = catch_panic(|| -> Result<(), RadixError> { panic!("boom") }).unwrap_err();
        assert_eq!(err.code(), "internal");
        assert_eq!(err.message(), "extract panicked");
    }
}
