//! Radiomics math. Feature names: docs/FEATURES.md (quoted from PyRadiomics).
#![allow(clippy::too_many_arguments)]

mod cshape;
mod cshape_tables;
pub mod discretize;
mod eig;
pub mod error;
pub mod extract;
pub mod filters;
pub mod firstorder;
pub mod glcm;
pub mod gldm;
pub mod glrlm;
pub mod glszm;
mod ibsi_types;
mod itk_log;
mod log_wavelet;
mod mc_tables;
pub mod ngtdm;
pub mod pyrad_filters;
mod pyrad_local;
pub mod registry;
pub mod resample;
pub mod reseg;
pub mod shape;
mod spline;
mod swt;
pub mod util;

pub use error::RadixError;
pub use extract::{extract, ExtractSettings, VolumeView};
pub use filters::{
    gabor as gabor_filter, laws as laws_filter, log as log_filter, mean as mean_filter,
    wavelet as wavelet_filter, Padding,
};
pub use registry::{
    all_image_type_names, catch_panic_ref, feature_class_names, feature_classes,
    find_feature_class, find_image_type, image_type_names, image_types, FeatureClass,
    FeatureContext, ImageType,
};
pub use resample::{resample, resample_grid, resample_typed, Interpolator, Resampled};
pub use reseg::{intensity_mask, ResegmentMode};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Ibsi,
    Pyradiomics,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "ibsi" => Ok(Self::Ibsi),
            "pyradiomics" => Ok(Self::Pyradiomics),
            other => Err(format!("unknown mode: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ibsi => "ibsi",
            Self::Pyradiomics => "pyradiomics",
        }
    }
}

pub fn ping() -> &'static str {
    "radix-core"
}

/// Called when the Python extension loads.
///
/// Reserved for BLAS runtime tweaks. Windows no longer links OpenBLAS (MCC
/// uses faer there), so this is a no-op on every current target.
pub fn configure_blas() {}
