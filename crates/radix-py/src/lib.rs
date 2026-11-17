#![allow(unknown_lints)]
#![allow(unexpected_cfgs)]
// PyO3's `PyResult` trips `useless_conversion` on every exported signature.
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use radix_core::{
    all_image_type_names, extract, feature_class_names, feature_classes, find_image_type,
    gabor_filter, image_type_names, laws_filter, log_filter, mean_filter, ping, resample_grid,
    wavelet_filter, ExtractSettings, Interpolator, Mode, Padding, RadixError as CoreError,
    ResegmentMode, VolumeView, VERSION,
};

pyo3::create_exception!(_radix, RadixError, PyException);

fn raise_radix(err: CoreError) -> PyErr {
    let code = err.code();
    let message = err.message().to_string();
    let pyerr = RadixError::new_err(message.clone());
    Python::attach(|py| {
        let value = pyerr.value(py);
        let _ = value.setattr("code", code);
        let _ = value.setattr("message", message);
    });
    pyerr
}

#[pyfunction]
fn legal_image_types(mode: &str) -> PyResult<Vec<String>> {
    let mode = Mode::parse(mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(image_type_names(mode)
        .into_iter()
        .map(str::to_string)
        .collect())
}

#[pyfunction]
fn legal_image_types_all() -> Vec<String> {
    all_image_type_names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

#[pyfunction]
fn legal_feature_classes() -> Vec<String> {
    feature_class_names()
        .into_iter()
        .map(str::to_string)
        .collect()
}

#[pyfunction]
fn feature_catalog(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for class in feature_classes() {
        let entry = PyDict::new(py);
        entry.set_item("features", class.feature_names().to_vec())?;
        entry.set_item("deprecated", class.deprecated().to_vec())?;
        out.set_item(class.name(), entry)?;
    }
    Ok(out)
}

/// Named volumes from one registered image type. Spacing is (z, y, x).
#[pyfunction]
fn apply_image_type<'py>(
    py: Python<'py>,
    name: &str,
    mode: &str,
    image: PyReadonlyArray3<f64>,
    spacing: (f64, f64, f64),
    settings: Vec<(String, String)>,
) -> PyResult<Vec<(String, Bound<'py, PyArray3<f64>>)>> {
    let image = image.as_array();
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                image_vec[z * dim[1] * dim[2] + y * dim[2] + x] = image[[z, y, x]];
            }
        }
    }
    let mode = Mode::parse(mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let kind = find_image_type(name, mode).ok_or_else(|| {
        raise_radix(CoreError::image_type(
            name,
            format!("not available in {} mode", mode.as_str()),
        ))
    })?;
    let refs: Vec<(&str, &str)> = settings
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    let volumes = py
        .detach(|| kind.apply(&image_vec, dim, [spacing.0, spacing.1, spacing.2], &refs))
        .map_err(raise_radix)?;
    let mut out = Vec::with_capacity(volumes.len());
    for (prefix, volume) in volumes {
        let arr = numpy::ndarray::Array3::from_shape_vec((dim[0], dim[1], dim[2]), volume)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        out.push((prefix, arr.into_pyarray(py)));
    }
    Ok(out)
}

#[pyfunction]
fn engine_info() -> String {
    format!("{} {}", ping(), VERSION)
}

#[pyfunction]
fn parse_mode(mode: &str) -> PyResult<String> {
    Mode::parse(mode)
        .map(|m| m.as_str().to_string())
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

type Axis3<T> = (T, T, T);
type ResampledGeometry = (Axis3<usize>, Axis3<f64>, Axis3<f64>);
type ImageTypeSettings = Vec<(String, Vec<(String, String)>)>;

/// Return (dim_zyx, spacing_zyx, origin_zyx) for the resampled grid.
#[pyfunction]
#[pyo3(signature = (dim, spacing, origin, mode, out_spacing))]
fn resample_geometry(
    dim: (usize, usize, usize),
    spacing: (f64, f64, f64),
    origin: (f64, f64, f64),
    mode: &str,
    out_spacing: (f64, f64, f64),
) -> PyResult<ResampledGeometry> {
    let mode = Mode::parse(mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let g = resample_grid(
        [dim.0, dim.1, dim.2],
        [spacing.0, spacing.1, spacing.2],
        [origin.0, origin.1, origin.2],
        [out_spacing.0, out_spacing.1, out_spacing.2],
        mode,
    );
    Ok((
        (g.dim[0], g.dim[1], g.dim[2]),
        (g.spacing[0], g.spacing[1], g.spacing[2]),
        (g.origin[0], g.origin[1], g.origin[2]),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    image,
    mask,
    spacing,
    mode,
    bin_width=None,
    bin_count=None,
    voxel_array_shift=0.0,
    resample_spacing=None,
    interpolator="linear",
    origin=None,
    mask_interpolator=None,
    resegment_range=None,
    outlier_sigma=None,
    bin_min=None,
    normalize=false,
    normalize_scale=1.0,
    pad_distance=5,
    pre_crop=false,
    resegment_mode="absolute",
    resegment_shape=false,
    image_types=None,
    feature_classes=None,
    distances=None,
    weighting_norm=None,
    symmetrical_glcm=true,
    gldm_a=0,
    force_2d=false,
    force_2d_dimension=0,
    integer_pixels=None,
))]
fn extract_features(
    py: Python<'_>,
    image: PyReadonlyArray3<f64>,
    mask: PyReadonlyArray3<bool>,
    spacing: (f64, f64, f64),
    mode: &str,
    bin_width: Option<f64>,
    bin_count: Option<u32>,
    voxel_array_shift: f64,
    resample_spacing: Option<(f64, f64, f64)>,
    interpolator: &str,
    origin: Option<(f64, f64, f64)>,
    mask_interpolator: Option<&str>,
    resegment_range: Option<(f64, f64)>,
    outlier_sigma: Option<f64>,
    bin_min: Option<f64>,
    normalize: bool,
    normalize_scale: f64,
    pad_distance: i32,
    pre_crop: bool,
    resegment_mode: &str,
    resegment_shape: bool,
    image_types: Option<ImageTypeSettings>,
    feature_classes: Option<Vec<(String, Vec<String>)>>,
    distances: Option<Vec<u32>>,
    weighting_norm: Option<String>,
    symmetrical_glcm: bool,
    gldm_a: u32,
    force_2d: bool,
    force_2d_dimension: usize,
    integer_pixels: Option<bool>,
) -> PyResult<BTreeMap<String, f64>> {
    let image = image.as_array();
    let mask = mask.as_array();
    if image.shape() != mask.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image and mask shapes differ",
        ));
    }
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    let mut mask_vec = vec![false; image_vec.len()];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                let i = z * dim[1] * dim[2] + y * dim[2] + x;
                image_vec[i] = image[[z, y, x]];
                mask_vec[i] = mask[[z, y, x]];
            }
        }
    }
    let mode = Mode::parse(mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let interpolator =
        Interpolator::parse(interpolator).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mask_interpolator = match mask_interpolator {
        Some(s) => Some(Interpolator::parse(s).map_err(pyo3::exceptions::PyValueError::new_err)?),
        None => None,
    };
    let origin = origin.unwrap_or((0.0, 0.0, 0.0));
    let resegment_mode = ResegmentMode::parse(resegment_mode).map_err(raise_radix)?;
    let vol = VolumeView {
        image: &image_vec,
        mask: &mask_vec,
        dim,
        spacing: [spacing.0, spacing.1, spacing.2],
        origin: [origin.0, origin.1, origin.2],
    };
    let settings = ExtractSettings {
        mode,
        bin_width,
        bin_count,
        voxel_array_shift,
        resample_to: resample_spacing.map(|s| [s.0, s.1, s.2]),
        interpolator,
        mask_interpolator,
        resegment: resegment_range,
        resegment_mode,
        resegment_shape,
        outlier_sigma,
        bin_min,
        normalize,
        normalize_scale,
        pad_distance,
        pre_crop,
        image_types: image_types.unwrap_or_default(),
        features: feature_classes,
        distances: distances.unwrap_or_else(|| vec![1]),
        weighting_norm,
        symmetrical_glcm,
        gldm_a,
        force_2d,
        force_2d_dimension,
        integer_pixels,
    };
    // Drop the GIL for the duration of the extract. Holding it while OpenBLAS
    // (or Accelerate) takes its lock deadlocks a NumPy thread that already
    // holds that lock and is waiting to re-enter Python.
    py.detach(|| extract(&vol, &settings)).map_err(raise_radix)
}

/// IBSI-2 mean filter S60F. `image` is (z, y, x). `ndims` is 2 (axial) or 3.
#[pyfunction]
#[pyo3(signature = (image, support, padding="zero", ndims=3))]
fn apply_mean<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    support: usize,
    padding: &str,
    ndims: u8,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let image = image.as_array();
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                image_vec[z * dim[1] * dim[2] + y * dim[2] + x] = image[[z, y, x]];
            }
        }
    }
    let pad = Padding::parse(padding).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = py
        .detach(|| mean_filter(&image_vec, dim, support, pad, ndims))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((dim[0], dim[1], dim[2]), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// IBSI-2 LoG L6PA (Eq. 4.2). `image` is (z, y, x). `sigma_mm` is σ* (physical).
/// `spacing` is (z, y, x) millimetres. `ndims` is 2 (axial) or 3.
#[pyfunction]
#[pyo3(signature = (image, sigma_mm, spacing, padding="zero", ndims=3, cutoff=4.0))]
fn apply_log<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    sigma_mm: f64,
    spacing: (f64, f64, f64),
    padding: &str,
    ndims: u8,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let image = image.as_array();
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                image_vec[z * dim[1] * dim[2] + y * dim[2] + x] = image[[z, y, x]];
            }
        }
    }
    let pad = Padding::parse(padding).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = py
        .detach(|| {
            log_filter(
                &image_vec,
                dim,
                [spacing.0, spacing.1, spacing.2],
                sigma_mm,
                pad,
                ndims,
                cutoff,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((dim[0], dim[1], dim[2]), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// IBSI-2 Laws JTXT. `kernels` is e.g. `E5L5S5` (g1 along k1=x).
/// Energy, if set, is the mean of |h| after optional Appendix-A max pooling.
#[pyfunction]
#[pyo3(signature = (image, kernels, padding="zero", rotation_invariant=false, energy_delta=None))]
fn apply_laws<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    kernels: &str,
    padding: &str,
    rotation_invariant: bool,
    energy_delta: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let image = image.as_array();
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                image_vec[z * dim[1] * dim[2] + y * dim[2] + x] = image[[z, y, x]];
            }
        }
    }
    let pad = Padding::parse(padding).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = py
        .detach(|| {
            laws_filter(
                &image_vec,
                dim,
                kernels,
                pad,
                rotation_invariant,
                energy_delta,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((dim[0], dim[1], dim[2]), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// IBSI-2 Gabor Q88H (Eq. 4.3). Returns the modulus. `image` is (z, y, x).
/// `spacing` is (z, y, x) millimetres. Rotation invariance averages |h| over
/// θ = 0, Δθ, … on [0, π). `orthogonal_planes` then averages axial/coronal/sagittal.
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma_mm,
    lambda_mm,
    gamma,
    spacing,
    theta=0.0,
    padding="zero",
    rotation_invariant=false,
    delta_theta=None,
    orthogonal_planes=false,
))]
fn apply_gabor<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    sigma_mm: f64,
    lambda_mm: f64,
    gamma: f64,
    spacing: (f64, f64, f64),
    theta: f64,
    padding: &str,
    rotation_invariant: bool,
    delta_theta: Option<f64>,
    orthogonal_planes: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let image = image.as_array();
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                image_vec[z * dim[1] * dim[2] + y * dim[2] + x] = image[[z, y, x]];
            }
        }
    }
    let pad = Padding::parse(padding).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = py
        .detach(|| {
            gabor_filter(
                &image_vec,
                dim,
                [spacing.0, spacing.1, spacing.2],
                sigma_mm,
                lambda_mm,
                gamma,
                theta,
                pad,
                rotation_invariant,
                delta_theta,
                orthogonal_planes,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((dim[0], dim[1], dim[2]), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// IBSI-2 undecimated separable wavelet. `combo` is e.g. `LHL` (g1 along k1=x).
/// Level > 1 filters the previous LLL approximation with à-trous kernels.
#[pyfunction]
#[pyo3(signature = (image, family, combo, level=1, padding="zero", rotation_invariant=false))]
fn apply_wavelet<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    family: &str,
    combo: &str,
    level: u32,
    padding: &str,
    rotation_invariant: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let image = image.as_array();
    let dim = [image.shape()[0], image.shape()[1], image.shape()[2]];
    let mut image_vec = vec![0.0; dim[0] * dim[1] * dim[2]];
    for z in 0..dim[0] {
        for y in 0..dim[1] {
            for x in 0..dim[2] {
                image_vec[z * dim[1] * dim[2] + y * dim[2] + x] = image[[z, y, x]];
            }
        }
    }
    let pad = Padding::parse(padding).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let out = py
        .detach(|| {
            wavelet_filter(
                &image_vec,
                dim,
                family,
                combo,
                level,
                pad,
                rotation_invariant,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((dim[0], dim[1], dim[2]), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// 26-connected foreground regions of a 3D mask (PyRadiomics `VolumeNum`).
#[pyfunction]
fn count_components_26(py: Python<'_>, mask: PyReadonlyArray3<bool>) -> usize {
    let mask = mask.as_array();
    let dim = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let flat: Vec<bool> = mask.iter().copied().collect();
    py.detach(|| radix_core::util::count_components_26(&flat, dim))
}

#[pymodule]
fn _radix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    radix_core::configure_blas();
    m.add("__lib_version__", VERSION)?;
    m.add("RadixError", m.py().get_type::<RadixError>())?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mode, m)?)?;
    m.add_function(wrap_pyfunction!(legal_image_types, m)?)?;
    m.add_function(wrap_pyfunction!(legal_image_types_all, m)?)?;
    m.add_function(wrap_pyfunction!(legal_feature_classes, m)?)?;
    m.add_function(wrap_pyfunction!(feature_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(resample_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(apply_image_type, m)?)?;
    m.add_function(wrap_pyfunction!(apply_mean, m)?)?;
    m.add_function(wrap_pyfunction!(apply_log, m)?)?;
    m.add_function(wrap_pyfunction!(apply_laws, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gabor, m)?)?;
    m.add_function(wrap_pyfunction!(apply_wavelet, m)?)?;
    m.add_function(wrap_pyfunction!(count_components_26, m)?)?;
    Ok(())
}
