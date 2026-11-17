//! Real parts of eigenvalues of a small dense matrix (GLCM MCC).
//! PyRadiomics uses `numpy.linalg.eigvals` on unsymmetric Q — do not symmetrize.
//!
//! On macOS and Linux the default path is LAPACK `dgeev` (Accelerate / OpenBLAS),
//! matching NumPy. faer is the fallback if `dgeev` does not converge.
//!
//! On Windows the build uses faer only. MSVC OpenBLAS from vcpkg does not
//! provide the gfortran `dgeev_` symbol the `lapack` crate expects, so a
//! Windows OpenBLAS link fails. MCC on Windows is still computed; hard cases
//! may differ from NumPy/LAPACK by a near-zero eigenvalue choice.

#[cfg(not(target_os = "windows"))]
extern crate lapack_src as _;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

use faer::complex_native::c64;
use faer::Mat;

use crate::error::RadixError;

/// Serializes the process-global panic hook used by the faer path.
static FAER_EIG: Mutex<()> = Mutex::new(());

/// Real parts of all eigenvalues of row-major `a` (n×n).
pub fn eigvals_real(a: &[f64], n: usize) -> Result<Vec<f64>, RadixError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if a.len() < n * n {
        return Err(mcc_error("the Q matrix is the wrong size"));
    }
    #[cfg(not(target_os = "windows"))]
    {
        // NumPy's MCC eigenvalues come from LAPACK dgeev. faer runs only when
        // dgeev does not return a finite spectrum.
        if let Ok(vals) = lapack_eigs(a, n) {
            if vals.len() == n && vals.iter().all(|value| value.is_finite()) {
                return Ok(vals);
            }
        }
    }
    faer_eigs_caught(a, n)
}

fn mcc_error(detail: &str) -> RadixError {
    RadixError::feature("glcm", "MCC", detail)
}

fn faer_eigs_caught(a: &[f64], n: usize) -> Result<Vec<f64>, RadixError> {
    let _guard = FAER_EIG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = catch_unwind(AssertUnwindSafe(|| faer_eigs(a, n)));
    std::panic::set_hook(hook);
    drop(_guard);
    if let Ok(vals) = caught {
        if vals.len() == n && vals.iter().all(|v| v.is_finite()) {
            return Ok(vals);
        }
    }
    Err(mcc_error("eigenvalues did not converge"))
}

fn faer_eigs(a: &[f64], n: usize) -> Vec<f64> {
    let m = Mat::<f64>::from_fn(n, n, |i, j| a[i * n + j]);
    m.eigenvalues::<c64>().into_iter().map(|z| z.re).collect()
}

/// LAPACK `dgeev`. Eigenvalues of `A` and `Aᵀ` match, so the row-major buffer
/// can be passed straight through the column-major routine.
#[cfg(not(target_os = "windows"))]
fn lapack_eigs(a: &[f64], n: usize) -> Result<Vec<f64>, RadixError> {
    if n > i32::MAX as usize {
        return Err(mcc_error("the Q matrix is too large"));
    }
    let n_i = n as i32;
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    let mut vl = [0.0];
    let mut vr = [0.0];
    let mut info = 0i32;
    let mut query = [0.0];
    let mut scratch = a.to_vec();
    unsafe {
        lapack::dgeev(
            b'N',
            b'N',
            n_i,
            &mut scratch,
            n_i,
            &mut wr,
            &mut wi,
            &mut vl,
            1,
            &mut vr,
            1,
            &mut query,
            -1,
            &mut info,
        );
    }
    if info != 0 || !query[0].is_finite() || query[0] < 1.0 {
        return Err(mcc_error("eigenvalues did not converge"));
    }
    let lwork = query[0] as i32;
    if lwork < 1 {
        return Err(mcc_error("eigenvalues did not converge"));
    }
    let mut work = vec![0.0; lwork as usize];
    scratch = a.to_vec();
    info = 0;
    unsafe {
        lapack::dgeev(
            b'N',
            b'N',
            n_i,
            &mut scratch,
            n_i,
            &mut wr,
            &mut wi,
            &mut vl,
            1,
            &mut vr,
            1,
            &mut work,
            lwork,
            &mut info,
        );
    }
    if info != 0 || wr.iter().any(|value| !value.is_finite()) {
        return Err(mcc_error("eigenvalues did not converge"));
    }
    Ok(wr)
}

#[cfg(test)]
mod tests {
    use super::eigvals_real;

    #[test]
    fn eigvals_2x2_nonsym() {
        let mut e = eigvals_real(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        e.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let s = 33f64.sqrt();
        assert!((e[0] - 0.5 * (5.0 + s)).abs() < 1e-8, "{e:?}");
        assert!((e[1] - 0.5 * (5.0 - s)).abs() < 1e-8, "{e:?}");
    }

    #[test]
    fn eigvals_3x3_nonsym() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut e = eigvals_real(&a, 3).unwrap();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e[0] + 1.116843969807043).abs() < 1e-6, "{e:?}");
        assert!(e[1].abs() < 1e-6, "{e:?}");
        assert!((e[2] - 16.116843969807043).abs() < 1e-6, "{e:?}");
    }

    #[test]
    fn eigvals_wide_range_returns() {
        let n = 40;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let scale = if i % 5 == 0 { 1e6 } else { 1e-4 };
                a[i * n + j] = scale * (((i * 13 + j * 7) % 9) as f64 - 4.0);
            }
            a[i * n + i] += 1.0 + i as f64;
        }
        let e = eigvals_real(&a, n).unwrap();
        assert_eq!(e.len(), n);
        assert!(e.iter().all(|v| v.is_finite()), "{e:?}");
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn lapack_matches_numpy_on_a_nonsymmetric_matrix() {
        // numpy.linalg.eigvals real parts, sorted, for this row-major matrix.
        let a = [
            0.2, 0.5, 0.1, 0.0, 1.4, 0.3, 0.2, 0.7, 0.0, 0.4, 2.0, 0.1, 0.6, 0.0, 0.8, 0.5,
        ];
        let mut e = super::lapack_eigs(&a, 4).unwrap();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let numpy = [
            -0.3779851992641026,
            0.1468195513769735,
            1.017521168916335,
            2.213644478970789,
        ];
        for (got, expected) in e.iter().zip(numpy) {
            assert!((got - expected).abs() < 1e-8, "got {e:?}");
        }
    }
}
