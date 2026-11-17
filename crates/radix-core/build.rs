use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    let target = std::env::var("TARGET").unwrap_or_default();
    // pkg-config pulls -lgfortran for OpenBLAS. rust-lld does not search the
    // GCC library directory, where libgfortran.so actually lives.
    if target.contains("linux") {
        add_gfortran_search_path();
    }
}

fn add_gfortran_search_path() {
    let Ok(output) = Command::new("gfortran")
        .args(["-print-file-name=libgfortran.so"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let path = text.trim();
    if path.is_empty() || path == "libgfortran.so" {
        return;
    }
    if let Some(dir) = Path::new(path).parent() {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
}
