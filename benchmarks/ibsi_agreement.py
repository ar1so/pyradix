#!/usr/bin/env python3
"""Compare IBSI mode with the reference tables shipped in tests/references.

IBSI-2 has no feature-value CSV in this repository. Filter checks are the
analytical tests under tests/ibsi, plus the maximum absolute LoG impulse
error from the same kernel formula as tests/ibsi/test_log_filter.py.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from libbench import (
    RADIX_PYTHON,
    RESULTS,
    ROOT,
    ensure_dirs,
    published_tol,
    read_reference_csv,
    write_json,
)

EXAMPLES = ROOT / "examples"
DIGITAL_CSV = ROOT / "tests" / "references" / "ibsi1_digital_phantom.csv"
PAT1_CSV = ROOT / "tests" / "references" / "ibsi1_pat1_config_cde.csv"
FILTER_TESTS = [
    "tests/ibsi/test_mean_filter.py",
    "tests/ibsi/test_log_filter.py",
    "tests/ibsi/test_laws_filter.py",
    "tests/ibsi/test_gabor_filter.py",
    "tests/ibsi/test_wavelet_filter.py",
    "tests/ibsi/test_resample.py",
    "tests/ibsi/test_g4_filters.py",
]


def _all_features(params):
    from pyradix._radix import feature_catalog

    catalog = feature_catalog()
    params.feature_classes = {
        name: tuple(entry["features"]) for name, entry in catalog.items()
    }
    return params


def _score(rows: list[dict[str, str]], values: dict[str, float], key_field: str) -> dict:
    scored = []
    n_pass = n_fail = n_missing = 0
    worst = 0.0
    worst_key = ""
    for row in rows:
        key = row["radix_key"]
        expected = float(row["value"])
        tol = published_tol(row["value"], row.get("tol", ""))
        if key not in values:
            n_missing += 1
            scored.append({"feature": key, "pass": False, "reason": "missing", "expected": expected})
            continue
        got = float(values[key])
        err = abs(got - expected)
        ok = err <= max(tol, 1e-12)
        n_pass += int(ok)
        n_fail += int(not ok)
        if err >= worst:
            worst = err
            worst_key = key
        scored.append(
            {
                "feature": key,
                "ibsi_id": row.get("ibsi_id", ""),
                "ibsi_name": row.get("ibsi_name", ""),
                "family": key.split("_")[1] if key.count("_") >= 2 else "",
                "expected": expected,
                "got": got,
                "tol": tol,
                "abs_err": err,
                "pass": ok,
                "config": row.get("config", key_field),
            }
        )
    return {
        "n_rows": len(rows),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_missing": n_missing,
        "worst_abs_err": worst,
        "worst_feature": worst_key,
        "rows": scored,
    }


def _digital() -> dict:
    from pyradix import Params, RadixExtractor

    image = ROOT / "data" / "ibsi1" / "digital_phantom" / "image.nii.gz"
    mask = ROOT / "data" / "ibsi1" / "digital_phantom" / "mask.nii.gz"
    if not image.exists():
        return {"error": "digital phantom not fetched"}
    params = _all_features(Params(mode="ibsi", bin_width=1.0, additional_info=False))
    result = RadixExtractor(mode="ibsi", params=params).extract(image, mask, label=1)
    comments, rows = read_reference_csv(DIGITAL_CSV)
    by_gate = {}
    for gate in ("G1", "G2"):
        by_gate[gate] = _score([row for row in rows if row["gate"] == gate], result.values, gate)
    return {
        "dataset": "d1_digital_phantom",
        "settings": "mode=ibsi, bin_width=1, no resample, all catalog names including deprecated",
        "source_comments": comments,
        "config_hash": result.provenance.get("config_hash"),
        "n_features": len(result.values),
        "roi_voxels": result.provenance.get("roi_voxels"),
        "gates": {gate: {k: v for k, v in block.items() if k != "rows"} for gate, block in by_gate.items()},
        "rows": [row for block in by_gate.values() for row in block["rows"]],
    }


def _pat1() -> dict:
    from pyradix import Params, RadixExtractor

    image = ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "image.nii.gz"
    mask = ROOT / "data" / "ibsi1" / "lung_ct_pat1" / "mask.nii.gz"
    if not image.exists():
        return {"error": "PAT1 not fetched"}
    comments, rows = read_reference_csv(PAT1_CSV)
    configs = {}
    flat_rows = []
    yamls = {"C": "ibsi_pat1.yaml", "D": "ibsi_pat1_d.yaml", "E": "ibsi_pat1_e.yaml"}
    for config, name in yamls.items():
        params = _all_features(Params.from_yaml(EXAMPLES / name))
        params.additional_info = False
        result = RadixExtractor(mode="ibsi", params=params).extract(image, mask, label=1)
        block = _score([row for row in rows if row["config"] == config], result.values, config)
        configs[config] = {
            "yaml": name,
            "config_hash": result.provenance.get("config_hash"),
            "n_features": len(result.values),
            "roi_voxels": result.provenance.get("roi_voxels"),
            **{k: v for k, v in block.items() if k != "rows"},
        }
        flat_rows.extend(block["rows"])
        print(
            f"PAT1 {config}: {block['n_pass']}/{block['n_rows']} pass",
            file=sys.stderr,
        )
    return {
        "dataset": "d2_pat1",
        "source_comments": comments,
        "configs": configs,
        "rows": flat_rows,
    }


def _log_impulse_error() -> dict:
    path = ROOT / "tests" / "ibsi" / "test_log_filter.py"
    spec = importlib.util.spec_from_file_location("ibsi_log_filter_benchmark", path)
    if spec is None or spec.loader is None:
        return {"error": "could not load test_log_filter.py"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = ROOT / "data" / "ibsi2" / "digital_phantom" / "impulse.nii.gz"
    if not image.exists():
        return {"error": "IBSI-2 impulse phantom not fetched"}
    import numpy as np

    from pyradix import apply_log, load_volume

    vol = load_volume(image)
    img = np.ascontiguousarray(vol.array, dtype=np.float64)
    sigma_mm = 3.0
    sigma = module._sigma_vox(sigma_mm, vol.spacing)
    radius = module._support_radius(sigma)
    out = np.asarray(apply_log(img, sigma_mm, vol.spacing, padding="zero", ndims=3))
    z0, y0, x0 = module.IMPULSE_IJK
    worst = 0.0
    n = 0
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                expected = module.IMPULSE_I * module._log_weight(
                    np.array([dz, dy, dx], dtype=np.float64), sigma, 3
                )
                worst = max(worst, abs(float(out[z0 + dz, y0 + dy, x0 + dx]) - float(expected)))
                n += 1
    return {
        "case": "IBSI-2 L6PA impulse, sigma*=3 mm, zero padding, 3D",
        "n_kernel_voxels": n,
        "max_abs_err": worst,
        "support_radius_voxels": int(radius),
        "sigma_voxels": float(sigma),
        "formula_module": "tests/ibsi/test_log_filter.py",
    }


def _pytest_filters() -> dict:
    cmd = [str(RADIX_PYTHON), "-m", "pytest", "-q", "--tb=line", *FILTER_TESTS]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "summary": tail[-5:] if tail else [],
    }


def main() -> int:
    ensure_dirs()
    print("IBSI-1 digital phantom", file=sys.stderr)
    digital = _digital()
    if "gates" in digital:
        for gate, block in digital["gates"].items():
            print(f"  {gate}: {block['n_pass']}/{block['n_rows']}", file=sys.stderr)
    print("IBSI-1 PAT1 C/D/E", file=sys.stderr)
    pat1 = _pat1()
    print("IBSI-2 LoG impulse", file=sys.stderr)
    log_err = _log_impulse_error()
    print(log_err, file=sys.stderr)
    print("IBSI-2 pytest filter modules", file=sys.stderr)
    filters = _pytest_filters()
    print(filters["summary"], file=sys.stderr)
    payload = {
        "digital_phantom": {k: v for k, v in digital.items() if k != "rows"},
        "digital_phantom_rows": digital.get("rows", []),
        "pat1": {k: v for k, v in pat1.items() if k != "rows"},
        "pat1_rows": pat1.get("rows", []),
        "ibsi2_log_impulse": log_err,
        "ibsi2_pytest": filters,
    }
    write_json(RESULTS / "ibsi.json", payload)
    print(f"wrote {RESULTS / 'ibsi.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
