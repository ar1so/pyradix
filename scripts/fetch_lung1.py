#!/usr/bin/env python3
"""Fetch Aerts 2014 Lung1 (TCIA NSCLC-Radiomics): clinical CSV + CT/SEG → NIfTI.

Study: docs/datasets/LUNG1.md
  Aerts et al. Nat Commun 2014. https://doi.org/10.1038/ncomms5006
  TCIA NSCLC-Radiomics v4. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI

License: CC BY-NC 3.0 (non-commercial).

    .venv/bin/python scripts/fetch_lung1.py --clinical
    .venv/bin/python scripts/fetch_lung1.py --limit 10
    .venv/bin/python scripts/fetch_lung1.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "lung1"
CLINICAL_URL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
)
SKIP_NO_GTV = frozenset({"LUNG1-128"})
MISSING_SLICES = frozenset({"LUNG1-014", "LUNG1-021", "LUNG1-085"})


def _download_clinical(dest: Path) -> Path:
    from http_get import download

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"clinical → {dest}", file=sys.stderr)
    download(CLINICAL_URL, dest)
    return dest


def _nbia():
    try:
        from tcia_utils import nbia
    except ImportError as exc:
        raise SystemExit(
            "pip install tcia_utils pandas  # needed to list/download TCIA series"
        ) from exc
    return nbia


def _inventory(path: Path):
    nbia = _nbia()
    print("listing NSCLC-Radiomics series…", file=sys.stderr)
    df = nbia.getSeries(collection="NSCLC-Radiomics", format="df")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _want_patients(df, limit: int | None) -> list[str]:
    ids = sorted({str(p) for p in df["PatientID"].tolist() if str(p) not in SKIP_NO_GTV})
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def _series_for_patient(df, patient: str) -> list[str]:
    sub = df[df["PatientID"] == patient]
    uids = []
    for modality in ("CT", "SEG"):
        rows = sub[sub["Modality"] == modality]
        if rows.empty:
            print(f"  skip {patient}: no {modality}", file=sys.stderr)
            return []
        # one series per modality; if several, take the first
        uids.append(str(rows.iloc[0]["SeriesInstanceUID"]))
    return uids


def _convert_patient(dicom_root: Path, out_dir: Path) -> bool:
    """CT series + SEG → image.nii.gz / mask.nii.gz. Returns True on success."""
    try:
        import SimpleITK as sitk
    except ImportError:
        print("  SimpleITK missing; leave DICOM in place (pip install SimpleITK)", file=sys.stderr)
        return False

    ct_dir = None
    seg_file = None
    for child in dicom_root.rglob("*"):
        if child.is_dir():
            dcms = list(child.glob("*.dcm"))
            if len(dcms) > 5:
                ct_dir = child
        elif child.suffix.lower() == ".dcm" and child.stat().st_size > 0:
            # SEG is usually a single larger object; detect later
            pass
    # SEG: a folder with 1 dcm, or any .dcm whose modality we check
    import pydicom

    for dcm in dicom_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") == "SEG":
            seg_file = dcm
        elif getattr(ds, "Modality", "") == "CT" and ct_dir is None:
            ct_dir = dcm.parent
    if ct_dir is None or seg_file is None:
        print(f"  convert fail {dicom_root}: ct={ct_dir} seg={seg_file}", file=sys.stderr)
        return False

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(ct_dir))
    reader.SetFileNames(names)
    image = reader.Execute()
    source_uids = []
    for path in names:
        source_uids.append(str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID))

    try:
        import highdicom as hd
        import numpy as np
    except ImportError:
        print("  highdicom missing; leave DICOM (pip install highdicom)", file=sys.stderr)
        return False
    seg = hd.seg.segread(str(seg_file))
    chosen = None
    for num in seg.get_segment_numbers():
        desc = str(seg.get_segment_description(num).SegmentDescription)
        if "GTV" in desc.upper():
            chosen = num
            break
    if chosen is None:
        chosen = seg.get_segment_numbers()[0]
    try:
        mask_arr = np.asarray(
            seg.get_pixels_by_source_instance(
                source_sop_instance_uids=source_uids,
                segment_numbers=[chosen],
                combine_segments=True,
                assert_missing_frames_are_empty=True,
                ignore_spatial_locations=True,
            )
        )
    except Exception as exc:
        print(f"  highdicom pixels: {exc}", file=sys.stderr)
        return False
    while mask_arr.ndim > 3:
        mask_arr = mask_arr[..., 0]
    mask_arr = (mask_arr > 0).astype("uint8")
    if mask_arr.shape[0] != int(image.GetSize()[2]):
        print(f"  mask shape {mask_arr.shape} vs CT {image.GetSize()}", file=sys.stderr)
        return False
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(image)

    out_dir.mkdir(parents=True, exist_ok=True)
    from image_paths import image_io_path

    sitk.WriteImage(image, image_io_path(out_dir / "image.nii.gz"), True)
    sitk.WriteImage(mask, image_io_path(out_dir / "mask.nii.gz"), True)
    return True


def _download_series(uids: list[str], dest: Path, timeout: int = 360) -> bool:
    """Child process so a stalled TCIA socket cannot freeze the whole cohort."""
    script = (
        "from tcia_utils import nbia\n"
        f"nbia.downloadSeries({uids!r}, path={str(dest)!r}, input_type='list', max_workers=2)\n"
    )
    env = os.environ.copy()
    for attempt in (1, 2):
        try:
            subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                timeout=timeout,
                check=True,
            )
            return True
        except subprocess.TimeoutExpired:
            print(f"  download timeout ({timeout}s) attempt {attempt}", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            print(f"  download error attempt {attempt}: {exc}", file=sys.stderr)
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
    return False


def _write_pairs(dest: Path) -> None:
    clinical = dest / "Lung1.clinical.csv"
    rows = []
    for patient_dir in sorted(dest.iterdir()):
        if not patient_dir.is_dir():
            continue
        image = patient_dir / "image.nii.gz"
        mask = patient_dir / "mask.nii.gz"
        if image.is_file() and mask.is_file():
            rows.append(
                {
                    "patient_id": patient_dir.name,
                    "image": str(image.relative_to(ROOT)),
                    "mask": str(mask.relative_to(ROOT)),
                    "missing_slices": patient_dir.name in MISSING_SLICES,
                }
            )
    out = dest / "pairs.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patient_id", "image", "mask", "missing_slices"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} pairs)", file=sys.stderr)
    if clinical.is_file():
        print(f"clinical {clinical} ({clinical.stat().st_size} bytes)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", action="store_true", help="download the CSV only")
    parser.add_argument("--limit", type=int, default=None, help="first N patients (after skipping LUNG1-128)")
    parser.add_argument("--keep-dicom", action="store_true", help="do not delete DICOM after NIfTI convert")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    clinical = DEST / "Lung1.clinical.csv"
    if not clinical.is_file() or args.clinical:
        _download_clinical(clinical)
    if args.clinical:
        return 0

    inv = DEST / "series_inventory.csv"
    df = _inventory(inv)
    ct = int((df["Modality"] == "CT").sum())
    seg = int((df["Modality"] == "SEG").sum())
    print(f"inventory: {df['PatientID'].nunique()} patients, {ct} CT, {seg} SEG → {inv}", file=sys.stderr)

    patients = _want_patients(df, args.limit)
    print(f"download {len(patients)} patients (CT+SEG only)", file=sys.stderr)
    for i, patient in enumerate(patients, 1):
        out_dir = DEST / patient
        if (out_dir / "image.nii.gz").is_file() and (out_dir / "mask.nii.gz").is_file():
            print(f"[{i}/{len(patients)}] {patient} already converted", file=sys.stderr)
            continue
        uids = _series_for_patient(df, patient)
        if not uids:
            continue
        dicom_root = DEST / "_dicom" / patient
        if dicom_root.exists():
            shutil.rmtree(dicom_root, ignore_errors=True)
        dicom_root.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(patients)}] {patient} download…", file=sys.stderr)
        ok_dl = _download_series(uids, dicom_root)
        if not ok_dl:
            print(f"[{i}/{len(patients)}] {patient} download failed", file=sys.stderr)
            shutil.rmtree(dicom_root, ignore_errors=True)
            continue
        ok = _convert_patient(dicom_root, out_dir)
        if ok and not args.keep_dicom:
            shutil.rmtree(dicom_root, ignore_errors=True)
        note = " missing-slices" if patient in MISSING_SLICES else ""
        print(f"[{i}/{len(patients)}] {patient} {'ok' if ok else 'dicom-only'}{note}", file=sys.stderr)
        if ok:
            _write_pairs(DEST)

    _write_pairs(DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
