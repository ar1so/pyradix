"""Command line: ``pyradix extract IMAGE MASK``.

Writes one row of features per case. JSON goes to stdout unless ``--out``
names a ``.json`` or ``.csv`` file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from pyradix._radix import engine_info
from pyradix._version import __version__
from pyradix.extractor import RadixExtractor
from pyradix.params import Params


def _load_params(path: Path, mode: str | None) -> Params:
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    is_pyradiomics = isinstance(raw, dict) and ("setting" in raw or "imageType" in raw)
    if is_pyradiomics:
        if path.suffix.lower() == ".json":
            return Params.from_pyradiomics_json(path)
        return Params.from_pyradiomics_yaml(path)
    return Params.from_yaml(path, default_mode=mode or "ibsi")


def _extract(args: argparse.Namespace) -> int:
    if args.params:
        params = _load_params(Path(args.params), args.mode)
        if args.mode and args.mode != params.mode:
            print(
                f"error: --mode {args.mode} disagrees with the parameter file ({params.mode})",
                file=sys.stderr,
            )
            return 2
        extractor = RadixExtractor(mode=params.mode, params=params)
    else:
        extractor = RadixExtractor(mode=args.mode or "ibsi")

    result = extractor.extract(args.image, args.mask, label=args.label)
    row: dict[str, Any] = {"image": args.image, "mask": args.mask}
    if args.provenance:
        row.update({f"provenance_{k}": v for k, v in result.provenance.items()})
    row.update(result.values)

    out = Path(args.out) if args.out else None
    if out is None or out.suffix.lower() == ".json":
        text = json.dumps(row, indent=2, default=str)
        if out is None:
            print(text)
        else:
            out.write_text(text + "\n", encoding="utf-8")
    elif out.suffix.lower() == ".csv":
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    else:
        print("error: --out must end in .json or .csv", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyradix", description="Radiomics feature extraction (IBSI or PyRadiomics 3.1.0)."
    )
    parser.add_argument("--version", action="version",
                        version=f"pyradix {__version__} ({engine_info()})")
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="extract features from one image and mask")
    ex.add_argument("image", help="image file (NIfTI or NRRD)")
    ex.add_argument("mask", help="mask file (NIfTI or NRRD)")
    ex.add_argument("--mode", choices=["ibsi", "pyradiomics"],
                    help="standard to follow (default ibsi, or the parameter file's)")
    ex.add_argument("--params", help="pyradix or PyRadiomics parameter file (YAML or JSON)")
    ex.add_argument("--label", type=int, help="mask label to use (default from the settings, 1)")
    ex.add_argument("--out", help="write to a .json or .csv file instead of stdout")
    ex.add_argument("--provenance", action="store_true",
                    help="also write the settings and configuration hash")
    ex.set_defaults(func=_extract)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # report library errors as one clean line
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
