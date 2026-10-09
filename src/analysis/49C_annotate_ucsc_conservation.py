#!/usr/bin/env python3
"""Step 49C: add GRCh38 phyloP/phastCons conservation to Step 49B variants.

The script uses the official UCSC REST API and queries only the short assayed
element intervals, avoiding multi-gigabyte whole-genome BigWig downloads.
Coordinates sent to UCSC are 0-based half-open intervals.
"""

from __future__ import annotations
import os

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT_CANDIDATES = [
    ROOT / "results" / "human_encode_promoter_annotation" / "development_variant_encode_promoter_annotations.tsv",
    ROOT / "development_variant_encode_promoter_annotations.tsv",
]
OUT = ROOT / "results" / "human_conservation_annotation"
CACHE = OUT / "ucsc_api_cache"
API = "https://api.genome.ucsc.edu"
GENOME = "hg38"
TRACK_CANDIDATES = {
    "phyloP100way": ["phyloP100way", "phyloP100wayAll"],
    "phastCons100way": ["phastCons100way", "phastCons100wayAll"],
}


def locate_input() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Step 49B variant annotations were not found. Expected one of:\n  "
        + "\n  ".join(map(str, INPUT_CANDIDATES))
    )


session = requests.Session()
session.headers.update({"User-Agent": "CausalRegNet-HIERA-VEP/1.0 conservation annotation"})


def get_json(url: str, params: dict, attempts: int = 6) -> dict:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < attempts:
                delay = min(30, 2 ** (attempt - 1))
                print(f"  request attempt {attempt}/{attempts} failed: {exc}; retrying in {delay}s")
                time.sleep(delay)
    raise RuntimeError(f"UCSC request failed after {attempts} attempts: {last}")


def find_record_lists(obj) -> list[list[dict]]:
    found = []
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and {"start", "end"}.issubset(obj[0]):
            found.append(obj)
        else:
            for value in obj:
                found.extend(find_record_lists(value))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key not in {"downloadTime", "downloadTimeStamp", "dataTime", "itemsReturned"}:
                found.extend(find_record_lists(value))
    return found


def extract_value(record: dict) -> float | None:
    for key in ("value", "score", "dataValue"):
        if key in record:
            try:
                value = float(record[key])
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                pass
    return None


def query_interval(track: str, chrom: str, start: int, end: int) -> dict[int, float]:
    cache_file = CACHE / f"{track}__{chrom}__{start}__{end}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        payload = get_json(
            f"{API}/getData/track",
            {"genome": GENOME, "track": track, "chrom": chrom,
             "start": start, "end": end, "maxItemsOutput": 100000},
        )
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(0.20)

    values: dict[int, float] = {}
    for records in find_record_lists(payload):
        for record in records:
            value = extract_value(record)
            if value is None:
                continue
            rec_start, rec_end = int(record["start"]), int(record["end"])
            for position in range(max(start, rec_start), min(end, rec_end)):
                values[position] = value
    return values


def resolve_track(label: str, intervals: pd.DataFrame) -> str:
    probe = intervals.iloc[0]
    errors = []
    for track in TRACK_CANDIDATES[label]:
        try:
            query_interval(track, probe.chromosome, int(probe.start0), int(probe.end0))
            return track
        except Exception as exc:  # continue through explicitly listed UCSC aliases
            errors.append(f"{track}: {exc}")
    raise RuntimeError(f"No usable UCSC track for {label}. " + " | ".join(errors))


def annotate_track(data: pd.DataFrame, intervals: pd.DataFrame, label: str) -> tuple[pd.Series, str]:
    track = resolve_track(label, intervals)
    print(f"Using UCSC track for {label}: {track}")
    lookup: dict[tuple[str, int], float] = {}
    for i, row in enumerate(intervals.itertuples(index=False), 1):
        scores = query_interval(track, row.chromosome, int(row.start0), int(row.end0))
        lookup.update({(row.chromosome, pos): value for pos, value in scores.items()})
        print(f"  {label}: element interval {i}/{len(intervals)}")
    values = [lookup.get((chrom, int(pos) - 1), np.nan)
              for chrom, pos in zip(data.chromosome, data.variant_start_1based)]
    return pd.Series(values, index=data.index, dtype=float), track


def finite_summary(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    return {
        "nonmissing": int(valid.size),
        "available_fraction": float(valid.size / len(x)),
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "sd": float(valid.std(ddof=1)) if len(valid) > 1 else None,
        "minimum": float(valid.min()) if len(valid) else None,
        "maximum": float(valid.max()) if len(valid) else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    source = locate_input()
    print(f"Loading Step 49B annotations: {source}")
    data = pd.read_csv(source, sep="\t")
    required = {"source_row", "element_id", "chromosome", "variant_start_1based"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing required Step 49B columns: {sorted(missing)}")
    if data.source_row.duplicated().any():
        raise RuntimeError("source_row is not unique; conservation merge would be ambiguous")

    intervals = (data.groupby(["element_id", "chromosome"], as_index=False)
                 .variant_start_1based.agg(["min", "max"]).reset_index())
    intervals["start0"] = intervals["min"].astype(int) - 1
    intervals["end0"] = intervals["max"].astype(int)
    print(f"Variants: {len(data):,} | elements: {data.element_id.nunique()} | API intervals: {len(intervals)}")

    tracks = {}
    for label in TRACK_CANDIDATES:
        data[label], tracks[label] = annotate_track(data, intervals, label)

    # Interpretable bins; continuous scores remain the primary model inputs.
    data["phyloP_conserved"] = data.phyloP100way >= 2.0
    data["phyloP_accelerated"] = data.phyloP100way <= -2.0
    data["phastCons_high"] = data.phastCons100way >= 0.8

    element_summary = (data.groupby("element_id", as_index=False)
        .agg(variants=("source_row", "size"),
             phyloP_available_fraction=("phyloP100way", lambda x: x.notna().mean()),
             phyloP_median=("phyloP100way", "median"),
             phyloP_conserved_fraction=("phyloP_conserved", "mean"),
             phastCons_available_fraction=("phastCons100way", lambda x: x.notna().mean()),
             phastCons_median=("phastCons100way", "median"),
             phastCons_high_fraction=("phastCons_high", "mean")))

    summary = {
        "annotated_variants": int(len(data)),
        "elements": int(data.element_id.nunique()),
        "assembly": GENOME,
        "ucsc_tracks": tracks,
        "phyloP100way": finite_summary(data.phyloP100way),
        "phastCons100way": finite_summary(data.phastCons100way),
        "coordinate_conversion": "1-based SNV position queried at GRCh38 0-based base [position-1, position).",
        "interpretation": "Conservation is an external genomic annotation, not experimental validation of variant effect.",
    }

    data.to_csv(OUT / "development_variant_context_conservation_annotations.tsv", sep="\t", index=False)
    element_summary.to_csv(OUT / "element_conservation_summary.tsv", sep="\t", index=False)
    (OUT / "conservation_annotation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\nUCSC CONSERVATION ANNOTATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nELEMENT CONSERVATION SUMMARY")
    print(element_summary.to_string(index=False))
    print("\nStep 49C UCSC conservation annotation: COMPLETED")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
