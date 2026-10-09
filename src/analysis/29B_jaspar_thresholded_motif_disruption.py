"""Step 29B: thresholded JASPAR motif gain/loss analysis.

This corrects the max-over-all-motifs saturation problem in Step 29 by retaining
only motif placements with a strong reference or alternative affinity.
All resources used by this script are free. JASPAR is downloaded only when the
local motif file is absent.
"""

from __future__ import annotations
import os

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import motifs
from scipy.stats import mannwhitneyu, spearmanr


JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/2026/CORE/"
    "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
)
BASES = "ACGT"
BASE_TO_INT = {b: i for i, b in enumerate(BASES)}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2])))
    p.add_argument("--variants", type=Path, default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--affinity-threshold", type=float, default=0.80)
    p.add_argument("--top-effect-fraction", type=float, default=0.10)
    return p.parse_args()


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {label}. Checked:\n" + "\n".join(map(str, paths)))


def choose_column(df: pd.DataFrame, candidates: list[str], label: str, required=True):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(f"No {label} column found. Available columns: {list(df.columns)}")
    return None


def load_test_data(args: argparse.Namespace) -> pd.DataFrame:
    variants_path = args.variants or first_existing(
        [
            args.project / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_within_element_splits.tsv",
            args.project / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_consensus_within_element_splits.tsv",
            args.project / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_consensus_splits.tsv",
        ],
        "within-element variant table",
    )
    variants_df = pd.read_csv(variants_path, sep="\t")
    # This analysis uses the corrected within-element held-out test set.
    # Keep generic "split" only as a fallback for explicitly supplied tables.
    split_col = choose_column(
        variants_df,
        ["within_element_split", "split"],
        "within-element split",
    )
    variants_df = variants_df.loc[variants_df[split_col].astype(str).str.lower().eq("test")].copy()

    ref_col = choose_column(variants_df, ["reference_sequence", "ref_sequence", "sequence_ref"], "reference sequence")
    alt_col = choose_column(variants_df, ["alternative_sequence", "alternate_sequence", "alt_sequence", "sequence_alt"], "alternative sequence")
    score_col = choose_column(
        variants_df,
        ["consensus_score_z", "consensus_score", "normalized_score", "score"],
        "observed score",
    )
    id_col = choose_column(variants_df, ["accession", "variant_id", "id"], "variant identifier", required=False)
    element_col = choose_column(variants_df, ["element_group", "element_name"], "element", required=False)

    out = pd.DataFrame({
        "variant_id": variants_df[id_col].astype(str).to_numpy() if id_col else variants_df.index.astype(str),
        "element_group": variants_df[element_col].astype(str).to_numpy() if element_col else "unknown",
        "reference_sequence": variants_df[ref_col].astype(str).str.upper().to_numpy(),
        "alternative_sequence": variants_df[alt_col].astype(str).str.upper().to_numpy(),
        "observed_score": pd.to_numeric(variants_df[score_col], errors="coerce").to_numpy(),
    })

    prediction_candidates = [
        args.project / "results/within_element_kmer_baseline/within_element_kmer_test_predictions.tsv",
        args.project / "results/within_element_kmer_baseline/test_predictions.tsv",
        args.project / "results/within_element_model_comparison/kmer_test_predictions.tsv",
    ]
    prediction_path = args.predictions
    if prediction_path is None:
        prediction_path = next((p for p in prediction_candidates if p.exists()), None)
    out["predicted_score"] = np.nan
    if prediction_path is not None and prediction_path.exists():
        pred = pd.read_csv(prediction_path, sep="\t")
        pred_col = choose_column(pred, ["predicted_score", "prediction", "y_pred"], "prediction")
        pred_id = choose_column(pred, ["accession", "variant_id", "id"], "prediction identifier", required=False)
        if pred_id is not None:
            mapping = pred.drop_duplicates(pred_id).set_index(pred_id)[pred_col]
            out["predicted_score"] = pd.to_numeric(out["variant_id"].map(mapping), errors="coerce")
        elif len(pred) == len(out):
            out["predicted_score"] = pd.to_numeric(pred[pred_col], errors="coerce").to_numpy()
        else:
            raise ValueError("Prediction table lacks an ID column and its row count differs from the test set.")
    print("Variant table:", variants_path)
    print("Prediction table:", prediction_path if prediction_path else "not supplied; prediction tests skipped")
    return out


def mutation_indices(refs: list[str], alts: list[str]) -> np.ndarray:
    result = []
    for i, (ref, alt) in enumerate(zip(refs, alts)):
        if len(ref) != len(alt):
            raise ValueError(f"Variant row {i} has unequal sequence lengths.")
        changed = [j for j, (a, b) in enumerate(zip(ref, alt)) if a != b]
        if len(changed) != 1:
            raise ValueError(f"Variant row {i} contains {len(changed)} changed bases; expected one SNV.")
        result.append(changed[0])
    return np.asarray(result, dtype=np.int32)


def encode_sequences(sequences: list[str]) -> np.ndarray:
    lengths = {len(x) for x in sequences}
    if len(lengths) != 1:
        raise ValueError(f"Sequences do not all have equal length: {sorted(lengths)}")
    encoded = np.full((len(sequences), len(sequences[0])), -1, dtype=np.int8)
    for base, value in BASE_TO_INT.items():
        chars = np.asarray([list(x) for x in sequences])
        encoded[chars == base] = value
    if (encoded < 0).any():
        raise ValueError("Sequences contain characters other than A, C, G and T.")
    return encoded


def motif_log_odds(motif) -> np.ndarray:
    counts = np.vstack([np.asarray(motif.counts[b], dtype=float) for b in BASES]).T
    probabilities = (counts + 0.25) / (counts.sum(axis=1, keepdims=True) + 1.0)
    return np.log2(probabilities / 0.25)


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    return pwm[::-1][:, [3, 2, 1, 0]]


def safe_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(np.asarray(x)[mask]).size < 2 or np.unique(np.asarray(y)[mask]).size < 2:
        return math.nan, math.nan
    result = spearmanr(np.asarray(x)[mask], np.asarray(y)[mask])
    return float(result.statistic), float(result.pvalue)


def effect_test(effect, disruption, fraction):
    effect = np.abs(np.asarray(effect, dtype=float))
    disruption = np.asarray(disruption, dtype=float)
    mask = np.isfinite(effect) & np.isfinite(disruption)
    effect, disruption = effect[mask], disruption[mask]
    cutoff = float(np.quantile(effect, 1.0 - fraction))
    high = effect >= cutoff
    mw = mannwhitneyu(disruption[high], disruption[~high], alternative="greater")
    rho, rho_p = safe_spearman(effect, disruption)
    return {
        "n": int(len(effect)),
        "top_effect_cutoff": cutoff,
        "high_effect_n": int(high.sum()),
        "high_effect_median_disruption": float(np.median(disruption[high])),
        "other_median_disruption": float(np.median(disruption[~high])),
        "mann_whitney_u": float(mw.statistic),
        "mann_whitney_p_one_sided": float(mw.pvalue),
        "spearman_abs_effect_vs_disruption": rho,
        "spearman_p": rho_p,
    }


def main() -> None:
    args = arguments()
    if not 0 < args.affinity_threshold < 1:
        raise ValueError("--affinity-threshold must be between 0 and 1.")
    data = load_test_data(args)
    refs = data["reference_sequence"].tolist()
    alts = data["alternative_sequence"].tolist()
    centers = mutation_indices(refs, alts)
    ref_encoded = encode_sequences(refs)
    alt_encoded = encode_sequences(alts)

    jaspar_dir = args.project / "data/external/jaspar"
    jaspar_dir.mkdir(parents=True, exist_ok=True)
    jaspar_path = jaspar_dir / Path(JASPAR_URL).name
    if not jaspar_path.exists():
        print("Downloading free JASPAR 2026 CORE motifs...")
        urllib.request.urlretrieve(JASPAR_URL, jaspar_path)
    with jaspar_path.open() as handle:
        motif_list = list(motifs.parse(handle, "jaspar"))

    n = len(data)
    best_delta = np.zeros(n, dtype=np.float64)
    best_ref_affinity = np.zeros(n, dtype=np.float64)
    best_alt_affinity = np.zeros(n, dtype=np.float64)
    best_motif = np.full(n, "none", dtype=object)
    best_motif_id = np.full(n, "none", dtype=object)
    best_strand = np.full(n, ".", dtype=object)
    qualifying_count = np.zeros(n, dtype=np.int32)
    motif_rows = []

    print(f"Variants: {n}")
    print(f"Motifs: {len(motif_list)}")
    print(f"Affinity threshold: {args.affinity_threshold:.2f}")
    rows = np.arange(n)
    for motif_number, motif in enumerate(motif_list, start=1):
        pwm_plus = motif_log_odds(motif)
        length = pwm_plus.shape[0]
        min_score = float(pwm_plus.min(axis=1).sum())
        max_score = float(pwm_plus.max(axis=1).sum())
        score_range = max_score - min_score
        if score_range <= 0:
            continue
        motif_hit_variants = np.zeros(n, dtype=bool)
        motif_best_abs_delta = np.zeros(n, dtype=float)
        for strand, pwm in (("+", pwm_plus), ("-", reverse_complement_pwm(pwm_plus))):
            for offset in range(length):
                starts = centers - offset
                valid = (starts >= 0) & (starts + length <= ref_encoded.shape[1])
                if not valid.any():
                    continue
                idx = rows[valid]
                positions = starts[valid, None] + np.arange(length)[None, :]
                ref_window = ref_encoded[idx[:, None], positions]
                alt_window = alt_encoded[idx[:, None], positions]
                columns = np.arange(length)[None, :]
                ref_score = pwm[columns, ref_window].sum(axis=1)
                alt_score = pwm[columns, alt_window].sum(axis=1)
                ref_aff = np.clip((ref_score - min_score) / score_range, 0.0, 1.0)
                alt_aff = np.clip((alt_score - min_score) / score_range, 0.0, 1.0)
                qualifies = np.maximum(ref_aff, alt_aff) >= args.affinity_threshold
                if not qualifies.any():
                    continue
                q_idx = idx[qualifies]
                delta = alt_aff[qualifies] - ref_aff[qualifies]
                abs_delta = np.abs(delta)
                qualifying_count[q_idx] += 1
                motif_hit_variants[q_idx] = True
                motif_best_abs_delta[q_idx] = np.maximum(motif_best_abs_delta[q_idx], abs_delta)
                replace = abs_delta > np.abs(best_delta[q_idx])
                target = q_idx[replace]
                best_delta[target] = delta[replace]
                best_ref_affinity[target] = ref_aff[qualifies][replace]
                best_alt_affinity[target] = alt_aff[qualifies][replace]
                best_motif[target] = motif.name or motif.matrix_id
                best_motif_id[target] = motif.matrix_id
                best_strand[target] = strand
        motif_rows.append({
            "motif_id": motif.matrix_id,
            "motif_name": motif.name,
            "length": length,
            "variants_with_qualifying_site": int(motif_hit_variants.sum()),
            "maximum_abs_affinity_change": float(motif_best_abs_delta.max()),
            "mean_abs_affinity_change_among_hits": float(motif_best_abs_delta[motif_hit_variants].mean()) if motif_hit_variants.any() else 0.0,
        })
        if motif_number % 100 == 0:
            print(f"  Scanned {motif_number}/{len(motif_list)} motifs")

    data["qualifying_placements"] = qualifying_count
    data["top_motif_id"] = best_motif_id
    data["top_motif_name"] = best_motif
    data["top_motif_strand"] = best_strand
    data["reference_affinity"] = best_ref_affinity
    data["alternative_affinity"] = best_alt_affinity
    data["affinity_delta"] = best_delta
    data["absolute_affinity_change"] = np.abs(best_delta)
    data["motif_event"] = np.select(
        [
            (best_ref_affinity >= args.affinity_threshold) & (best_alt_affinity < args.affinity_threshold),
            (best_alt_affinity >= args.affinity_threshold) & (best_ref_affinity < args.affinity_threshold),
            qualifying_count > 0,
        ],
        ["loss", "gain", "affinity_change"],
        default="no_qualifying_site",
    )

    statistics = {
        "variants": n,
        "motifs": len(motif_list),
        "affinity_threshold": args.affinity_threshold,
        "variants_with_qualifying_site": int((qualifying_count > 0).sum()),
        "variants_without_qualifying_site": int((qualifying_count == 0).sum()),
        "event_counts": {str(k): int(v) for k, v in data["motif_event"].value_counts().items()},
        "observed_effect_test": effect_test(data["observed_score"], data["absolute_affinity_change"], args.top_effect_fraction),
        "jaspar_url": JASPAR_URL,
        "interpretation_note": "Thresholded PWM changes are computational predictions, not direct TF-binding or causal evidence.",
    }
    if data["predicted_score"].notna().sum() >= 3:
        statistics["predicted_effect_test"] = effect_test(
            data["predicted_score"], data["absolute_affinity_change"], args.top_effect_fraction
        )

    output_dir = args.project / "results/jaspar_motif_disruption_thresholded"
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_path = output_dir / "thresholded_variant_motif_disruption.tsv"
    motif_path = output_dir / "thresholded_motif_summary.tsv"
    stats_path = output_dir / "thresholded_motif_statistics.json"
    data.drop(columns=["reference_sequence", "alternative_sequence"]).to_csv(variant_path, sep="\t", index=False)
    pd.DataFrame(motif_rows).sort_values(
        ["variants_with_qualifying_site", "maximum_abs_affinity_change"], ascending=False
    ).to_csv(motif_path, sep="\t", index=False)
    stats_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    print("\nTHRESHOLDED MOTIF DISRUPTION STATISTICS")
    print(json.dumps(statistics, indent=2))
    print("\nStep 29B thresholded JASPAR motif disruption: COMPLETED")
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
