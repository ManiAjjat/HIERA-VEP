import os
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import requests
from Bio import motifs
from scipy import stats


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/condition_differential_regulatory_model/paired_sul1_condition_effects.tsv"
OUTPUT = ROOT / "results/fungal_motif_context_interpretation"
OUTPUT.mkdir(parents=True, exist_ok=True)
JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/2026/CORE/"
    "JASPAR2026_CORE_fungi_non-redundant_pfms_jaspar.txt"
)
LOCAL_JASPAR = OUTPUT / "JASPAR2026_CORE_fungi_nonredundant.jaspar"
AFFINITY_THRESHOLD = 0.80
TOP_EFFECT_FRACTION = 0.10
PSEUDOCOUNT = 0.5


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values without statsmodels."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return adjusted
    valid = p[finite]
    order = np.argsort(valid)
    ranked = valid[order]
    correction = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    correction = np.minimum.accumulate(correction[::-1])[::-1]
    correction = np.clip(correction, 0.0, 1.0)
    restored = np.empty_like(correction)
    restored[order] = correction
    adjusted[finite] = restored
    return adjusted


def download(url, path):
    if path.exists() and path.stat().st_size > 1000:
        return
    error = None
    for attempt in range(1, 6):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if len(response.content) < 1000:
                raise RuntimeError("Downloaded motif file is unexpectedly small")
            path.write_bytes(response.content)
            return
        except Exception as exc:
            error = exc
            print(f"Download attempt {attempt}/5 failed: {exc}")
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Could not download fungal JASPAR motifs: {error}")


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def prepare_motif(record):
    matrix = np.vstack([np.asarray(record.counts[base], float) for base in "ACGT"])
    probability = (matrix + PSEUDOCOUNT) / (matrix.sum(axis=0) + 4 * PSEUDOCOUNT)
    log_odds = np.log2(probability / 0.25)
    minimum = float(log_odds.min(axis=0).sum())
    maximum = float(log_odds.max(axis=0).sum())
    return {
        "motif_id": str(record.matrix_id or record.name),
        "motif_name": str(record.name or record.matrix_id),
        "length": matrix.shape[1], "log_odds": log_odds,
        "minimum": minimum, "maximum": maximum,
    }


BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def affinity(sequence, motif):
    if any(base not in BASE_INDEX for base in sequence):
        return np.nan
    indices = np.array([BASE_INDEX[x] for x in sequence], dtype=int)
    score = float(motif["log_odds"][indices, np.arange(motif["length"])].sum())
    denominator = motif["maximum"] - motif["minimum"]
    return (score - motif["minimum"]) / denominator if denominator > 0 else np.nan


def best_overlapping_change(reference, alternative, center, motif):
    length = motif["length"]
    best = None
    for start in range(center - length + 1, center + 1):
        if start < 0 or start + length > len(reference):
            continue
        ref_window = reference[start:start + length]
        alt_window = alternative[start:start + length]
        for strand, ref_seq, alt_seq in [
            ("+", ref_window, alt_window),
            ("-", reverse_complement(ref_window), reverse_complement(alt_window)),
        ]:
            ref_affinity = affinity(ref_seq, motif)
            alt_affinity = affinity(alt_seq, motif)
            if not np.isfinite(ref_affinity) or not np.isfinite(alt_affinity):
                continue
            if max(ref_affinity, alt_affinity) < AFFINITY_THRESHOLD:
                continue
            delta = alt_affinity - ref_affinity
            candidate = (abs(delta), delta, ref_affinity, alt_affinity, start, strand)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


print("Loading paired SUL1 condition effects...")
d = pd.read_csv(INPUT, sep="\t", low_memory=False)
required = {"position_1based", "reference_allele", "alternative_allele",
            "reference_sequence", "alternative_sequence", "condition_difference"}
missing = required.difference(d.columns)
if missing:
    raise KeyError(f"Missing columns: {sorted(missing)}")
d["reference_sequence"] = d.reference_sequence.astype(str).str.upper()
d["alternative_sequence"] = d.alternative_sequence.astype(str).str.upper()
d["mutation_index"] = d.reference_sequence.str.len() // 2
for row in d.itertuples():
    if (row.reference_sequence[row.mutation_index] != row.reference_allele or
            row.alternative_sequence[row.mutation_index] != row.alternative_allele):
        raise ValueError(f"Mutation alignment failed at table row {row.Index}")

print("Obtaining JASPAR fungal CORE motifs...")
download(JASPAR_URL, LOCAL_JASPAR)
with LOCAL_JASPAR.open(encoding="utf-8") as handle:
    motif_records = list(motifs.parse(handle, "jaspar"))
prepared = [prepare_motif(record) for record in motif_records]
print("Fungal motifs:", len(prepared))

event_rows = []
summary_rows = []
print("Scanning motif changes overlapping each tested nucleotide...")
for motif_number, motif in enumerate(prepared, start=1):
    qualifying = 0
    for row in d.itertuples():
        change = best_overlapping_change(
            row.reference_sequence, row.alternative_sequence,
            row.mutation_index, motif)
        if change is None:
            continue
        absolute_delta, delta, ref_affinity, alt_affinity, start, strand = change
        qualifying += 1
        if ref_affinity < AFFINITY_THRESHOLD <= alt_affinity:
            event = "gain"
        elif alt_affinity < AFFINITY_THRESHOLD <= ref_affinity:
            event = "loss"
        else:
            event = "affinity_change"
        event_rows.append({
            "variant_row": row.Index, "position_1based": row.position_1based,
            "reference_allele": row.reference_allele,
            "alternative_allele": row.alternative_allele,
            "condition_difference": row.condition_difference,
            "absolute_condition_difference": abs(row.condition_difference),
            "motif_id": motif["motif_id"], "motif_name": motif["motif_name"],
            "motif_length": motif["length"], "strand": strand,
            "window_start": start, "reference_affinity": ref_affinity,
            "alternative_affinity": alt_affinity, "affinity_delta": delta,
            "absolute_affinity_delta": absolute_delta, "event": event,
        })
    summary_rows.append({"motif_id": motif["motif_id"],
                         "motif_name": motif["motif_name"],
                         "motif_length": motif["length"],
                         "qualifying_variants": qualifying})
    if motif_number % 50 == 0 or motif_number == len(prepared):
        print(f"  Scanned {motif_number}/{len(prepared)} motifs")

events = pd.DataFrame(event_rows)
motif_summary = pd.DataFrame(summary_rows)
if events.empty:
    raise RuntimeError("No qualifying motif-overlap events were found")

# One strongest motif event per variant for global association testing.
best = (events.sort_values("absolute_affinity_delta", ascending=False)
        .drop_duplicates("variant_row"))
analysis = d.copy()
analysis["variant_row"] = analysis.index
analysis = analysis.merge(
    best[["variant_row", "motif_id", "motif_name", "event", "affinity_delta",
          "absolute_affinity_delta"]], on="variant_row", how="left")
analysis["absolute_affinity_delta"] = analysis.absolute_affinity_delta.fillna(0.0)
cutoff = analysis.absolute_condition_difference.quantile(1 - TOP_EFFECT_FRACTION)
analysis["top_condition_effect"] = analysis.absolute_condition_difference.ge(cutoff)

rho = stats.spearmanr(
    analysis.absolute_condition_difference, analysis.absolute_affinity_delta)
high = analysis.loc[analysis.top_condition_effect, "absolute_affinity_delta"]
other = analysis.loc[~analysis.top_condition_effect, "absolute_affinity_delta"]
mw = stats.mannwhitneyu(high, other, alternative="greater")

# Motif-specific enrichment among the top 10% condition-dependent variants.
top_rows = set(analysis.loc[analysis.top_condition_effect, "variant_row"])
all_rows = set(analysis.variant_row)
enrichment_rows = []
for (motif_id, motif_name), x in events.groupby(["motif_id", "motif_name"]):
    affected = set(x.variant_row)
    a = len(affected & top_rows)
    b = len(top_rows - affected)
    c = len((affected & all_rows) - top_rows)
    e = len((all_rows - top_rows) - affected)
    odds, p_value = stats.fisher_exact([[a, b], [c, e]], alternative="greater")
    enrichment_rows.append({
        "motif_id": motif_id, "motif_name": motif_name,
        "top_effect_with_motif": a, "top_effect_without_motif": b,
        "other_with_motif": c, "other_without_motif": e,
        "odds_ratio": odds, "fisher_p": p_value,
    })
enrichment = pd.DataFrame(enrichment_rows)
enrichment["fdr_bh"] = benjamini_hochberg(enrichment.fisher_p)
enrichment = enrichment.sort_values(["fdr_bh", "fisher_p"])

events.to_csv(OUTPUT / "sul1_fungal_motif_events.tsv", sep="\t", index=False)
analysis.to_csv(OUTPUT / "variant_best_fungal_motif_event.tsv", sep="\t", index=False)
motif_summary.to_csv(OUTPUT / "fungal_motif_scan_summary.tsv", sep="\t", index=False)
enrichment.to_csv(OUTPUT / "condition_effect_motif_enrichment.tsv", sep="\t", index=False)

statistics = {
    "variants": len(d), "fungal_motifs": len(prepared),
    "affinity_threshold": AFFINITY_THRESHOLD,
    "variants_with_qualifying_motif": int(analysis.motif_id.notna().sum()),
    "top_effect_fraction": TOP_EFFECT_FRACTION,
    "top_effect_cutoff": float(cutoff),
    "spearman_abs_condition_effect_vs_abs_motif_disruption": float(rho.statistic),
    "spearman_p": float(rho.pvalue),
    "top_effect_median_abs_motif_disruption": float(high.median()),
    "other_median_abs_motif_disruption": float(other.median()),
    "mann_whitney_u": float(mw.statistic),
    "mann_whitney_p_one_sided": float(mw.pvalue),
    "fdr_significant_enriched_motifs": int(enrichment.fdr_bh.lt(0.05).sum()),
    "jaspar_url": JASPAR_URL,
    "warning": "PWM matches are computational annotations, not direct TF-binding evidence."
}
(OUTPUT / "fungal_motif_context_statistics.json").write_text(
    json.dumps(statistics, indent=2), encoding="utf-8")

print("\nFUNGAL MOTIF CONTEXT STATISTICS")
print(json.dumps(statistics, indent=2))
print("\nTOP MOTIF ENRICHMENTS")
print(enrichment.head(20).to_string(index=False, float_format=lambda z: f"{z:.4g}"))
print("\nStep 41 fungal motif context interpretation: COMPLETED")
print("Saved:", OUTPUT)
