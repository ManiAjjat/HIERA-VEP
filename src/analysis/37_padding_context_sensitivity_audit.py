import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/frozen_nt_external_validation/frozen_nt_external_predictions.tsv"
OUTPUT = ROOT / "results/external_context_sensitivity_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "frozen_nt_ridge": "predicted_frozen_nt_ridge",
    "shared_kmer_ridge": "predicted_shared_kmer_ridge",
    "mutation_class_baseline": "predicted_mutation_class_baseline",
}
BIN_ORDER = ["0%", ">0-10%", ">10-25%", ">25-50%", ">50%"]


def safe_corr(y, p, method="pearson"):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(fn(y, p).statistic)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "n": len(y),
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "r2": r2_score(y, p),
        "pearson_r": safe_corr(y, p),
        "spearman_rho": safe_corr(y, p, "spearman"),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_sd": float(np.std(p, ddof=1)),
    }


def padding_bin(x):
    if x == 0:
        return "0%"
    if x <= 0.10:
        return ">0-10%"
    if x <= 0.25:
        return ">10-25%"
    if x <= 0.50:
        return ">25-50%"
    return ">50%"


print("Loading Step 36 external predictions...")
d = pd.read_csv(INPUT, sep="\t", low_memory=False)
required = {"validation_tier", "element_name", "position_1based", "reference_sequence",
            "alternative_sequence", "observed_score", *MODELS.values()}
missing = required.difference(d.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

d["sequence_length"] = d.reference_sequence.astype(str).str.len()
d["ambiguous_bases"] = d.reference_sequence.astype(str).str.count(r"[^ACGTacgt]")
d["ambiguous_fraction"] = d.ambiguous_bases / d.sequence_length
d["padding_bin"] = pd.Categorical(
    d.ambiguous_fraction.map(padding_bin), categories=BIN_ORDER, ordered=True)

composition = (d.groupby(["validation_tier", "element_name", "padding_bin"],
                         observed=True)
                 .agg(variants=("observed_score", "size"),
                      positions=("position_1based", "nunique"),
                      mean_ambiguous_fraction=("ambiguous_fraction", "mean"),
                      median_ambiguous_fraction=("ambiguous_fraction", "median"),
                      maximum_ambiguous_fraction=("ambiguous_fraction", "max"))
                 .reset_index())

rows = []
groupings = [
    ("overall_bin", ["padding_bin"]),
    ("tier_bin", ["validation_tier", "padding_bin"]),
    ("element_bin", ["validation_tier", "element_name", "padding_bin"]),
    ("element_all", ["validation_tier", "element_name"]),
]
for analysis, columns in groupings:
    for keys, x in d.groupby(columns, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(columns, keys))
        for model, prediction_column in MODELS.items():
            rows.append({"analysis": analysis, **labels, "model": model,
                         **metrics(x.observed_score, x[prediction_column])})
performance = pd.DataFrame(rows)

# Does missing context track prediction error? Spearman is used because padding
# and absolute errors are non-normal and strongly tied within assay targets.
error_rows = []
for (tier, element), x in d.groupby(["validation_tier", "element_name"]):
    for model, prediction_column in MODELS.items():
        absolute_error = np.abs(x.observed_score.to_numpy(float) -
                                x[prediction_column].to_numpy(float))
        error_rows.append({
            "validation_tier": tier,
            "element_name": element,
            "model": model,
            "n": len(x),
            "spearman_padding_vs_absolute_error": safe_corr(
                x.ambiguous_fraction, absolute_error, "spearman"),
        })
error_association = pd.DataFrame(error_rows)

# Direct paired error comparison between the two sequence representations.
paired_rows = []
for (tier, element, bin_name), x in d.groupby(
        ["validation_tier", "element_name", "padding_bin"], observed=True):
    nt_error = np.abs(x.observed_score - x.predicted_frozen_nt_ridge)
    kmer_error = np.abs(x.observed_score - x.predicted_shared_kmer_ridge)
    difference = nt_error - kmer_error
    if len(difference) >= 10 and np.any(difference != 0):
        test = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    else:
        statistic = p_value = np.nan
    paired_rows.append({
        "validation_tier": tier, "element_name": element,
        "padding_bin": str(bin_name), "n": len(x),
        "mean_absolute_error_nt_minus_kmer": float(difference.mean()),
        "median_absolute_error_nt_minus_kmer": float(difference.median()),
        "wilcoxon_statistic": statistic, "wilcoxon_p": p_value,
    })
paired = pd.DataFrame(paired_rows)

d.to_csv(OUTPUT / "external_variants_with_padding_metrics.tsv", sep="\t", index=False)
composition.to_csv(OUTPUT / "padding_composition.tsv", sep="\t", index=False)
performance.to_csv(OUTPUT / "performance_by_padding_context.tsv", sep="\t", index=False)
error_association.to_csv(OUTPUT / "padding_error_associations.tsv", sep="\t", index=False)
paired.to_csv(OUTPUT / "paired_nt_vs_kmer_by_padding.tsv", sep="\t", index=False)

zero_context = d.ambiguous_fraction.eq(0)
low_context = d.ambiguous_fraction.le(0.10)
settings = {
    "analysis": "external sequence-context and padding sensitivity audit",
    "variants": len(d),
    "elements": int(d.element_name.nunique()),
    "total_sequence_bases": int(d.sequence_length.sum()),
    "ambiguous_bases": int(d.ambiguous_bases.sum()),
    "overall_ambiguous_fraction": float(d.ambiguous_bases.sum() / d.sequence_length.sum()),
    "zero_padding_variants": int(zero_context.sum()),
    "at_most_10pct_padding_variants": int(low_context.sum()),
    "interpretation_rule": (
        "This is a prespecified diagnostic sensitivity analysis, not a replacement "
        "for the complete external evaluation. Subgroups must not be presented as "
        "the primary result."
    ),
}
(OUTPUT / "context_sensitivity_settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8")

print("\nPADDING COMPOSITION")
print(composition.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nPERFORMANCE BY PADDING BIN")
show = performance.query("analysis == 'tier_bin'")
print(show.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nPADDING VERSUS ABSOLUTE ERROR")
print(error_association.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nStep 37 padding/context sensitivity audit: COMPLETED")
print("Saved:", OUTPUT)
