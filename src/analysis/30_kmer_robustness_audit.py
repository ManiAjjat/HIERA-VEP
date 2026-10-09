import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/within_element_kmer_baseline/within_element_test_predictions.tsv"
OUTPUT = ROOT / "results/kmer_robustness_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260928)
N_BOOT = 2000


def safe_corr(a, b, method="pearson"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(a[keep], b[keep]).statistic)
    return float(stats.spearmanr(a[keep], b[keep]).statistic)


def metrics(d):
    y = d["observed_score"].to_numpy(float)
    p = d["predicted_score"].to_numpy(float)
    return {
        "n": len(d),
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "r2": r2_score(y, p),
        "pearson_r": safe_corr(y, p),
        "spearman_rho": safe_corr(y, p, "spearman"),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_sd": float(np.std(p, ddof=1)),
    }


def calibration(d):
    y = d["observed_score"].to_numpy(float)
    p = d["predicted_score"].to_numpy(float)
    slope, intercept = np.polyfit(p, y, 1)
    residual = y - p
    return {
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "residual_sd": float(np.std(residual, ddof=1)),
        "absolute_error_vs_absolute_observed_spearman": safe_corr(
            np.abs(y), np.abs(residual), "spearman"
        ),
    }


d = pd.read_csv(INPUT, sep="\t")
required = {"element_group", "observed_score", "predicted_score"}
missing = required.difference(d.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")
d = d.dropna(subset=list(required)).copy()
d["residual"] = d["observed_score"] - d["predicted_score"]
d["absolute_observed_effect"] = d["observed_score"].abs()
d["absolute_error"] = d["residual"].abs()

# Overall and sensitivity analyses after removing the most extreme observations.
sensitivity_rows = []
for label, fraction_removed in [("all_variants", 0.00), ("exclude_top_1pct_abs_effect", 0.01),
                                ("exclude_top_5pct_abs_effect", 0.05)]:
    if fraction_removed == 0:
        subset = d
    else:
        cutoff = d["absolute_observed_effect"].quantile(1 - fraction_removed)
        subset = d[d["absolute_observed_effect"] <= cutoff]
    sensitivity_rows.append({"analysis": label, **metrics(subset), **calibration(subset)})
sensitivity = pd.DataFrame(sensitivity_rows)

# Performance by observed-effect quartile; bins are defined without using predictions.
d["observed_effect_quartile"] = pd.qcut(
    d["absolute_observed_effect"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop"
)
effect_rows = []
for group, x in d.groupby("observed_effect_quartile", observed=False):
    effect_rows.append({"effect_bin": str(group),
                        "min_abs_effect": x.absolute_observed_effect.min(),
                        "max_abs_effect": x.absolute_observed_effect.max(),
                        **metrics(x)})
effect_table = pd.DataFrame(effect_rows)

# Element heterogeneity and leave-one-element-out pooled sensitivity.
element_table = pd.DataFrame([
    {"element_group": name, **metrics(x), **calibration(x)}
    for name, x in d.groupby("element_group", sort=True)
])
loo_table = pd.DataFrame([
    {"excluded_element": name, **metrics(d[d.element_group != name])}
    for name in sorted(d.element_group.unique())
])

# Cluster bootstrap by genomic position within element when coordinates exist;
# otherwise use dataset_row as an explicitly reported fallback.
if "variant_start" in d.columns:
    d["cluster_id"] = d["element_group"].astype(str) + ":" + d["variant_start"].astype(str)
    cluster_definition = "element_group + variant_start"
elif "dataset_row" in d.columns:
    d["cluster_id"] = d["dataset_row"].astype(str)
    cluster_definition = "dataset_row fallback"
    warnings.warn("variant_start absent; bootstrap uses dataset_row rather than genomic positions.")
else:
    raise KeyError("Need variant_start or dataset_row for bootstrap clustering.")

groups = {k: x.index.to_numpy() for k, x in d.groupby("cluster_id")}
cluster_ids = np.array(list(groups), dtype=object)
boot = []
for i in range(N_BOOT):
    sampled = RNG.choice(cluster_ids, len(cluster_ids), replace=True)
    idx = np.concatenate([groups[k] for k in sampled])
    x = d.loc[idx]
    m = metrics(x)
    c = calibration(x)
    boot.append({"iteration": i + 1, **m, **c})
bootstrap = pd.DataFrame(boot)

ci_rows = []
for col in [c for c in bootstrap.columns if c not in {"iteration", "n"}]:
    values = bootstrap[col].dropna().to_numpy(float)
    ci_rows.append({"measure": col, "point_estimate": {**metrics(d), **calibration(d)}[col],
                    "ci_lower_95": np.quantile(values, 0.025),
                    "ci_upper_95": np.quantile(values, 0.975)})
ci = pd.DataFrame(ci_rows)

sensitivity.to_csv(OUTPUT / "robustness_sensitivity.tsv", sep="\t", index=False)
effect_table.to_csv(OUTPUT / "performance_by_effect_quartile.tsv", sep="\t", index=False)
element_table.to_csv(OUTPUT / "element_heterogeneity.tsv", sep="\t", index=False)
loo_table.to_csv(OUTPUT / "leave_one_element_out_sensitivity.tsv", sep="\t", index=False)
bootstrap.to_csv(OUTPUT / "robustness_cluster_bootstrap_iterations.tsv", sep="\t", index=False)
ci.to_csv(OUTPUT / "robustness_confidence_intervals.tsv", sep="\t", index=False)

settings = {"input": str(INPUT), "variants": len(d), "elements": d.element_group.nunique(),
            "clusters": len(cluster_ids), "bootstrap_iterations": N_BOOT,
            "cluster_definition": cluster_definition, "random_seed": 20260928}
(OUTPUT / "robustness_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

print("\nOVERALL ROBUSTNESS AND EXTREME-VARIANT SENSITIVITY")
print(sensitivity.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nPERFORMANCE BY OBSERVED-EFFECT QUARTILE")
print(effect_table.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nPOSITION-CLUSTERED 95% CONFIDENCE INTERVALS")
print(ci.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nStep 30 k-mer robustness audit: COMPLETED")
print("Saved:", OUTPUT)
