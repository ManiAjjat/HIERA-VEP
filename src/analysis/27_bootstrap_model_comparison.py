import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "results" / "within_element_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "K-mer Ridge": ROOT / "results/within_element_kmer_baseline/within_element_test_predictions.tsv",
    "Frozen NT": ROOT / "results/nucleotide_transformer/within_element_frozen_nt/within_element_frozen_nt_test_predictions.tsv",
    "Fine-tuned NT": ROOT / "results/nucleotide_transformer/within_element_supervised_finetuning/within_element_finetuned_test_predictions.tsv",
}
N_BOOT = 2000
SEED = 42

def safe_corr(fn, y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(fn(y, p).statistic)

def calculate(y, p):
    residual = y - p
    denom = np.sum((y - y.mean()) ** 2)
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.sum(residual ** 2) / denom) if denom > 0 else np.nan,
        "pearson_r": safe_corr(pearsonr, y, p),
        "spearman_rho": safe_corr(spearmanr, y, p),
    }

print("Loading predictions...")
tables = {}
for model, path in FILES.items():
    d = pd.read_csv(path, sep="\t")
    required = {"dataset_row", "element_group", "variant_start", "observed_score", "predicted_score"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{model} is missing columns: {sorted(missing)}")
    if d.dataset_row.duplicated().any():
        raise ValueError(f"Duplicate dataset_row values in {model}")
    tables[model] = d.sort_values("dataset_row").reset_index(drop=True)

reference = tables["K-mer Ridge"]
for model, d in tables.items():
    if not np.array_equal(reference.dataset_row.to_numpy(), d.dataset_row.to_numpy()):
        raise ValueError(f"Test rows do not align for {model}")
    for column in ["element_group", "variant_start"]:
        if not reference[column].astype(str).equals(d[column].astype(str)):
            raise ValueError(f"{column} does not align for {model}")
    if not np.allclose(reference.observed_score, d.observed_score, atol=1e-6):
        raise ValueError(f"Observed scores do not align for {model}")

print("Prediction alignment: PASSED")
print("Test variants:", len(reference))
print("Elements:", reference.element_group.nunique())
print("Genomic positions:", reference.groupby(["element_group", "variant_start"]).ngroups)

y = reference.observed_score.to_numpy(float)
predictions = {model: d.predicted_score.to_numpy(float) for model, d in tables.items()}
predictions["Zero baseline"] = np.zeros(len(reference), dtype=float)

# Point estimates overall and by regulatory element.
metric_rows = []
for model, pred in predictions.items():
    metric_rows.append({"element_group": "ALL", "model": model, **calculate(y, pred)})
    for element in sorted(reference.element_group.unique()):
        mask = reference.element_group.eq(element).to_numpy()
        metric_rows.append({"element_group": element, "model": model, **calculate(y[mask], pred[mask])})
metrics = pd.DataFrame(metric_rows)

# Cluster bootstrap: sample genomic positions, keeping all alleles from each position together.
cluster_rows = {}
for key, group in reference.groupby(["element_group", "variant_start"], sort=False):
    cluster_rows[key] = group.index.to_numpy()
clusters_by_element = {}
for key in cluster_rows:
    clusters_by_element.setdefault(key[0], []).append(key)

rng = np.random.default_rng(SEED)
boot_rows = []
print(f"Starting {N_BOOT:,} position-clustered bootstrap iterations...")
for iteration in range(1, N_BOOT + 1):
    sampled_parts = []
    for element, keys in clusters_by_element.items():
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        sampled_parts.extend(cluster_rows[keys[i]] for i in chosen)
    idx = np.concatenate(sampled_parts)
    for model, pred in predictions.items():
        boot_rows.append({"iteration": iteration, "model": model, **calculate(y[idx], pred[idx])})
    if iteration % 250 == 0:
        print(f"  Completed {iteration}/{N_BOOT}")

boot = pd.DataFrame(boot_rows)
# Save completed iterations before calculating summaries so they survive any
# later reporting problem and can be reused for inspection.
boot.to_csv(OUT / "position_clustered_bootstrap_iterations.tsv", sep="\t", index=False)
summary_rows = []
for model, d in boot.groupby("model"):
    for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_rho"]:
        values = d[metric].dropna().to_numpy()
        if len(values) == 0:
            lower = np.nan
            upper = np.nan
        else:
            lower = np.quantile(values, 0.025)
            upper = np.quantile(values, 0.975)
        summary_rows.append({
            "model": model,
            "metric": metric,
            "point_estimate": metrics.query("element_group == 'ALL' and model == @model")[metric].iloc[0],
            "ci_lower_95": lower,
            "ci_upper_95": upper,
        })
summary = pd.DataFrame(summary_rows)

# Paired comparisons against the leading k-mer model.
pivot = boot.pivot(index="iteration", columns="model", values=["rmse", "mae", "r2", "pearson_r"])
comparison_rows = []
for competitor in ["Frozen NT", "Fine-tuned NT", "Zero baseline"]:
    definitions = {
        "rmse_reduction": pivot[("rmse", competitor)] - pivot[("rmse", "K-mer Ridge")],
        "mae_reduction": pivot[("mae", competitor)] - pivot[("mae", "K-mer Ridge")],
        "r2_gain": pivot[("r2", "K-mer Ridge")] - pivot[("r2", competitor)],
        "pearson_gain": pivot[("pearson_r", "K-mer Ridge")] - pivot[("pearson_r", competitor)],
    }
    for measure, values in definitions.items():
        values = values.dropna().to_numpy()
        if len(values) == 0:
            mean_difference = np.nan
            lower = np.nan
            upper = np.nan
            p_two_sided = np.nan
        else:
            mean_difference = values.mean()
            lower = np.quantile(values, 0.025)
            upper = np.quantile(values, 0.975)
            p_two_sided = min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))
        comparison_rows.append({
            "reference_model": "K-mer Ridge", "competitor": competitor, "measure": measure,
            "mean_difference": mean_difference, "ci_lower_95": lower,
            "ci_upper_95": upper, "bootstrap_p_two_sided": p_two_sided,
        })
comparisons = pd.DataFrame(comparison_rows)

metrics.to_csv(OUT / "model_point_estimates.tsv", sep="\t", index=False)
summary.to_csv(OUT / "bootstrap_confidence_intervals.tsv", sep="\t", index=False)
comparisons.to_csv(OUT / "paired_model_comparisons.tsv", sep="\t", index=False)
(OUT / "bootstrap_settings.json").write_text(json.dumps({
    "iterations": N_BOOT, "seed": SEED, "resampling_unit": "genomic position within element",
    "test_variants": len(reference), "elements": reference.element_group.nunique(),
    "genomic_positions": len(cluster_rows), "test_labels_used_for_model_selection": False,
}, indent=2), encoding="utf-8")

print("\nOVERALL POINT ESTIMATES")
print(metrics.query("element_group == 'ALL'").to_string(index=False))
print("\nPAIRED COMPARISONS AGAINST K-MER RIDGE")
print(comparisons.to_string(index=False))
print("\nStep 27 bootstrap comparison: COMPLETED")
print("Saved:", OUT)
