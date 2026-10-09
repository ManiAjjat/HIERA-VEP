import os
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
DEVELOPMENT_FILE = ROOT / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_within_element_splits.tsv"
EXTERNAL_FILE = ROOT / "results/external_validation_harmonized/external_model_ready_501bp.tsv"
OUTPUT = ROOT / "results/shared_kmer_external_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET = "consensus_score_z"
EXTERNAL_TARGET = "robust_z"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
NGRAM_RANGE = (3, 6)
N_BOOT = 2000
SEED = 20261006
RNG = np.random.default_rng(SEED)
MUTATION_CLASSES = [a + ">" + b for a in "ACGT" for b in "ACGT" if a != b]


def require_columns(data, columns, label):
    missing = set(columns).difference(data.columns)
    if missing:
        raise KeyError(f"{label} is missing required columns: {sorted(missing)}")


def clean_sequence(values):
    return values.fillna("").astype(str).str.upper().tolist()


def validate_sequences(data, label):
    bad_length = data.reference_sequence.str.len().ne(data.alternative_sequence.str.len())
    if bad_length.any():
        raise ValueError(f"{label}: {int(bad_length.sum())} reference/alternative length mismatches")
    idx = data.mutation_index.to_numpy(int)
    reference = data.reference_sequence.astype(str).str.upper().to_numpy()
    alternative = data.alternative_sequence.astype(str).str.upper().to_numpy()
    ref_allele = data.reference_allele.astype(str).str.upper().to_numpy()
    alt_allele = data.alternative_allele.astype(str).str.upper().to_numpy()
    failures = []
    for i in range(len(data)):
        if idx[i] < 0 or idx[i] >= len(reference[i]):
            failures.append(i)
        elif reference[i][idx[i]] != ref_allele[i] or alternative[i][idx[i]] != alt_allele[i]:
            failures.append(i)
    if failures:
        raise ValueError(f"{label}: mutation alignment failed for {len(failures)} rows; first indices={failures[:10]}")


def delta_kmers(vectorizer, data):
    reference = vectorizer.transform(clean_sequence(data.reference_sequence))
    alternative = vectorizer.transform(clean_sequence(data.alternative_sequence))
    return (alternative - reference).astype(np.float32).tocsr()


def mutation_matrix(data):
    labels = (data.reference_allele.astype(str).str.upper() + ">" +
              data.alternative_allele.astype(str).str.upper())
    lookup = {name: i for i, name in enumerate(MUTATION_CLASSES)}
    unknown = sorted(set(labels).difference(lookup))
    if unknown:
        raise ValueError(f"Unexpected mutation classes: {unknown}")
    rows = np.arange(len(data))
    cols = np.array([lookup[x] for x in labels], dtype=int)
    return sparse.csr_matrix((np.ones(len(data), dtype=np.float32), (rows, cols)),
                             shape=(len(data), len(MUTATION_CLASSES)))


def safe_corr(y, prediction, method):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    keep = np.isfinite(y) & np.isfinite(prediction)
    if keep.sum() < 3 or np.std(y[keep]) == 0 or np.std(prediction[keep]) == 0:
        return np.nan
    function = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(function(y[keep], prediction[keep]).statistic)


def calculate_metrics(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
        "pearson_r": safe_corr(y, prediction, "pearson"),
        "spearman_rho": safe_corr(y, prediction, "spearman"),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(prediction))),
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_sd": float(np.std(prediction, ddof=1)),
    }


def macro_element_rmse(data, y, prediction):
    values = []
    elements = data.element_group.astype(str).to_numpy()
    for element in sorted(np.unique(elements)):
        mask = elements == element
        values.append(mean_squared_error(y[mask], prediction[mask]) ** 0.5)
    return float(np.mean(values))


def add_point_estimates(rows, data, y, predictions, evaluation_group):
    for model, prediction in predictions.items():
        rows.append({"evaluation_group": evaluation_group, "element_name": "ALL",
                     "model": model, **calculate_metrics(y, prediction)})
        for element in sorted(data.element_name.unique()):
            mask = data.element_name.eq(element).to_numpy()
            rows.append({"evaluation_group": evaluation_group, "element_name": element,
                         "model": model, **calculate_metrics(y[mask], prediction[mask])})


def clustered_bootstrap(data, y, predictions, evaluation_group):
    # Alleles at the same assayed position remain together. Sampling is stratified
    # by element so large assays do not erase small elements from an iteration.
    cluster_rows = {}
    clusters_by_element = {}
    for key, group in data.groupby(["element_name", "position_1based"], sort=False):
        cluster_rows[key] = group.index.to_numpy()
        clusters_by_element.setdefault(key[0], []).append(key)
    output = []
    for iteration in range(1, N_BOOT + 1):
        parts = []
        for element, keys in clusters_by_element.items():
            selected = RNG.integers(0, len(keys), size=len(keys))
            parts.extend(cluster_rows[keys[i]] for i in selected)
        idx = np.concatenate(parts)
        for model, prediction in predictions.items():
            output.append({"evaluation_group": evaluation_group,
                           "iteration": iteration, "model": model,
                           **calculate_metrics(y[idx], prediction[idx])})
    return pd.DataFrame(output), len(cluster_rows)


print("Loading development and external datasets...")
development = pd.read_csv(DEVELOPMENT_FILE, sep="\t", low_memory=False)
external = pd.read_csv(EXTERNAL_FILE, sep="\t", low_memory=False)
require_columns(development,
                ["element_group", "within_element_split", TARGET, "reference_sequence",
                 "alternative_sequence", "reference_allele", "alternative_allele"],
                "Development dataset")
require_columns(external,
                ["validation_tier", "element_name", "position_1based", EXTERNAL_TARGET,
                 "reference_sequence", "alternative_sequence", "reference_allele",
                 "alternative_allele", "mutation_index"], "External dataset")

development = development.dropna(subset=[TARGET, "reference_sequence", "alternative_sequence"]).copy()
external = external.dropna(subset=[EXTERNAL_TARGET, "reference_sequence", "alternative_sequence"]).copy()
development["reference_sequence"] = development.reference_sequence.astype(str).str.upper()
development["alternative_sequence"] = development.alternative_sequence.astype(str).str.upper()
external["reference_sequence"] = external.reference_sequence.astype(str).str.upper()
external["alternative_sequence"] = external.alternative_sequence.astype(str).str.upper()

# Older development tables may not explicitly store the centered mutation index.
if "mutation_index" not in development.columns:
    development["mutation_index"] = development.reference_sequence.str.len() // 2
validate_sequences(development, "Development dataset")
validate_sequences(external, "External dataset")
print("Sequence and mutation alignment: PASSED")

split = development.within_element_split.astype(str).str.lower()
train_idx = np.flatnonzero(split.eq("train"))
validation_idx = np.flatnonzero(split.isin(["validation", "val"]))
test_idx = np.flatnonzero(split.eq("test"))
if min(len(train_idx), len(validation_idx), len(test_idx)) == 0:
    raise ValueError(f"Invalid development split counts: {split.value_counts().to_dict()}")

# Fit vocabulary only on development training sequences. External sequences and
# outcomes cannot influence which features exist.
print("Fitting development-only k-mer vocabulary...")
vectorizer = CountVectorizer(analyzer="char", ngram_range=NGRAM_RANGE,
                             lowercase=False, dtype=np.float32)
training_sequences = (clean_sequence(development.iloc[train_idx].reference_sequence) +
                      clean_sequence(development.iloc[train_idx].alternative_sequence))
vectorizer.fit(training_sequences)
X_development = delta_kmers(vectorizer, development)
X_external = delta_kmers(vectorizer, external)
M_development = mutation_matrix(development)
M_external = mutation_matrix(external)
y_development = development[TARGET].to_numpy(float)
y_external = external[EXTERNAL_TARGET].to_numpy(float)

print("Development variants:", len(development))
print("Training / validation / internal test:", len(train_idx), len(validation_idx), len(test_idx))
print("Development elements:", development.element_group.nunique())
print("External variants:", len(external))
print("K-mer features:", X_development.shape[1])

print("\nSelecting shared-model alpha on development validation data only...")
tuning_rows = []
for alpha in ALPHAS:
    model = Ridge(alpha=alpha, solver="lsqr", max_iter=10000, tol=1e-5)
    model.fit(X_development[train_idx], y_development[train_idx])
    prediction = model.predict(X_development[validation_idx])
    row = {"alpha": alpha,
           "macro_element_rmse": macro_element_rmse(
               development.iloc[validation_idx], y_development[validation_idx], prediction),
           **calculate_metrics(y_development[validation_idx], prediction)}
    tuning_rows.append(row)
    print(f"alpha={alpha:<8g} macro-RMSE={row['macro_element_rmse']:.4f} "
          f"Pearson={row['pearson_r']:.4f}")
tuning = pd.DataFrame(tuning_rows).sort_values(["macro_element_rmse", "alpha"])
best_alpha = float(tuning.iloc[0].alpha)
print("Selected alpha:", best_alpha)

# Lock both baselines after fitting development train + validation rows.
train_validation_idx = np.concatenate([train_idx, validation_idx])
shared_model = Ridge(alpha=best_alpha, solver="lsqr", max_iter=10000, tol=1e-5)
shared_model.fit(X_development[train_validation_idx], y_development[train_validation_idx])
mutation_model = Ridge(alpha=1.0, solver="lsqr", max_iter=10000, tol=1e-5)
mutation_model.fit(M_development[train_validation_idx], y_development[train_validation_idx])

# Internal test is retained as a diagnostic; it is not external validation.
internal_data = development.iloc[test_idx].copy().reset_index(drop=True)
internal_data["element_name"] = internal_data.element_group.astype(str)
internal_data["position_1based"] = internal_data.variant_start
internal_y = y_development[test_idx]
internal_predictions = {
    "shared_kmer_ridge": shared_model.predict(X_development[test_idx]),
    "mutation_class_baseline": mutation_model.predict(M_development[test_idx]),
    "zero_baseline": np.zeros(len(test_idx)),
}

external_predictions = {
    "shared_kmer_ridge": shared_model.predict(X_external),
    "mutation_class_baseline": mutation_model.predict(M_external),
    "zero_baseline": np.zeros(len(external)),
}

point_rows = []
add_point_estimates(point_rows, internal_data, internal_y, internal_predictions,
                    "development_internal_test")
for tier in ["primary_human_external", "cross_species_transfer", "secondary_splicing_transfer"]:
    mask = external.validation_tier.eq(tier).to_numpy()
    if not mask.any():
        continue
    tier_data = external.loc[mask].reset_index(drop=True)
    tier_y = y_external[mask]
    tier_predictions = {name: values[mask] for name, values in external_predictions.items()}
    add_point_estimates(point_rows, tier_data, tier_y, tier_predictions, tier)
point_estimates = pd.DataFrame(point_rows)

prediction_table = external.copy()
prediction_table["observed_score"] = y_external
for name, values in external_predictions.items():
    prediction_table[f"predicted_{name}"] = values
prediction_table["shared_kmer_residual"] = (
    prediction_table.observed_score - prediction_table.predicted_shared_kmer_ridge)

print(f"\nStarting {N_BOOT:,} position-clustered external bootstrap iterations...")
bootstrap_parts = []
cluster_counts = {}
for tier in ["primary_human_external", "cross_species_transfer", "secondary_splicing_transfer"]:
    mask = external.validation_tier.eq(tier).to_numpy()
    if not mask.any():
        continue
    tier_data = external.loc[mask].reset_index(drop=True)
    tier_y = y_external[mask]
    tier_predictions = {name: values[mask] for name, values in external_predictions.items()}
    boot, clusters = clustered_bootstrap(tier_data, tier_y, tier_predictions, tier)
    bootstrap_parts.append(boot)
    cluster_counts[tier] = clusters
    print(f"  {tier}: {len(tier_data)} variants, {clusters} position clusters completed")
bootstrap = pd.concat(bootstrap_parts, ignore_index=True)

ci_rows = []
metrics_to_summarize = ["mae", "rmse", "r2", "pearson_r", "spearman_rho",
                        "direction_accuracy"]
for (tier, model), group in bootstrap.groupby(["evaluation_group", "model"]):
    point = point_estimates.query(
        "evaluation_group == @tier and element_name == 'ALL' and model == @model").iloc[0]
    for metric in metrics_to_summarize:
        values = group[metric].dropna().to_numpy(float)
        ci_rows.append({"evaluation_group": tier, "model": model, "metric": metric,
                        "point_estimate": float(point[metric]),
                        "ci_lower_95": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                        "ci_upper_95": float(np.quantile(values, 0.975)) if len(values) else np.nan})
confidence_intervals = pd.DataFrame(ci_rows)

# Paired bootstrap comparisons quantify whether sequence features outperform
# the two prespecified baselines under identical resamples.
comparison_rows = []
for tier, group in bootstrap.groupby("evaluation_group"):
    pivot = group.pivot(index="iteration", columns="model",
                        values=["rmse", "mae", "pearson_r", "spearman_rho"])
    for baseline in ["mutation_class_baseline", "zero_baseline"]:
        definitions = {
            "rmse_reduction": pivot[("rmse", baseline)] - pivot[("rmse", "shared_kmer_ridge")],
            "mae_reduction": pivot[("mae", baseline)] - pivot[("mae", "shared_kmer_ridge")],
            "pearson_gain": pivot[("pearson_r", "shared_kmer_ridge")] - pivot[("pearson_r", baseline)],
            "spearman_gain": pivot[("spearman_rho", "shared_kmer_ridge")] - pivot[("spearman_rho", baseline)],
        }
        for measure, series in definitions.items():
            values = series.dropna().to_numpy(float)
            if len(values):
                mean_difference = float(np.mean(values))
                lower = float(np.quantile(values, 0.025))
                upper = float(np.quantile(values, 0.975))
                p_value = float(min(1.0, 2 * min(
                    np.mean(values <= 0), np.mean(values >= 0))))
            else:
                mean_difference = lower = upper = p_value = np.nan
            comparison_rows.append({
                "evaluation_group": tier, "model": "shared_kmer_ridge",
                "baseline": baseline, "measure": measure,
                "mean_difference": mean_difference,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "bootstrap_p_two_sided": p_value,
            })
paired_comparisons = pd.DataFrame(comparison_rows)

tuning.to_csv(OUTPUT / "shared_kmer_alpha_tuning.tsv", sep="\t", index=False)
point_estimates.to_csv(OUTPUT / "external_validation_point_estimates.tsv", sep="\t", index=False)
prediction_table.to_csv(OUTPUT / "external_variant_predictions.tsv", sep="\t", index=False)
bootstrap.to_csv(OUTPUT / "external_position_clustered_bootstrap.tsv", sep="\t", index=False)
confidence_intervals.to_csv(OUTPUT / "external_validation_confidence_intervals.tsv", sep="\t", index=False)
paired_comparisons.to_csv(OUTPUT / "external_paired_model_comparisons.tsv", sep="\t", index=False)
joblib.dump({"vectorizer": vectorizer, "shared_model": shared_model,
             "mutation_model": mutation_model, "selected_alpha": best_alpha,
             "target": TARGET, "external_target": EXTERNAL_TARGET,
             "feature_definition": "alternative minus reference character 3-6-mer counts"},
            OUTPUT / "shared_kmer_external_model.joblib", compress=3)

settings = {
    "analysis": "locked shared-model external regulatory variant validation",
    "development_file": str(DEVELOPMENT_FILE),
    "external_file": str(EXTERNAL_FILE),
    "development_variants": len(development),
    "development_elements": int(development.element_group.nunique()),
    "training_variants": len(train_idx),
    "validation_variants": len(validation_idx),
    "internal_test_variants": len(test_idx),
    "external_variants": len(external),
    "external_elements": int(external.element_name.nunique()),
    "selected_alpha": best_alpha,
    "alphas_tested": ALPHAS,
    "kmer_range": list(NGRAM_RANGE),
    "kmer_features": int(X_development.shape[1]),
    "bootstrap_iterations": N_BOOT,
    "bootstrap_seed": SEED,
    "external_position_clusters": cluster_counts,
    "external_labels_used_for_training_or_tuning": False,
    "external_vocabulary_leakage": False,
    "primary_claim_tier": "primary_human_external",
    "interpretation_warning": (
        "Cross-species and splicing tiers are transfer stress tests, not equivalent "
        "to independent human transcriptional validation. Association is predictive, not causal."
    ),
}
(OUTPUT / "external_validation_settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8")

print("\nEXTERNAL VALIDATION POINT ESTIMATES")
display = point_estimates.query(
    "evaluation_group != 'development_internal_test' and element_name == 'ALL'")
print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPAIRED EXTERNAL MODEL COMPARISONS")
print(paired_comparisons.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nStep 35 shared-model external validation: COMPLETED")
print("Saved:", OUTPUT)
