import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/frozen_nt_external_validation/frozen_nt_external_predictions.tsv"
KMER_BUNDLE = ROOT / "results/shared_kmer_external_validation/shared_kmer_external_model.joblib"
OUTPUT = ROOT / "results/few_shot_adaptation_learning_curves"
OUTPUT.mkdir(parents=True, exist_ok=True)

SUPPORT_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.80]
TEST_FRACTION = 0.20
REPEATS = 25
ALPHA = 100.0  # Fixed before this analysis; no fraction-specific tuning.
SEED = 20261007


def safe_corr(y, p, method="pearson"):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(fn(y, p).statistic)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "n": len(y), "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5, "r2": r2_score(y, p),
        "pearson_r": safe_corr(y, p),
        "spearman_rho": safe_corr(y, p, "spearman"),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
    }


print("Loading external variants and development-defined vocabulary...")
d = pd.read_csv(INPUT, sep="\t", low_memory=False)
required = {"element_name", "validation_tier", "position_1based", "observed_score",
            "reference_sequence", "alternative_sequence", "predicted_shared_kmer_ridge"}
missing = required.difference(d.columns)
if missing:
    raise KeyError(f"Missing columns: {sorted(missing)}")
bundle = joblib.load(KMER_BUNDLE)
vectorizer = bundle["vectorizer"]
reference = vectorizer.transform(d.reference_sequence.astype(str).str.upper().tolist())
alternative = vectorizer.transform(d.alternative_sequence.astype(str).str.upper().tolist())
X = (alternative - reference).astype(np.float32).tocsr()
y = d.observed_score.to_numpy(float)
zero_shot = d.predicted_shared_kmer_ridge.to_numpy(float)

prediction_rows = []
split_rows = []
print("Running repeated few-shot position-held-out evaluation...")
for element_number, element in enumerate(sorted(d.element_name.unique()), start=1):
    element_rows = np.flatnonzero(d.element_name.eq(element).to_numpy())
    positions = d.position_1based.to_numpy()[element_rows]
    unique_positions = np.unique(positions)
    tier = d.validation_tier.iloc[element_rows[0]]
    print(f"  {element}: {len(element_rows)} variants, {len(unique_positions)} positions")
    for repeat in range(1, REPEATS + 1):
        rng = np.random.default_rng(SEED + element_number * 1000 + repeat)
        shuffled = rng.permutation(unique_positions)
        test_n = max(1, int(np.ceil(TEST_FRACTION * len(shuffled))))
        test_positions = set(shuffled[:test_n])
        support_pool = shuffled[test_n:]
        test_rows = element_rows[np.isin(positions, list(test_positions))]
        for fraction in SUPPORT_FRACTIONS:
            support_n = max(2, int(np.ceil(fraction * len(support_pool))))
            support_positions = set(support_pool[:support_n])
            support_rows = element_rows[np.isin(positions, list(support_positions))]
            if set(d.position_1based.iloc[support_rows]).intersection(
                    set(d.position_1based.iloc[test_rows])):
                raise RuntimeError("Position leakage detected")
            model = Ridge(alpha=ALPHA, solver="lsqr", max_iter=10000, tol=1e-5)
            model.fit(X[support_rows], y[support_rows])
            adapted = model.predict(X[test_rows])
            split_rows.append({
                "element_name": element, "validation_tier": tier, "repeat": repeat,
                "support_fraction_of_available_pool": fraction,
                "support_positions": len(support_positions),
                "support_variants": len(support_rows), "test_positions": len(test_positions),
                "test_variants": len(test_rows),
            })
            for local_i, row in enumerate(test_rows):
                prediction_rows.append({
                    "dataset_row": int(row), "element_name": element,
                    "validation_tier": tier, "repeat": repeat,
                    "support_fraction_of_available_pool": fraction,
                    "position_1based": d.position_1based.iloc[row],
                    "observed_score": y[row], "adapted_prediction": adapted[local_i],
                    "zero_shot_prediction": zero_shot[row],
                })

predictions = pd.DataFrame(prediction_rows)
metric_rows = []
for (fraction, repeat, tier), x in predictions.groupby(
        ["support_fraction_of_available_pool", "repeat", "validation_tier"]):
    for model, column in [("few_shot_adapted_kmer", "adapted_prediction"),
                          ("zero_shot_kmer", "zero_shot_prediction")]:
        metric_rows.append({
            "support_fraction_of_available_pool": fraction, "repeat": repeat,
            "validation_tier": tier, "element_name": "ALL", "model": model,
            **metrics(x.observed_score, x[column])})
for (fraction, repeat, tier, element), x in predictions.groupby(
        ["support_fraction_of_available_pool", "repeat", "validation_tier", "element_name"]):
    for model, column in [("few_shot_adapted_kmer", "adapted_prediction"),
                          ("zero_shot_kmer", "zero_shot_prediction")]:
        metric_rows.append({
            "support_fraction_of_available_pool": fraction, "repeat": repeat,
            "validation_tier": tier, "element_name": element, "model": model,
            **metrics(x.observed_score, x[column])})
repeat_metrics = pd.DataFrame(metric_rows)

summary_rows = []
for keys, x in repeat_metrics.groupby(
        ["support_fraction_of_available_pool", "validation_tier", "element_name", "model"]):
    fraction, tier, element, model = keys
    for measure in ["mae", "rmse", "r2", "pearson_r", "spearman_rho", "direction_accuracy"]:
        values = x[measure].dropna().to_numpy(float)
        summary_rows.append({
            "support_fraction_of_available_pool": fraction,
            "validation_tier": tier, "element_name": element, "model": model,
            "measure": measure, "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "lower_95_across_repeats": float(np.quantile(values, 0.025)),
            "upper_95_across_repeats": float(np.quantile(values, 0.975)),
        })
summary = pd.DataFrame(summary_rows)

# Paired repeat-level improvements use identical held-out positions.
wide = repeat_metrics.pivot(
    index=["support_fraction_of_available_pool", "repeat", "validation_tier", "element_name"],
    columns="model", values=["rmse", "mae", "r2", "pearson_r", "spearman_rho"])
gain_rows = []
for index, x in wide.groupby(level=[0, 2, 3]):
    fraction, tier, element = index
    definitions = {
        "rmse_reduction": x[("rmse", "zero_shot_kmer")] - x[("rmse", "few_shot_adapted_kmer")],
        "mae_reduction": x[("mae", "zero_shot_kmer")] - x[("mae", "few_shot_adapted_kmer")],
        "r2_gain": x[("r2", "few_shot_adapted_kmer")] - x[("r2", "zero_shot_kmer")],
        "pearson_gain": x[("pearson_r", "few_shot_adapted_kmer")] - x[("pearson_r", "zero_shot_kmer")],
        "spearman_gain": x[("spearman_rho", "few_shot_adapted_kmer")] - x[("spearman_rho", "zero_shot_kmer")],
    }
    for measure, series in definitions.items():
        values = series.dropna().to_numpy(float)
        gain_rows.append({
            "support_fraction_of_available_pool": fraction,
            "validation_tier": tier, "element_name": element, "measure": measure,
            "mean_improvement": float(np.mean(values)),
            "lower_95_across_repeats": float(np.quantile(values, 0.025)),
            "upper_95_across_repeats": float(np.quantile(values, 0.975)),
            "sign_consistency": float(np.mean(values > 0)),
        })
gains = pd.DataFrame(gain_rows)

pd.DataFrame(split_rows).to_csv(OUTPUT / "few_shot_split_registry.tsv", sep="\t", index=False)
predictions.to_csv(OUTPUT / "few_shot_test_predictions.tsv", sep="\t", index=False)
repeat_metrics.to_csv(OUTPUT / "few_shot_repeat_metrics.tsv", sep="\t", index=False)
summary.to_csv(OUTPUT / "few_shot_learning_curve_summary.tsv", sep="\t", index=False)
gains.to_csv(OUTPUT / "few_shot_improvement_over_zero_shot.tsv", sep="\t", index=False)
(OUTPUT / "few_shot_settings.json").write_text(json.dumps({
    "analysis": "repeated position-held-out few-shot assay adaptation",
    "support_fractions_of_non_test_pool": SUPPORT_FRACTIONS,
    "test_position_fraction": TEST_FRACTION, "repeats": REPEATS,
    "fixed_ridge_alpha": ALPHA, "seed": SEED,
    "support_sets_nested_within_each_repeat": True,
    "position_leakage": False,
    "claim_boundary": "Within-assay label-efficiency analysis, not independent external validation."
}, indent=2), encoding="utf-8")

print("\nPRIMARY HUMAN FEW-SHOT LEARNING CURVE")
show = summary.query(
    "validation_tier == 'primary_human_external' and element_name == 'ALL' and "
    "model == 'few_shot_adapted_kmer' and measure in ['r2','pearson_r','spearman_rho']")
print(show.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nStep 39 few-shot adaptation learning curves: COMPLETED")
print("Saved:", OUTPUT)
