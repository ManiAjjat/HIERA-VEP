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
OUTPUT = ROOT / "results/condition_differential_regulatory_model"
OUTPUT.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT = 2000
SEED = 20261007
RNG = np.random.default_rng(SEED)


def safe_corr(y, p, method="pearson"):
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
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_sd": float(np.std(p, ddof=1)),
    }


def group_folds(groups, n_folds, rng):
    unique = rng.permutation(np.unique(groups))
    assignment = {group: i % n_folds for i, group in enumerate(unique)}
    return np.array([assignment[x] for x in groups])


print("Loading paired SUL1 conditions...")
d = pd.read_csv(INPUT, sep="\t", low_memory=False)
keys = ["position_1based", "reference_allele", "alternative_allele"]
columns = keys + ["reference_sequence", "alternative_sequence", "observed_score"]
sulfate = d.loc[d.element_name.eq("SUL1_sulfate"), columns].rename(
    columns={"observed_score": "sulfate_score"})
glucose = d.loc[d.element_name.eq("SUL1_glucose"), keys + ["observed_score"]].rename(
    columns={"observed_score": "glucose_score"})
paired = sulfate.merge(glucose, on=keys, how="inner", validate="one_to_one")
if len(paired) < 100:
    raise ValueError(f"Too few paired variants: {len(paired)}")
paired["condition_difference"] = paired.sulfate_score - paired.glucose_score
paired["absolute_condition_difference"] = paired.condition_difference.abs()
paired["condition_sign_switch"] = (
    np.sign(paired.sulfate_score) != np.sign(paired.glucose_score))

agreement = {
    "paired_variants": len(paired),
    "paired_positions": int(paired.position_1based.nunique()),
    "pearson_sulfate_vs_glucose": safe_corr(paired.sulfate_score, paired.glucose_score),
    "spearman_sulfate_vs_glucose": safe_corr(
        paired.sulfate_score, paired.glucose_score, "spearman"),
    "sign_concordance": float(np.mean(
        np.sign(paired.sulfate_score) == np.sign(paired.glucose_score))),
    "sign_switch_fraction": float(paired.condition_sign_switch.mean()),
    "condition_difference_sd": float(paired.condition_difference.std(ddof=1)),
}

print("Building development-defined delta-k-mer features...")
bundle = joblib.load(KMER_BUNDLE)
vectorizer = bundle["vectorizer"]
ref = vectorizer.transform(paired.reference_sequence.astype(str).str.upper().tolist())
alt = vectorizer.transform(paired.alternative_sequence.astype(str).str.upper().tolist())
X = (alt - ref).astype(np.float32).tocsr()
y = paired.condition_difference.to_numpy(float)
groups = paired.position_1based.to_numpy()

outer = group_folds(groups, OUTER_FOLDS, np.random.default_rng(SEED))
prediction = np.full(len(paired), np.nan)
tuning_rows = []
print("Running nested position-grouped differential-effect prediction...")
for outer_fold in range(OUTER_FOLDS):
    test = np.flatnonzero(outer == outer_fold)
    train = np.flatnonzero(outer != outer_fold)
    inner = group_folds(groups[train], INNER_FOLDS,
                        np.random.default_rng(SEED + outer_fold + 1))
    alpha_scores = []
    for alpha in ALPHAS:
        rmses = []
        for inner_fold in range(INNER_FOLDS):
            val = train[inner == inner_fold]
            fit = train[inner != inner_fold]
            model = Ridge(alpha=alpha, solver="lsqr", max_iter=10000, tol=1e-5)
            model.fit(X[fit], y[fit])
            rmses.append(mean_squared_error(y[val], model.predict(X[val])) ** 0.5)
        alpha_scores.append((float(np.mean(rmses)), alpha))
    alpha_scores.sort(key=lambda z: (z[0], z[1]))
    selected_alpha = float(alpha_scores[0][1])
    model = Ridge(alpha=selected_alpha, solver="lsqr", max_iter=10000, tol=1e-5)
    model.fit(X[train], y[train])
    prediction[test] = model.predict(X[test])
    tuning_rows.append({
        "outer_fold": outer_fold + 1, "selected_alpha": selected_alpha,
        "inner_validation_rmse": alpha_scores[0][0],
        "training_positions": len(np.unique(groups[train])),
        "test_positions": len(np.unique(groups[test])),
        "training_variants": len(train), "test_variants": len(test),
    })
if not np.isfinite(prediction).all():
    raise RuntimeError("Incomplete out-of-fold predictions")

point = pd.DataFrame([
    {"model": "condition_differential_kmer", **metrics(y, prediction)},
    {"model": "zero_difference_baseline", **metrics(y, np.zeros(len(y)))},
])
paired["outer_fold"] = outer + 1
paired["predicted_condition_difference"] = prediction
paired["residual"] = y - prediction

# Position-clustered bootstrap of the already out-of-fold predictions.
clusters = {position: x.index.to_numpy()
            for position, x in paired.groupby("position_1based")}
cluster_ids = np.array(list(clusters))
boot_rows = []
for iteration in range(1, N_BOOT + 1):
    sampled = RNG.choice(cluster_ids, len(cluster_ids), replace=True)
    idx = np.concatenate([clusters[x] for x in sampled])
    for model, values in [("condition_differential_kmer", prediction),
                          ("zero_difference_baseline", np.zeros(len(y)))]:
        boot_rows.append({"iteration": iteration, "model": model,
                          **metrics(y[idx], values[idx])})
bootstrap = pd.DataFrame(boot_rows)

ci_rows = []
for model, x in bootstrap.groupby("model"):
    current = point.loc[point.model.eq(model)].iloc[0]
    for measure in ["mae", "rmse", "r2", "pearson_r", "spearman_rho",
                    "direction_accuracy"]:
        values = x[measure].dropna().to_numpy(float)
        ci_rows.append({"model": model, "measure": measure,
                        "point_estimate": current[measure],
                        "ci_lower_95": np.quantile(values, 0.025) if len(values) else np.nan,
                        "ci_upper_95": np.quantile(values, 0.975) if len(values) else np.nan})
confidence = pd.DataFrame(ci_rows)

# Refit only for interpretation after unbiased OOF evaluation is complete.
final_alpha = float(pd.Series([x["selected_alpha"] for x in tuning_rows]).mode().iloc[0])
final_model = Ridge(alpha=final_alpha, solver="lsqr", max_iter=10000, tol=1e-5)
final_model.fit(X, y)
features = vectorizer.get_feature_names_out()
coefficients = pd.DataFrame({"kmer": features,
                             "coefficient": np.asarray(final_model.coef_).ravel()})
coefficients["absolute_coefficient"] = coefficients.coefficient.abs()
coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)

paired.sort_values("absolute_condition_difference", ascending=False).to_csv(
    OUTPUT / "paired_sul1_condition_effects.tsv", sep="\t", index=False)
pd.DataFrame(tuning_rows).to_csv(OUTPUT / "condition_model_nested_tuning.tsv", sep="\t", index=False)
point.to_csv(OUTPUT / "condition_model_point_estimates.tsv", sep="\t", index=False)
bootstrap.to_csv(OUTPUT / "condition_model_position_bootstrap.tsv", sep="\t", index=False)
confidence.to_csv(OUTPUT / "condition_model_confidence_intervals.tsv", sep="\t", index=False)
coefficients.head(500).to_csv(OUTPUT / "top_condition_differential_kmers.tsv", sep="\t", index=False)
joblib.dump({"vectorizer": vectorizer, "ridge_model": final_model,
             "alpha": final_alpha, "target": "sulfate robust_z minus glucose robust_z"},
            OUTPUT / "condition_differential_kmer_model.joblib", compress=3)
(OUTPUT / "condition_model_settings.json").write_text(json.dumps({
    **agreement, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
    "alphas": ALPHAS, "bootstrap_iterations": N_BOOT, "seed": SEED,
    "position_leakage": False,
    "interpretation": "Predicts context-dependent score difference, not causal response."
}, indent=2), encoding="utf-8")

print("\nSUL1 CONDITION AGREEMENT")
print(json.dumps(agreement, indent=2))
print("\nCONDITION-DIFFERENTIAL MODEL")
print(point.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nStep 40 condition-differential regulatory modeling: COMPLETED")
print("Saved:", OUTPUT)
