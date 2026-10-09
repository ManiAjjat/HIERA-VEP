import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
PREDICTION_FILE = ROOT / "results/frozen_nt_external_validation/frozen_nt_external_predictions.tsv"
NT_FEATURE_FILE = ROOT / "results/frozen_nt_external_validation/external_nt_delta_embeddings.npy"
KMER_MODEL_FILE = ROOT / "results/shared_kmer_external_validation/shared_kmer_external_model.joblib"
OUTPUT = ROOT / "results/assay_adaptive_hybrid_model"
OUTPUT.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT = 2000
SEED = 20261006
RNG = np.random.default_rng(SEED)
ADAPTIVE_MODELS = ["adapted_kmer", "adapted_frozen_nt", "adapted_hybrid"]


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


def balanced_group_folds(groups, n_folds, rng):
    unique, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(unique))
    # Greedy allocation balances variant counts while keeping every position intact.
    fold_load = np.zeros(n_folds, dtype=int)
    assignment = {}
    for i in order[np.argsort(-counts[order], kind="stable")]:
        fold = int(np.argmin(fold_load))
        assignment[unique[i]] = fold
        fold_load[fold] += counts[i]
    return np.array([assignment[g] for g in groups], dtype=int)


def delta_kmers(vectorizer, data):
    ref = vectorizer.transform(data.reference_sequence.astype(str).str.upper().tolist())
    alt = vectorizer.transform(data.alternative_sequence.astype(str).str.upper().tolist())
    return (alt - ref).astype(np.float32).tocsr()


def make_features(kind, kmer, nt, fit_rows, apply_rows, scaler=None):
    if kind == "adapted_kmer":
        return kmer[apply_rows], scaler
    if scaler is None:
        scaler = StandardScaler().fit(nt[fit_rows])
    nt_scaled = scaler.transform(nt[apply_rows]).astype(np.float32)
    if kind == "adapted_frozen_nt":
        return nt_scaled, scaler
    hybrid = sparse.hstack([kmer[apply_rows], sparse.csr_matrix(nt_scaled)], format="csr")
    return hybrid, scaler


def tune_alpha(kind, kmer, nt, y, positions, train_rows, seed):
    local_groups = positions[train_rows]
    inner_assignment = balanced_group_folds(
        local_groups, INNER_FOLDS, np.random.default_rng(seed))
    scores = []
    for alpha in ALPHAS:
        fold_rmse = []
        for fold in range(INNER_FOLDS):
            inner_val = train_rows[inner_assignment == fold]
            inner_train = train_rows[inner_assignment != fold]
            X_train, scaler = make_features(kind, kmer, nt, inner_train, inner_train)
            X_val, _ = make_features(kind, kmer, nt, inner_train, inner_val, scaler)
            model = Ridge(alpha=alpha, solver="lsqr", max_iter=10000, tol=1e-5)
            model.fit(X_train, y[inner_train])
            prediction = model.predict(X_val)
            fold_rmse.append(mean_squared_error(y[inner_val], prediction) ** 0.5)
        scores.append((float(np.mean(fold_rmse)), alpha))
    scores.sort(key=lambda x: (x[0], x[1]))
    return float(scores[0][1]), float(scores[0][0])


print("Loading external predictions and frozen embeddings...")
d = pd.read_csv(PREDICTION_FILE, sep="\t", low_memory=False)
nt = np.load(NT_FEATURE_FILE, mmap_mode="r")
if len(nt) != len(d):
    raise ValueError(f"Embedding/prediction row mismatch: {len(nt)} versus {len(d)}")
required = {"element_name", "validation_tier", "position_1based", "observed_score",
            "reference_sequence", "alternative_sequence", "predicted_frozen_nt_ridge",
            "predicted_shared_kmer_ridge"}
missing = required.difference(d.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

print("Loading development-defined k-mer vocabulary...")
kmer_bundle = joblib.load(KMER_MODEL_FILE)
if "vectorizer" not in kmer_bundle:
    raise KeyError("Step 35 model bundle does not contain its vectorizer")
kmer = delta_kmers(kmer_bundle["vectorizer"], d)
y = d.observed_score.to_numpy(float)

predictions = {name: np.full(len(d), np.nan, dtype=float) for name in ADAPTIVE_MODELS}
fold_labels = np.full(len(d), -1, dtype=int)
tuning_rows = []

print("Running nested position-grouped assay adaptation...")
for element_number, element in enumerate(sorted(d.element_name.unique()), start=1):
    rows = np.flatnonzero(d.element_name.eq(element).to_numpy())
    positions = d.position_1based.to_numpy()
    outer_assignment = balanced_group_folds(
        positions[rows], OUTER_FOLDS,
        np.random.default_rng(SEED + element_number))
    fold_labels[rows] = outer_assignment
    print(f"  {element}: {len(rows)} variants, {len(np.unique(positions[rows]))} positions")
    for fold in range(OUTER_FOLDS):
        test_rows = rows[outer_assignment == fold]
        train_rows = rows[outer_assignment != fold]
        if set(positions[test_rows]).intersection(set(positions[train_rows])):
            raise RuntimeError(f"Position leakage detected for {element}, fold {fold}")
        for model_number, kind in enumerate(ADAPTIVE_MODELS):
            alpha, inner_rmse = tune_alpha(
                kind, kmer, nt, y, positions, train_rows,
                SEED + element_number * 100 + fold * 10 + model_number)
            X_train, scaler = make_features(kind, kmer, nt, train_rows, train_rows)
            X_test, _ = make_features(kind, kmer, nt, train_rows, test_rows, scaler)
            model = Ridge(alpha=alpha, solver="lsqr", max_iter=10000, tol=1e-5)
            model.fit(X_train, y[train_rows])
            predictions[kind][test_rows] = model.predict(X_test)
            tuning_rows.append({
                "element_name": element, "outer_fold": fold + 1, "model": kind,
                "selected_alpha": alpha, "inner_validation_rmse": inner_rmse,
                "training_variants": len(train_rows), "test_variants": len(test_rows),
                "training_positions": len(np.unique(positions[train_rows])),
                "test_positions": len(np.unique(positions[test_rows])),
            })

for name, values in predictions.items():
    if not np.isfinite(values).all():
        raise RuntimeError(f"Incomplete out-of-fold predictions for {name}")

# Prespecified equal-weight ensemble; no external test fold is used to choose weights.
predictions["adapted_equal_weight_ensemble"] = (
    predictions["adapted_kmer"] + predictions["adapted_frozen_nt"] +
    predictions["adapted_hybrid"]) / 3.0
predictions["zero_shot_kmer"] = d.predicted_shared_kmer_ridge.to_numpy(float)
predictions["zero_shot_frozen_nt"] = d.predicted_frozen_nt_ridge.to_numpy(float)

metric_rows = []
for element, x in d.groupby("element_name", sort=True):
    idx = x.index.to_numpy()
    for model, values in predictions.items():
        metric_rows.append({"validation_tier": x.validation_tier.iloc[0],
                            "element_name": element, "model": model,
                            **metrics(y[idx], values[idx])})
for tier, x in d.groupby("validation_tier", sort=True):
    idx = x.index.to_numpy()
    for model, values in predictions.items():
        metric_rows.append({"validation_tier": tier, "element_name": "ALL",
                            "model": model, **metrics(y[idx], values[idx])})
point_estimates = pd.DataFrame(metric_rows)

prediction_table = d.copy()
prediction_table["outer_fold"] = fold_labels + 1
for model, values in predictions.items():
    prediction_table[f"predicted_{model}"] = values

print(f"Starting {N_BOOT:,} position-clustered bootstrap iterations...")
boot_rows = []
for tier, tier_data in d.groupby("validation_tier", sort=True):
    clusters = {key: group.index.to_numpy()
                for key, group in tier_data.groupby(["element_name", "position_1based"])}
    by_element = {}
    for key in clusters:
        by_element.setdefault(key[0], []).append(key)
    for iteration in range(1, N_BOOT + 1):
        parts = []
        for element, keys in by_element.items():
            chosen = RNG.integers(0, len(keys), size=len(keys))
            parts.extend(clusters[keys[i]] for i in chosen)
        idx = np.concatenate(parts)
        for model, values in predictions.items():
            boot_rows.append({"validation_tier": tier, "iteration": iteration,
                              "model": model, **metrics(y[idx], values[idx])})
bootstrap = pd.DataFrame(boot_rows)

comparison_rows = []
for tier, group in bootstrap.groupby("validation_tier"):
    pivot = group.pivot(index="iteration", columns="model",
                        values=["rmse", "mae", "r2", "pearson_r", "spearman_rho"])
    for adaptive in [*ADAPTIVE_MODELS, "adapted_equal_weight_ensemble"]:
        for baseline in ["zero_shot_kmer", "zero_shot_frozen_nt"]:
            definitions = {
                "rmse_reduction": pivot[("rmse", baseline)] - pivot[("rmse", adaptive)],
                "mae_reduction": pivot[("mae", baseline)] - pivot[("mae", adaptive)],
                "r2_gain": pivot[("r2", adaptive)] - pivot[("r2", baseline)],
                "pearson_gain": pivot[("pearson_r", adaptive)] - pivot[("pearson_r", baseline)],
                "spearman_gain": pivot[("spearman_rho", adaptive)] - pivot[("spearman_rho", baseline)],
            }
            for measure, series in definitions.items():
                values = series.dropna().to_numpy(float)
                comparison_rows.append({
                    "validation_tier": tier, "adaptive_model": adaptive,
                    "baseline": baseline, "measure": measure,
                    "mean_difference": float(np.mean(values)),
                    "ci_lower_95": float(np.quantile(values, 0.025)),
                    "ci_upper_95": float(np.quantile(values, 0.975)),
                    "bootstrap_p_two_sided": float(min(
                        1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0))))
                })
comparisons = pd.DataFrame(comparison_rows)

pd.DataFrame(tuning_rows).to_csv(
    OUTPUT / "nested_tuning_results.tsv", sep="\t", index=False)
point_estimates.to_csv(
    OUTPUT / "assay_adaptive_point_estimates.tsv", sep="\t", index=False)
prediction_table.to_csv(
    OUTPUT / "assay_adaptive_oof_predictions.tsv", sep="\t", index=False)
bootstrap.to_csv(
    OUTPUT / "assay_adaptive_position_bootstrap.tsv", sep="\t", index=False)
comparisons.to_csv(
    OUTPUT / "assay_adaptive_paired_comparisons.tsv", sep="\t", index=False)

settings = {
    "analysis": "nested position-grouped assay-adaptive sequence modeling",
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "alphas": ALPHAS,
    "bootstrap_iterations": N_BOOT,
    "seed": SEED,
    "variants": len(d),
    "elements": int(d.element_name.nunique()),
    "nt_features": int(nt.shape[1]),
    "kmer_features": int(kmer.shape[1]),
    "position_leakage": False,
    "claim_boundary": (
        "These are within-assay adaptation results from nested grouped cross-validation. "
        "They are not independent zero-shot external validation and must be reported "
        "separately from Steps 35-36."
    ),
}
(OUTPUT / "assay_adaptive_settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8")

print("\nASSAY-ADAPTIVE POINT ESTIMATES")
print(point_estimates.query("element_name == 'ALL'").to_string(
    index=False, float_format=lambda z: f"{z:.4f}"))
print("\nADAPTATION VERSUS ZERO-SHOT K-MER")
show = comparisons.query(
    "baseline == 'zero_shot_kmer' and adaptive_model == 'adapted_hybrid'")
print(show.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nStep 38 assay-adaptive hybrid modeling: COMPLETED")
print("Saved:", OUTPUT)
