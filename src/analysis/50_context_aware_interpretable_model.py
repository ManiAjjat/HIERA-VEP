import os
from pathlib import Path
from collections import Counter
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
TABLE = ROOT / "results/hiera_unified_training_table/hiera_unified_variant_table.tsv"
ANNOT = ROOT / "results/human_conservation_annotation/development_variant_context_conservation_annotations.tsv"
OUT = ROOT / "results/context_aware_interpretable_model"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261007
BOOTSTRAPS = 2000

# Prespecified before inspecting the locked test.
FAMILIES = ["sequence_only", "context_only", "sequence_plus_context", "context_gated_sequence"]
ALPHAS = [10.0, 100.0, 1000.0, 10000.0]
CCRE_LEVELS = ["none", "PLS", "pELS", "dELS", "CA-H3K4me3"]
GATING_COLUMNS = ["ccre_overlap", "promoter_overlap", "phyloP100way", "phastCons100way"]


def clean(x):
    return x.fillna("").astype(str).str.upper().tolist()


def delta(vectorizer, frame):
    return (vectorizer.transform(clean(frame.alternative_sequence))
            - vectorizer.transform(clean(frame.reference_sequence))).astype(np.float32).tocsr()


def connected_groups(frame):
    n = len(frame)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for column in ["position_group", "sequence_pair_hash"]:
        first = {}
        for i, value in enumerate(frame[column].astype(str)):
            if value in first:
                union(i, first[value])
            else:
                first[value] = i
    roots = [find(i) for i in range(n)]
    codes = {value: i for i, value in enumerate(sorted(set(roots)))}
    return np.array([codes[value] for value in roots])


def safe_corr(fn, y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(fn(y, p).statistic)


def metrics(y, p):
    return {
        "n": len(y),
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "r2": r2_score(y, p),
        "pearson_r": safe_corr(stats.pearsonr, y, p),
        "spearman_rho": safe_corr(stats.spearmanr, y, p),
        "direction_accuracy": np.mean(np.sign(y) == np.sign(p)),
        "observed_sd": np.std(y, ddof=1),
        "predicted_sd": np.std(p, ddof=1),
    }


def macro_rmse(y, p, elements):
    values = []
    for element in np.unique(elements):
        keep = elements == element
        if keep.sum() >= 2:
            values.append(mean_squared_error(y[keep], p[keep]) ** 0.5)
    return float(np.mean(values))


def prepare_context(annotation):
    c = pd.DataFrame(index=annotation.index)
    c["ccre_overlap"] = annotation.ccre_overlap.astype(float)
    c["promoter_overlap"] = annotation.promoter_overlap.astype(float)
    c["log1p_abs_tss_distance"] = np.log1p(pd.to_numeric(
        annotation.absolute_distance_to_nearest_tss, errors="coerce"))
    c["phyloP100way"] = pd.to_numeric(annotation.phyloP100way, errors="coerce")
    c["phastCons100way"] = pd.to_numeric(annotation.phastCons100way, errors="coerce")
    c["phyloP_conserved"] = annotation.phyloP_conserved.astype(float)
    c["phyloP_accelerated"] = annotation.phyloP_accelerated.astype(float)
    c["phastCons_high"] = annotation.phastCons_high.astype(float)
    primary = annotation.primary_ccre_class.fillna("none").astype(str)
    for level in CCRE_LEVELS[1:]:  # 'none' is the reference category
        c[f"ccre_class_{level}"] = primary.eq(level).astype(float)
    if c.isna().any().any():
        raise RuntimeError(f"Missing context values: {c.isna().sum()[c.isna().sum()>0].to_dict()}")
    return c.astype(np.float32)


def scale_context(raw, train_idx, eval_idx):
    train = raw.iloc[train_idx].to_numpy(np.float32)
    evaluate = raw.iloc[eval_idx].to_numpy(np.float32)
    continuous = [raw.columns.get_loc(x) for x in
                  ["log1p_abs_tss_distance", "phyloP100way", "phastCons100way"]]
    mean = train[:, continuous].mean(axis=0)
    sd = train[:, continuous].std(axis=0)
    sd[sd == 0] = 1.0
    train[:, continuous] = (train[:, continuous] - mean) / sd
    evaluate[:, continuous] = (evaluate[:, continuous] - mean) / sd
    return sparse.csr_matrix(train), sparse.csr_matrix(evaluate), {"mean": mean, "sd": sd}


def feature_pair(family, xseq, craw, train_idx, eval_idx):
    ctrain, ceval, scaling = scale_context(craw, train_idx, eval_idx)
    xtrain, xeval = xseq[train_idx], xseq[eval_idx]
    if family == "sequence_only":
        return xtrain, xeval, scaling
    if family == "context_only":
        return ctrain, ceval, scaling
    if family == "sequence_plus_context":
        return sparse.hstack([xtrain, ctrain], format="csr"), sparse.hstack([xeval, ceval], format="csr"), scaling
    # Interpretable context-gated k-mer effects: x_j * context_k.
    gate_indices = [craw.columns.get_loc(name) for name in GATING_COLUMNS]
    train_blocks = [xtrain, ctrain]
    eval_blocks = [xeval, ceval]
    for j in gate_indices:
        train_blocks.append(xtrain.multiply(ctrain[:, j].toarray()))
        eval_blocks.append(xeval.multiply(ceval[:, j].toarray()))
    return sparse.hstack(train_blocks, format="csr"), sparse.hstack(eval_blocks, format="csr"), scaling


def feature_names(family, kmer_names, context_names):
    if family == "sequence_only":
        return list(kmer_names)
    if family == "context_only":
        return list(context_names)
    names = list(kmer_names) + list(context_names)
    if family == "context_gated_sequence":
        for gate in GATING_COLUMNS:
            names.extend([f"{kmer} x {gate}" for kmer in kmer_names])
    return names


def fit_predict(family, alpha, xseq, craw, y, train_idx, eval_idx):
    xtrain, xeval, scaling = feature_pair(family, xseq, craw, train_idx, eval_idx)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
    model.fit(xtrain, y[train_idx])
    return model.predict(xeval), model, scaling


def clustered_bootstrap(frame, y, context_pred, sequence_pred):
    rng = np.random.default_rng(SEED)
    groups = frame.position_group.astype(str).to_numpy()
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rows = []
    for iteration in range(1, BOOTSTRAPS + 1):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[g] for g in sampled])
        yc, pc, ps = y[take], context_pred[take], sequence_pred[take]
        mc, ms = metrics(yc, pc), metrics(yc, ps)
        rows.append({"iteration": iteration,
                     "rmse_reduction": ms["rmse"] - mc["rmse"],
                     "mae_reduction": ms["mae"] - mc["mae"],
                     "r2_gain": mc["r2"] - ms["r2"],
                     "pearson_gain": mc["pearson_r"] - ms["pearson_r"],
                     "spearman_gain": mc["spearman_rho"] - ms["spearman_rho"]})
    return pd.DataFrame(rows)


print("Loading unified variants and Step 49 context annotations...")
d = pd.read_csv(TABLE, sep="\t", low_memory=False)
d = d[d.source_partition.eq("development")].reset_index(drop=True)
a = pd.read_csv(ANNOT, sep="\t", low_memory=False).sort_values("source_row").reset_index(drop=True)
if len(d) != len(a) or not np.array_equal(a.source_row.to_numpy(), np.arange(len(d))):
    raise RuntimeError("Step 49 annotations do not align one-to-one with unified development rows")
if not np.array_equal(d.element_id.astype(str).to_numpy(), a.element_id.astype(str).to_numpy()):
    raise RuntimeError("Element identity mismatch between Step 44 and Step 49")

y = d.target_model.to_numpy(np.float32)
split = d.original_split.astype(str).str.lower()
pool = np.flatnonzero(split.isin(["train", "validation", "val"]))
locked = np.flatnonzero(split.eq("test"))
groups_all = connected_groups(d)
craw = prepare_context(a)

print("Fitting development-only k-mer vocabulary...")
original_train = np.flatnonzero(split.eq("train"))
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 6), lowercase=False, dtype=np.float32)
vectorizer.fit(clean(d.iloc[original_train].reference_sequence) + clean(d.iloc[original_train].alternative_sequence))
xseq = delta(vectorizer, d)
print(f"Pool: {len(pool):,} | locked test: {len(locked):,} | k-mers: {xseq.shape[1]:,} | context features: {craw.shape[1]}")

# Nested connected-group selection; locked test is untouched.
outer = GroupKFold(n_splits=3)
tuning_rows, outer_rows, selections = [], [], []
for outer_fold, (otr_rel, ova_rel) in enumerate(outer.split(pool, groups=groups_all[pool]), 1):
    otr, ova = pool[otr_rel], pool[ova_rel]
    inner = GroupKFold(n_splits=2)
    candidates = {(family, alpha): [] for family in FAMILIES for alpha in ALPHAS}
    print(f"Outer fold {outer_fold}/3: nested feature-family tuning...")
    for inner_fold, (itr_rel, iva_rel) in enumerate(inner.split(otr, groups=groups_all[otr]), 1):
        itr, iva = otr[itr_rel], otr[iva_rel]
        for family in FAMILIES:
            for alpha in ALPHAS:
                pred, _, _ = fit_predict(family, alpha, xseq, craw, y, itr, iva)
                score = macro_rmse(y[iva], pred, d.iloc[iva].element_id.to_numpy())
                candidates[(family, alpha)].append(score)
                tuning_rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold,
                                    "feature_family": family, "alpha": alpha,
                                    "macro_element_rmse": score})
    winner = min(candidates, key=lambda key: np.mean(candidates[key]))
    family, alpha = winner
    pred, _, _ = fit_predict(family, alpha, xseq, craw, y, otr, ova)
    row = {"outer_fold": outer_fold, "selected_family": family, "selected_alpha": alpha,
           "macro_element_rmse": macro_rmse(y[ova], pred, d.iloc[ova].element_id.to_numpy()),
           **metrics(y[ova], pred)}
    outer_rows.append(row)
    selections.append(winner)
    print(f"  selected {family}, alpha={alpha:g}, macro-RMSE={row['macro_element_rmse']:.4f}")

outer_df = pd.DataFrame(outer_rows)
frequency = Counter(selections)
final_family, final_alpha = sorted(
    frequency,
    key=lambda key: (-frequency[key],
                     outer_df.loc[(outer_df.selected_family == key[0]) &
                                  (outer_df.selected_alpha == key[1]), "macro_element_rmse"].mean(),
                     FAMILIES.index(key[0]), key[1]))[0]

# Independently choose the sequence-only comparator from nested inner results.
tuning_df = pd.DataFrame(tuning_rows)
seq_scores = (tuning_df[tuning_df.feature_family.eq("sequence_only")]
              .groupby("alpha").macro_element_rmse.mean())
seq_alpha = float(seq_scores.idxmin())
print(f"Final selection: {final_family}, alpha={final_alpha:g}; sequence comparator alpha={seq_alpha:g}")

best_pred, best_model, scaling = fit_predict(final_family, final_alpha, xseq, craw, y, pool, locked)
seq_pred, seq_model, _ = fit_predict("sequence_only", seq_alpha, xseq, craw, y, pool, locked)
zero_pred = np.zeros(len(locked), dtype=float)

point = pd.DataFrame([
    {"model": "context_selected", "feature_family": final_family, "alpha": final_alpha,
     **metrics(y[locked], best_pred)},
    {"model": "sequence_only", "feature_family": "sequence_only", "alpha": seq_alpha,
     **metrics(y[locked], seq_pred)},
    {"model": "zero_baseline", "feature_family": "none", "alpha": np.nan,
     **metrics(y[locked], zero_pred)},
])

predictions = d.iloc[locked][["row_id", "element_id", "assay_id", "position_group",
                              "sequence_pair_hash", "target_model"]].copy()
predictions["context_selected_prediction"] = best_pred
predictions["sequence_only_prediction"] = seq_pred

boot = clustered_bootstrap(predictions, y[locked], best_pred, seq_pred)
comparison_rows = []
gain_metric = {"r2_gain": "r2", "pearson_gain": "pearson_r",
               "spearman_gain": "spearman_rho"}
for measure in ["rmse_reduction", "mae_reduction", "r2_gain", "pearson_gain", "spearman_gain"]:
    values = boot[measure].dropna().to_numpy()
    comparison_rows.append({"measure": measure, "point_estimate":
        (point.loc[point.model.eq("sequence_only"), "rmse"].iat[0] - point.loc[point.model.eq("context_selected"), "rmse"].iat[0])
        if measure == "rmse_reduction" else
        (point.loc[point.model.eq("sequence_only"), "mae"].iat[0] - point.loc[point.model.eq("context_selected"), "mae"].iat[0])
        if measure == "mae_reduction" else
        (point.loc[point.model.eq("context_selected"), gain_metric[measure]].iat[0] -
         point.loc[point.model.eq("sequence_only"), gain_metric[measure]].iat[0]),
        "ci_lower_95": np.quantile(values, .025), "ci_upper_95": np.quantile(values, .975),
        "bootstrap_p_two_sided": min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))})
comparison = pd.DataFrame(comparison_rows)

kmer_names = vectorizer.get_feature_names_out()
names = feature_names(final_family, kmer_names, craw.columns)
coef = pd.DataFrame({"feature": names, "coefficient": best_model.coef_})
coef["absolute_coefficient"] = coef.coefficient.abs()
coef = coef.sort_values("absolute_coefficient", ascending=False)

tuning_df.to_csv(OUT / "nested_context_tuning.tsv", sep="\t", index=False)
outer_df.to_csv(OUT / "nested_context_outer_performance.tsv", sep="\t", index=False)
point.to_csv(OUT / "locked_context_point_estimates.tsv", sep="\t", index=False)
predictions.to_csv(OUT / "locked_context_predictions.tsv", sep="\t", index=False)
boot.to_csv(OUT / "locked_context_position_bootstrap.tsv", sep="\t", index=False)
comparison.to_csv(OUT / "context_vs_sequence_paired_comparison.tsv", sep="\t", index=False)
coef.to_csv(OUT / "context_model_coefficients.tsv", sep="\t", index=False)
joblib.dump(vectorizer, OUT / "context_model_vectorizer.joblib")
joblib.dump(best_model, OUT / "context_selected_ridge.joblib")

summary = {
    "selected_feature_family": final_family,
    "selected_alpha": final_alpha,
    "selection_frequency": {f"{family}|{alpha:g}": count for (family, alpha), count in frequency.items()},
    "sequence_comparator_alpha": seq_alpha,
    "locked_test_used_for_selection": False,
    "outer_folds": 3,
    "inner_folds": 2,
    "bootstrap_position_clusters": BOOTSTRAPS,
    "context_features": list(craw.columns),
    "gating_features": GATING_COLUMNS,
    "locked_test_metrics": point.to_dict("records"),
    "claim_rule": "Context improves prediction only if paired confidence intervals support improvement on the locked test.",
}
(OUT / "context_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nLOCKED CONTEXT-AWARE RESULTS")
print(point.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPAIRED CONTEXT VERSUS SEQUENCE-ONLY")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nStep 50 context-aware interpretable model: COMPLETED")
print("Saved:", OUT)
