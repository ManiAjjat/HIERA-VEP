import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
TABLE = ROOT / "results/hiera_unified_training_table/hiera_unified_variant_table.tsv"
ANNOT = ROOT / "results/human_conservation_annotation/development_variant_context_conservation_annotations.tsv"
MODEL_DIR = ROOT / "results/context_aware_interpretable_model"
OUT = ROOT / "results/exact_attribution_counterfactual_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261007
PERMUTATIONS = 1000
CCRE_LEVELS = ["none", "PLS", "pELS", "dELS", "CA-H3K4me3"]
GATES = ["ccre_overlap", "promoter_overlap", "phyloP100way", "phastCons100way"]


def clean(x): return x.fillna("").astype(str).str.upper().tolist()


def delta(vectorizer, frame):
    return (vectorizer.transform(clean(frame.alternative_sequence))
            - vectorizer.transform(clean(frame.reference_sequence))).astype(np.float32).tocsr()


def context_frame(annotation):
    c = pd.DataFrame(index=annotation.index)
    c["ccre_overlap"] = annotation.ccre_overlap.astype(float)
    c["promoter_overlap"] = annotation.promoter_overlap.astype(float)
    c["log1p_abs_tss_distance"] = np.log1p(pd.to_numeric(annotation.absolute_distance_to_nearest_tss))
    c["phyloP100way"] = pd.to_numeric(annotation.phyloP100way)
    c["phastCons100way"] = pd.to_numeric(annotation.phastCons100way)
    c["phyloP_conserved"] = annotation.phyloP_conserved.astype(float)
    c["phyloP_accelerated"] = annotation.phyloP_accelerated.astype(float)
    c["phastCons_high"] = annotation.phastCons_high.astype(float)
    primary = annotation.primary_ccre_class.fillna("none").astype(str)
    for level in CCRE_LEVELS[1:]: c[f"ccre_class_{level}"] = primary.eq(level).astype(float)
    if c.isna().any().any(): raise RuntimeError("Context table contains missing values")
    return c.astype(np.float32)


def scale_context(raw, train, evaluate):
    tr = raw.iloc[train].to_numpy(np.float32); ev = raw.iloc[evaluate].to_numpy(np.float32)
    continuous = [raw.columns.get_loc(x) for x in
                  ["log1p_abs_tss_distance", "phyloP100way", "phastCons100way"]]
    mean = tr[:, continuous].mean(0); sd = tr[:, continuous].std(0); sd[sd == 0] = 1
    ev[:, continuous] = (ev[:, continuous] - mean) / sd
    tr[:, continuous] = (tr[:, continuous] - mean) / sd
    return tr, ev, mean, sd


def safe_corr(y, p):
    return np.nan if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0 else float(stats.pearsonr(y, p).statistic)


def metrics(y, p):
    return {"mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** .5,
            "r2": r2_score(y, p), "pearson_r": safe_corr(y, p),
            "spearman_rho": np.nan if np.std(p) == 0 else float(stats.spearmanr(y, p).statistic),
            "direction_accuracy": np.mean(np.sign(y) == np.sign(p))}


def canonical_kmer(kmer):
    rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return min(kmer, rc)


print("Loading locked context model and rebuilding exact design blocks...")
d = pd.read_csv(TABLE, sep="\t", low_memory=False)
d = d[d.source_partition.eq("development")].reset_index(drop=True)
a = pd.read_csv(ANNOT, sep="\t", low_memory=False).sort_values("source_row").reset_index(drop=True)
summary = json.loads((MODEL_DIR/"context_model_summary.json").read_text(encoding="utf-8"))
if summary["selected_feature_family"] != "context_gated_sequence":
    raise RuntimeError("Step 52 currently requires the selected context_gated_sequence model")
if len(d) != len(a) or not np.array_equal(a.source_row.to_numpy(), np.arange(len(d))):
    raise RuntimeError("Step 44/49 row alignment failed")

vectorizer = joblib.load(MODEL_DIR/"context_model_vectorizer.joblib")
model = joblib.load(MODEL_DIR/"context_selected_ridge.joblib")
split = d.original_split.astype(str).str.lower()
pool = np.flatnonzero(split.isin(["train", "validation", "val"]))
locked = np.flatnonzero(split.eq("test"))
x_all = delta(vectorizer, d); x = x_all[locked]
craw = context_frame(a)
_, c, means, sds = scale_context(craw, pool, locked)
y = d.iloc[locked].target_model.to_numpy(float)

n_seq, n_context = x.shape[1], c.shape[1]
expected = n_seq + n_context + len(GATES) * n_seq
coef = np.asarray(model.coef_).reshape(-1)
if len(coef) != expected:
    raise RuntimeError(f"Coefficient layout mismatch: {len(coef)} versus expected {expected}")

offset = 0
sequence_coef = coef[offset:offset+n_seq]; offset += n_seq
context_coef = coef[offset:offset+n_context]; offset += n_context
gate_coef = {}
for gate in GATES:
    gate_coef[gate] = coef[offset:offset+n_seq]; offset += n_seq

contributions = pd.DataFrame(index=np.arange(len(locked)))
contributions["intercept"] = float(model.intercept_)
contributions["sequence_main"] = np.asarray(x @ sequence_coef).reshape(-1)
contributions["context_additive"] = c @ context_coef
for gate in GATES:
    j = craw.columns.get_loc(gate)
    contributions[f"gate_{gate}"] = np.asarray(x.multiply(c[:, j][:, None]) @ gate_coef[gate]).reshape(-1)
contribution_columns = ["intercept", "sequence_main", "context_additive"] + [f"gate_{g}" for g in GATES]
reconstructed = contributions[contribution_columns].sum(1).to_numpy()
direct_blocks = [x, sparse.csr_matrix(c)] + [x.multiply(c[:, craw.columns.get_loc(g)][:, None]) for g in GATES]
direct = model.predict(sparse.hstack(direct_blocks, format="csr"))
max_error = float(np.max(np.abs(reconstructed-direct)))
if max_error > 1e-5: raise RuntimeError(f"Attribution additivity failed: max error={max_error}")

identity = d.iloc[locked][["row_id", "element_id", "assay_id", "position_group",
                            "reference_allele", "alternative_allele", "target_model"]].reset_index(drop=True)
variant = pd.concat([identity, a.iloc[locked][["primary_ccre_class", "promoter_overlap",
                    "absolute_distance_to_nearest_tss", "phyloP100way", "phastCons100way"]].reset_index(drop=True),
                    contributions], axis=1)
variant["prediction"] = direct
variant["residual"] = variant.target_model - variant.prediction

# Block-removal predictions. These are model sensitivity analyses, not causal interventions.
ablation_rows = [{"removed_block": "none", **metrics(y, direct)}]
for column in contribution_columns[1:]:
    p = direct - contributions[column].to_numpy()
    ablation_rows.append({"removed_block": column, **metrics(y, p),
                          "mean_absolute_prediction_change": np.mean(np.abs(direct-p))})
ablation = pd.DataFrame(ablation_rows)

# Shuffle all context at position-group level while keeping sequence and observed effects fixed.
rng = np.random.default_rng(SEED)
groups = identity.position_group.astype(str).to_numpy(); unique = np.unique(groups)
group_rows = {g: np.flatnonzero(groups == g) for g in unique}
representative = np.array([group_rows[g][0] for g in unique])
permutation_rows = []
observed_metric = metrics(y, direct)
for iteration in range(1, PERMUTATIONS+1):
    shuffled = rng.permutation(len(unique)); cp = np.empty_like(c)
    for target_i, source_i in enumerate(shuffled): cp[group_rows[unique[target_i]]] = c[representative[source_i]]
    blocks = [x, sparse.csr_matrix(cp)] + [x.multiply(cp[:, craw.columns.get_loc(g)][:, None]) for g in GATES]
    p = model.predict(sparse.hstack(blocks, format="csr")); m = metrics(y, p)
    permutation_rows.append({"iteration": iteration, **m,
                             "rmse_increase_vs_observed": m["rmse"]-observed_metric["rmse"],
                             "pearson_decrease_vs_observed": observed_metric["pearson_r"]-m["pearson_r"]})
permutations = pd.DataFrame(permutation_rows)

# K-mer and context-gated attribution magnitudes on the locked test.
kmers = vectorizer.get_feature_names_out()
kmer_rows = []
blocks = {"sequence_main": (np.ones(len(locked)), sequence_coef)}
for gate in GATES: blocks[f"gate_{gate}"] = (c[:, craw.columns.get_loc(gate)], gate_coef[gate])
for block, (weights, coefficients) in blocks.items():
    weighted = x.multiply(weights[:, None]).multiply(coefficients)
    signed = np.asarray(weighted.mean(0)).reshape(-1)
    mean_abs = np.asarray(abs(weighted).mean(0)).reshape(-1)
    for kmer, coefficient, s, magnitude in zip(kmers, coefficients, signed, mean_abs):
        kmer_rows.append({"block": block, "kmer": kmer, "canonical_kmer": canonical_kmer(kmer),
                          "coefficient": coefficient, "mean_signed_contribution": s,
                          "mean_absolute_contribution": magnitude})
kmer = pd.DataFrame(kmer_rows)
canonical = (kmer.groupby(["block", "canonical_kmer"], as_index=False)
             .agg(mean_signed_contribution=("mean_signed_contribution", "sum"),
                  mean_absolute_contribution=("mean_absolute_contribution", "sum"),
                  member_kmers=("kmer", lambda z:";".join(sorted(z))))
             .sort_values("mean_absolute_contribution", ascending=False))

block_summary = []
for column in contribution_columns:
    values = contributions[column].to_numpy()
    block_summary.append({"block": column, "mean": np.mean(values), "sd": np.std(values, ddof=1),
                          "mean_absolute_contribution": np.mean(np.abs(values)),
                          "median_absolute_contribution": np.median(np.abs(values)),
                          "pearson_with_prediction": safe_corr(values, direct)})
block_summary = pd.DataFrame(block_summary)

variant.to_csv(OUT/"locked_variant_exact_attributions.tsv", sep="\t", index=False)
ablation.to_csv(OUT/"context_block_counterfactual_ablation.tsv", sep="\t", index=False)
permutations.to_csv(OUT/"position_context_shuffle_null.tsv", sep="\t", index=False)
kmer.to_csv(OUT/"kmer_attribution_components.tsv", sep="\t", index=False)
canonical.to_csv(OUT/"canonical_kmer_attribution_ranking.tsv", sep="\t", index=False)
block_summary.to_csv(OUT/"attribution_block_summary.tsv", sep="\t", index=False)

audit = {"variants": len(locked), "selected_model": "context_gated_sequence",
         "exact_additivity_max_absolute_error": max_error, "permutations": PERMUTATIONS,
         "observed_metrics": observed_metric,
         "shuffle_rmse_increase_mean": float(permutations.rmse_increase_vs_observed.mean()),
         "shuffle_rmse_increase_95_interval": [float(permutations.rmse_increase_vs_observed.quantile(.025)),
                                               float(permutations.rmse_increase_vs_observed.quantile(.975))],
         "shuffle_pearson_decrease_mean": float(permutations.pearson_decrease_vs_observed.mean()),
         "warning": "Block removal and context shuffling quantify model reliance; they are not biological causal effects."}
(OUT/"attribution_counterfactual_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\nEXACT ATTRIBUTION AUDIT")
print(json.dumps(audit, indent=2))
print("\nBLOCK CONTRIBUTIONS")
print(block_summary.to_string(index=False, float_format=lambda z:f"{z:.5f}"))
print("\nMODEL BLOCK-REMOVAL SENSITIVITY")
print(ablation.to_string(index=False, float_format=lambda z:f"{z:.5f}"))
print("\nStep 52 exact attribution and counterfactual audit: COMPLETED")
print("Saved:", OUT)
