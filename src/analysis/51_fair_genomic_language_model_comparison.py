import os
from pathlib import Path
from collections import Counter
import gc, json, time

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
TABLE = ROOT / "results/hiera_unified_training_table/hiera_unified_variant_table.tsv"
CONTEXT_PRED = ROOT / "results/context_aware_interpretable_model/locked_context_predictions.tsv"
OUT = ROOT / "results/fair_genomic_language_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)
EMBEDDING_FILE = OUT / "development_nt_delta_embeddings.npy"
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
BATCH_SIZE = 64
MAX_TOKEN_LENGTH = 128
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
BOOTSTRAPS = 2000
SEED = 20261007


def connected_groups(frame):
    n = len(frame); parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for column in ["position_group", "sequence_pair_hash"]:
        first = {}
        for i, value in enumerate(frame[column].astype(str)):
            if value in first: union(i, first[value])
            else: first[value] = i
    roots = [find(i) for i in range(n)]
    code = {v: i for i, v in enumerate(sorted(set(roots)))}
    return np.array([code[v] for v in roots])


def safe_corr(fn, y, p):
    return np.nan if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0 else float(fn(y, p).statistic)


def metrics(y, p):
    return {"n": len(y), "mae": mean_absolute_error(y, p),
            "rmse": mean_squared_error(y, p) ** .5, "r2": r2_score(y, p),
            "pearson_r": safe_corr(stats.pearsonr, y, p),
            "spearman_rho": safe_corr(stats.spearmanr, y, p),
            "direction_accuracy": np.mean(np.sign(y) == np.sign(p)),
            "observed_sd": np.std(y, ddof=1), "predicted_sd": np.std(p, ddof=1)}


def macro_rmse(y, p, elements):
    values = []
    for element in np.unique(elements):
        keep = elements == element
        if keep.sum() >= 2: values.append(mean_squared_error(y[keep], p[keep]) ** .5)
    return float(np.mean(values))


def fit_nt(x, y, train, evaluate, alpha):
    scaler = StandardScaler().fit(x[train])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
    return model.predict(scaler.transform(x[evaluate])), scaler, model


def extract_delta(data, tokenizer, encoder, device):
    features, changed_counts = [], {}
    start_time = time.time(); encoder.eval()
    for left in range(0, len(data), BATCH_SIZE):
        right = min(left + BATCH_SIZE, len(data))
        ref = tokenizer(data.model_reference_sequence.iloc[left:right].tolist(), padding=True,
                        truncation=True, max_length=MAX_TOKEN_LENGTH, return_tensors="pt")
        alt = tokenizer(data.model_alternative_sequence.iloc[left:right].tolist(), padding=True,
                        truncation=True, max_length=MAX_TOKEN_LENGTH, return_tensors="pt")
        changed = ref["input_ids"].ne(alt["input_ids"])
        values, counts = torch.unique(changed.sum(1), return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            changed_counts[int(value)] = changed_counts.get(int(value), 0) + int(count)
        if not torch.all(changed.sum(1).eq(1)):
            raise RuntimeError(f"Each SNV must alter one token; cumulative distribution={changed_counts}")
        ids = torch.cat([ref["input_ids"], alt["input_ids"]]).to(device)
        mask_ids = torch.cat([ref["attention_mask"], alt["attention_mask"]]).to(device)
        with torch.inference_mode():
            hidden = encoder(ids, attention_mask=mask_ids,
                             encoder_attention_mask=mask_ids, return_dict=True).last_hidden_state
        n = right - left
        mask = changed.to(device).unsqueeze(-1).to(hidden.dtype)
        local_ref = (hidden[:n] * mask).sum(1)
        local_alt = (hidden[n:] * mask).sum(1)
        features.append((local_alt - local_ref).float().cpu().numpy())
        if right % 640 == 0 or right == len(data):
            print(f"  Embedded {right:,}/{len(data):,} | {(time.time()-start_time)/60:.1f} minutes")
    return np.concatenate(features).astype(np.float32), changed_counts


def bootstrap(frame, y, context, nt):
    rng = np.random.default_rng(SEED)
    groups = frame.position_group.astype(str).to_numpy(); unique = np.unique(groups)
    mapping = {g: np.flatnonzero(groups == g) for g in unique}; rows = []
    for iteration in range(1, BOOTSTRAPS + 1):
        selected = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([mapping[g] for g in selected])
        mc, mn = metrics(y[idx], context[idx]), metrics(y[idx], nt[idx])
        rows.append({"iteration": iteration,
                     "rmse_reduction_context_vs_nt": mn["rmse"]-mc["rmse"],
                     "mae_reduction_context_vs_nt": mn["mae"]-mc["mae"],
                     "r2_gain_context_vs_nt": mc["r2"]-mn["r2"],
                     "pearson_gain_context_vs_nt": mc["pearson_r"]-mn["pearson_r"],
                     "spearman_gain_context_vs_nt": mc["spearman_rho"]-mn["spearman_rho"]})
    return pd.DataFrame(rows)


print("Loading development variants and locked Step 50 predictions...")
d = pd.read_csv(TABLE, sep="\t", low_memory=False)
d = d[d.source_partition.eq("development")].reset_index(drop=True)
context = pd.read_csv(CONTEXT_PRED, sep="\t", low_memory=False)
split = d.original_split.astype(str).str.lower()
pool = np.flatnonzero(split.isin(["train", "validation", "val"]))
locked = np.flatnonzero(split.eq("test"))
expected_ids = d.iloc[locked].row_id.astype(str).to_numpy()
if not np.array_equal(context.row_id.astype(str).to_numpy(), expected_ids):
    raise RuntimeError("Step 50 predictions do not match the current locked test order")
y = d.target_model.to_numpy(np.float32); groups = connected_groups(d)

if EMBEDDING_FILE.exists():
    print("Loading cached frozen NT embeddings:", EMBEDDING_FILE)
    x = np.load(EMBEDDING_FILE)
    changed_distribution = {"cached": len(x)}
else:
    print("Loading frozen Nucleotide Transformer...")
    d["model_reference_sequence"] = d.reference_sequence.astype(str).str.upper().str.replace(r"[^ACGT]", "A", regex=True)
    d["model_alternative_sequence"] = d.alternative_sequence.astype(str).str.upper().str.replace(r"[^ACGT]", "A", regex=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    else: print("WARNING: GPU unavailable; embedding extraction will be slow.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    masked = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if not hasattr(masked, "esm"): raise AttributeError("Expected ESM encoder is absent")
    encoder = masked.esm.to(device); del masked; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    x, changed_distribution = extract_delta(d, tokenizer, encoder, device)
    np.save(EMBEDDING_FILE, x)
    del encoder; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
if x.shape[0] != len(d): raise RuntimeError(f"Embedding row mismatch: {x.shape[0]} versus {len(d)}")
print("Embeddings:", x.shape)

outer = GroupKFold(n_splits=3); tuning_rows, outer_rows, selected = [], [], []
for outer_fold, (otr_rel, ova_rel) in enumerate(outer.split(pool, groups=groups[pool]), 1):
    otr, ova = pool[otr_rel], pool[ova_rel]; inner = GroupKFold(n_splits=2)
    scores = {alpha: [] for alpha in ALPHAS}
    print(f"Outer fold {outer_fold}/3: nested NT Ridge tuning...")
    for inner_fold, (itr_rel, iva_rel) in enumerate(inner.split(otr, groups=groups[otr]), 1):
        itr, iva = otr[itr_rel], otr[iva_rel]
        for alpha in ALPHAS:
            p, _, _ = fit_nt(x, y, itr, iva, alpha)
            score = macro_rmse(y[iva], p, d.iloc[iva].element_id.to_numpy())
            scores[alpha].append(score)
            tuning_rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold,
                                "alpha": alpha, "macro_element_rmse": score})
    alpha = min(ALPHAS, key=lambda a: np.mean(scores[a])); selected.append(alpha)
    p, _, _ = fit_nt(x, y, otr, ova, alpha)
    outer_rows.append({"outer_fold": outer_fold, "selected_alpha": alpha,
                       "macro_element_rmse": macro_rmse(y[ova], p, d.iloc[ova].element_id.to_numpy()),
                       **metrics(y[ova], p)})
    print(f"  selected alpha={alpha:g}, macro-RMSE={outer_rows[-1]['macro_element_rmse']:.4f}")

freq = Counter(selected)
outer_df = pd.DataFrame(outer_rows)
alpha = sorted(freq, key=lambda a: (-freq[a], outer_df.loc[outer_df.selected_alpha.eq(a), "macro_element_rmse"].mean(), a))[0]
nt_pred, scaler, model = fit_nt(x, y, pool, locked, alpha)
context_pred = context.context_selected_prediction.to_numpy(float)
sequence_pred = context.sequence_only_prediction.to_numpy(float)
point = pd.DataFrame([
    {"model": "context_gated_kmer", **metrics(y[locked], context_pred)},
    {"model": "frozen_nt_ridge", **metrics(y[locked], nt_pred)},
    {"model": "sequence_only_kmer", **metrics(y[locked], sequence_pred)},
    {"model": "zero_baseline", **metrics(y[locked], np.zeros(len(locked)))},
])

pred = context.copy()
pred["frozen_nt_prediction"] = nt_pred
boot = bootstrap(pred, y[locked], context_pred, nt_pred)
comparison = []
point_values = {
    "rmse_reduction_context_vs_nt": point.loc[point.model.eq("frozen_nt_ridge"), "rmse"].iat[0]-point.loc[point.model.eq("context_gated_kmer"), "rmse"].iat[0],
    "mae_reduction_context_vs_nt": point.loc[point.model.eq("frozen_nt_ridge"), "mae"].iat[0]-point.loc[point.model.eq("context_gated_kmer"), "mae"].iat[0],
    "r2_gain_context_vs_nt": point.loc[point.model.eq("context_gated_kmer"), "r2"].iat[0]-point.loc[point.model.eq("frozen_nt_ridge"), "r2"].iat[0],
    "pearson_gain_context_vs_nt": point.loc[point.model.eq("context_gated_kmer"), "pearson_r"].iat[0]-point.loc[point.model.eq("frozen_nt_ridge"), "pearson_r"].iat[0],
    "spearman_gain_context_vs_nt": point.loc[point.model.eq("context_gated_kmer"), "spearman_rho"].iat[0]-point.loc[point.model.eq("frozen_nt_ridge"), "spearman_rho"].iat[0],
}
for measure, estimate in point_values.items():
    values = boot[measure].dropna().to_numpy()
    comparison.append({"measure": measure, "point_estimate": estimate,
                       "ci_lower_95": np.quantile(values, .025), "ci_upper_95": np.quantile(values, .975),
                       "bootstrap_p_two_sided": min(1., 2*min(np.mean(values <= 0), np.mean(values >= 0)))})
comparison = pd.DataFrame(comparison)

pd.DataFrame(tuning_rows).to_csv(OUT/"nested_nt_tuning.tsv", sep="\t", index=False)
outer_df.to_csv(OUT/"nested_nt_outer_performance.tsv", sep="\t", index=False)
point.to_csv(OUT/"locked_model_comparison.tsv", sep="\t", index=False)
pred.to_csv(OUT/"locked_model_predictions.tsv", sep="\t", index=False)
boot.to_csv(OUT/"context_vs_nt_position_bootstrap.tsv", sep="\t", index=False)
comparison.to_csv(OUT/"context_vs_nt_paired_comparison.tsv", sep="\t", index=False)
joblib.dump({"scaler": scaler, "ridge_model": model, "alpha": alpha,
             "model_name": MODEL_NAME, "representation": "mutation_token_delta"}, OUT/"locked_nt_ridge.joblib")
summary = {"model_name": MODEL_NAME, "representation": "mutation_token_delta",
           "selected_alpha": alpha, "selection_frequency": dict(freq),
           "embedding_shape": list(x.shape), "changed_token_distribution": changed_distribution,
           "locked_test_used_for_selection": False, "bootstrap_position_clusters": BOOTSTRAPS,
           "interpretation": "Frozen local-token NT is a pretrained-language-model baseline, not a fine-tuned foundation model."}
(OUT/"fair_glm_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nFAIR LOCKED MODEL COMPARISON")
print(point.to_string(index=False, float_format=lambda z:f"{z:.4f}"))
print("\nCONTEXT MODEL VERSUS FROZEN NT")
print(comparison.to_string(index=False, float_format=lambda z:f"{z:.4f}"))
print("\nStep 51 fair genomic-language-model comparison: COMPLETED")
print("Saved:", OUT)
