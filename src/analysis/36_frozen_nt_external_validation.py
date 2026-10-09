import os
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
EXTERNAL_FILE = ROOT / "results/external_validation_harmonized/external_model_ready_501bp.tsv"
KMER_PREDICTION_FILE = ROOT / "results/shared_kmer_external_validation/external_variant_predictions.tsv"
OUTPUT = ROOT / "results/frozen_nt_external_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
BATCH_SIZE = 64
MAX_TOKEN_LENGTH = 128
N_BOOT = 2000
SEED = 20261006
RNG = np.random.default_rng(SEED)
TIERS = ["primary_human_external", "cross_species_transfer", "secondary_splicing_transfer"]


def locate_locked_model():
    matches = list((ROOT / "results").glob("**/local_nt_ridge_model.joblib"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one locked local_nt_ridge_model.joblib under results; "
            f"found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]


def require_columns(data, columns, label):
    missing = set(columns).difference(data.columns)
    if missing:
        raise KeyError(f"{label} is missing columns: {sorted(missing)}")


def safe_corr(y, prediction, method):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return np.nan
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(fn(y, prediction).statistic)


def metrics(y, prediction):
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


def extract_local_delta(data, tokenizer, encoder, device):
    features = []
    changed_counts = {}
    encoder.eval()
    start = time.time()
    for left in range(0, len(data), BATCH_SIZE):
        right = min(left + BATCH_SIZE, len(data))
        ref = tokenizer(
            data.model_reference_sequence.iloc[left:right].tolist(), padding=True,
            truncation=True, max_length=MAX_TOKEN_LENGTH, return_tensors="pt"
        )
        alt = tokenizer(
            data.model_alternative_sequence.iloc[left:right].tolist(), padding=True,
            truncation=True, max_length=MAX_TOKEN_LENGTH, return_tensors="pt"
        )
        ref_ids = ref["input_ids"]
        alt_ids = alt["input_ids"]
        changed = ref_ids.ne(alt_ids)
        values, counts = torch.unique(changed.sum(dim=1), return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            changed_counts[int(value)] = changed_counts.get(int(value), 0) + int(count)
        if not torch.all(changed.sum(dim=1).eq(1)):
            raise RuntimeError(
                f"Every SNV must alter exactly one token; cumulative distribution={changed_counts}"
            )
        combined_ids = torch.cat([ref_ids, alt_ids], dim=0).to(device)
        combined_mask = torch.cat([ref["attention_mask"], alt["attention_mask"]], dim=0).to(device)
        with torch.inference_mode():
            output = encoder(
                combined_ids, attention_mask=combined_mask,
                encoder_attention_mask=combined_mask, return_dict=True
            )
        hidden = output.last_hidden_state
        batch_n = right - left
        mask = changed.to(device).unsqueeze(-1).to(hidden.dtype)
        local_ref = (hidden[:batch_n] * mask).sum(dim=1)
        local_alt = (hidden[batch_n:] * mask).sum(dim=1)
        features.append((local_alt - local_ref).float().cpu().numpy())
        if right % 640 == 0 or right == len(data):
            elapsed = (time.time() - start) / 60
            print(f"  Embedded {right:,}/{len(data):,} variants | {elapsed:.1f} minutes")
    return np.concatenate(features).astype(np.float32), changed_counts


def point_rows(data, observed, predictions):
    rows = []
    for tier in TIERS:
        tier_mask = data.validation_tier.eq(tier).to_numpy()
        if not tier_mask.any():
            continue
        for model, prediction in predictions.items():
            rows.append({"evaluation_group": tier, "element_name": "ALL", "model": model,
                         **metrics(observed[tier_mask], prediction[tier_mask])})
            for element in sorted(data.loc[tier_mask, "element_name"].unique()):
                mask = tier_mask & data.element_name.eq(element).to_numpy()
                rows.append({"evaluation_group": tier, "element_name": element, "model": model,
                             **metrics(observed[mask], prediction[mask])})
    return pd.DataFrame(rows)


def bootstrap_tier(data, observed, predictions, tier):
    tier_mask = data.validation_tier.eq(tier).to_numpy()
    local = data.loc[tier_mask].reset_index(drop=True)
    y = observed[tier_mask]
    pred = {name: values[tier_mask] for name, values in predictions.items()}
    cluster_rows = {}
    by_element = {}
    for key, group in local.groupby(["element_name", "position_1based"], sort=False):
        cluster_rows[key] = group.index.to_numpy()
        by_element.setdefault(key[0], []).append(key)
    rows = []
    for iteration in range(1, N_BOOT + 1):
        pieces = []
        for element, keys in by_element.items():
            selected = RNG.integers(0, len(keys), size=len(keys))
            pieces.extend(cluster_rows[keys[i]] for i in selected)
        idx = np.concatenate(pieces)
        for model, values in pred.items():
            rows.append({"evaluation_group": tier, "iteration": iteration,
                         "model": model, **metrics(y[idx], values[idx])})
    return pd.DataFrame(rows), len(cluster_rows)


print("Loading harmonized external variants...")
data = pd.read_csv(EXTERNAL_FILE, sep="\t", low_memory=False)
require_columns(data, ["validation_tier", "element_name", "position_1based", "robust_z",
                       "reference_sequence", "alternative_sequence", "reference_allele",
                       "alternative_allele"], "External dataset")
data = data.dropna(subset=["robust_z", "reference_sequence", "alternative_sequence"]).copy()
data["reference_sequence"] = data.reference_sequence.astype(str).str.upper()
data["alternative_sequence"] = data.alternative_sequence.astype(str).str.upper()
if data.reference_sequence.str.len().ne(data.alternative_sequence.str.len()).any():
    raise ValueError("Reference and alternative sequence lengths differ")

# Nucleotide Transformer uses an unknown token for ambiguous k-mers. External
# assay targets shorter than 501 bp were padded with N, which can make a true
# SNV pair tokenize identically. Replace ambiguous padding identically in both
# alleles before tokenization; retain the original sequences in all outputs.
data["model_reference_sequence"] = data.reference_sequence.str.replace(
    r"[^ACGT]", "A", regex=True)
data["model_alternative_sequence"] = data.alternative_sequence.str.replace(
    r"[^ACGT]", "A", regex=True)
ambiguous_reference_bases = int(
    data.reference_sequence.str.count(r"[^ACGT]").sum())
ambiguous_alternative_bases = int(
    data.alternative_sequence.str.count(r"[^ACGT]").sum())
print("Ambiguous padding bases replaced for model input:", ambiguous_reference_bases)
if ambiguous_reference_bases != ambiguous_alternative_bases:
    raise ValueError("Reference/alternative ambiguous-base counts do not match")

model_file = locate_locked_model()
print("Locked NT model:", model_file)
bundle = joblib.load(model_file)
required_bundle = {"scaler", "ridge_model", "representation", "alpha", "embedding_dimension"}
if required_bundle.difference(bundle):
    raise KeyError(f"Unexpected locked model structure: {sorted(bundle.keys())}")
if bundle["representation"] != "delta":
    raise ValueError(f"This script requires the locked delta representation, got {bundle['representation']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected; embedding extraction will be slow.")

print("Loading pretrained Nucleotide Transformer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
masked_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
if not hasattr(masked_model, "esm"):
    raise AttributeError("Expected ESM encoder not found in pretrained model")
encoder = masked_model.esm.to(device)
del masked_model
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

print(f"Extracting locked mutation-token representations for {len(data):,} variants...")
embedding_start = time.time()
X_external, changed_distribution = extract_local_delta(data, tokenizer, encoder, device)
embedding_minutes = (time.time() - embedding_start) / 60
if X_external.shape[1] != int(bundle["embedding_dimension"]):
    raise ValueError(
        f"Embedding dimension {X_external.shape[1]} does not match locked model "
        f"dimension {bundle['embedding_dimension']}"
    )
np.save(OUTPUT / "external_nt_delta_embeddings.npy", X_external)

print("Applying locked development-trained NT Ridge model...")
nt_prediction = bundle["ridge_model"].predict(bundle["scaler"].transform(X_external))

print("Loading Step 35 predictions for paired comparison...")
kmer = pd.read_csv(KMER_PREDICTION_FILE, sep="\t", low_memory=False)
if len(kmer) != len(data):
    raise ValueError(f"Step 35/external row mismatch: {len(kmer)} versus {len(data)}")
alignment_columns = ["element_name", "position_1based", "reference_allele", "alternative_allele"]
for column in alignment_columns:
    if not kmer[column].astype(str).reset_index(drop=True).equals(
            data[column].astype(str).reset_index(drop=True)):
        raise ValueError(f"Step 35 alignment failed for {column}")
require_columns(kmer, ["predicted_shared_kmer_ridge", "predicted_mutation_class_baseline"],
                "Step 35 predictions")

observed = data.robust_z.to_numpy(float)
predictions = {
    "frozen_nt_ridge": np.asarray(nt_prediction, float),
    "shared_kmer_ridge": kmer.predicted_shared_kmer_ridge.to_numpy(float),
    "mutation_class_baseline": kmer.predicted_mutation_class_baseline.to_numpy(float),
    "zero_baseline": np.zeros(len(data), dtype=float),
}
points = point_rows(data, observed, predictions)

prediction_table = data.copy()
# The temporary sanitized model-input columns are not biological sequences and
# must not be exported as if they were observed reference alleles.
prediction_table = prediction_table.drop(
    columns=["model_reference_sequence", "model_alternative_sequence"])
prediction_table["observed_score"] = observed
for model, prediction in predictions.items():
    prediction_table[f"predicted_{model}"] = prediction
prediction_table["frozen_nt_residual"] = observed - nt_prediction

print(f"Starting {N_BOOT:,} paired position-clustered bootstrap iterations per tier...")
bootstrap_parts = []
cluster_counts = {}
for tier in TIERS:
    boot, clusters = bootstrap_tier(data, observed, predictions, tier)
    bootstrap_parts.append(boot)
    cluster_counts[tier] = clusters
    print(f"  {tier}: {clusters} position clusters completed")
bootstrap = pd.concat(bootstrap_parts, ignore_index=True)

ci_rows = []
for (tier, model), group in bootstrap.groupby(["evaluation_group", "model"]):
    point = points.query(
        "evaluation_group == @tier and element_name == 'ALL' and model == @model").iloc[0]
    for metric in ["mae", "rmse", "r2", "pearson_r", "spearman_rho", "direction_accuracy"]:
        values = group[metric].dropna().to_numpy(float)
        ci_rows.append({"evaluation_group": tier, "model": model, "metric": metric,
                        "point_estimate": float(point[metric]),
                        "ci_lower_95": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                        "ci_upper_95": float(np.quantile(values, 0.975)) if len(values) else np.nan})
confidence = pd.DataFrame(ci_rows)

comparison_rows = []
for tier, group in bootstrap.groupby("evaluation_group"):
    pivot = group.pivot(index="iteration", columns="model",
                        values=["rmse", "mae", "pearson_r", "spearman_rho"])
    for baseline in ["shared_kmer_ridge", "mutation_class_baseline", "zero_baseline"]:
        definitions = {
            "rmse_reduction": pivot[("rmse", baseline)] - pivot[("rmse", "frozen_nt_ridge")],
            "mae_reduction": pivot[("mae", baseline)] - pivot[("mae", "frozen_nt_ridge")],
            "pearson_gain": pivot[("pearson_r", "frozen_nt_ridge")] - pivot[("pearson_r", baseline)],
            "spearman_gain": pivot[("spearman_rho", "frozen_nt_ridge")] - pivot[("spearman_rho", baseline)],
        }
        for measure, series in definitions.items():
            values = series.dropna().to_numpy(float)
            if len(values):
                mean_value = float(np.mean(values))
                lower, upper = np.quantile(values, [0.025, 0.975]).astype(float)
                p_value = float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0))))
            else:
                mean_value = lower = upper = p_value = np.nan
            comparison_rows.append({"evaluation_group": tier, "model": "frozen_nt_ridge",
                                    "baseline": baseline, "measure": measure,
                                    "mean_difference": mean_value, "ci_lower_95": lower,
                                    "ci_upper_95": upper,
                                    "bootstrap_p_two_sided": p_value})
comparisons = pd.DataFrame(comparison_rows)

points.to_csv(OUTPUT / "frozen_nt_external_point_estimates.tsv", sep="\t", index=False)
prediction_table.to_csv(OUTPUT / "frozen_nt_external_predictions.tsv", sep="\t", index=False)
bootstrap.to_csv(OUTPUT / "frozen_nt_external_bootstrap.tsv", sep="\t", index=False)
confidence.to_csv(OUTPUT / "frozen_nt_external_confidence_intervals.tsv", sep="\t", index=False)
comparisons.to_csv(OUTPUT / "frozen_nt_paired_comparisons.tsv", sep="\t", index=False)
metadata = data[["score_set_urn", "element_name", "position_1based", "reference_allele",
                 "alternative_allele"]].copy()
metadata.insert(0, "embedding_row", np.arange(len(metadata)))
metadata.to_csv(OUTPUT / "external_nt_embedding_metadata.tsv", sep="\t", index=False)

settings = {
    "analysis": "locked frozen Nucleotide Transformer external validation",
    "pretrained_model": MODEL_NAME,
    "locked_ridge_model": str(model_file),
    "locked_representation": bundle["representation"],
    "locked_alpha": float(bundle["alpha"]),
    "embedding_dimension": int(X_external.shape[1]),
    "external_variants": len(data),
    "batch_size": BATCH_SIZE,
    "max_token_length": MAX_TOKEN_LENGTH,
    "changed_token_distribution": changed_distribution,
    "ambiguous_padding_policy": (
        "Non-ACGT padding replaced identically with A in reference and alternative "
        "model inputs to prevent unknown-token collapse; original sequences retained."
    ),
    "ambiguous_reference_bases_replaced": ambiguous_reference_bases,
    "bootstrap_iterations": N_BOOT,
    "bootstrap_seed": SEED,
    "position_clusters": cluster_counts,
    "external_labels_used_for_model_selection_or_training": False,
    "embedding_extraction_minutes": round(embedding_minutes, 3),
    "device": str(device),
    "warning": "External performance is reported without post-hoc score flipping or external tuning."
    }
(OUTPUT / "frozen_nt_external_settings.json").write_text(
    json.dumps(settings, indent=2), encoding="utf-8")

print("\nFROZEN NT EXTERNAL RESULTS")
print(points.query("element_name == 'ALL'").to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPAIRED COMPARISON OF FROZEN NT AGAINST STEP 35")
print(comparisons.query("baseline == 'shared_kmer_ridge'").to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
print("\nStep 36 frozen NT external validation: COMPLETED")
print("Saved:", OUTPUT)
