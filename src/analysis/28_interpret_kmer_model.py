import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
PRED_FILE = ROOT / "results/within_element_kmer_baseline/within_element_test_predictions.tsv"
MODEL_FILE = ROOT / "results/within_element_kmer_baseline/within_element_kmer_models.joblib"
OUT = ROOT / "results/kmer_interpretation"
OUT.mkdir(parents=True, exist_ok=True)

def corr(fn, y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(fn(y, p).statistic)

def metrics(d):
    y = d.observed_score.to_numpy(float)
    p = d.predicted_score.to_numpy(float)
    residual = y - p
    denom = np.sum((y-y.mean())**2)
    return pd.Series({"n":len(d), "mae":np.mean(np.abs(residual)),
                      "rmse":np.sqrt(np.mean(residual**2)),
                      "r2":1-np.sum(residual**2)/denom if denom else np.nan,
                      "pearson_r":corr(pearsonr,y,p),
                      "spearman_rho":corr(spearmanr,y,p)})

print("Loading k-mer predictions and model...")
pred = pd.read_csv(PRED_FILE, sep="\t")
bundle = joblib.load(MODEL_FILE)
required = {"mutation_class", "element_group", "observed_score", "predicted_score"}
if required - set(pred.columns):
    raise ValueError(f"Missing prediction columns: {sorted(required-set(pred.columns))}")
if not {"vectorizer", "models"}.issubset(bundle):
    raise ValueError("Unexpected model-bundle structure")

vectorizer = bundle["vectorizer"]
models = bundle["models"]
features = np.asarray(vectorizer.get_feature_names_out())
print("Test variants:", len(pred))
print("Elements:", pred.element_group.nunique())
print("K-mer features:", len(features))

# Performance across the 12 directional SNV classes.
mutation_metrics = (pred.groupby("mutation_class", sort=True, observed=True)
                       .apply(metrics, include_groups=False).reset_index())
mutation_metrics.to_csv(OUT / "mutation_class_performance.tsv", sep="\t", index=False)

# Mean observed and predicted effects for each mutation class and element.
mutation_effects = (pred.groupby(["element_group", "mutation_class"], observed=True)
                    .agg(n=("observed_score","size"),
                         mean_observed=("observed_score","mean"),
                         mean_predicted=("predicted_score","mean"),
                         median_observed=("observed_score","median"),
                         median_predicted=("predicted_score","median"))
                    .reset_index())
mutation_effects.to_csv(OUT / "element_mutation_class_effects.tsv", sep="\t", index=False)

# Extract element-specific Ridge coefficients using the shared vocabulary.
coefficient_rows = []
for element, model in models.items():
    coef = np.asarray(model.coef_).reshape(-1)
    if len(coef) != len(features):
        raise ValueError(f"Coefficient length mismatch for {element}")
    coefficient_rows.extend({"element_group":element, "kmer":kmer,
                             "kmer_length":len(kmer), "coefficient":float(value)}
                            for kmer, value in zip(features, coef))
coefficients = pd.DataFrame(coefficient_rows)
coefficients.to_csv(OUT / "element_specific_kmer_coefficients.tsv", sep="\t", index=False)

# Cross-element coefficient consistency. These are model features, not causal motifs.
consensus = (coefficients.groupby("kmer")
             .agg(kmer_length=("kmer_length","first"),
                  mean_coefficient=("coefficient","mean"),
                  median_coefficient=("coefficient","median"),
                  sd_coefficient=("coefficient","std"),
                  positive_elements=("coefficient",lambda x:int((x>0).sum())),
                  negative_elements=("coefficient",lambda x:int((x<0).sum())),
                  mean_absolute_coefficient=("coefficient",lambda x:float(np.mean(np.abs(x)))))
             .reset_index())
consensus["sign_consistency"] = consensus[["positive_elements","negative_elements"]].max(axis=1) / len(models)
consensus = consensus.sort_values(["sign_consistency","mean_absolute_coefficient"], ascending=False)
consensus.to_csv(OUT / "cross_element_kmer_consistency.tsv", sep="\t", index=False)

top_positive = coefficients.sort_values("coefficient", ascending=False).groupby("element_group").head(25)
top_negative = coefficients.sort_values("coefficient").groupby("element_group").head(25)
top_features = pd.concat([top_positive.assign(direction="positive"),
                          top_negative.assign(direction="negative")], ignore_index=True)
top_features.to_csv(OUT / "top_element_specific_kmer_features.tsv", sep="\t", index=False)

# Test-set variants prioritized by predicted effect magnitude.
prioritized = pred.copy()
prioritized["predicted_absolute_effect"] = prioritized.predicted_score.abs()
prioritized["observed_absolute_effect"] = prioritized.observed_score.abs()
prioritized["prediction_direction_correct"] = np.sign(prioritized.predicted_score) == np.sign(prioritized.observed_score)
prioritized = prioritized.sort_values("predicted_absolute_effect", ascending=False)
prioritized.to_csv(OUT / "test_variants_ranked_by_predicted_effect.tsv", sep="\t", index=False)
prioritized.groupby("element_group").head(20).to_csv(
    OUT / "top20_predicted_variants_per_element.tsv", sep="\t", index=False)

settings = {"interpretation_model":"element_specific_kmer_ridge",
            "test_variants":len(pred), "elements":len(models), "features":len(features),
            "warning":"K-mer coefficients are predictive associations, not causal motif effects."}
(OUT / "interpretation_settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")

print("\nMUTATION-CLASS PERFORMANCE")
print(mutation_metrics.round(4).to_string(index=False))
print("\nTop cross-element consistent features:")
print(consensus.head(20).round(5).to_string(index=False))
print("\nStep 28 k-mer interpretation: COMPLETED")
print("Saved:", OUT)
