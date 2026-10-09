import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
ATTR = ROOT / "results/exact_attribution_counterfactual_audit/locked_variant_exact_attributions.tsv"
MOTIF = ROOT / "results/jaspar_motif_disruption_thresholded/thresholded_variant_motif_disruption.tsv"
OUT = ROOT / "results/position_level_motif_conservation_validation"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261007
BOOTSTRAPS = 5000
TOP_FRACTION = 0.10


def benjamini_hochberg(p_values):
    """Dependency-free Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0: return np.nan
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def bootstrap_correlations(position):
    rng = np.random.default_rng(SEED)
    elements = {e:g.index.to_numpy() for e,g in position.groupby("element_id")}
    rows = []
    pairs = {
        "abs_observed_vs_motif": ("max_abs_observed", "max_abs_motif_disruption"),
        "abs_prediction_vs_motif": ("max_abs_prediction", "max_abs_motif_disruption"),
        "abs_context_attribution_vs_motif": ("max_abs_context_attribution", "max_abs_motif_disruption"),
        "abs_observed_vs_phyloP": ("max_abs_observed", "phyloP100way"),
        "abs_observed_vs_phastCons": ("max_abs_observed", "phastCons100way"),
    }
    for iteration in range(1, BOOTSTRAPS+1):
        pieces = []
        for idx in elements.values(): pieces.append(rng.choice(idx, size=len(idx), replace=True))
        take = np.concatenate(pieces); row = {"iteration": iteration}
        for name, (x, y) in pairs.items(): row[name] = safe_spearman(position.loc[take, x], position.loc[take, y])
        rows.append(row)
    return pd.DataFrame(rows), pairs


print("Loading exact attributions and thresholded JASPAR annotations...")
attr = pd.read_csv(ATTR, sep="\t", low_memory=False)
motif = pd.read_csv(MOTIF, sep="\t", low_memory=False)
attr["source_row"] = attr.row_id.astype(str).str.extract(r"(\d+)$")[0].astype(int)
motif["source_row"] = pd.to_numeric(motif.variant_id, errors="raise").astype(int)
if attr.source_row.duplicated().any() or motif.source_row.duplicated().any():
    raise RuntimeError("Variant identifiers are not unique")
data = attr.merge(motif.drop(columns=["element_group", "observed_score", "predicted_score"], errors="ignore"),
                  on="source_row", how="inner", validate="one_to_one")
if len(data) != len(attr): raise RuntimeError(f"Motif/attribution merge retained {len(data)} of {len(attr)} variants")
data["absolute_observed"] = data.target_model.abs()
data["absolute_prediction"] = data.prediction.abs()
gate_columns = [c for c in data if c.startswith("gate_")]
data["context_attribution"] = data.context_additive + data[gate_columns].sum(axis=1)
data["absolute_context_attribution"] = data.context_attribution.abs()

# Independent unit is a tested genomic position, not each alternative allele.
position = (data.groupby(["element_id", "position_group"], as_index=False)
    .agg(variants=("row_id", "size"),
         max_abs_observed=("absolute_observed", "max"),
         mean_abs_observed=("absolute_observed", "mean"),
         max_abs_prediction=("absolute_prediction", "max"),
         max_abs_context_attribution=("absolute_context_attribution", "max"),
         max_abs_motif_disruption=("absolute_affinity_change", "max"),
         mean_abs_motif_disruption=("absolute_affinity_change", "mean"),
         phyloP100way=("phyloP100way", "first"),
         phastCons100way=("phastCons100way", "first")))
cutoff = float(position.max_abs_observed.quantile(1-TOP_FRACTION))
position["top_effect_position"] = position.max_abs_observed >= cutoff

boot, pairs = bootstrap_correlations(position)
association_rows = []
for name, (x, y) in pairs.items():
    values = boot[name].dropna().to_numpy()
    association_rows.append({"association": name, "x": x, "y": y,
        "point_spearman": safe_spearman(position[x], position[y]),
        "ci_lower_95": np.quantile(values, .025), "ci_upper_95": np.quantile(values, .975),
        "bootstrap_sign_p_two_sided": min(1., 2*min(np.mean(values <= 0), np.mean(values >= 0)))})
associations = pd.DataFrame(association_rows)

# Position-level motif enrichment: a motif is present if it is the top disrupted motif
# for any alternate allele at that position. This avoids allele pseudo-replication.
presence = (data.assign(present=1).pivot_table(index=["element_id", "position_group"],
            columns=["top_motif_id", "top_motif_name"], values="present", aggfunc="max", fill_value=0))
presence.columns = [f"{mid}|{name}" for mid,name in presence.columns]
motif_columns = list(presence.columns)
presence = presence.reset_index().merge(position[["element_id", "position_group", "top_effect_position"]],
                                         on=["element_id", "position_group"], validate="one_to_one")
enrichment = []
top = presence.top_effect_position.to_numpy(bool)
for column in motif_columns:
    hit = presence[column].to_numpy(bool)
    a = int(np.sum(top & hit)); b = int(np.sum(top & ~hit))
    c = int(np.sum(~top & hit)); d = int(np.sum(~top & ~hit))
    if a < 5: continue
    odds, p = stats.fisher_exact([[a,b],[c,d]], alternative="greater")
    motif_id, motif_name = column.split("|", 1)
    enrichment.append({"motif_id":motif_id, "motif_name":motif_name,
                       "top_positions_with_motif":a, "top_positions_without_motif":b,
                       "other_positions_with_motif":c, "other_positions_without_motif":d,
                       "odds_ratio":odds, "fisher_p":p})
enrichment = pd.DataFrame(enrichment)
if len(enrichment):
    enrichment["fdr_bh"] = benjamini_hochberg(enrichment.fisher_p.to_numpy())
    enrichment = enrichment.sort_values(["fdr_bh", "fisher_p"])

top_mask = position.top_effect_position
effect_summary = {
    "top_effect_position_cutoff": cutoff,
    "top_effect_positions": int(top_mask.sum()),
    "other_positions": int((~top_mask).sum()),
    "top_median_motif_disruption": float(position.loc[top_mask,"max_abs_motif_disruption"].median()),
    "other_median_motif_disruption": float(position.loc[~top_mask,"max_abs_motif_disruption"].median()),
    "top_median_phyloP": float(position.loc[top_mask,"phyloP100way"].median()),
    "other_median_phyloP": float(position.loc[~top_mask,"phyloP100way"].median()),
    "top_median_phastCons": float(position.loc[top_mask,"phastCons100way"].median()),
    "other_median_phastCons": float(position.loc[~top_mask,"phastCons100way"].median()),
}
summary = {"variants":len(data), "independent_positions":len(position), "elements":position.element_id.nunique(),
           "bootstrap_iterations":BOOTSTRAPS, "top_fraction":TOP_FRACTION, **effect_summary,
           "fdr_significant_position_level_motifs": int((enrichment.fdr_bh < .05).sum()) if len(enrichment) else 0,
           "interpretation_limits":[
               "JASPAR PWM disruption is computational annotation, not direct TF-binding evidence.",
               "Observed-effect versus motif/conservation tests are annotation concordance, not causal validation.",
               "Attribution versus conservation is not independent because conservation was a model input.",
               "eQTL validation is deferred until tissue and gene mappings are prespecified; it is not inferred from proximity."
           ]}

data.to_csv(OUT/"variant_attribution_motif_conservation_merged.tsv", sep="\t", index=False)
position.to_csv(OUT/"position_level_validation_table.tsv", sep="\t", index=False)
boot.to_csv(OUT/"position_clustered_correlation_bootstrap.tsv", sep="\t", index=False)
associations.to_csv(OUT/"motif_conservation_associations.tsv", sep="\t", index=False)
enrichment.to_csv(OUT/"position_level_motif_enrichment.tsv", sep="\t", index=False)
(OUT/"motif_conservation_validation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

print("\nPOSITION-LEVEL MOTIF/CONSERVATION VALIDATION")
print(json.dumps(summary, indent=2))
print("\nASSOCIATIONS")
print(associations.to_string(index=False, float_format=lambda z:f"{z:.4g}"))
print("\nTOP MOTIF ENRICHMENTS")
print(enrichment.head(20).to_string(index=False, float_format=lambda z:f"{z:.4g}") if len(enrichment) else "No motifs met the minimum-hit rule.")
print("\nStep 53A position-level motif/conservation validation: COMPLETED")
print("Saved:", OUT)
