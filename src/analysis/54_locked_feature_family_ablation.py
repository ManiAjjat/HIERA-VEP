#!/usr/bin/env python3
"""Step 54: locked feature-family and interaction ablation for HIERA-VEP.

Uses the exact additive decomposition produced in Step 52. No parameters are
refit and the locked test outcomes do not influence feature selection. Position
clusters are resampled within elements. Multiplicity is controlled across the
prespecified RMSE ablations using Holm's procedure.
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BLOCKS = [
    "sequence_main", "context_additive", "gate_ccre_overlap",
    "gate_promoter_overlap", "gate_phyloP100way", "gate_phastCons100way",
]
CONTEXT_BLOCKS = BLOCKS[1:]
GATE_BLOCKS = BLOCKS[2:]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2])))
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=5401)
    return p.parse_args()


def choose(df, names, label):
    lut = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lut:
            return lut[n.lower()]
    raise KeyError(f"No {label} column. Available: {list(df.columns)}")


def finite_corr(fn, y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(fn(y, p).statistic)


def metrics(y, p):
    return {
        "n": len(y),
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "r2": r2_score(y, p),
        "pearson_r": finite_corr(stats.pearsonr, y, p),
        "spearman_rho": finite_corr(stats.spearmanr, y, p),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_sd": float(np.std(p, ddof=1)),
    }


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if not len(vals): return out
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = np.maximum.accumulate((len(ranked)-np.arange(len(ranked))) * ranked)
    tmp = np.empty(len(ranked)); tmp[order] = np.minimum(adjusted, 1.0)
    out[ok] = tmp
    return out


def two_sided_sign_p(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return np.nan
    return float(min(1.0, 2 * min((np.sum(x <= 0)+1)/(len(x)+1),
                                  (np.sum(x >= 0)+1)/(len(x)+1))))


def main():
    a = parse_args(); root = a.project
    src = root/"results"/"exact_attribution_counterfactual_audit"/"locked_variant_exact_attributions.tsv"
    if not src.exists():
        raise FileNotFoundError(f"Missing Step 52 attribution table: {src}")
    out = root/"results"/"locked_feature_family_ablation"; out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(src, sep="\t", low_memory=False)
    for c in ["intercept", "prediction", *BLOCKS]:
        if c not in d:
            raise KeyError(f"Required Step 52 column missing: {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    target = choose(d, ["target_model", "observed", "observed_effect", "y_true"], "observed effect")
    element = choose(d, ["element_id", "element_group", "element_name"], "element")
    position = choose(d, ["position_group", "position_group_id", "position_key"], "position group")
    d["observed_effect"] = pd.to_numeric(d[target], errors="coerce")
    if d[["observed_effect", "prediction", "intercept", *BLOCKS]].isna().any().any():
        raise ValueError("Missing numeric values in locked effects, predictions, or attribution blocks.")

    exact = d["intercept"] + d[BLOCKS].sum(axis=1)
    add_err = float(np.max(np.abs(exact-d.prediction)))
    if add_err > 1e-5:
        raise ValueError(f"Exact attribution reconstruction failed; maximum error={add_err}")

    pred = {"full_locked_model": d.prediction.to_numpy(float)}
    for b in BLOCKS:
        pred[f"without_{b}"] = (d.prediction-d[b]).to_numpy(float)
    pred["without_all_context"] = (d.prediction-d[CONTEXT_BLOCKS].sum(axis=1)).to_numpy(float)
    pred["without_all_gates"] = (d.prediction-d[GATE_BLOCKS].sum(axis=1)).to_numpy(float)
    pred["without_conservation_gates"] = (d.prediction-d[["gate_phyloP100way","gate_phastCons100way"]].sum(axis=1)).to_numpy(float)
    pred["without_regulatory_class_gates"] = (d.prediction-d[["gate_ccre_overlap","gate_promoter_overlap"]].sum(axis=1)).to_numpy(float)
    pred["context_without_sequence"] = (d.prediction-d.sequence_main).to_numpy(float)
    pred["intercept_only"] = d.intercept.to_numpy(float)

    y = d.observed_effect.to_numpy(float)
    point = []
    for name, values in pred.items():
        row = {"model": name, **metrics(y, values)}
        if name != "full_locked_model":
            row["mean_absolute_prediction_change"] = float(np.mean(np.abs(values-pred["full_locked_model"])))
        point.append(row)
    point = pd.DataFrame(point)
    point.to_csv(out/"locked_ablation_point_estimates.tsv", sep="\t", index=False)

    # Position clusters nested within elements preserve the three-allele structure.
    d["__cluster"] = d[element].astype(str) + "|" + d[position].astype(str)
    cluster_rows = {k: np.asarray(v, int) for k, v in d.groupby("__cluster", sort=False).indices.items()}
    element_clusters = {}
    for key in cluster_rows:
        el = key.split("|", 1)[0]
        element_clusters.setdefault(el, []).append(key)
    print(f"Locked variants: {len(d):,}; elements: {len(element_clusters)}; position clusters: {len(cluster_rows):,}")
    print(f"Exact reconstruction maximum error: {add_err:.3e}")
    print(f"Starting {a.bootstrap:,} element-stratified position-clustered bootstrap iterations...")
    rng = np.random.default_rng(a.seed)
    comparisons = [x for x in pred if x != "full_locked_model"]
    boot = {x: {"rmse_loss": [], "mae_loss": [], "pearson_loss": [], "spearman_loss": []} for x in comparisons}
    full = pred["full_locked_model"]
    for it in range(a.bootstrap):
        sampled = []
        for keys in element_clusters.values():
            draws = rng.integers(0, len(keys), len(keys))
            sampled.extend(cluster_rows[keys[j]] for j in draws)
        ix = np.concatenate(sampled)
        yy = y[ix]; pf = full[ix]
        mf = metrics(yy, pf)
        for name in comparisons:
            ma = metrics(yy, pred[name][ix])
            boot[name]["rmse_loss"].append(ma["rmse"]-mf["rmse"])
            boot[name]["mae_loss"].append(ma["mae"]-mf["mae"])
            boot[name]["pearson_loss"].append(mf["pearson_r"]-ma["pearson_r"])
            boot[name]["spearman_loss"].append(mf["spearman_rho"]-ma["spearman_rho"])
        if (it+1) % 500 == 0 or it+1 == a.bootstrap:
            print(f"  Bootstrap {it+1:,}/{a.bootstrap:,}")

    rows = []
    distributions = []
    for name in comparisons:
        for measure, vals in boot[name].items():
            z = np.asarray(vals, float)
            rows.append({
                "ablated_model": name, "measure": measure,
                "point_estimate": float(np.nanmean(z)),
                "ci_lower_95": float(np.nanquantile(z, .025)),
                "ci_upper_95": float(np.nanquantile(z, .975)),
                "bootstrap_p_two_sided": two_sided_sign_p(z),
            })
            distributions.extend({"iteration": i+1, "ablated_model": name,
                                  "measure": measure, "value": v} for i, v in enumerate(z))
    comp = pd.DataFrame(rows)
    primary = comp.measure.eq("rmse_loss")
    comp["holm_p_across_primary_rmse_ablations"] = np.nan
    comp.loc[primary, "holm_p_across_primary_rmse_ablations"] = holm(comp.loc[primary,"bootstrap_p_two_sided"])
    comp.to_csv(out/"locked_ablation_paired_bootstrap.tsv", sep="\t", index=False)
    pd.DataFrame(distributions).to_csv(out/"locked_ablation_bootstrap_distributions.tsv", sep="\t", index=False)

    # Element-level effect consistency; descriptive because several elements are small.
    element_rows = []
    for el, sub in d.groupby(element):
        ix = sub.index.to_numpy()
        mf = metrics(y[ix], full[ix])
        for name in comparisons:
            ma = metrics(y[ix], pred[name][ix])
            element_rows.append({"element_id": el, "ablated_model": name, "n": len(ix),
                                 "rmse_loss": ma["rmse"]-mf["rmse"],
                                 "pearson_loss": mf["pearson_r"]-ma["pearson_r"]})
    element_df = pd.DataFrame(element_rows)
    element_df.to_csv(out/"element_level_ablation_consistency.tsv", sep="\t", index=False)

    primary_table = comp[primary].sort_values("point_estimate", ascending=False)
    context_row = primary_table[primary_table.ablated_model.eq("without_all_context")].iloc[0]
    summary = {
        "variants": int(len(d)), "elements": int(d[element].nunique()),
        "position_clusters": int(d["__cluster"].nunique()), "bootstrap_iterations": a.bootstrap,
        "exact_additivity_max_absolute_error": add_err,
        "prespecified_primary_ablation": "without_all_context",
        "without_all_context_rmse_loss": float(context_row.point_estimate),
        "without_all_context_ci_95": [float(context_row.ci_lower_95), float(context_row.ci_upper_95)],
        "without_all_context_holm_p": float(context_row.holm_p_across_primary_rmse_ablations),
        "interpretation": "Positive RMSE loss means the locked full model performs better than the ablated prediction.",
        "warning": "These are locked-model functional ablations, not refitted-model comparisons and not causal biological effects."
    }
    (out/"locked_feature_family_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nLOCKED FEATURE-FAMILY ABLATION")
    print(point.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPRIMARY RMSE LOSSES")
    print(primary_table.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("\nStep 54 locked feature-family ablation: COMPLETED")
    print("Saved:", out)


if __name__ == "__main__":
    main()
