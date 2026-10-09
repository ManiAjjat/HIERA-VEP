#!/usr/bin/env python3
"""Step 55: locked calibration, uncertainty-proxy, and failure analysis.

Calibration is diagnostic only and is never fitted back onto locked predictions.
Uncertainty is represented by disagreement among three already-locked models;
it is not described as a calibrated predictive interval.
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--project",type=Path,default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2])))
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=5501)
    return p.parse_args()


def first_existing(paths):
    for p in paths:
        if p.exists(): return p
    raise FileNotFoundError("Expected file not found:\n"+"\n".join(map(str,paths)))


def safe_spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])==0 or np.std(y[ok])==0: return np.nan
    return float(stats.spearmanr(x[ok],y[ok]).statistic)


def safe_pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])==0 or np.std(y[ok])==0: return np.nan
    return float(stats.pearsonr(x[ok],y[ok]).statistic)


def calibration(y,p):
    x=np.asarray(p,float); y=np.asarray(y,float)
    slope=np.cov(x,y,ddof=1)[0,1]/np.var(x,ddof=1)
    intercept=float(np.mean(y)-slope*np.mean(x))
    return intercept,float(slope)


def metric_row(y,p):
    return {"n":len(y),"mae":mean_absolute_error(y,p),
            "rmse":mean_squared_error(y,p)**.5,"r2":r2_score(y,p),
            "pearson_r":safe_pearson(y,p),"spearman_rho":safe_spearman(y,p),
            "direction_accuracy":float(np.mean(np.sign(y)==np.sign(p))),
            "observed_sd":float(np.std(y,ddof=1)),"predicted_sd":float(np.std(p,ddof=1))}


def sign_p(x,null=0):
    x=np.asarray(x,float); x=x[np.isfinite(x)]-null
    if not len(x): return np.nan
    return float(min(1,2*min((np.sum(x<=0)+1)/(len(x)+1),(np.sum(x>=0)+1)/(len(x)+1))))


def main():
    a=parse_args(); root=a.project
    pred_path=first_existing([
        root/"results"/"fair_genomic_language_model_comparison"/"locked_model_predictions.tsv",
        root/"results"/"fair_genomic_language_model_comparison"/"locked_predictions.tsv"])
    attr_path=root/"results"/"exact_attribution_counterfactual_audit"/"locked_variant_exact_attributions.tsv"
    if not attr_path.exists(): raise FileNotFoundError(attr_path)
    out=root/"results"/"calibration_uncertainty_failure_analysis"; out.mkdir(parents=True,exist_ok=True)
    pr=pd.read_csv(pred_path,sep="\t",low_memory=False)
    at=pd.read_csv(attr_path,sep="\t",low_memory=False)
    required=["row_id","element_id","position_group","target_model","context_selected_prediction",
              "sequence_only_prediction","frozen_nt_prediction"]
    missing=[c for c in required if c not in pr]
    if missing: raise KeyError(f"Step 51 prediction columns missing: {missing}; available={list(pr.columns)}")
    keep=[c for c in at.columns if c not in pr.columns or c=="row_id"]
    d=pr.merge(at[keep],on="row_id",how="left",validate="one_to_one")
    if d["prediction"].isna().any(): raise ValueError("Step 52 attribution merge was incomplete.")
    if np.max(np.abs(d.context_selected_prediction-d.prediction))>1e-5:
        raise ValueError("Step 51 and Step 52 locked context predictions disagree.")
    numeric=["target_model","context_selected_prediction","sequence_only_prediction","frozen_nt_prediction"]
    for c in numeric: d[c]=pd.to_numeric(d[c],errors="coerce")
    if d[numeric].isna().any().any(): raise ValueError("Missing locked prediction values.")

    y=d.target_model.to_numpy(float); p=d.context_selected_prediction.to_numpy(float)
    d["residual"]=y-p; d["absolute_error"]=np.abs(d.residual)
    models=d[["context_selected_prediction","sequence_only_prediction","frozen_nt_prediction"]].to_numpy(float)
    d["model_disagreement_sd"]=np.std(models,axis=1,ddof=1)
    d["model_disagreement_range"]=np.ptp(models,axis=1)
    cutoff=float(d.absolute_error.quantile(.90)); d["high_error_top10"]=d.absolute_error>=cutoff
    intercept,slope=calibration(y,p)

    # Equal-frequency calibration bins.
    d["prediction_decile"]=pd.qcut(d.context_selected_prediction,10,labels=False,duplicates="drop")
    cal=[]
    for b,s in d.groupby("prediction_decile"):
        cal.append({"prediction_decile":int(b)+1,"n":len(s),
                    "mean_prediction":s.context_selected_prediction.mean(),
                    "mean_observed":s.target_model.mean(),"observed_sd":s.target_model.std(ddof=1),
                    "rmse":mean_squared_error(s.target_model,s.context_selected_prediction)**.5})
    pd.DataFrame(cal).to_csv(out/"calibration_deciles.tsv",sep="\t",index=False)

    # Selective-risk curve: lower-disagreement predictions are retained first.
    sel=[]
    ordered=np.argsort(d.model_disagreement_sd.to_numpy())
    for frac in np.arange(.1,1.01,.1):
        n=max(1,int(np.ceil(frac*len(d)))); ix=ordered[:n]
        row={"retained_fraction":float(frac),"n":n,
             "maximum_disagreement_retained":float(d.model_disagreement_sd.iloc[ix].max()),
             **metric_row(y[ix],p[ix])}
        sel.append(row)
    pd.DataFrame(sel).to_csv(out/"uncertainty_selective_risk_curve.tsv",sep="\t",index=False)

    # Prespecified error strata.
    d["absolute_effect_bin"]=pd.cut(np.abs(y),[-np.inf,.5,1,2,np.inf],
                                    labels=["<=0.5","0.5-1","1-2",">2"])
    d["phyloP_bin"]=pd.qcut(pd.to_numeric(d.phyloP100way,errors="coerce"),4,
                             labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
    d["phastCons_bin"]=pd.cut(pd.to_numeric(d.phastCons100way,errors="coerce"),
                               [-np.inf,0,.5,.9,np.inf],labels=["0","0-0.5","0.5-0.9",">0.9"])
    strata=[]
    for variable in ["element_id","primary_ccre_class","promoter_overlap","absolute_effect_bin","phyloP_bin","phastCons_bin"]:
        for level,s in d.groupby(variable,observed=True,dropna=False):
            row={"stratification":variable,"level":str(level),**metric_row(s.target_model.to_numpy(),s.context_selected_prediction.to_numpy())}
            row["mean_residual"]=s.residual.mean(); row["median_absolute_error"]=s.absolute_error.median()
            row["high_error_fraction"]=s.high_error_top10.mean(); strata.append(row)
    strata=pd.DataFrame(strata); strata.to_csv(out/"structured_failure_strata.tsv",sep="\t",index=False)

    # Clustered bootstrap by positions within elements.
    d["__cluster"]=d.element_id.astype(str)+"|"+d.position_group.astype(str)
    rows={k:np.asarray(v,int) for k,v in d.groupby("__cluster",sort=False).indices.items()}
    by_element={}
    for k in rows: by_element.setdefault(k.split("|",1)[0],[]).append(k)
    rng=np.random.default_rng(a.seed)
    boot=[]
    print(f"Variants: {len(d):,}; elements: {d.element_id.nunique()}; position clusters: {len(rows):,}")
    print(f"Starting {a.bootstrap:,} element-stratified position-clustered bootstrap iterations...")
    for it in range(a.bootstrap):
        parts=[]
        for keys in by_element.values():
            draw=rng.integers(0,len(keys),len(keys)); parts.extend(rows[keys[j]] for j in draw)
        ix=np.concatenate(parts); yy=y[ix]; pp=p[ix]; uu=d.model_disagreement_sd.to_numpy()[ix]
        ci,cs=calibration(yy,pp)
        he=d.high_error_top10.to_numpy()[ix]
        auc=roc_auc_score(he,uu) if np.unique(he).size==2 else np.nan
        boot.append({"iteration":it+1,"calibration_intercept":ci,"calibration_slope":cs,
                     "mean_residual":float(np.mean(yy-pp)),
                     "disagreement_vs_absolute_error_spearman":safe_spearman(uu,np.abs(yy-pp)),
                     "disagreement_high_error_auc":auc})
        if (it+1)%500==0 or it+1==a.bootstrap: print(f"  Bootstrap {it+1:,}/{a.bootstrap:,}")
    boot=pd.DataFrame(boot); boot.to_csv(out/"clustered_calibration_uncertainty_bootstrap.tsv",sep="\t",index=False)
    estimates=[]
    point={"calibration_intercept":intercept,"calibration_slope":slope,
           "mean_residual":float(np.mean(y-p)),
           "disagreement_vs_absolute_error_spearman":safe_spearman(d.model_disagreement_sd,d.absolute_error),
           "disagreement_high_error_auc":float(roc_auc_score(d.high_error_top10,d.model_disagreement_sd))}
    nulls={"calibration_intercept":0,"calibration_slope":1,"mean_residual":0,
           "disagreement_vs_absolute_error_spearman":0,"disagreement_high_error_auc":.5}
    for measure,value in point.items():
        z=boot[measure].dropna().to_numpy()
        estimates.append({"measure":measure,"point_estimate":value,
                          "ci_lower_95":np.quantile(z,.025),"ci_upper_95":np.quantile(z,.975),
                          "bootstrap_p_two_sided_vs_null":sign_p(z,nulls[measure]),"null_value":nulls[measure]})
    pd.DataFrame(estimates).to_csv(out/"calibration_uncertainty_estimates.tsv",sep="\t",index=False)

    # Preserve the largest errors for biological/manual audit.
    failure_cols=[c for c in ["row_id","element_id","assay_id","position_group","reference_allele",
                  "alternative_allele","target_model","context_selected_prediction","sequence_only_prediction",
                  "frozen_nt_prediction","residual","absolute_error","model_disagreement_sd","primary_ccre_class",
                  "promoter_overlap","phyloP100way","phastCons100way","sequence_main","context_additive",
                  "gate_ccre_overlap","gate_promoter_overlap","gate_phyloP100way","gate_phastCons100way"] if c in d]
    d.nlargest(100,"absolute_error")[failure_cols].to_csv(out/"top_100_locked_failures.tsv",sep="\t",index=False)
    d.to_csv(out/"locked_prediction_uncertainty_diagnostics.tsv",sep="\t",index=False)

    full_metrics=metric_row(y,p)
    summary={"variants":len(d),"elements":int(d.element_id.nunique()),"position_clusters":int(d.__cluster.nunique()),
             "locked_metrics":full_metrics,"calibration_intercept":intercept,"calibration_slope":slope,
             "mean_residual":float(np.mean(y-p)),"median_residual":float(np.median(y-p)),
             "high_error_cutoff_absolute_error":cutoff,
             "disagreement_vs_absolute_error_spearman":point["disagreement_vs_absolute_error_spearman"],
             "disagreement_high_error_auc":point["disagreement_high_error_auc"],
             "limitations":["Calibration estimates are locked-test diagnostics and are not used to recalibrate predictions.",
                            "Model disagreement is an uncertainty proxy, not a calibrated predictive interval.",
                            "Failure-stratum results are descriptive and should not be treated as independent confirmatory tests."]}
    (out/"calibration_uncertainty_failure_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nCALIBRATION AND UNCERTAINTY SUMMARY")
    print(json.dumps(summary,indent=2))
    print("\nWORST ERROR STRATA (minimum n=30)")
    print(strata[strata.n>=30].nlargest(15,"rmse")[["stratification","level","n","mae","rmse","mean_residual","high_error_fraction"]].to_string(index=False))
    print("\nStep 55 calibration, uncertainty and failure analysis: COMPLETED")
    print("Saved:",out)


if __name__=="__main__": main()
