#!/usr/bin/env python3
"""Standalone batch scorer for the locked HIERA-VEP context-gated Ridge model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy import sparse

CCRE_LEVELS=["none","PLS","pELS","dELS","CA-H3K4me3"]
CONTEXT_COLUMNS=["ccre_overlap","promoter_overlap","log1p_abs_tss_distance","phyloP100way",
 "phastCons100way","phyloP_conserved","phyloP_accelerated","phastCons_high",
 "ccre_class_PLS","ccre_class_pELS","ccre_class_dELS","ccre_class_CA-H3K4me3"]
GATES=["ccre_overlap","promoter_overlap","phyloP100way","phastCons100way"]

def as_bool(s):
    if pd.api.types.is_bool_dtype(s): return s.astype(float)
    x=s.astype(str).str.strip().str.lower()
    out=x.map({"true":1.,"false":0.,"1":1.,"0":0.,"yes":1.,"no":0.})
    if out.isna().any(): raise ValueError(f"Invalid boolean values in {s.name}: {sorted(x[out.isna()].unique())}")
    return out

def prepare_context(d):
    required=["ccre_overlap","promoter_overlap","absolute_distance_to_nearest_tss","phyloP100way",
              "phastCons100way","phyloP_conserved","phyloP_accelerated","phastCons_high","primary_ccre_class"]
    missing=[c for c in required if c not in d]
    if missing: raise KeyError(f"Missing context columns: {missing}")
    c=pd.DataFrame(index=d.index)
    for col in ["ccre_overlap","promoter_overlap","phyloP_conserved","phyloP_accelerated","phastCons_high"]:
        c[col]=as_bool(d[col])
    c["log1p_abs_tss_distance"]=np.log1p(pd.to_numeric(d.absolute_distance_to_nearest_tss,errors="coerce"))
    c["phyloP100way"]=pd.to_numeric(d.phyloP100way,errors="coerce")
    c["phastCons100way"]=pd.to_numeric(d.phastCons100way,errors="coerce")
    if d.primary_ccre_class.isna().any(): raise ValueError("Missing context values: primary_ccre_class")
    primary=d.primary_ccre_class.astype(str)
    unknown=sorted(set(primary)-set(CCRE_LEVELS))
    if unknown: raise ValueError(f"Unsupported cCRE classes: {unknown}")
    for level in CCRE_LEVELS[1:]: c[f"ccre_class_{level}"]=primary.eq(level).astype(float)
    c=c[CONTEXT_COLUMNS]
    if not np.isfinite(c.to_numpy()).all(): raise ValueError("Missing or non-finite context values")
    if (pd.to_numeric(d.absolute_distance_to_nearest_tss,errors="coerce")<0).any(): raise ValueError("TSS distance must be nonnegative")
    if c.isna().any().any(): raise ValueError(f"Missing context values: {c.isna().sum()[c.isna().sum()>0].to_dict()}")
    return c.astype(np.float32)

def validate_sequences(d):
    for c in ["reference_sequence","alternative_sequence"]:
        if c not in d: raise KeyError(f"Missing sequence column: {c}")
    ref=d.reference_sequence.fillna("").astype(str).str.upper()
    alt=d.alternative_sequence.fillna("").astype(str).str.upper()
    bad_alpha=~ref.str.fullmatch("[ACGTN]+")|~alt.str.fullmatch("[ACGTN]+")
    unequal=ref.str.len()!=alt.str.len()
    changes=np.array([sum(a!=b for a,b in zip(r,t)) for r,t in zip(ref,alt)])
    bad=bad_alpha|unequal|(changes!=1)
    if bad.any(): raise ValueError(f"{int(bad.sum())} rows are not aligned single-nucleotide substitutions")
    return ref.tolist(),alt.tolist()

def build_design(d,vectorizer,spec):
    if spec["context_columns"] != CONTEXT_COLUMNS or spec["gating_columns"] != GATES:
        raise ValueError("Preprocessing feature order does not match scorer")
    expected_order=["delta_kmer_5440","context_12"]+[f"delta_kmer_5440_x_{g}" for g in GATES]
    if spec["feature_order"] != expected_order or spec["total_features"] != 27212:
        raise ValueError("Invalid saved feature specification")
    ref,alt=validate_sequences(d)
    xseq=(vectorizer.transform(alt)-vectorizer.transform(ref)).astype(np.float32).tocsr()
    c=prepare_context(d).to_numpy(np.float32)
    for i,name in enumerate(spec["continuous_context_columns"]):
        j=CONTEXT_COLUMNS.index(name)
        c[:,j]=(c[:,j]-np.float32(spec["continuous_mean"][i]))/np.float32(spec["continuous_sd"][i])
    cs=sparse.csr_matrix(c)
    blocks=[xseq,cs]
    for name in GATES:
        j=CONTEXT_COLUMNS.index(name); blocks.append(xseq.multiply(cs[:,j].toarray()))
    return sparse.hstack(blocks,format="csr"),xseq,cs

def score_frame(d,model,vectorizer,spec):
    X,xseq,cs=build_design(d,vectorizer,spec)
    if X.shape[1]!=model.n_features_in_: raise ValueError(f"Feature dimension {X.shape[1]} != model {model.n_features_in_}")
    coef=np.asarray(model.coef_); k=xseq.shape[1]; q=len(CONTEXT_COLUMNS)
    out=d.copy(); out["hiera_vep_prediction"]=model.predict(X)
    out["contribution_intercept"]=float(model.intercept_)
    out["contribution_sequence_main"]=np.asarray(xseq.multiply(coef[:k]).sum(axis=1)).ravel()
    out["contribution_context_additive"]=np.asarray(cs.multiply(coef[k:k+q]).sum(axis=1)).ravel()
    start=k+q
    for gate in GATES:
        j=CONTEXT_COLUMNS.index(gate); block=xseq.multiply(cs[:,j].toarray())
        out[f"contribution_gate_{gate}"]=np.asarray(block.multiply(coef[start:start+k]).sum(axis=1)).ravel(); start+=k
    pieces=["contribution_intercept","contribution_sequence_main","contribution_context_additive"]+[f"contribution_gate_{g}" for g in GATES]
    out["reconstructed_prediction"]=out[pieces].sum(axis=1)
    out["reconstruction_error"]=(out.hiera_vep_prediction-out.reconstructed_prediction).abs()
    return out

def find_asset(bundle,name):
    hits=list(bundle.rglob(name))
    if len(hits)!=1: raise FileNotFoundError(f"Expected exactly one {name} under {bundle}; found {len(hits)}")
    return hits[0]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle",type=Path,required=True); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); model=joblib.load(find_asset(a.bundle,"context_selected_ridge.joblib")); vec=joblib.load(find_asset(a.bundle,"context_model_vectorizer.joblib")); spec=json.loads(find_asset(a.bundle,"preprocessing_spec.json").read_text())
    d=pd.read_csv(a.input,sep="\t",low_memory=False); out=score_frame(d,model,vec,spec);
    if len(out)==0 or not np.isfinite(out.hiera_vep_prediction).all() or out.reconstruction_error.max()>=1e-5: raise ValueError("Inference/additivity validation failed")
    a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,sep="\t",index=False)
    print(f"Scored {len(out):,} variants; maximum additivity error={out.reconstruction_error.max():.3e}"); print("Saved:",a.output)
if __name__=="__main__": main()
