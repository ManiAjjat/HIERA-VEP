import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#!/usr/bin/env python3
"""Derive preprocessing state, verify locked equivalence, and finalize release."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
import joblib,numpy as np,pandas as pd
from src.hiera_vep_score import prepare_context,score_frame,CONTEXT_COLUMNS,GATES

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))); ap.add_argument("--release-name",default="HIERA-VEP_reproducibility_release_v1"); ap.add_argument("--tolerance",type=float,default=1e-5); a=ap.parse_args()
    root=a.project.resolve(); model_dir=root/"results/context_aware_interpretable_model"; release=root/"release"/a.release_name
    table=root/"results/hiera_unified_training_table/hiera_unified_variant_table.tsv"; annot=root/"results/human_conservation_annotation/development_variant_context_conservation_annotations.tsv"
    d=pd.read_csv(table,sep="\t",low_memory=False); d=d[d.source_partition.eq("development")].reset_index(drop=True)
    an=pd.read_csv(annot,sep="\t",low_memory=False).sort_values("source_row").reset_index(drop=True)
    if len(d)!=len(an) or not np.array_equal(an.source_row.to_numpy(),np.arange(len(d))): raise RuntimeError("Step 44/49 alignment failed")
    if not np.array_equal(d.element_id.astype(str),an.element_id.astype(str)): raise RuntimeError("Element alignment failed")
    raw=prepare_context(an); split=d.original_split.astype(str).str.lower(); pool=np.flatnonzero(split.isin(["train","validation","val"])); locked=np.flatnonzero(split.eq("test"))
    continuous=["log1p_abs_tss_distance","phyloP100way","phastCons100way"]
    arr=raw.iloc[pool][continuous].to_numpy(np.float32); mean=arr.mean(axis=0); sd=arr.std(axis=0); sd[sd==0]=1
    vec=joblib.load(model_dir/"context_model_vectorizer.joblib"); model=joblib.load(model_dir/"context_selected_ridge.joblib")
    spec={"schema_version":"1.0","sequence_operation":"CountVectorizer alternative minus reference",
          "vectorizer":{"analyzer":"char","ngram_range":[3,6],"lowercase":False,"dtype":"float32","vocabulary_size":len(vec.vocabulary_)},
          "context_columns":CONTEXT_COLUMNS,"continuous_context_columns":continuous,"continuous_mean":[float(x) for x in mean],"continuous_sd":[float(x) for x in sd],
          "gating_columns":GATES,"feature_order":["delta_kmer_5440","context_12"]+[f"delta_kmer_5440_x_{x}" for x in GATES],"total_features":int(model.n_features_in_),
          "scaling_fit_partition":"development train+validation pool only","locked_test_used_for_scaling":False}
    (model_dir/"preprocessing_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    input_cols=["row_id","reference_sequence","alternative_sequence"]
    inp=d.iloc[locked][input_cols].reset_index(drop=True)
    for c in ["ccre_overlap","promoter_overlap","absolute_distance_to_nearest_tss","phyloP100way","phastCons100way","phyloP_conserved","phyloP_accelerated","phastCons_high","primary_ccre_class"]: inp[c]=an.iloc[locked][c].reset_index(drop=True)
    scored=score_frame(inp,model,vec,spec)
    saved=pd.read_csv(model_dir/"locked_context_predictions.tsv",sep="\t",low_memory=False)
    check=scored[["row_id","hiera_vep_prediction"]].merge(saved[["row_id","context_selected_prediction"]],on="row_id",validate="one_to_one")
    diff=np.abs(check.hiera_vep_prediction-check.context_selected_prediction); maxdiff=float(diff.max()); add=float(scored.reconstruction_error.max())
    passed=maxdiff<=a.tolerance and add<=a.tolerance
    smoke={"rows":len(check),"prediction_max_absolute_difference":maxdiff,"additivity_max_absolute_error":add,"tolerance":a.tolerance,"status":"PASS" if passed else "FAIL"}
    (model_dir/"locked_inference_smoke_test.json").write_text(json.dumps(smoke,indent=2),encoding="utf-8")
    scored.head(12).drop(columns=[c for c in scored.columns if c.startswith("contribution_") or c in ["hiera_vep_prediction","reconstructed_prediction","reconstruction_error"]]).to_csv(model_dir/"example_scoring_input.tsv",sep="\t",index=False)
    if not passed: raise RuntimeError(f"Deployment blocked: {smoke}")
    if not release.exists(): raise FileNotFoundError("Run Step 56 before Step 56B")
    target=release/"model/context_aware_interpretable_model"; target.mkdir(parents=True,exist_ok=True)
    for name in ["context_model_vectorizer.joblib","context_selected_ridge.joblib","preprocessing_spec.json","locked_inference_smoke_test.json","example_scoring_input.tsv"]: shutil.copy2(model_dir/name,target/name)
    shutil.copy2(root/"hiera_vep_score.py",release/"hiera_vep_score.py")
    audit_path=release/"metadata/release_readiness_audit.json"; audit=json.loads(audit_path.read_text()); audit["deployment_gates"]={"model_parameters":True,"sequence_vocabulary":True,"preprocessing_object_or_spec":True,"context_feature_specification":True,"locked_end_to_end_smoke_test":True}; audit["release_status"]="DEPLOYABLE_INFERENCE_ASSETS_PRESENT"; audit["locked_smoke_test"]=smoke; audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    for name in ["README.md","MODEL_CARD.md"]:
        p=release/name; txt=p.read_text(encoding="utf-8").replace("REPRODUCIBILITY_BUNDLE_ONLY_INFERENCE_ASSETS_INCOMPLETE","DEPLOYABLE_INFERENCE_ASSETS_PRESENT"); p.write_text(txt,encoding="utf-8")
    (release/"docs/INFERENCE.md").write_text("# Inference\n\n```powershell\npython hiera_vep_score.py --bundle . --input model/context_aware_interpretable_model/example_scoring_input.tsv --output predictions.tsv\n```\n\nInputs require aligned single-SNV sequences and all raw context columns. Outputs include prediction and exact block contributions.\n",encoding="utf-8")
    provenance=release/"metadata/provenance.json"; pro=json.loads(provenance.read_text()); pro["release_status"]="DEPLOYABLE_INFERENCE_ASSETS_PRESENT"; pro["locked_smoke_test"]=smoke; provenance.write_text(json.dumps(pro,indent=2),encoding="utf-8")
    checks=release/"metadata/checksums.sha256"; files=sorted(p for p in release.rglob("*") if p.is_file() and p!=checks); checks.write_text("\n".join(f"{sha(p)}  {p.relative_to(release).as_posix()}" for p in files)+"\n",encoding="utf-8")
    zpath=root/"release"/(a.release_name+".zip"); zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(release.rglob("*")):
            if p.is_file(): z.write(p,Path(a.release_name)/p.relative_to(release))
    print("\nLOCKED INFERENCE SMOKE TEST"); print(json.dumps(smoke,indent=2)); print("Release status: DEPLOYABLE_INFERENCE_ASSETS_PRESENT"); print("Archive:",zpath)
if __name__=="__main__": main()
