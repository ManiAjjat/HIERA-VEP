#!/usr/bin/env python3
"""Step 56: build and audit the HIERA-VEP reproducibility release.

The script never labels the bundle deployable unless model, vocabulary and
preprocessing assets are all present. It excludes raw data and large bootstrap
replicates, creates checksums, records software versions, and emits a ZIP file.
"""

from __future__ import annotations
import os

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODEL_EXTENSIONS={".joblib",".pkl",".pickle",".pt",".pth",".npz",".npy",".safetensors"}
TEXT_ASSET_EXTENSIONS={".json",".yaml",".yml",".txt",".tsv",".csv"}
MAX_RESULT_BYTES=8*1024*1024


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--project",type=Path,default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2])))
    p.add_argument("--release-name",default="HIERA-VEP_reproducibility_release_v1")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def copy_file(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)


def safe_json(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None


def package_versions():
    wanted=["numpy","pandas","scipy","scikit-learn","torch","transformers","joblib",
            "biopython","requests","matplotlib","seaborn"]
    out={}
    for name in wanted:
        try: out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name]="not_installed"
    return out


def classify_model_assets(files):
    names=[p.name.lower() for p in files]
    has_model=any(p.suffix.lower() in MODEL_EXTENSIONS and
                  any(k in p.name.lower() for k in ["model","ridge","coefficient","coef"]) for p in files)
    has_vocabulary=any(any(k in n for k in ["vocab","vectorizer","kmer_features","feature_names"]) for n in names)
    has_preprocessing=any(any(k in n for k in ["preprocess","scaler","encoder","design_spec","feature_spec"]) for n in names)
    has_context_spec=any(any(k in n for k in ["context","annotation","feature_family"]) for n in names)
    return {"model_parameters":has_model,"sequence_vocabulary":has_vocabulary,
            "preprocessing_object_or_spec":has_preprocessing,"context_feature_specification":has_context_spec}


def main():
    a=args_parser(); root=a.project.resolve()
    release_root=root/"release"/a.release_name
    zip_path=root/"release"/(a.release_name+".zip")
    if release_root.exists():
        if not a.overwrite:
            raise FileExistsError(f"Release already exists: {release_root}\nRerun with --overwrite to rebuild it.")
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True)
    for d in ["src","model","metadata","results","examples","docs"]: (release_root/d).mkdir()

    # Include numbered pipeline scripts and preserve the step history.
    scripts=[]
    for p in root.glob("*.py"):
        m=re.match(r"^(\d+)[A-Za-z_].*\.py$",p.name)
        if m and 32<=int(m.group(1))<=56:
            copy_file(p,release_root/"src"/p.name); scripts.append(p)

    # Search only relevant trained-model directories; do not package downloaded foundation models.
    candidate_dirs=[root/"results"/"context_aware_interpretable_model",
                    root/"results"/"nested_grouped_hiera_selection",
                    root/"results"/"hiera_vep_v1"]
    model_assets=[]
    for folder in candidate_dirs:
        if not folder.exists(): continue
        for p in folder.rglob("*"):
            if not p.is_file(): continue
            low=p.name.lower()
            if p.suffix.lower() in MODEL_EXTENSIONS or (
                p.suffix.lower() in TEXT_ASSET_EXTENSIONS and
                any(k in low for k in ["vocab","feature","preprocess","scaler","encoder","model","context","design"])):
                rel=Path(folder.name)/p.relative_to(folder)
                copy_file(p,release_root/"model"/rel); model_assets.append(p)

    # Curated, compact outputs. Raw variants, embeddings and bootstrap replicate matrices are excluded.
    preferred=[
        ("context_aware_interpretable_model",["*summary*.json","*comparison*.tsv","*performance*.tsv"]),
        ("fair_genomic_language_model_comparison",["*summary*.json","locked_model_comparison.tsv"]),
        ("exact_attribution_counterfactual_audit",["*summary*.json","attribution_block_summary.tsv","context_block_counterfactual_ablation.tsv"]),
        ("position_level_motif_conservation_validation",["*summary*.json","motif_conservation_associations.tsv"]),
        ("exact_gtex_eqtl_concordance",["*summary*.json","gtex_concordance_tests.tsv"]),
        ("locked_feature_family_ablation",["*summary*.json","locked_ablation_point_estimates.tsv","locked_ablation_paired_bootstrap.tsv","element_level_ablation_consistency.tsv"]),
        ("calibration_uncertainty_failure_analysis",["*summary*.json","calibration_uncertainty_estimates.tsv","calibration_deciles.tsv","uncertainty_selective_risk_curve.tsv","structured_failure_strata.tsv"]),
    ]
    result_assets=[]
    for folder_name,patterns in preferred:
        folder=root/"results"/folder_name
        if not folder.exists(): continue
        seen=set()
        for pattern in patterns:
            for p in folder.glob(pattern):
                if p.is_file() and p.stat().st_size<=MAX_RESULT_BYTES and p.resolve() not in seen:
                    copy_file(p,release_root/"results"/folder_name/p.name)
                    result_assets.append(p); seen.add(p.resolve())

    # Create a target-free example schema from the unified table.
    unified=root/"results"/"hiera_unified_training_table"/"hiera_unified_variant_table.tsv"
    if unified.exists():
        u=pd.read_csv(unified,sep="\t",nrows=12,low_memory=False)
        cols=[c for c in ["row_id","element_id","assay_id","condition","cell_context","species",
             "position","reference_allele","alternative_allele","reference_sequence","alternative_sequence"] if c in u]
        u[cols].to_csv(release_root/"examples"/"example_input.tsv",sep="\t",index=False)
        (release_root/"examples"/"INPUT_SCHEMA.md").write_text(
            "# Input schema\n\n`example_input.tsv` is target-free. Coordinates alone are insufficient: "
            "the release expects aligned reference and alternative sequences plus explicitly supplied context.\n",
            encoding="utf-8")

    versions=package_versions()
    environment={"created_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
                 "platform":platform.platform(),"packages":versions}
    (release_root/"metadata"/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    (release_root/"metadata"/"requirements-lock.txt").write_text(
        "\n".join(f"{k}=={v}" for k,v in versions.items() if v!="not_installed")+"\n",encoding="utf-8")

    gates=classify_model_assets(model_assets)
    deployable=all(gates.values())
    release_status="DEPLOYABLE_INFERENCE_ASSETS_PRESENT" if deployable else "REPRODUCIBILITY_BUNDLE_ONLY_INFERENCE_ASSETS_INCOMPLETE"
    audit={"release_status":release_status,"deployment_gates":gates,
           "scripts_included":len(scripts),"model_assets_included":len(model_assets),
           "result_assets_included":len(result_assets),
           "excluded_by_policy":["raw source datasets","full variant tables","foundation-model weights",
                                 "large bootstrap replicate matrices","API caches"],
           "model_assets":[str(p.relative_to(root)) for p in model_assets]}
    (release_root/"metadata"/"release_readiness_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")

    model_card=f"""# HIERA-VEP model card

## Intended use
Research benchmarking and interpretation of regulatory-variant effects in the assay and annotation domains represented by the training data.

## Locked internal performance
- n = 4,408 variants
- RMSE = 0.9384
- R2 = 0.1162
- Pearson r = 0.3515
- Spearman rho = 0.2918
- Direction accuracy = 0.6030

## Evidence for context dependence
Joint removal of context increased RMSE by 0.0450 (95% clustered-bootstrap CI 0.0326–0.0577; Holm-adjusted P=0.0048).

## Limitations
- Zero-shot external transfer was weak and should not be described as successful generalization.
- Predictions are compressed toward zero (locked calibration slope 1.279; predicted SD 0.274 versus observed SD 0.998).
- Model disagreement has only modest error-detection ability (AUC 0.595).
- Generic motif disruption and exact GTEx eQTL concordance did not validate the largest effects.
- Predictions and attributions are computational associations, not causal evidence or clinical guidance.

## Release status
`{release_status}`. See `metadata/release_readiness_audit.json` for gate details.
"""
    (release_root/"MODEL_CARD.md").write_text(model_card,encoding="utf-8")
    readme=f"""# HIERA-VEP reproducibility release v1

This archive records the locked analysis state used for manuscript preparation.

## Contents
- `src/`: numbered analysis scripts (Steps 32–56)
- `model/`: discovered model and preprocessing assets
- `results/`: compact confirmatory summaries and tables
- `metadata/`: environment, checksums, provenance and release audit
- `examples/`: target-free example input schema

## Integrity
Run `python verify_release.py` from the release root. Every distributed file is checked against `metadata/checksums.sha256`.

## Important
Release status: **{release_status}**. A reproducibility archive is not automatically a working scoring tool. Do not advertise pretrained inference unless every deployment gate passes and an end-to-end independent smoke test is completed.
"""
    (release_root/"README.md").write_text(readme,encoding="utf-8")
    (release_root/"LICENSE_NOT_SELECTED.md").write_text(
        "No software license has been selected. The authors must choose and add a license before public release.\n",encoding="utf-8")
    (release_root/"CITATION.cff").write_text(
        "cff-version: 1.2.0\nmessage: Please cite the associated HIERA-VEP manuscript.\n"
        "title: HIERA-VEP reproducibility release\nversion: 1.0.0\ndate-released: "
        +datetime.now(timezone.utc).date().isoformat()+"\n",encoding="utf-8")

    verifier='''from pathlib import Path\nimport hashlib,sys\nroot=Path(__file__).resolve().parent\nbad=[]\nfor line in (root/"metadata/checksums.sha256").read_text().splitlines():\n    expected,rel=line.split("  ",1); p=root/rel\n    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected: bad.append(rel)\nprint("PASS: all release checksums verified" if not bad else "FAIL: "+", ".join(bad))\nsys.exit(bool(bad))\n'''
    (release_root/"verify_release.py").write_text(verifier,encoding="utf-8")

    # Checksums are calculated last and do not checksum the checksum file itself.
    files=sorted(p for p in release_root.rglob("*") if p.is_file() and p.name!="checksums.sha256")
    checksum_lines=[f"{sha256(p)}  {p.relative_to(release_root).as_posix()}" for p in files]
    (release_root/"metadata"/"checksums.sha256").write_text("\n".join(checksum_lines)+"\n",encoding="utf-8")
    provenance={"release_name":a.release_name,"root_file_count":len(files)+1,
                "source_project":str(root),"release_status":release_status,
                "checksums_file":"metadata/checksums.sha256"}
    (release_root/"metadata"/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    # Add provenance checksum after creation.
    with (release_root/"metadata"/"checksums.sha256").open("a",encoding="utf-8") as f:
        p=release_root/"metadata"/"provenance.json"; f.write(f"{sha256(p)}  metadata/provenance.json\n")

    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(release_root.rglob("*")):
            if p.is_file(): z.write(p,Path(a.release_name)/p.relative_to(release_root))

    print("\nHIERA-VEP RELEASE AUDIT")
    print(json.dumps(audit,indent=2))
    print("\nStep 56 reproducibility release: COMPLETED")
    print("Release directory:",release_root)
    print("Release archive:",zip_path)
    if not deployable:
        print("IMPORTANT: The bundle is reproducible but must not yet be advertised as a deployable pretrained scorer.")


if __name__=="__main__": main()
