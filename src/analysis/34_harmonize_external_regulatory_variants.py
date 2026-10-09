import os
from pathlib import Path
import json, re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/external_validation_strict_audit/external_regulatory_harmonization_manifest.tsv"
OUT = ROOT / "results/external_validation_harmonized"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 501
CENTER = WINDOW // 2
SNV = re.compile(r"^(?:[cnrmg]\.)?(?P<position>-?\d+|\*\d+)(?P<ref>[ACGT])>(?P<alt>[ACGT])$", re.I)


def robust_z(values):
    x = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x-med))
    scale = 1.4826*mad
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(x, ddof=1)
    return (x-med)/scale if np.isfinite(scale) and scale > 0 else np.full(len(x), np.nan)


def centered_window(sequence, position0, alt):
    left = position0-CENTER; right = position0+CENTER+1
    prefix = "N"*max(0,-left); suffix = "N"*max(0,right-len(sequence))
    ref_window = prefix + sequence[max(0,left):min(len(sequence),right)] + suffix
    if len(ref_window) != WINDOW: raise ValueError("Window construction failed")
    alt_window = ref_window[:CENTER] + alt + ref_window[CENTER+1:]
    return ref_window, alt_window


if not INPUT.exists(): raise FileNotFoundError(INPUT)
manifest = pd.read_csv(INPUT, sep="\t", low_memory=False)
manifest = manifest[manifest.final_dataset_status.eq("eligible_for_harmonization")].copy()
rows=[]; qc=[]
for m in manifest.itertuples(index=False):
    scores = pd.read_csv(Path(m.raw_score_file), low_memory=False)
    parsed = scores[m.variant_column].astype(str).str.extract(SNV)
    keep = parsed.position.notna() & pd.to_numeric(scores[m.score_column],errors="coerce").notna()
    s=scores.loc[keep].copy(); p=parsed.loc[keep].copy()
    p["position"]=pd.to_numeric(p.position,errors="coerce")
    # Negative/star HGVS positions are retained in the audit but excluded from sequence construction.
    valid_coordinate = p.position.between(1,len(m.target_sequence))
    s=s.loc[valid_coordinate].copy(); p=p.loc[valid_coordinate].copy()
    raw=pd.to_numeric(s[m.score_column],errors="coerce").to_numpy(float)
    z=robust_z(raw)
    reference_matches=[]; built=0
    for idx, ((_,sr),(_,pr)) in enumerate(zip(s.iterrows(),p.iterrows())):
        pos1=int(pr.position); pos0=pos1-1; ref=pr.ref.upper(); alt=pr.alt.upper()
        match=(m.target_sequence[pos0].upper()==ref); reference_matches.append(match)
        if not match: continue
        rw,aw=centered_window(m.target_sequence,pos0,alt)
        rows.append({"score_set_urn":m.score_set_urn,"title":m.title,
            "validation_tier":m.validation_tier,"species":m.species,"target_class":m.target_class,
            "element_name":("SUL1_sulfate" if m.score_set_urn.endswith("09-a-1") else
                            "SUL1_glucose" if m.score_set_urn.endswith("09-a-2") else
                            "LTV1_rep1" if m.score_set_urn.endswith("08-a-1") else
                            "LTV1_rep2" if m.score_set_urn.endswith("08-a-2") else
                            "ECR11" if m.score_set_urn.endswith("07-a-1") else
                            "ALDOB" if m.score_set_urn.endswith("06-a-1") else
                            "Splice_CG" if m.score_set_urn.endswith("83-h-1") else "Splice_AC"),
            "accession":sr.get("accession",pd.NA),"hgvs_nt":sr[m.variant_column],
            "position_1based":pos1,"reference_allele":ref,"alternative_allele":alt,
            "raw_score":raw[idx],"robust_z":z[idx],"target_sequence_length":len(m.target_sequence),
            "reference_sequence":rw,"alternative_sequence":aw,"mutation_index":CENTER})
        built+=1
    qc.append({"score_set_urn":m.score_set_urn,"title":m.title,"strict_snv_with_score":len(p),
               "reference_matches":int(np.sum(reference_matches)),
               "reference_mismatches":int(len(reference_matches)-np.sum(reference_matches)),
               "harmonized_rows":built,"raw_score_mean":float(np.mean(raw)) if len(raw) else np.nan,
               "raw_score_sd":float(np.std(raw,ddof=1)) if len(raw)>1 else np.nan,
               "robust_z_mean":float(np.nanmean(z)) if len(z) else np.nan,
               "robust_z_sd":float(np.nanstd(z,ddof=1)) if len(z)>1 else np.nan})

all_rows=pd.DataFrame(rows)
qc=pd.DataFrame(qc)
if all_rows.empty: raise RuntimeError("No variants passed harmonization")
if not all_rows.reference_sequence.str.len().eq(WINDOW).all(): raise ValueError("Reference windows are not 501 bp")
if not all_rows.alternative_sequence.str.len().eq(WINDOW).all(): raise ValueError("Alternative windows are not 501 bp")
changed=(all_rows.reference_sequence.str[CENTER] != all_rows.alternative_sequence.str[CENTER])
if not changed.all(): raise ValueError("Some alternative windows do not change the central nucleotide")

# LTV1 replicate agreement and consensus. Replicates are merged; biological conditions are not.
lv=all_rows[all_rows.element_name.isin(["LTV1_rep1","LTV1_rep2"])].copy()
pivot=lv.pivot_table(index=["position_1based","reference_allele","alternative_allele"],
                     columns="element_name",values=["raw_score","robust_z"],aggfunc="first").dropna()
rep_stats={"paired_variants":len(pivot),
 "raw_pearson_r":float(pearsonr(pivot[("raw_score","LTV1_rep1")],pivot[("raw_score","LTV1_rep2")]).statistic),
 "raw_spearman_rho":float(spearmanr(pivot[("raw_score","LTV1_rep1")],pivot[("raw_score","LTV1_rep2")]).statistic),
 "robust_z_pearson_r":float(pearsonr(pivot[("robust_z","LTV1_rep1")],pivot[("robust_z","LTV1_rep2")]).statistic)}
base=lv[lv.element_name.eq("LTV1_rep1")].set_index(["position_1based","reference_allele","alternative_allele"])
consensus=[]
for key in pivot.index:
    r=base.loc[key].to_dict(); r["element_name"]="LTV1_consensus"; r["score_set_urn"]="urn:mavedb:00000008-a-consensus"
    r["title"]="LTV1 mean across two biological score-set replicates"
    r["raw_score"]=float(pivot.loc[key,[("raw_score","LTV1_rep1"),("raw_score","LTV1_rep2")]].mean())
    r["robust_z"]=float(pivot.loc[key,[("robust_z","LTV1_rep1"),("robust_z","LTV1_rep2")]].mean())
    r.update({"position_1based":key[0],"reference_allele":key[1],"alternative_allele":key[2]}); consensus.append(r)
consensus=pd.DataFrame(consensus)

model_ready=pd.concat([all_rows[~all_rows.element_name.isin(["LTV1_rep1","LTV1_rep2"])],consensus],ignore_index=True)
primary=model_ready[model_ready.validation_tier.eq("primary_human_external")].copy()
cross=model_ready[model_ready.validation_tier.eq("cross_species_transfer")].copy()
splicing=model_ready[model_ready.validation_tier.eq("secondary_splicing_transfer")].copy()

all_rows.to_csv(OUT/"all_harmonized_score_set_rows.tsv",sep="\t",index=False)
model_ready.to_csv(OUT/"external_model_ready_501bp.tsv",sep="\t",index=False)
primary.to_csv(OUT/"primary_human_external_501bp.tsv",sep="\t",index=False)
cross.to_csv(OUT/"cross_species_external_501bp.tsv",sep="\t",index=False)
splicing.to_csv(OUT/"secondary_splicing_external_501bp.tsv",sep="\t",index=False)
qc.to_csv(OUT/"external_harmonization_qc.tsv",sep="\t",index=False)
(OUT/"ltv1_replicate_agreement.json").write_text(json.dumps(rep_stats,indent=2),encoding="utf-8")
summary={"window_length":WINDOW,"mutation_index_zero_based":CENTER,"score_set_rows":len(all_rows),
 "model_ready_rows_after_ltv1_consensus":len(model_ready),"primary_human_rows":len(primary),
 "cross_species_rows":len(cross),"splicing_rows":len(splicing),
 "elements_or_contexts":sorted(model_ready.element_name.unique()),"ltv1_replicate_agreement":rep_stats,
 "normalization":"median-centered and scaled by 1.4826*MAD within each score set; SD fallback only if MAD=0",
 "warning":"Primary human transcriptional, cross-species transcriptional, and splicing evaluations remain separate."}
(OUT/"external_harmonization_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nEXTERNAL HARMONIZATION SUMMARY")
print(json.dumps(summary,indent=2))
print("\nSCORE-SET QC")
print(qc.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\nStep 34 external regulatory harmonization: COMPLETED")
print("Saved:",OUT)
