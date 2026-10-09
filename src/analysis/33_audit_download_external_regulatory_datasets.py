import os
from pathlib import Path
from io import StringIO
import json, re, time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/external_validation_inventory/external_validation_candidates.tsv"
OUT = ROOT / "results/external_validation_strict_audit"
RAW = OUT / "raw_score_tables"
OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(exist_ok=True)
API = "https://api.mavedb.org/api/v1"

retry = Retry(total=8, connect=8, read=8, status=8, backoff_factor=1.5,
              status_forcelist=(429,500,502,503,504),
              allowed_methods=frozenset(("GET",)), raise_on_status=False)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))
session.headers.update({"User-Agent":"CausalRegNet-external-audit/1.0"})


def get(url, required=True):
    last=None
    for attempt in range(4):
        try:
            r=session.get(url,timeout=(30,180)); r.raise_for_status(); time.sleep(.3); return r
        except requests.RequestException as exc:
            last=exc; time.sleep(3*(attempt+1))
    if required: raise last
    return None


def extract_sequence(text):
    match=re.search(r"TargetGene\s+dna\s+([ACGTN]+)\s+TargetSequence",str(text),re.I)
    return match.group(1).upper() if match else ""


def species(text):
    text=str(text)
    if "NCBI:txid9606" in text: return "Homo sapiens"
    if "NCBI:txid10090" in text: return "Mus musculus"
    if "NCBI:txid559292" in text: return "Saccharomyces cerevisiae"
    return "unknown"


def target_class(text):
    x=str(text).lower()
    if "protein_coding" in x: return "protein_coding"
    if "promoter regulatory" in x: return "promoter"
    if "enhancer regulatory" in x: return "enhancer"
    if "other_noncoding" in x and "splicing" in x: return "splicing_regulatory"
    if "other_noncoding" in x: return "other_noncoding"
    return "uncertain"


if not INPUT.exists(): raise FileNotFoundError(INPUT)
d=pd.read_csv(INPUT,sep="\t",low_memory=False)
d["species"]=d.target_summary.map(species)
d["target_class"]=(d.target_summary.fillna("")+" "+d.title.fillna("")).map(target_class)
d["target_sequence"]=d.target_summary.map(extract_sequence)
d["target_sequence_length"]=d.target_sequence.str.len()
d["strict_noncoding_target"]=d.target_class.isin(["promoter","enhancer","splicing_regulatory","other_noncoding"])

def tier(row):
    if row.target_class in ("promoter","enhancer") and row.species=="Homo sapiens": return "primary_human_external"
    if row.target_class in ("promoter","enhancer") and row.species in ("Mus musculus","Saccharomyces cerevisiae"): return "cross_species_transfer"
    if row.target_class=="splicing_regulatory" and row.species=="Homo sapiens": return "secondary_splicing_transfer"
    return "exclude"
d["validation_tier"]=d.apply(tier,axis=1)
d["strict_decision"]=np.where(d.validation_tier.eq("exclude"),"exclude","retain_for_score_audit")
d["strict_exclusion_reason"]=np.where(d.validation_tier.eq("exclude"),
    np.where(d.target_class.eq("protein_coding"),"protein-coding DMS, not a regulatory-DNA target",
             "target is not an eligible noncoding regulatory element"),"")

strict=d[d.validation_tier.ne("exclude")].copy()
print("Initial candidates:",len(d))
print("Strict noncoding score sets:",len(strict))
print(strict.validation_tier.value_counts().to_string())

audit=[]; failures=[]
snv_pattern=re.compile(r"^(?:[cnrmg]\.)?(?:-?\d+|\*\d+)[ACGT]>[ACGT]$",re.I)
for i,row in enumerate(strict.itertuples(index=False),1):
    urn=row.score_set_urn
    safe=urn.replace(":","_")
    path=RAW/f"{safe}_scores.csv"
    if path.exists() and path.stat().st_size>20:
        text=path.read_text(encoding="utf-8-sig")
    else:
        response=get(f"{API}/score-sets/{urn}/scores",required=False)
        if response is None:
            failures.append({"score_set_urn":urn,"endpoint":"scores","status":"temporarily_unavailable"})
            print("  WARNING: score download unavailable",urn); continue
        text=response.text; path.write_text(text,encoding="utf-8")
    try: scores=pd.read_csv(StringIO(text))
    except Exception as exc:
        failures.append({"score_set_urn":urn,"endpoint":"scores_parse","status":str(exc)[:300]}); continue
    variant_col=next((c for c in ("hgvs_nt","hgvs","variant","variant_hgvs") if c in scores.columns),None)
    score_col=next((c for c in ("score","functional_score","effect","fitness") if c in scores.columns),None)
    if score_col is None:
        numeric=[c for c in scores.select_dtypes(include="number").columns if "count" not in c.lower() and "rep" not in c.lower()]
        score_col=numeric[0] if len(numeric)==1 else None
    single_snv_n=0
    if variant_col:
        single_snv_n=int(scores[variant_col].dropna().astype(str).str.match(snv_pattern).sum())
    score_nonmissing=int(pd.to_numeric(scores[score_col],errors="coerce").notna().sum()) if score_col else 0
    audit.append({"score_set_urn":urn,"title":row.title,"validation_tier":row.validation_tier,
        "species":row.species,"target_class":row.target_class,"target_sequence_length":row.target_sequence_length,
        "downloaded_rows":len(scores),"variant_column":variant_col,"score_column":score_col,
        "nonmissing_scores":score_nonmissing,"strict_single_snv_rows":single_snv_n,
        "single_snv_fraction":single_snv_n/len(scores) if len(scores) else np.nan,
        "raw_score_file":str(path),"score_table_columns":" | ".join(map(str,scores.columns))})
    print(f"[{i}/{len(strict)}] {urn}: rows={len(scores):,}, SNVs={single_snv_n:,}, score={score_col}")

audit=pd.DataFrame(audit)
if len(audit):
    audit["usable_score_table"]=((audit.nonmissing_scores>=100)&(audit.strict_single_snv_rows>=100)&audit.score_column.notna())
else:
    audit=pd.DataFrame(columns=["score_set_urn","usable_score_table"])
final=strict.merge(audit,on=["score_set_urn","title","validation_tier","species","target_class","target_sequence_length"],how="left")
final["final_dataset_status"]=np.where(final.usable_score_table.fillna(False),"eligible_for_harmonization","failed_score_table_requirements")

decision_cols=["score_set_urn","title","num_variants","species","target_class","validation_tier",
               "target_sequence_length","strict_decision","strict_exclusion_reason"]
d[decision_cols].to_csv(OUT/"all_candidate_eligibility_decisions.tsv",sep="\t",index=False)
audit.to_csv(OUT/"strict_score_table_audit.tsv",sep="\t",index=False)
final.to_csv(OUT/"external_regulatory_harmonization_manifest.tsv",sep="\t",index=False)
if failures: pd.DataFrame(failures).to_csv(OUT/"strict_audit_failures.tsv",sep="\t",index=False)
summary={"initial_candidates":len(d),"excluded_nonregulatory_or_protein_dms":int(d.validation_tier.eq('exclude').sum()),
         "strict_noncoding_score_sets":len(strict),"primary_human_score_sets":int(strict.validation_tier.eq('primary_human_external').sum()),
         "cross_species_score_sets":int(strict.validation_tier.eq('cross_species_transfer').sum()),
         "splicing_score_sets":int(strict.validation_tier.eq('secondary_splicing_transfer').sum()),
         "score_tables_passing_structure":int(audit.usable_score_table.sum()) if len(audit) else 0,
         "temporary_failures":len(failures),
         "rule":"Protein-coding DMS assays are excluded even when the phenotype is called expression."}
(OUT/"strict_external_audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nSTRICT EXTERNAL DATASET AUDIT")
print(json.dumps(summary,indent=2))
if len(final): print("\nHARMONIZATION MANIFEST\n",final[["score_set_urn","title","validation_tier","downloaded_rows","strict_single_snv_rows","usable_score_table","final_dataset_status"]].to_string(index=False))
print("\nStep 33 strict external dataset audit: COMPLETED")
print("Saved:",OUT)
