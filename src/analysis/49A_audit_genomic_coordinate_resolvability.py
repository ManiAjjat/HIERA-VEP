import os
from pathlib import Path
import json, re, time

import numpy as np
import pandas as pd
import requests

ROOT=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
DEV=ROOT/"data/processed/mavedb/regulatory_panel/mavedb_regulatory_within_element_splits.tsv"
EXT=ROOT/"results/external_validation_harmonized/external_model_ready_501bp.tsv"
OUTPUT=ROOT/"results/genomic_coordinate_resolvability_audit"; OUTPUT.mkdir(parents=True,exist_ok=True)
CACHE=OUTPUT/"mavedb_metadata_cache"; CACHE.mkdir(exist_ok=True)
API="https://api.mavedb.org/api/v1"

GRCH38_ACCESSIONS={
 "NC_000001.11":"chr1","NC_000002.12":"chr2","NC_000003.12":"chr3","NC_000004.12":"chr4",
 "NC_000005.10":"chr5","NC_000006.12":"chr6","NC_000007.14":"chr7","NC_000008.11":"chr8",
 "NC_000009.12":"chr9","NC_000010.11":"chr10","NC_000011.10":"chr11","NC_000012.12":"chr12",
 "NC_000013.11":"chr13","NC_000014.9":"chr14","NC_000015.10":"chr15","NC_000016.10":"chr16",
 "NC_000017.11":"chr17","NC_000018.10":"chr18","NC_000019.10":"chr19","NC_000020.11":"chr20",
 "NC_000021.9":"chr21","NC_000022.11":"chr22","NC_000023.11":"chrX","NC_000024.10":"chrY"}

def first(frame,names,default=np.nan):
 for n in names:
  if n in frame:return frame[n]
 return pd.Series(default,index=frame.index)

def split_urns(values):
 out=set()
 for value in values.dropna().astype(str):
  out.update(x.strip() for x in value.split(";") if x.strip().startswith("urn:mavedb:"))
 return sorted(out)

def request_json(urn):
 path=CACHE/(urn.replace(":","_")+".json")
 if path.exists():
  try:return json.loads(path.read_text(encoding="utf-8")),"cached"
  except Exception:pass
 last=None
 session=requests.Session()
 for attempt in range(1,7):
  try:
   r=session.get(f"{API}/score-sets/{urn}",timeout=90); r.raise_for_status(); obj=r.json()
   path.write_text(json.dumps(obj,indent=2),encoding="utf-8"); return obj,"downloaded"
  except Exception as exc:
   last=str(exc); print(f"  {urn}: attempt {attempt}/6 failed: {exc}"); time.sleep(min(2**attempt,30))
 return {},"unavailable: "+str(last)

def flatten_strings(obj):
 strings=[]
 def walk(x,path=""):
  if isinstance(x,dict):
   for k,v in x.items():walk(v,f"{path}.{k}" if path else str(k))
  elif isinstance(x,list):
   for i,v in enumerate(x):walk(v,f"{path}[{i}]")
  elif isinstance(x,(str,int,float,bool)) and x is not None:strings.append((path,str(x)))
 walk(obj); return strings

def metadata_evidence(obj):
 pairs=flatten_strings(obj); text=" | ".join(v for _,v in pairs)
 builds=sorted(set(re.findall(r"(?i)\b(?:GRCh38|hg38|GRCh37|hg19)\b",text)))
 accessions=sorted(set(re.findall(r"NC_\d{6}\.\d+",text)))
 coordinate_hits=[f"{p}={v}" for p,v in pairs if any(k in p.lower() for k in ["coordinate","chromosome","assembly","genome","accession"])]
 return builds,accessions,"; ".join(coordinate_hits[:30])

def chromosome_ok(x):
 x=str(x).replace("chr","").upper()
 return bool(re.fullmatch(r"(?:[1-9]|1[0-9]|2[0-2]|X|Y)",x))

print("Loading source variant tables...")
dev=pd.read_csv(DEV,sep="\t",low_memory=False); ext=pd.read_csv(EXT,sep="\t",low_memory=False)
sources=[("development",dev,"element_group",["score_set_urns","score_set_urn"]),
         ("external",ext,"element_name",["score_set_urn","score_set_urns"])]
all_urns=[]
for _,frame,_,urncols in sources:all_urns.extend(split_urns(first(frame,urncols)))
all_urns=sorted(set(all_urns)); print("Unique score sets requiring metadata review:",len(all_urns))

metadata={}; metadata_rows=[]
for i,urn in enumerate(all_urns,1):
 obj,status=request_json(urn); builds,accessions,evidence=metadata_evidence(obj); metadata[urn]=(builds,accessions,status)
 metadata_rows.append({"score_set_urn":urn,"request_status":status,"build_tokens":";".join(builds),
                       "accessions":";".join(accessions),"coordinate_metadata_evidence":evidence})
 if i%10==0 or i==len(all_urns):print(f"Metadata reviewed: {i}/{len(all_urns)}")

rows=[]; variant_rows=[]
for source,frame,element_col,urncols in sources:
 frame=frame.copy(); frame["_element"]=first(frame,[element_col]).astype(str)
 frame["_urns"]=first(frame,urncols).fillna("").astype(str)
 frame["_chromosome"]=first(frame,["chromosome","chr","seqname"]).astype(str)
 frame["_accession"]=first(frame,["chromosome_accession","accession"]).astype(str)
 frame["_start"]=pd.to_numeric(first(frame,["variant_start","position_1based","position"]),errors="coerce")
 frame["_end"]=pd.to_numeric(first(frame,["variant_end"]),errors="coerce")
 for element,g in frame.groupby("_element",dropna=False):
  urns=split_urns(g._urns); builds=set(); api_acc=set(); unavailable=0
  for urn in urns:
   b,a,status=metadata.get(urn,([],[],"not_checked")); builds.update(x.lower() for x in b);api_acc.update(a);unavailable+=status.startswith("unavailable")
  local_acc=set(g._accession.replace("nan",np.nan).dropna())
  grch38_acc=sorted((local_acc|api_acc)&set(GRCH38_ACCESSIONS))
  valid_chr=float(g._chromosome.map(chromosome_ok).mean())
  numeric=float(g._start.notna().mean()); minpos=float(g._start.min()) if g._start.notna().any() else np.nan; maxpos=float(g._start.max()) if g._start.notna().any() else np.nan
  looks_local=bool(np.isfinite(maxpos) and maxpos<100000)
  explicit38=bool({"grch38","hg38"}&builds or grch38_acc)
  explicit37=bool({"grch37","hg19"}&builds)
  if explicit38 and numeric==1 and (valid_chr==1 or grch38_acc):status="GRCh38_CONFIRMED"
  elif explicit37:status="NON_GRCh38_CONFIRMED"
  elif looks_local:status="LOCAL_ASSAY_COORDINATES"
  elif numeric==1 and valid_chr==1:status="GENOMIC_BUILD_UNRESOLVED"
  else:status="COORDINATES_UNRESOLVED"
  eligible=status=="GRCh38_CONFIRMED"
  rows.append({"source_partition":source,"element_id":element,"variants":len(g),"score_set_urns":";".join(urns),
   "chromosome_values":";".join(sorted(set(g._chromosome)-{"nan"})),"accession_values":";".join(sorted(local_acc)),
   "valid_chromosome_fraction":valid_chr,"numeric_start_fraction":numeric,"minimum_start":minpos,"maximum_start":maxpos,
   "metadata_build_tokens":";".join(sorted(builds)),"GRCh38_accessions":";".join(grch38_acc),
   "metadata_requests_unavailable":unavailable,"coordinate_status":status,"eligible_for_GRCh38_annotation":eligible})
  for idx in g.index:
   accession=str(frame.loc[idx,"_accession"]); chromosome=str(frame.loc[idx,"_chromosome"])
   if accession in GRCH38_ACCESSIONS:chromosome=GRCH38_ACCESSIONS[accession]
   elif chromosome_ok(chromosome):chromosome="chr"+chromosome.replace("chr","")
   variant_rows.append({"source_partition":source,"source_row":int(idx),"element_id":element,
     "chromosome":chromosome,"start":frame.loc[idx,"_start"],"end":frame.loc[idx,"_end"],
     "chromosome_accession":accession,"coordinate_status":status,"eligible_for_GRCh38_annotation":eligible})

manifest=pd.DataFrame(rows); variants=pd.DataFrame(variant_rows); meta=pd.DataFrame(metadata_rows)
manifest.to_csv(OUTPUT/"coordinate_eligibility_manifest.tsv",sep="\t",index=False)
variants.to_csv(OUTPUT/"variant_coordinate_registry.tsv",sep="\t",index=False)
meta.to_csv(OUTPUT/"mavedb_coordinate_metadata.tsv",sep="\t",index=False)
summary={"score_sets_reviewed":len(all_urns),"elements_reviewed":len(manifest),"variants_reviewed":len(variants),
 "status_counts":manifest.coordinate_status.value_counts().to_dict(),
 "GRCh38_eligible_elements":int(manifest.eligible_for_GRCh38_annotation.sum()),
 "GRCh38_eligible_variants":int(variants.eligible_for_GRCh38_annotation.sum()),
 "rule":"Only GRCh38_CONFIRMED rows proceed to ENCODE/GTEx/GRCh38 conservation intersection.",
 "warning":"Chromosome-like numbers alone do not establish a genome build; local reporter coordinates are excluded."}
(OUTPUT/"coordinate_resolvability_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nGENOMIC COORDINATE RESOLVABILITY SUMMARY");print(json.dumps(summary,indent=2));print("\nELEMENT ELIGIBILITY");print(manifest[["source_partition","element_id","variants","minimum_start","maximum_start","metadata_build_tokens","coordinate_status","eligible_for_GRCh38_annotation"]].to_string(index=False));print("\nStep 49A genomic coordinate resolvability audit: COMPLETED");print("Saved:",OUTPUT)
