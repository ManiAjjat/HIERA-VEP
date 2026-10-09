import os
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
REGISTRY=ROOT/"results/genomic_coordinate_resolvability_audit/variant_coordinate_registry.tsv"
OUTPUT=ROOT/"results/human_encode_promoter_annotation"; OUTPUT.mkdir(parents=True,exist_ok=True)

def locate(patterns,label):
 hits=[]
 for pattern in patterns:hits.extend(ROOT.rglob(pattern))
 hits=[p for p in hits if p.is_file() and "results/human_encode_promoter_annotation" not in str(p)]
 if not hits:raise FileNotFoundError(f"Could not locate {label}. Searched: {patterns}")
 hits=sorted(set(hits),key=lambda p:(0 if "processed" in str(p).lower() else 1,len(str(p))))
 print(f"Using {label}: {hits[0]}"); return hits[0]

CCRE=locate(["*GRCh38*cCRE*.bed","*grch38*ccre*.bed","*cCRE*v4*.bed","*ccre*v4*.bed"],"ENCODE cCRE BED")
PROMOTERS=locate(["gencode_v50_promoters.tsv","*v50*promoter*.tsv","*promoter*2501*.tsv"],"GENCODE v50 promoter table")

def norm_chr(x):
 x=str(x); return x if x.startswith("chr") else "chr"+x

def read_ccre(path):
 d=pd.read_csv(path,sep="\t",header=None,comment="#",low_memory=False)
 if d.shape[1]<3:raise ValueError("cCRE BED has fewer than three columns")
 out=pd.DataFrame({"chromosome":d.iloc[:,0].map(norm_chr),"start0":pd.to_numeric(d.iloc[:,1],errors="coerce"),"end0":pd.to_numeric(d.iloc[:,2],errors="coerce")})
 class_pattern=re.compile(r"(?:^|[-,;| ])(PLS|pELS|dELS|CA-CTCF|CA-H3K4me3|CA-TF|CA|TF)(?:$|[-,;| ])",re.I)
 best=None; best_hits=-1
 for col in d.columns[3:]:
  s=d[col].fillna("").astype(str); hits=s.map(lambda x:bool(class_pattern.search(x))).sum()
  if hits>best_hits:best,best_hits=col,hits
 if best is None or best_hits==0:raise ValueError("Could not identify cCRE class column")
 raw=d[best].fillna("unknown").astype(str)
 def cls(x):
  for token in ["PLS","pELS","dELS","CA-CTCF","CA-H3K4me3","CA-TF","TF","CA"]:
   if re.search(rf"(?i)(?:^|[-,;| ]){re.escape(token)}(?:$|[-,;| ])",x):return token
  return "other"
 out["ccre_class"]=raw.map(cls)
 # First BED-like identifier column, if present.
 out["ccre_id"]=d.iloc[:,3].astype(str) if d.shape[1]>3 else np.arange(len(d)).astype(str)
 return out.dropna(subset=["start0","end0"])

def read_promoters(path):
 d=pd.read_csv(path,sep="\t",low_memory=False)
 lower={c.lower():c for c in d.columns}
 def col(names,required=True):
  for n in names:
   if n.lower() in lower:return lower[n.lower()]
  if required:raise KeyError(f"Promoter table missing one of {names}; columns={list(d.columns)}")
 chrom=col(["chromosome","chrom","chr","seqname"]); start=col(["promoter_start","start"]); end=col(["promoter_end","end"])
 out=pd.DataFrame({"chromosome":d[chrom].map(norm_chr),"start0":pd.to_numeric(d[start],errors="coerce"),"end0":pd.to_numeric(d[end],errors="coerce")})
 gene=col(["gene_name","gene","symbol"],False); gid=col(["gene_id"],False); strand=col(["strand"],False); tss=col(["tss","tss_1based","transcription_start_site"],False)
 out["gene_name"]=d[gene].astype(str) if gene else "unknown"; out["gene_id"]=d[gid].astype(str) if gid else "unknown"; out["strand"]=d[strand].astype(str) if strand else "."
 if tss:out["tss0"]=pd.to_numeric(d[tss],errors="coerce")-1
 else:out["tss0"]=((out.start0+out.end0-1)/2).round()
 return out.dropna(subset=["start0","end0","tss0"])

def interval_index(frame):
 result={}
 for chrom,g in frame.groupby("chromosome"):
  g=g.sort_values("start0").reset_index(drop=True); result[chrom]=(g.start0.to_numpy(int),g)
 return result

print("Loading confirmed GRCh38 variants and annotations...")
v=pd.read_csv(REGISTRY,sep="\t",low_memory=False); v=v[(v.source_partition=="development")&v.eligible_for_GRCh38_annotation].copy()
v["chromosome"]=v.chromosome.map(norm_chr); v["variant_start0"]=pd.to_numeric(v.start,errors="raise").astype(int)-1; v["variant_end0"]=v.variant_start0+1
ccre=read_ccre(CCRE); promoters=read_promoters(PROMOTERS); ci=interval_index(ccre); pi=interval_index(promoters)
print("Eligible variants:",len(v),"| cCREs:",len(ccre),"| promoters:",len(promoters))

rows=[]
for row in v.itertuples():
 pos=int(row.variant_start0); classes=[]; ids=[]
 if row.chromosome in ci:
  starts,g=ci[row.chromosome]; stop=np.searchsorted(starts,pos,side="right")
  hit=g.iloc[:stop]; hit=hit[hit.end0.to_numpy()>pos]
  classes=sorted(set(hit.ccre_class)); ids=hit.ccre_id.astype(str).tolist()
 promoter_hits=[]; nearest_gene="unknown"; nearest_gene_id="unknown"; nearest_distance=np.nan; strand="."
 if row.chromosome in pi:
  starts,g=pi[row.chromosome]; stop=np.searchsorted(starts,pos,side="right"); hit=g.iloc[:stop]; hit=hit[hit.end0.to_numpy()>pos]
  promoter_hits=sorted(set(hit.gene_name.astype(str)))
  distances=np.abs(g.tss0.to_numpy(float)-pos); j=int(np.argmin(distances)); nearest=g.iloc[j]
  nearest_gene=str(nearest.gene_name);nearest_gene_id=str(nearest.gene_id);nearest_distance=float(pos-nearest.tss0);strand=str(nearest.strand)
  if strand=="-":nearest_distance=-nearest_distance
 priority=[x for x in ["PLS","pELS","dELS","CA-H3K4me3","CA-CTCF","CA-TF","TF","CA","other"] if x in classes]
 rows.append({"source_row":int(row.source_row),"element_id":row.element_id,"chromosome":row.chromosome,
  "variant_start_1based":int(row.start),"ccre_overlap":bool(classes),"ccre_count":len(ids),"ccre_classes":";".join(classes),
  "primary_ccre_class":priority[0] if priority else "none","ccre_ids":";".join(ids),"promoter_overlap":bool(promoter_hits),
  "promoter_genes":";".join(promoter_hits),"nearest_tss_gene":nearest_gene,"nearest_tss_gene_id":nearest_gene_id,
  "signed_distance_to_nearest_tss":nearest_distance,"absolute_distance_to_nearest_tss":abs(nearest_distance) if np.isfinite(nearest_distance) else np.nan})

ann=pd.DataFrame(rows); ann.to_csv(OUTPUT/"development_variant_encode_promoter_annotations.tsv",sep="\t",index=False)
element=(ann.groupby("element_id").agg(variants=("source_row","size"),ccre_overlap_fraction=("ccre_overlap","mean"),promoter_overlap_fraction=("promoter_overlap","mean"),median_abs_tss_distance=("absolute_distance_to_nearest_tss","median"),ccre_classes=("ccre_classes",lambda x:";".join(sorted(set(";".join(x).split(";"))-{""})))).reset_index())
element.to_csv(OUTPUT/"element_encode_promoter_summary.tsv",sep="\t",index=False)
summary={"annotated_variants":len(ann),"elements":int(ann.element_id.nunique()),"variants_overlapping_ccre":int(ann.ccre_overlap.sum()),"ccre_overlap_fraction":float(ann.ccre_overlap.mean()),"variants_overlapping_promoter":int(ann.promoter_overlap.sum()),"promoter_overlap_fraction":float(ann.promoter_overlap.mean()),"ccre_class_counts":ann.primary_ccre_class.value_counts().to_dict(),"ccre_file":str(CCRE),"promoter_file":str(PROMOTERS),"coordinate_conversion":"MaveDB 1-based SNV start converted to 0-based half-open BED [start-1,start)."}
(OUTPUT/"encode_promoter_annotation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nENCODE/GENCODE ANNOTATION SUMMARY");print(json.dumps(summary,indent=2));print("\nELEMENT SUMMARY");print(element.to_string(index=False));print("\nStep 49B ENCODE cCRE/promoter annotation: COMPLETED");print("Saved:",OUTPUT)
