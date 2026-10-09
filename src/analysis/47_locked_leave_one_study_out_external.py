import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
TABLE=ROOT/"results/hiera_unified_training_table/hiera_unified_variant_table.tsv"
MODEL_FILE=ROOT/"results/nested_grouped_hiera_selection/hiera_nested_selected_model.pt"
VECTORIZER_FILE=ROOT/"results/nested_grouped_hiera_selection/hiera_nested_vectorizer.joblib"
OUTPUT=ROOT/"results/locked_leave_one_study_out_external"; OUTPUT.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOTSTRAPS=2000; SEED=20261007

def clean(x): return x.fillna("").astype(str).str.upper().tolist()
def delta(v,d): return (v.transform(clean(d.alternative_sequence))-v.transform(clean(d.reference_sequence))).astype(np.float32).tocsr()
def tsparse(x):
 c=x.tocoo(); ix=torch.tensor(np.vstack([c.row,c.col]),dtype=torch.long,device=DEVICE); val=torch.tensor(c.data,dtype=torch.float32,device=DEVICE)
 return torch.sparse_coo_tensor(ix,val,c.shape,device=DEVICE).coalesce()
def encode(x,m): return torch.tensor([m.get(str(v),0) for v in x],dtype=torch.long,device=DEVICE)
def safe_corr(y,p,kind):
 if np.std(y)==0 or np.std(p)==0:return np.nan
 return float((stats.pearsonr if kind=="pearson" else stats.spearmanr)(y,p).statistic)
def metrics(y,p):
 return {"n":len(y),"mae":mean_absolute_error(y,p),"rmse":mean_squared_error(y,p)**.5,"r2":r2_score(y,p),
 "pearson_r":safe_corr(y,p,"pearson"),"spearman_rho":safe_corr(y,p,"spearman"),
 "direction_accuracy":np.mean(np.sign(y)==np.sign(p)),"observed_sd":np.std(y,ddof=1),"predicted_sd":np.std(p,ddof=1)}

class ResidualHiera(torch.nn.Module):
 def __init__(self,nf,n_inter,n_assay,n_element,cfg):
  super().__init__(); self.cfg=cfg; self.bias=torch.nn.Parameter(torch.zeros(())); self.delta_beta=torch.nn.Parameter(torch.zeros(nf))
  self.assay=torch.nn.Embedding(n_assay+1,1,padding_idx=0); self.element=torch.nn.Embedding(n_element+1,1,padding_idx=0)
  if cfg["rank"]:
   self.basis=torch.nn.Parameter(torch.zeros(nf,cfg["rank"])); self.embedding=torch.nn.Embedding(n_inter+1,cfg["rank"],padding_idx=0)
  else:self.basis=None;self.embedding=None
 def forward(self,x,base,inter,assay,element):
  seq=torch.sparse.mm(x,self.delta_beta[:,None]).squeeze(1); interaction=0.
  if self.basis is not None:interaction=(torch.sparse.mm(x,self.basis)*self.embedding(inter)).sum(1)
  groups=self.assay(assay).squeeze(1)+self.element(element).squeeze(1) if self.cfg["l_group"]>0 else 0.
  return base+self.bias+seq+interaction+groups

print("Loading locked Step 46 model and unified datasets...")
bundle=torch.load(MODEL_FILE,map_location=DEVICE,weights_only=False); cfg=bundle["config"]; maps=bundle["mappings"]
v=joblib.load(VECTORIZER_FILE); data=pd.read_csv(TABLE,sep="\t",low_memory=False)
dev=data[data.source_partition.eq("development")].copy(); ext=data[data.source_partition.eq("external")].copy().reset_index(drop=True)
pool=dev.original_split.astype(str).str.lower().isin(["train","validation","val"])
Xd=delta(v,dev.loc[pool]); Xe=delta(v,ext); yd=dev.loc[pool,"target_model"].to_numpy(float); ye=ext.target_model.to_numpy(float)

# Refit only the locked Ridge anchor on all allowed development rows. External outcomes are untouched.
ridge_anchor=Ridge(alpha=cfg["ridge_alpha"]).fit(Xd,yd); base=ridge_anchor.predict(Xe).astype(np.float32)
ridge100=Ridge(alpha=100.).fit(Xd,yd); ridge100_prediction=ridge100.predict(Xe)
model=ResidualHiera(bundle["n_features"],len(maps["interaction"]),len(maps["assay"]),len(maps["element"]),cfg).to(DEVICE)
model.load_state_dict(bundle["state_dict"]); model.eval()
key=maps["interaction_key"]; inter=encode(ext[key],maps["interaction"]); assay=encode(ext.assay_id,maps["assay"]); element=encode(ext.element_group,maps["element"])
with torch.no_grad():hiera=model(tsparse(Xe),torch.tensor(base,device=DEVICE),inter,assay,element).cpu().numpy()

predictions=ext[["row_id","validation_tier","assay_id","element_id","species","assay_family","position_group","sequence_pair_hash","target_model"]].copy()
predictions["hiera_prediction"]=hiera;predictions["ridge_anchor_prediction"]=base;predictions["ridge100_prediction"]=ridge100_prediction;predictions["zero_prediction"]=0.
models={"locked_hiera":hiera,"ridge_anchor":base,"ridge_alpha100":ridge100_prediction,"zero_baseline":np.zeros(len(ext))}
rows=[]
for level,column in [("study","element_id"),("tier","validation_tier")]:
 for group,idx in ext.groupby(column).groups.items():
  ii=np.asarray(list(idx),int)
  for name,p in models.items():rows.append({"evaluation_level":level,"evaluation_group":group,"model":name,**metrics(ye[ii],p[ii])})
for name,p in models.items():rows.append({"evaluation_level":"all_external","evaluation_group":"ALL","model":name,**metrics(ye,p)})
point=pd.DataFrame(rows)

# Position-clustered paired bootstrap: HIERA minus alpha-100 Ridge.
rng=np.random.default_rng(SEED); boot=[]
for group,idx in ext.groupby("validation_tier").groups.items():
 ii=np.asarray(list(idx),int); groups=ext.iloc[ii].position_group.to_numpy(); unique=np.unique(groups)
 for b in range(BOOTSTRAPS):
  sampled=rng.choice(unique,size=len(unique),replace=True); take=np.concatenate([ii[groups==g] for g in sampled])
  mh=metrics(ye[take],hiera[take]); mr=metrics(ye[take],ridge100_prediction[take])
  boot.append({"validation_tier":group,"iteration":b,"rmse_reduction":mr["rmse"]-mh["rmse"],"mae_reduction":mr["mae"]-mh["mae"],"pearson_gain":mh["pearson_r"]-mr["pearson_r"],"spearman_gain":mh["spearman_rho"]-mr["spearman_rho"]})
boot=pd.DataFrame(boot); comparisons=[]
for (tier,measure),x in boot.melt(id_vars=["validation_tier","iteration"],var_name="measure",value_name="difference").groupby(["validation_tier","measure"]):
 vals=x.difference.dropna().to_numpy(); comparisons.append({"validation_tier":tier,"measure":measure,"mean_difference":vals.mean(),"ci_lower_95":np.quantile(vals,.025),"ci_upper_95":np.quantile(vals,.975),"bootstrap_p_two_sided":2*min(np.mean(vals<=0),np.mean(vals>=0))})
comparisons=pd.DataFrame(comparisons)

predictions.to_csv(OUTPUT/"locked_external_predictions.tsv",sep="\t",index=False);point.to_csv(OUTPUT/"locked_external_point_estimates.tsv",sep="\t",index=False);boot.to_csv(OUTPUT/"locked_external_position_bootstrap.tsv",sep="\t",index=False);comparisons.to_csv(OUTPUT/"locked_hiera_vs_ridge100.tsv",sep="\t",index=False)
summary={"development_training_rows":int(pool.sum()),"external_rows":len(ext),"external_studies":int(ext.element_id.nunique()),"configuration":cfg,"fixed_epochs":bundle["epochs"],"external_outcomes_used_for_training_or_selection":False,"unseen_assay_codes":int((assay==0).sum().item()),"unseen_element_codes":int((element==0).sum().item()),"unseen_interaction_codes":int((inter==0).sum().item()),"bootstrap_iterations":BOOTSTRAPS}
(OUTPUT/"locked_external_settings.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nLOCKED LEAVE-ONE-STUDY-OUT EXTERNAL RESULTS");print(point.query("evaluation_level in ['tier','all_external']").to_string(index=False,float_format=lambda z:f"{z:.4f}"));print("\nPAIRED HIERA VERSUS RIDGE-100");print(comparisons.to_string(index=False,float_format=lambda z:f"{z:.4f}"));print("\nStep 47 locked leave-one-study-out external evaluation: COMPLETED");print("Saved:",OUTPUT)
