import os
from pathlib import Path
from collections import Counter
import copy, json, random

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/hiera_unified_training_table/hiera_unified_variant_table.tsv"
OUTPUT = ROOT / "results/nested_grouped_hiera_selection"
OUTPUT.mkdir(parents=True, exist_ok=True)
SEED, MAX_EPOCHS, PATIENCE = 20261007, 350, 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small prespecified search space: shared anchor strength and residual hierarchy.
CONFIGS = [
 {"name":"ridge_anchor", "ridge_alpha":100., "scope":"none", "rank":0, "lr":0.003, "l_seq":0.01, "l_int":0., "l_group":0.},
 {"name":"group_residual", "ridge_alpha":100., "scope":"none", "rank":0, "lr":0.003, "l_seq":0.02, "l_int":0., "l_group":0.05},
 {"name":"context_rank2", "ridge_alpha":100., "scope":"context", "rank":2, "lr":0.003, "l_seq":0.02, "l_int":0.01, "l_group":0.05},
 {"name":"context_rank4", "ridge_alpha":100., "scope":"context", "rank":4, "lr":0.001, "l_seq":0.02, "l_int":0.02, "l_group":0.10},
 {"name":"element_rank2", "ridge_alpha":100., "scope":"element", "rank":2, "lr":0.003, "l_seq":0.02, "l_int":0.01, "l_group":0.05},
 {"name":"element_rank4", "ridge_alpha":100., "scope":"element", "rank":4, "lr":0.001, "l_seq":0.02, "l_int":0.02, "l_group":0.10},
 {"name":"ridge_strong_context", "ridge_alpha":1000., "scope":"context", "rank":2, "lr":0.003, "l_seq":0.01, "l_int":0.01, "l_group":0.05},
]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def clean(x): return x.fillna("").astype(str).str.upper().tolist()
def delta(v, d): return (v.transform(clean(d.alternative_sequence))-v.transform(clean(d.reference_sequence))).astype(np.float32).tocsr()
def t_sparse(x):
    c=x.tocoo(); ix=torch.tensor(np.vstack([c.row,c.col]),dtype=torch.long,device=DEVICE)
    val=torch.tensor(c.data,dtype=torch.float32,device=DEVICE)
    return torch.sparse_coo_tensor(ix,val,c.shape,device=DEVICE).coalesce()
def cmap(x): return {v:i+1 for i,v in enumerate(sorted(set(x.astype(str))))}
def enc(x,m): return torch.tensor([m.get(str(v),0) for v in x],dtype=torch.long,device=DEVICE)

def connected_groups(d):
    n=len(d); parent=np.arange(n)
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for col in ["position_group","sequence_pair_hash"]:
        first={}
        for i,v in enumerate(d[col].astype(str)):
            if v in first: union(i,first[v])
            else: first[v]=i
    roots=[find(i) for i in range(n)]; code={v:i for i,v in enumerate(sorted(set(roots)))}
    return np.array([code[v] for v in roots])

def macro_rmse(y,p,elements):
    vals=[]
    for e in np.unique(elements):
        z=elements==e
        if z.sum()>=2: vals.append(mean_squared_error(y[z],p[z])**0.5)
    return float(np.mean(vals))

def metrics(y,p):
    return {"n":len(y),"mae":mean_absolute_error(y,p),"rmse":mean_squared_error(y,p)**.5,
      "r2":r2_score(y,p),"pearson_r":stats.pearsonr(y,p).statistic,
      "spearman_rho":stats.spearmanr(y,p).statistic,"direction_accuracy":np.mean(np.sign(y)==np.sign(p)),
      "observed_sd":np.std(y,ddof=1),"predicted_sd":np.std(p,ddof=1)}

class ResidualHiera(torch.nn.Module):
    def __init__(self,nf,n_inter,n_assay,n_element,cfg):
        super().__init__(); self.cfg=cfg
        self.bias=torch.nn.Parameter(torch.zeros(())); self.delta_beta=torch.nn.Parameter(torch.zeros(nf))
        self.assay=torch.nn.Embedding(n_assay+1,1,padding_idx=0); self.element=torch.nn.Embedding(n_element+1,1,padding_idx=0)
        torch.nn.init.zeros_(self.assay.weight); torch.nn.init.zeros_(self.element.weight)
        if cfg["rank"]:
            self.basis=torch.nn.Parameter(torch.randn(nf,cfg["rank"])*0.001)
            self.embedding=torch.nn.Embedding(n_inter+1,cfg["rank"],padding_idx=0); torch.nn.init.zeros_(self.embedding.weight)
        else: self.basis=None; self.embedding=None
    def forward(self,x,base,inter,assay,element):
        seq=torch.sparse.mm(x,self.delta_beta[:,None]).squeeze(1)
        interaction=0.
        if self.basis is not None: interaction=(torch.sparse.mm(x,self.basis)*self.embedding(inter)).sum(1)
        groups=self.assay(assay).squeeze(1)+self.element(element).squeeze(1) if self.cfg["l_group"]>0 else 0.
        return base+self.bias+seq+interaction+groups
    def penalty(self):
        p=self.cfg["l_seq"]*self.delta_beta.square().sum()
        if self.basis is not None: p=p+self.cfg["l_int"]*(self.basis.square().sum()+self.embedding.weight[1:].square().sum())
        if self.cfg["l_group"]>0: p=p+self.cfg["l_group"]*(self.assay.weight[1:].square().sum()+self.element.weight[1:].square().sum())
        return p

def prepare_categories(d,tr,cfg):
    ma=cmap(d.iloc[tr].assay_id); me=cmap(d.iloc[tr].element_group)
    key="context_group" if cfg["scope"]=="context" else "element_group"
    mi=cmap(d.iloc[tr][key]) if cfg["scope"]!="none" else {}
    return ma,me,mi,key

def fit_once(d,X,y,tr,va,cfg,fixed_epochs=None,return_objects=False):
    ridge=Ridge(alpha=cfg["ridge_alpha"]).fit(X[tr],y[tr]); btr=ridge.predict(X[tr]).astype(np.float32); bva=ridge.predict(X[va]).astype(np.float32)
    ma,me,mi,key=prepare_categories(d,tr,cfg)
    atr,ava=enc(d.iloc[tr].assay_id,ma),enc(d.iloc[va].assay_id,ma); etr,eva=enc(d.iloc[tr].element_group,me),enc(d.iloc[va].element_group,me)
    itr=enc(d.iloc[tr][key],mi) if mi else torch.zeros(len(tr),dtype=torch.long,device=DEVICE)
    iva=enc(d.iloc[va][key],mi) if mi else torch.zeros(len(va),dtype=torch.long,device=DEVICE)
    xt,xv=t_sparse(X[tr]),t_sparse(X[va]); yt=torch.tensor(y[tr],dtype=torch.float32,device=DEVICE)
    bt=torch.tensor(btr,device=DEVICE); bv=torch.tensor(bva,device=DEVICE)
    model=ResidualHiera(X.shape[1],len(mi),len(ma),len(me),cfg).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"])
    best=copy.deepcopy(model.state_dict()); best_score=macro_rmse(y[va],bva,d.iloc[va].element_id.to_numpy()); best_epoch=0; wait=0
    epochs=fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad(); pred=model(xt,bt,itr,atr,etr); loss=((pred-yt)**2).mean()+model.penalty(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        if fixed_epochs is None:
            model.eval()
            with torch.no_grad(): pv=model(xv,bv,iva,ava,eva).cpu().numpy()
            score=macro_rmse(y[va],pv,d.iloc[va].element_id.to_numpy())
            if score<best_score-1e-5: best_score,best_epoch,best,wait=score,ep,copy.deepcopy(model.state_dict()),0
            else: wait+=1
            if wait>=PATIENCE: break
    if fixed_epochs is None: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): pv=model(xv,bv,iva,ava,eva).cpu().numpy()
    result={"macro_element_rmse":macro_rmse(y[va],pv,d.iloc[va].element_id.to_numpy()),"rmse":mean_squared_error(y[va],pv)**.5,"pearson_r":stats.pearsonr(y[va],pv).statistic,"best_epoch":best_epoch if fixed_epochs is None else fixed_epochs}
    if return_objects: return result,pv,model,ridge,{"assay":ma,"element":me,"interaction":mi,"interaction_key":key}
    return result

print("Loading unified data and constructing connected leakage groups...")
d=pd.read_csv(INPUT,sep="\t",low_memory=False); d=d[d.source_partition.eq("development")].reset_index(drop=True)
split=d.original_split.astype(str).str.lower(); pool=np.flatnonzero(split.isin(["train","validation","val"])); locked=np.flatnonzero(split.eq("test"))
groups_all=connected_groups(d); groups=groups_all[pool]; y=d.target_model.to_numpy(np.float32)
print("Device:",DEVICE,"| pool:",len(pool),"| locked test:",len(locked),"| connected groups:",len(np.unique(groups)))

vectorizer=CountVectorizer(analyzer="char",ngram_range=(3,6),lowercase=False,dtype=np.float32)
original_train=np.flatnonzero(split.eq("train")); vectorizer.fit(clean(d.iloc[original_train].reference_sequence)+clean(d.iloc[original_train].alternative_sequence)); X=delta(vectorizer,d)

outer=GroupKFold(n_splits=3); tuning_rows=[]; outer_rows=[]; chosen=[]; chosen_epochs=[]
for outer_fold,(otr_rel,ova_rel) in enumerate(outer.split(pool,groups=groups),1):
    otr,ova=pool[otr_rel],pool[ova_rel]; inner_groups=groups_all[otr]; inner=GroupKFold(n_splits=2)
    config_scores={c["name"]:[] for c in CONFIGS}; config_epochs={c["name"]:[] for c in CONFIGS}
    print(f"Outer fold {outer_fold}/3: nested tuning...")
    for inner_fold,(itr_rel,iva_rel) in enumerate(inner.split(otr,groups=inner_groups),1):
        itr,iva=otr[itr_rel],otr[iva_rel]
        for cfg in CONFIGS:
            z=fit_once(d,X,y,itr,iva,cfg); config_scores[cfg["name"]].append(z["macro_element_rmse"]); config_epochs[cfg["name"]].append(z["best_epoch"])
            tuning_rows.append({"outer_fold":outer_fold,"inner_fold":inner_fold,"config":cfg["name"],**z})
    winner=min(CONFIGS,key=lambda c:np.mean(config_scores[c["name"]])); epochs=int(np.median(config_epochs[winner["name"]]))
    chosen.append(winner["name"]); chosen_epochs.append(epochs)
    z,pv,_,_,_=fit_once(d,X,y,otr,ova,winner,fixed_epochs=epochs,return_objects=True)
    outer_rows.append({"outer_fold":outer_fold,"selected_config":winner["name"],"fixed_epochs":epochs,**z})
    print("  selected",winner["name"],"epochs",epochs,"outer macro-RMSE",round(z["macro_element_rmse"],4))

# Final choice: lowest mean outer performance among selections; deterministic tie break by frequency.
outer_df=pd.DataFrame(outer_rows); frequency=Counter(chosen)
candidate_names=sorted(frequency,key=lambda n:(-frequency[n],outer_df.loc[outer_df.selected_config.eq(n),"macro_element_rmse"].mean()))
final_name=candidate_names[0]; final_cfg=next(c for c in CONFIGS if c["name"]==final_name); final_epochs=int(np.median([e for n,e in zip(chosen,chosen_epochs) if n==final_name]))
print("Final nested-CV selection:",final_name,"epochs:",final_epochs)
z,pred,model,ridge,mappings=fit_once(d,X,y,pool,locked,final_cfg,fixed_epochs=final_epochs,return_objects=True)
ridge_pred=ridge.predict(X[locked]); rows=[]
for name,p in [("hiera_nested_selected",pred),("ridge_anchor",ridge_pred),("zero_baseline",np.zeros(len(locked)))]: rows.append({"model":name,**metrics(y[locked],p)})
metric=pd.DataFrame(rows)

predtab=d.iloc[locked][["row_id","element_id","assay_id","position_group","sequence_pair_hash","target_model"]].copy(); predtab["hiera_prediction"]=pred; predtab["ridge_prediction"]=ridge_pred
pd.DataFrame(tuning_rows).to_csv(OUTPUT/"nested_inner_tuning.tsv",sep="\t",index=False); outer_df.to_csv(OUTPUT/"nested_outer_performance.tsv",sep="\t",index=False); metric.to_csv(OUTPUT/"locked_test_point_estimates.tsv",sep="\t",index=False); predtab.to_csv(OUTPUT/"locked_test_predictions.tsv",sep="\t",index=False)
joblib.dump(vectorizer,OUTPUT/"hiera_nested_vectorizer.joblib"); joblib.dump(ridge,OUTPUT/"hiera_nested_ridge_anchor.joblib")
torch.save({"state_dict":model.state_dict(),"config":final_cfg,"mappings":mappings,"n_features":X.shape[1],"epochs":final_epochs},OUTPUT/"hiera_nested_selected_model.pt")
summary={"selected_config":final_name,"selected_epochs":final_epochs,"selection_frequency":dict(frequency),"outer_folds":3,"inner_folds":2,"connected_groups":int(len(np.unique(groups_all))),"locked_test_used_for_selection":False,"candidate_configs":CONFIGS,"locked_test_metrics":metric.to_dict("records")}
(OUTPUT/"nested_selection_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nNESTED GROUPED LOCKED-TEST RESULTS"); print(metric.to_string(index=False,float_format=lambda x:f"{x:.4f}")); print("\nStep 46 nested grouped HIERA selection: COMPLETED"); print("Saved:",OUTPUT)
