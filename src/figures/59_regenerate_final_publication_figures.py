import os
#!/usr/bin/env python3
"""Regenerate final HIERA-VEP publication figures directly from final result tables."""
from pathlib import Path
import json, textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import r2_score

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "upload"
RESULTS = ROOT / "results"
OUT = ROOT / "redesigned_final_publication_figures"
MAIN = OUT / "main_manuscript_figures"
SUPP = OUT / "supplementary_figures"
for p in (MAIN, SUPP): p.mkdir(parents=True, exist_ok=True)

NAVY="#173F5F"; BLUE="#20639B"; CYAN="#3CAEA3"; GREEN="#2A9D8F"; GOLD="#F6AE2D"
ORANGE="#F26419"; RED="#C44E52"; PURPLE="#6F4E9C"; GREY="#6B7280"; LIGHT="#E8EEF3"
MODEL_COLORS={"context_gated_kmer":NAVY,"context_selected":NAVY,"sequence_only_kmer":CYAN,
              "sequence_only":CYAN,"frozen_nt_ridge":ORANGE,"zero_baseline":"#B9BDC5",
              "locked_hiera":NAVY,"ridge_alpha100":CYAN,"ridge_anchor":PURPLE,
              "adapted_kmer":NAVY,"adapted_frozen_nt":ORANGE,"adapted_hybrid":PURPLE,
              "adapted_equal_weight_ensemble":GREEN,"zero_shot_kmer":"#9AA0A6","zero_shot_frozen_nt":"#D0D3D8"}

mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":10.5,
 "axes.labelsize":9,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.8,
 "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
sns.set_style("ticks")

def find_input(name):
 """Resolve an exported table from upload/ or any existing results/ subfolder."""
 direct = DATA / name
 if direct.exists():
  return direct
 matches = sorted(RESULTS.rglob(name)) if RESULTS.exists() else []
 if not matches:
  raise FileNotFoundError(
   f"Required input '{name}' was not found. Searched:\n"
   f"  {direct}\n  {RESULTS}\\**\\{name}\n"
   "Run the analysis step that creates this table or place a copy in the upload folder."
  )
 if len(matches) > 1:
  print(f"WARNING: multiple copies of {name}; using {matches[0]}")
 return matches[0]

def read(name): return pd.read_csv(find_input(name),sep="\t",low_memory=False)
def label(ax, letter): ax.text(-.10,1.06,letter,transform=ax.transAxes,fontsize=15,fontweight="bold",va="top")
def clean(ax): ax.grid(axis="y",color="#E5E7EB",lw=.6,zorder=0); ax.set_axisbelow(True)
def save(fig, folder, stem):
 for ext,kwargs in [("pdf",{}),("svg",{}),("png",{"dpi":600}),("tiff",{"dpi":600,"pil_kwargs":{"compression":"tiff_lzw"}})]:
  with open(folder/f"{stem}.{ext}","wb") as fh:
   fig.savefig(fh,format=ext,bbox_inches="tight",**kwargs)
 plt.close(fig)

def forest(ax,d,label_col,point="point_estimate",lo="ci_lower_95",hi="ci_upper_95",null=0,colors=None):
 d=d.dropna(subset=[point,lo,hi]).copy().iloc[::-1]; y=np.arange(len(d))
 if colors is None: colors=[NAVY]*len(d)
 for i,(_,r) in enumerate(d.iterrows()):
  ax.errorbar(r[point],i,xerr=[[r[point]-r[lo]],[r[hi]-r[point]]],fmt="o",ms=4,
              color=colors[i] if isinstance(colors,(list,np.ndarray,pd.Series)) else colors,
              ecolor=colors[i] if isinstance(colors,(list,np.ndarray,pd.Series)) else colors,capsize=2,lw=1.2)
 ax.axvline(null,ls="--",lw=.8,c=GREY);ax.set_yticks(y,d[label_col].astype(str));return ax

def fig1():
 manifest=read("hiera_unified_dataset_manifest.tsv");leak=read("hiera_sequence_leakage_audit.tsv")
 enc=read("element_encode_promoter_summary.tsv");cons=read("element_conservation_summary.tsv")
 fig=plt.figure(figsize=(13.2,8.8));gs=GridSpec(2,3,figure=fig,height_ratios=[.85,1],hspace=.38,wspace=.34)
 ax=fig.add_subplot(gs[0,:]);ax.axis("off");label(ax,"a")
 boxes=[("31 assays\n36,188 variants",.02),("Δ 3–6-mer\n5,440 features",.21),("cCRE · promoter\nconservation",.40),("context-gated\nsequence effects",.59),("grouped locked tests\nexternal transfer",.78)]
 for i,(txt,x) in enumerate(boxes):
  patch=FancyBboxPatch((x,.30),.16,.38,boxstyle="round,pad=.02,rounding_size=.025",fc=LIGHT,ec=NAVY,lw=1.5)
  ax.add_patch(patch);ax.text(x+.08,.49,txt,ha="center",va="center",fontsize=10,fontweight="bold" if i in [0,3] else None)
  if i<len(boxes)-1:ax.add_patch(FancyArrowPatch((x+.16,.49),(boxes[i+1][1],.49),arrowstyle="-|>",mutation_scale=13,lw=1.2,color=GREY))
 ax.text(.02,.83,"HIERA-VEP: transparent sequence modeling conditioned on genomic context",fontsize=14,fontweight="bold")
 ax=fig.add_subplot(gs[1,0]);label(ax,"b")
 q=manifest.groupby(["source_partition","assay_family"],as_index=False).variants.sum();pivot=q.pivot(index="source_partition",columns="assay_family",values="variants").fillna(0)
 pivot.plot(kind="bar",stacked=True,ax=ax,color=[NAVY,CYAN,ORANGE,PURPLE],width=.65);ax.set(title="Cohort and assay composition",xlabel="",ylabel="Variants");ax.tick_params(axis="x",rotation=0);ax.legend(frameon=False,title="",fontsize=6.5);clean(ax)
 ax=fig.add_subplot(gs[1,1]);label(ax,"c")
 y=np.arange(len(leak));ax.barh(y,leak.development_unique,color=BLUE,label="development unique");ax.barh(y,leak.external_unique.fillna(0),left=leak.development_unique,color=GOLD,label="external unique")
 ax.set_yticks(y,["External sequence-pair overlap","Within-element split overlap"]);ax.set_xlabel("Unique sequence pairs");ax.set_title("Leakage audit: zero observed overlap")
 for i,r in leak.iterrows():ax.text(max(r.development_unique*.5,100),i,f"overlap = {int(r.overlap)}  ·  {r.status}",ha="center",va="center",color="white",fontweight="bold",fontsize=8)
 ax.legend(frameon=False,fontsize=6.5);clean(ax)
 ax=fig.add_subplot(gs[1,2]);label(ax,"d")
 m=enc.merge(cons,on=["element_id","variants"],how="inner").set_index("element_id")
 cols=["ccre_overlap_fraction","promoter_overlap_fraction","phyloP_conserved_fraction","phastCons_high_fraction"]
 sns.heatmap(m[cols],cmap="crest",vmin=0,vmax=1,ax=ax,cbar_kws={"label":"Variant fraction","shrink":.78});ax.set(title="Element-resolved genomic context",xlabel="",ylabel="");ax.set_xticklabels(["cCRE","promoter","phyloP > 2","phastCons > 0.9"],rotation=28,ha="right");ax.tick_params(axis="y",labelsize=6)
 fig.suptitle("Figure 1 | Study architecture and genomic context",x=.01,ha="left",fontsize=15,fontweight="bold",y=.995);save(fig,MAIN,"Figure_1_study_architecture")

def fig2():
 pred=read("locked_context_predictions.tsv");models=read("locked_model_comparison.tsv");p1=read("context_vs_sequence_paired_comparison.tsv");p2=read("context_vs_nt_paired_comparison.tsv");cal=read("calibration_deciles.tsv")
 fig=plt.figure(figsize=(13.2,9.4));gs=GridSpec(2,3,figure=fig,hspace=.40,wspace=.38)
 ax=fig.add_subplot(gs[0,0]);label(ax,"a");h=ax.hexbin(pred.target_model,pred.context_selected_prediction,gridsize=50,mincnt=1,bins="log",cmap="viridis");lim=np.quantile(np.abs(pred.target_model),.995);ax.plot([-lim,lim],[-lim,lim],"--",c=ORANGE,lw=1);ax.set(xlabel="Observed standardized effect",ylabel="Predicted effect",title="Locked test: observed versus predicted");fig.colorbar(h,ax=ax,label="log count",shrink=.78)
 ax=fig.add_subplot(gs[0,1]);label(ax,"b");metrics=["r2","pearson_r","spearman_rho","direction_accuracy"];long=models.melt(id_vars="model",value_vars=metrics,var_name="metric",value_name="value")
 for i,m in enumerate(models.model):
  q=long[long.model.eq(m)];ax.plot(q.metric,q.value,"o-",label=m.replace("_"," "),color=MODEL_COLORS.get(m,GREY),lw=1.5)
 ax.axhline(0,c=GREY,lw=.7);ax.set(title="Locked model comparison",xlabel="",ylabel="Performance");ax.set_xticks(range(len(metrics)),["R²","Pearson r","Spearman ρ","Direction\naccuracy"]);ax.legend(frameon=False,fontsize=6.2);clean(ax)
 ax=fig.add_subplot(gs[0,2]);label(ax,"c");a=p1.assign(comparator="sequence only");b=p2.assign(comparator="frozen NT");b["measure"]=b.measure.str.replace("_context_vs_nt","",regex=False);q=pd.concat([a,b]);q["label"]=q.measure.str.replace("_"," ")+" | "+q.comparator
 colors=[NAVY if "sequence" in x else ORANGE for x in q.iloc[::-1].label];forest(ax,q,"label",colors=colors);ax.set(title="Paired gains of context gating",xlabel="Improvement (95% clustered CI)");ax.tick_params(axis="y",labelsize=6)
 ax=fig.add_subplot(gs[1,0]);label(ax,"d");ax.errorbar(cal.mean_prediction,cal.mean_observed,yerr=cal.observed_sd/np.sqrt(cal.n)*1.96,fmt="o-",color=NAVY,capsize=2);lo=min(cal.mean_prediction.min(),cal.mean_observed.min());hi=max(cal.mean_prediction.max(),cal.mean_observed.max());ax.plot([lo,hi],[lo,hi],"--",c=ORANGE);ax.set(xlabel="Mean prediction",ylabel="Mean observed effect",title="Calibration by prediction decile");clean(ax)
 ax=fig.add_subplot(gs[1,1]);label(ax,"e");vals=[]
 for el,g in pred.groupby("element_id"):
  if len(g)>=30:vals.append((el,r2_score(g.target_model,g.context_selected_prediction),len(g)))
 ed=pd.DataFrame(vals,columns=["element","r2","n"]).sort_values("r2");colors=np.where(ed.r2>=0,BLUE,ORANGE);ax.barh(ed.element,ed.r2,color=colors);ax.axvline(0,c=GREY,lw=.8);ax.set(title="Element-level locked performance",xlabel="R²",ylabel="");ax.tick_params(axis="y",labelsize=6);clean(ax)
 ax=fig.add_subplot(gs[1,2]);label(ax,"f");q=models[models.model.ne("zero_baseline")];x=np.arange(len(q));ax.bar(x-.18,q.observed_sd,.36,label="observed SD",color="#B8C4CE");ax.bar(x+.18,q.predicted_sd,.36,label="predicted SD",color=[MODEL_COLORS.get(m,GREY) for m in q.model]);ax.set_xticks(x,[m.replace("_","\n") for m in q.model]);ax.set_ylabel("Standard deviation");ax.set_title("Prediction variance remains compressed");ax.legend(frameon=False);clean(ax)
 fig.suptitle("Figure 2 | Locked predictive performance and calibration",x=.01,ha="left",fontsize=15,fontweight="bold",y=.995);save(fig,MAIN,"Figure_2_locked_performance")

def fig3():
 ext=read("locked_external_point_estimates.tsv");paired=read("locked_hiera_vs_ridge100.tsv");adapt=read("assay_adaptive_point_estimates.tsv");few=read("few_shot_learning_curve_summary.tsv");cond=read("condition_model_confidence_intervals.tsv")
 fig=plt.figure(figsize=(13.2,9.5));gs=GridSpec(2,3,figure=fig,hspace=.43,wspace=.48)
 tier_names={"cross_species_transfer":"cross-species","primary_human_external":"primary human","secondary_splicing_transfer":"splicing"};model_names={"locked_hiera":"HIERA-VEP","ridge_alpha100":"Ridge-100","ridge_anchor":"Ridge anchor","zero_baseline":"zero baseline"}
 ax=fig.add_subplot(gs[0,0]);label(ax,"a");q=ext[ext.evaluation_level.eq("tier")].pivot(index="model",columns="evaluation_group",values="pearson_r").rename(index=model_names,columns=tier_names);sns.heatmap(q,cmap="vlag",center=0,annot=True,fmt=".2f",ax=ax,cbar_kws={"label":"Pearson r","shrink":.68,"pad":.02});ax.set(title="Locked zero-shot external transfer",xlabel="",ylabel="");ax.set_xticklabels(ax.get_xticklabels(),rotation=22,ha="right");ax.tick_params(axis="y",labelsize=6.5)
 ax=fig.add_subplot(gs[0,1]);label(ax,"b");q=paired.copy();metric_names={"mae_reduction":"MAE reduction","pearson_gain":"Pearson gain","rmse_reduction":"RMSE reduction","spearman_gain":"Spearman gain"};q["label"]=q.measure.map(metric_names)+" | "+q.validation_tier.map(tier_names);forest(ax,q,"label",point="mean_difference");ax.set(title="HIERA-VEP versus Ridge-100",xlabel="Paired difference (95% CI)");ax.tick_params(axis="y",labelsize=5.7)
 ax=fig.add_subplot(gs[0,2]);label(ax,"c");adapt_names={"adapted_equal_weight_ensemble":"ensemble","adapted_frozen_nt":"adapted NT","adapted_hybrid":"hybrid","adapted_kmer":"adapted k-mer","zero_shot_frozen_nt":"zero-shot NT","zero_shot_kmer":"zero-shot k-mer"};q=adapt[adapt.element_name.eq("ALL")].pivot(index="model",columns="validation_tier",values="pearson_r").rename(index=adapt_names,columns=tier_names);sns.heatmap(q,cmap="crest",annot=True,fmt=".2f",vmin=0,vmax=.65,ax=ax,cbar_kws={"label":"Pearson r","shrink":.68,"pad":.02});ax.set(title="Assay adaptation",xlabel="",ylabel="");ax.set_xticklabels(ax.get_xticklabels(),rotation=22,ha="right");ax.tick_params(axis="y",labelsize=6.5)
 ax=fig.add_subplot(gs[1,0]);label(ax,"d");q=few[(few.measure.eq("pearson_r")) & few.element_name.eq("ALL")]
 for (tier,metric),g in q.groupby(["validation_tier","measure"]):
  style="-";color={"primary_human_external":NAVY,"cross_species_transfer":ORANGE,"secondary_splicing_transfer":PURPLE}.get(tier,GREY);g=g.sort_values("support_fraction_of_available_pool");ax.plot(g.support_fraction_of_available_pool,g["mean"],marker="o",ls=style,color=color,label=tier_names.get(tier,tier));ax.fill_between(g.support_fraction_of_available_pool,g.lower_95_across_repeats,g.upper_95_across_repeats,color=color,alpha=.10)
 ax.axhline(0,c=GREY,lw=.7);ax.set(title="Few-shot adaptation learning curves",xlabel="Support fraction",ylabel="Correlation");ax.legend(frameon=False,fontsize=5.5);clean(ax)
 ax=fig.add_subplot(gs[1,1]);label(ax,"e");q=cond[cond.model.eq("condition_differential_kmer") & cond.measure.isin(["r2","pearson_r","spearman_rho","direction_accuracy"])].copy();q["label"]=q.measure.str.replace("_"," ");forest(ax,q,"label");ax.set(title="Condition-differential SUL1 prediction",xlabel="Estimate (95% clustered CI)")
 ax=fig.add_subplot(gs[1,2]);label(ax,"f");q=adapt[adapt.element_name.eq("ALL") & adapt.model.isin(["adapted_kmer","zero_shot_kmer"])];p=q.pivot(index="validation_tier",columns="model",values="pearson_r");p=p.sort_values("adapted_kmer");y=np.arange(len(p));ax.hlines(y,p.zero_shot_kmer,p.adapted_kmer,color="#B7C3CC",lw=3);ax.scatter(p.zero_shot_kmer,y,color=GREY,label="zero shot",zorder=3);ax.scatter(p.adapted_kmer,y,color=NAVY,label="adapted",zorder=3);ax.set_yticks(y,[x.replace("_"," ") for x in p.index]);ax.set(xlabel="Pearson r",title="Adaptation benefit by tier");ax.legend(frameon=False);clean(ax)
 fig.suptitle("Figure 3 | External transfer and assay adaptation",x=.01,ha="left",fontsize=15,fontweight="bold",y=.995);save(fig,MAIN,"Figure_3_external_transfer")

def fig4():
 block=read("attribution_block_summary.tsv");attr=read("locked_variant_exact_attributions.tsv");abl=read("locked_ablation_paired_bootstrap.tsv");km=read("canonical_kmer_attribution_ranking.tsv");cf=read("context_block_counterfactual_ablation.tsv")
 fig=plt.figure(figsize=(13.2,9.5));gs=GridSpec(2,3,figure=fig,hspace=.43,wspace=.43)
 ax=fig.add_subplot(gs[0,0]);label(ax,"a");q=block[block.block.ne("intercept")].sort_values("mean_absolute_contribution");ax.barh(q.block.str.replace("_"," "),q.mean_absolute_contribution,color=BLUE);ax.set(title="Exact additive contribution magnitude",xlabel="Mean absolute contribution");clean(ax)
 ax=fig.add_subplot(gs[0,1]);label(ax,"b");q=block[block.block.ne("intercept")].sort_values("pearson_with_prediction");ax.barh(q.block.str.replace("_"," "),q.pearson_with_prediction,color=np.where(q.pearson_with_prediction>0,NAVY,ORANGE));ax.axvline(0,c=GREY,lw=.8);ax.set(title="Contribution–prediction coupling",xlabel="Pearson r");clean(ax)
 ax=fig.add_subplot(gs[0,2]);label(ax,"c");q=abl[abl.measure.eq("rmse_loss")].copy();q["label"]=q.ablated_model.str.replace("without_","− ").str.replace("_"," ");forest(ax,q,"label");ax.set(title="Locked feature-family ablation",xlabel="RMSE loss (95% CI)");ax.tick_params(axis="y",labelsize=6)
 ax=fig.add_subplot(gs[1,0]);label(ax,"d");q=km.nlargest(14,"mean_absolute_contribution").sort_values("mean_signed_contribution");colors=np.where(q.mean_signed_contribution>=0,CYAN,ORANGE);ax.barh(q.block.str.replace("gate_","")+" | "+q.canonical_kmer,q.mean_signed_contribution,color=colors);ax.axvline(0,c=GREY,lw=.8);ax.set(title="Most influential canonical k-mers",xlabel="Mean signed contribution");ax.tick_params(axis="y",labelsize=6);clean(ax)
 ax=fig.add_subplot(gs[1,1]);label(ax,"e");cols=["sequence_main","context_additive","gate_ccre_overlap","gate_promoter_overlap","gate_phyloP100way","gate_phastCons100way"];sample=attr[cols].sample(min(2500,len(attr)),random_state=7).melt(var_name="block",value_name="contribution");sns.violinplot(data=sample,y="block",x="contribution",orient="h",cut=0,inner="quartile",palette="crest",hue="block",legend=False,ax=ax);ax.axvline(0,c=GREY,lw=.8);ax.set(title="Variant-level attribution distributions",xlabel="Contribution",ylabel="");ax.tick_params(axis="y",labelsize=6)
 ax=fig.add_subplot(gs[1,2]);label(ax,"f");q=cf[cf.removed_block.ne("none")].sort_values("mean_absolute_prediction_change");ax.barh(q.removed_block.str.replace("_"," "),q.mean_absolute_prediction_change,color=PURPLE);ax.set(title="Counterfactual prediction sensitivity",xlabel="Mean |prediction change|");ax.tick_params(axis="y",labelsize=6);clean(ax)
 fig.suptitle("Figure 4 | Exact interpretation and feature dependence",x=.01,ha="left",fontsize=15,fontweight="bold",y=.995);save(fig,MAIN,"Figure_4_exact_interpretation")

def fig5():
 enc=read("element_encode_promoter_summary.tsv");cons=read("element_conservation_summary.tsv");assoc=read("motif_conservation_associations.tsv");fung=read("condition_effect_motif_enrichment.tsv");gtex=read("gtex_concordance_tests.tsv");unc=read("calibration_uncertainty_estimates.tsv");risk=read("uncertainty_selective_risk_curve.tsv")
 fig=plt.figure(figsize=(13.2,9.6));gs=GridSpec(2,3,figure=fig,hspace=.44,wspace=.56)
 ax=fig.add_subplot(gs[0,0]);label(ax,"a");q=enc.set_index("element_id")[["ccre_overlap_fraction","promoter_overlap_fraction"]];sns.heatmap(q,cmap="crest",vmin=0,vmax=1,ax=ax,cbar_kws={"label":"Variant fraction","shrink":.75});ax.set(title="ENCODE/GENCODE context",xlabel="",ylabel="");ax.set_xticklabels(["cCRE overlap","promoter overlap"],rotation=25,ha="right");ax.tick_params(axis="y",labelsize=6)
 ax=fig.add_subplot(gs[0,1]);label(ax,"b");q=cons.set_index("element_id")[["phyloP_conserved_fraction","phastCons_high_fraction"]];sns.heatmap(q,cmap="mako",vmin=0,vmax=1,ax=ax,cbar_kws={"label":"Variant fraction","shrink":.75});ax.set(title="Conservation context",xlabel="",ylabel="");ax.set_xticklabels(["phyloP > 2","phastCons > 0.9"],rotation=25,ha="right");ax.tick_params(axis="y",labelsize=6)
 ax=fig.add_subplot(gs[0,2]);label(ax,"c");q=assoc.copy();short={"abs_observed_vs_motif":"observed vs motif","abs_prediction_vs_motif":"prediction vs motif","abs_context_attribution_vs_motif":"context attribution vs motif","abs_observed_vs_phyloP":"observed vs phyloP","abs_observed_vs_phastCons":"observed vs phastCons"};q["label"]=q.association.map(short).fillna(q.association.str.replace("_"," "));forest(ax,q,"label",point="point_spearman");ax.set(title="Position-level annotation concordance",xlabel="Spearman ρ (95% clustered CI)");ax.tick_params(axis="y",labelsize=5.8)
 ax=fig.add_subplot(gs[1,0]);label(ax,"d");q=fung.nsmallest(12,"fdr_bh").sort_values("odds_ratio");colors=np.where(q.fdr_bh<.05,RED,"#B9BDC5");ax.barh(q.motif_name,q.odds_ratio,color=colors);ax.axvline(1,ls="--",c=GREY,lw=.8);ax.set(title="SUL1 condition-effect motif enrichment",xlabel="Odds ratio");clean(ax)
 ax=fig.add_subplot(gs[1,1]);label(ax,"e");ax.axis("off");summary=json.loads(find_input("gtex_eqtl_concordance_summary.json").read_text());ax.add_patch(FancyBboxPatch((.05,.12),.90,.75,boxstyle="round,pad=.025",fc="#F7F8FA",ec=GREY,lw=1.2));ax.text(.5,.78,"Exact GTEx concordance",ha="center",fontsize=12,fontweight="bold");ax.text(.5,.59,f"{summary.get('exact_variants_with_significant_gtex_eqtl',1):,}",ha="center",fontsize=30,fontweight="bold",color=ORANGE);ax.text(.5,.47,"exact significant eQTL match",ha="center",fontsize=9);ax.text(.5,.30,"4,408 locked variants · 1,474 positions",ha="center",fontsize=9);ax.text(.5,.20,"Sparse overlap; not broad validation",ha="center",fontsize=9,color=RED,fontweight="bold")
 ax=fig.add_subplot(gs[1,2]);label(ax,"f");q=unc.copy();short={"calibration_intercept":"calibration intercept","calibration_slope":"calibration slope","mean_residual":"mean residual","disagreement_vs_absolute_error_spearman":"disagreement–error ρ","disagreement_high_error_auc":"high-error AUC"};q["label"]=q.measure.map(short).fillna(q.measure.str.replace("_"," "));forest(ax,q,"label",null=0);ax.set(title="Calibration and uncertainty diagnostics",xlabel="Estimate (95% clustered CI)");ax.tick_params(axis="y",labelsize=6)
 fig.suptitle("Figure 5 | Biological concordance and model limitations",x=.01,ha="left",fontsize=15,fontweight="bold",y=.995);save(fig,MAIN,"Figure_5_biological_concordance_limits")

def supplements():
 # S1 sequence and robustness
 mut=read("mutation_class_performance.tsv");quart=read("performance_by_effect_quartile.tsv");sens=read("robustness_sensitivity.tsv");cons=read("cross_element_kmer_consistency.tsv")
 fig,axs=plt.subplots(2,2,figsize=(11,8));
 q=mut.sort_values("r2");axs[0,0].barh(q.mutation_class,q.r2,color=GREEN);axs[0,0].set(title="Performance by mutation class",xlabel="R²");label(axs[0,0],"a")
 for metric,color in [("pearson_r",BLUE),("direction_accuracy",ORANGE)]:axs[0,1].plot(quart.effect_bin,quart[metric],"o-",label=metric.replace("_"," "),color=color)
 axs[0,1].legend(frameon=False);axs[0,1].set(title="Performance by effect-size quartile",ylabel="Performance");label(axs[0,1],"b")
 metrics=[c for c in ["r2","pearson_r","direction_accuracy"] if c in sens];sens.set_index(sens.columns[0])[metrics].plot(kind="bar",ax=axs[1,0],color=[BLUE,CYAN,ORANGE]);axs[1,0].set(title="Extreme-effect sensitivity",xlabel="");label(axs[1,0],"c")
 num=cons.select_dtypes(include=np.number);sns.heatmap(num.corr(),cmap="vlag",center=0,ax=axs[1,1],cbar_kws={"shrink":.7});axs[1,1].set_title("Cross-element feature consistency");label(axs[1,1],"d")
 fig.suptitle("Supplementary Figure S1 | Sequence robustness and heterogeneity",x=.02,ha="left",fontsize=14,fontweight="bold");fig.tight_layout(rect=[0,0,1,.96]);save(fig,SUPP,"Supplementary_Figure_S1_sequence_robustness")
 # S2 external context
 pad=read("performance_by_padding_context.tsv");pa=read("padding_error_associations.tsv");fs=read("few_shot_improvement_over_zero_shot.tsv");ext=read("external_validation_point_estimates.tsv")
 fig,axs=plt.subplots(2,2,figsize=(11,8));
 q=pad[pad.model.isin(["shared_kmer_ridge","frozen_nt_ridge"])].pivot_table(index="padding_bin",columns="model",values="pearson_r",aggfunc="mean");q.plot(kind="bar",ax=axs[0,0],color=[ORANGE,NAVY]);axs[0,0].set(title="Performance by ambiguous-padding bin",ylabel="Pearson r",xlabel="");label(axs[0,0],"a")
 q=pa.pivot(index="element_name",columns="model",values="spearman_padding_vs_absolute_error");sns.heatmap(q,cmap="vlag",center=0,annot=True,fmt=".2f",ax=axs[0,1]);axs[0,1].set_title("Padding–error association");label(axs[0,1],"b")
 q=fs[(fs.measure.eq("pearson_gain"))&fs.element_name.eq("ALL")];
 for tier,g in q.groupby("validation_tier"):axs[1,0].plot(g.support_fraction_of_available_pool,g.mean_improvement,"o-",label=tier.replace("_"," "))
 axs[1,0].axhline(0,c=GREY,lw=.7);axs[1,0].legend(frameon=False,fontsize=6);axs[1,0].set(title="Few-shot gain over zero shot",xlabel="Support fraction",ylabel="Pearson gain");label(axs[1,0],"c")
 q=ext[ext.model.eq("shared_kmer_ridge") & ext.evaluation_group.ne("development_internal_test")];sns.scatterplot(data=q,x="pearson_r",y="element_name",hue="evaluation_group",s=60,ax=axs[1,1]);axs[1,1].axvline(0,c=GREY,lw=.7);axs[1,1].set_title("Zero-shot external performance");axs[1,1].legend(frameon=False,fontsize=6);label(axs[1,1],"d")
 fig.suptitle("Supplementary Figure S2 | External context sensitivity",x=.02,ha="left",fontsize=14,fontweight="bold");fig.tight_layout(rect=[0,0,1,.96]);save(fig,SUPP,"Supplementary_Figure_S2_external_context")
 # S3 motif details
 pos=read("position_level_motif_enrichment.tsv");fung=read("condition_effect_motif_enrichment.tsv");m=read("motif_conservation_associations.tsv");g=read("gtex_concordance_tests.tsv")
 fig,axs=plt.subplots(2,2,figsize=(11,8));
 q=pos.nsmallest(15,"fdr_bh").sort_values("odds_ratio");axs[0,0].barh(q.motif_name,q.odds_ratio,color=np.where(q.fdr_bh<.05,RED,"#B9BDC5"));axs[0,0].axvline(1,ls="--",c=GREY);axs[0,0].set_title("Human motif enrichment (no FDR hits)");label(axs[0,0],"a")
 q=fung.nsmallest(15,"fdr_bh").sort_values("odds_ratio");axs[0,1].barh(q.motif_name,q.odds_ratio,color=RED);axs[0,1].axvline(1,ls="--",c=GREY);axs[0,1].set_title("Fungal motif enrichment");label(axs[0,1],"b")
 q=m.copy();q["label"]=q.association.str.replace("_"," ");forest(axs[1,0],q,"label",point="point_spearman");axs[1,0].set_title("Motif/conservation associations");label(axs[1,0],"c")
 axs[1,1].axis("off");txt="\n".join([f"{r.analysis.replace('_',' ')}\nOR={r.odds_ratio if pd.notna(r.odds_ratio) else 'NA'}; P={r.p_value:.3g}; FDR={r.fdr_bh:.3g}" for _,r in g.iterrows()]);axs[1,1].text(.03,.92,"Exact GTEx concordance tests",fontsize=11,fontweight="bold",va="top");axs[1,1].text(.03,.72,txt,fontsize=9,va="top",linespacing=1.6);label(axs[1,1],"d")
 fig.suptitle("Supplementary Figure S3 | Motif and eQTL concordance",x=.02,ha="left",fontsize=14,fontweight="bold");fig.tight_layout(rect=[0,0,1,.96]);save(fig,SUPP,"Supplementary_Figure_S3_motif_eqtl")
 # S4 failure details
 dec=read("calibration_deciles.tsv");risk=read("uncertainty_selective_risk_curve.tsv");fail=read("structured_failure_strata.tsv");top=read("top_100_locked_failures.tsv")
 fig,axs=plt.subplots(2,2,figsize=(11,8));axs[0,0].plot(dec.mean_prediction,dec.mean_observed,"o-",color=NAVY);axs[0,0].plot([dec.mean_prediction.min(),dec.mean_prediction.max()],[dec.mean_prediction.min(),dec.mean_prediction.max()],"--",color=ORANGE);axs[0,0].set(title="Calibration deciles",xlabel="Prediction",ylabel="Observed");label(axs[0,0],"a")
 axs[0,1].plot(risk.retained_fraction,risk.rmse,"o-",label="RMSE",color=RED);axs[0,1].plot(risk.retained_fraction,risk.mae,"o-",label="MAE",color=BLUE);axs[0,1].legend(frameon=False);axs[0,1].set(title="Selective-risk curve",xlabel="Retained fraction",ylabel="Error");label(axs[0,1],"b")
 q=fail[fail.stratification.eq("element_id")].nlargest(12,"rmse").sort_values("rmse");axs[1,0].barh(q.level,q.rmse,color=ORANGE);axs[1,0].set(title="Worst element-level RMSE",xlabel="RMSE");label(axs[1,0],"c")
 if "absolute_error" not in top and {"target_model","prediction"}.issubset(top):top["absolute_error"]=(top.target_model-top.prediction).abs()
 axs[1,1].hist(top.absolute_error,bins=20,color=PURPLE,alpha=.85);axs[1,1].set(title="Top-100 locked failures",xlabel="Absolute error",ylabel="Variants");label(axs[1,1],"d")
 fig.suptitle("Supplementary Figure S4 | Calibration and failure analysis",x=.02,ha="left",fontsize=14,fontweight="bold");fig.tight_layout(rect=[0,0,1,.96]);save(fig,SUPP,"Supplementary_Figure_S4_failure_analysis")

def main():
 fig1();fig2();fig3();fig4();fig5();supplements()
 manifest={"main_figures":5,"supplementary_figures":4,"formats":["PDF","SVG","PNG 600 dpi","TIFF 600 dpi"],"source":"Final uploaded result tables","policy":"Null, weak external and failure results retained."}
 (OUT/"figure_set_summary.json").write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2));print("Saved:",OUT)
if __name__=="__main__":main()
