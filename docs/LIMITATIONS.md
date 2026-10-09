# Scientific limitations

External zero-shot transfer was weak and heterogeneous; external R² values were around zero or negative across several tiers. HIERA-VEP did not consistently outperform Ridge externally. Assay-specific adaptation is a secondary analysis requiring target-assay labels and is not zero-shot generalization.

Predicted effects were compressed relative to observed effects, and large-effect variants were a major failure mode. Model disagreement is only an uncertainty proxy, not a calibrated predictive interval.

Position-level motif enrichment did not yield broadly FDR-supported motifs (zero significant position-level motifs in `results/interpretation/motif_conservation_validation_summary.json`). Exact significant GTEx overlap was extremely sparse: one variant/position in `results/interpretation/gtex_eqtl_concordance_summary.json`. Absence of an overlap is not evidence of absent regulatory activity.

Conservation and motif analyses are concordance or computational annotation, not causal validation. Conservation was itself a model input, so attribution–conservation agreement is not independent validation. Earlier Step 53A language deferring eQTL analysis is retained as historical provenance; Step 53B provides the subsequent sparse exact-match analysis.

The software is research-only and is not a clinical pathogenicity classifier. No universal generalization, experimental validation, or causal interpretation is claimed.
