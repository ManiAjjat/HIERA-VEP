# HIERA-VEP model card

## Intended use

Research prediction and inspection of regulatory SNV effects in aligned sequence pairs with complete genomic annotations. CPU batch inference uses a locked Ridge model, alpha 1000, with 27,212 features. It is not intended for clinical diagnosis, pathogenicity classification, or unsupported tissues and assays.

## Training and evaluation

The study includes 36,188 variants: 29,444 development and 6,744 external, from 31 assay groups, 28 elements, and 15 context groups (`results/locked/hiera_unified_table_summary.json`). Grouped nested selection was performed before locked evaluation; saved preprocessing uses only the development train+validation pool. The locked test has 4,408 variants and 1,476 position clusters (`results/locked/calibration_uncertainty_failure_summary.json`). Test outcomes were not used to package or select the released model.

## Assets and performance

`models/context_model_summary.json` identifies the selected context-gated sequence family and alpha. `models/preprocessing_spec.json` gives the exact feature order and saved scaling. See the README's source-linked table for locked metrics and `docs/RESULT_PROVENANCE.md` for evidence roles. Contributions are an exact algebraic decomposition of the model prediction, not causal effects.

## Limitations

External zero-shot transfer was weak and heterogeneous; external R² values were around zero or negative across several tiers. HIERA-VEP did not consistently outperform Ridge externally. Assay-specific adaptation is a secondary analysis requiring target-assay labels and is not zero-shot generalization.

Predicted effects were compressed relative to observed effects, and large-effect variants were a major failure mode. Model disagreement is only an uncertainty proxy, not a calibrated predictive interval.

Position-level motif enrichment did not yield broadly FDR-supported motifs (zero significant position-level motifs in `results/interpretation/motif_conservation_validation_summary.json`). Exact significant GTEx overlap was extremely sparse: one variant/position in `results/interpretation/gtex_eqtl_concordance_summary.json`. Absence of an overlap is not evidence of absent regulatory activity.

Conservation and motif analyses are concordance or computational annotation, not causal validation. Conservation was itself a model input, so attribution–conservation agreement is not independent validation. Earlier Step 53A language deferring eQTL analysis is retained as historical provenance; Step 53B provides the subsequent sparse exact-match analysis.

The software is research-only and is not a clinical pathogenicity classifier. No universal generalization, experimental validation, or causal interpretation is claimed.

## Serialization and licensing

Model and vectorizer scikit-learn serialization version: 1.9.1. Check `models/MODEL_ASSET_MANIFEST.tsv` before loading trusted joblib assets. Original model and vectorizer bytes are preserved. Code is MIT licensed; external data retain their original terms. Author and manuscript citation metadata are placeholders pending authoritative final metadata.
