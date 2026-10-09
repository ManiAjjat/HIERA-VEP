# Scoring input and output dictionary

Input: UTF-8 TSV, one aligned SNV per row. Extra metadata columns, such as `row_id`, are preserved.

| Column | Meaning / accepted values |
|---|---|
| reference_sequence | Nonempty aligned sequence containing A/C/G/T/N |
| alternative_sequence | Same length as reference; exactly one changed nucleotide |
| ccre_overlap | Boolean: true/false, 1/0, yes/no |
| promoter_overlap | Same Boolean encoding |
| absolute_distance_to_nearest_tss | Finite nonnegative distance in base pairs |
| phyloP100way | Finite numeric conservation score |
| phastCons100way | Finite numeric conservation score |
| phyloP_conserved | Boolean flag from original annotation workflow |
| phyloP_accelerated | Boolean flag from original annotation workflow |
| phastCons_high | Boolean flag from original annotation workflow |
| primary_ccre_class | none, PLS, pELS, dELS, or CA-H3K4me3 |

All annotations are required; missing annotations must not be replaced by fabricated zeros or `none`. Sequences are uppercased before vectorization. The scorer does not query genomic annotations, validate a genome coordinate, or infer tissue-specific context.

The 12 features are cCRE overlap, promoter overlap, log1p absolute TSS distance, phyloP100way, phastCons100way, three conservation flags, and four cCRE class indicators. Continuous values use the saved train+validation means and standard deviations, including conservation gate variables. Order is recorded in `models/preprocessing_spec.json`.

Outputs retain input columns and add `hiera_vep_prediction`, `contribution_intercept`, `contribution_sequence_main`, `contribution_context_additive`, four `contribution_gate_<gate>` columns, `reconstructed_prediction`, and `reconstruction_error`. Predictions use the study's standardized assay-effect scale; they are not probabilities or clinical classifications.
