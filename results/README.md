# Summary results

- `locked/`: locked evaluation, paired comparisons, ablation, grouped selection and failure diagnostics; descriptive diagnostics must not be used to recalibrate the locked model.
- `external/`: zero-shot transfer and separately labeled adaptive secondary analyses. Assay adaptation uses target-assay labels.
- `interpretation/`: exact attribution, annotation, motif/conservation and sparse GTEx concordance. These are not causal validation.

Each export is mapped to its original project-relative path, generating analysis and evidence role in [RESULT_PROVENANCE.md](../docs/RESULT_PROVENANCE.md). Full predictions, raw inputs and massive bootstrap replicates are excluded. Original JSON NaN values are represented as null. The model summary is stored under `models/` to avoid duplicate copies.
