# Reproducibility

## Locked inference

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python src/hiera_vep_score.py --bundle . --input examples/example_scoring_input.tsv --output examples/example_predictions.tsv
python -m compileall -q src HIERA_VEP_COMPLETE_CODE.py
```

`tests/smoke_expected.json` contains the original Step 56B reference row IDs and scalar predictions; tests compare against those original values at absolute tolerance 1e-5. The examples are the original 12-row public release example, not newly selected variants. `models/locked_inference_smoke_test.json` records the prior 4,408-row validation; a separate local audit reruns locked equivalence without publishing the full variant table. Core tests need no dataset downloads.

The scalar prediction is reconstructed from seven contribution blocks, with a maximum absolute error strictly below 1e-5. Floating-point accumulation may differ slightly by platform. Tests fail on scikit-learn serialization version mismatch.

## Historical analyses

See `ANALYSIS_WORKFLOW.md` and `data/README.md`. Historical scripts retain their scientific calculations. Copies change only runtime paths and imports; `SOURCE_PROVENANCE.tsv` records original and public checksums. Outputs are summary-level exports from existing analysis runs. Undefined JSON statistics were converted from non-standard NaN to null, and private runtime paths were replaced by PROJECT_ROOT; no finite statistic was changed. Raw genomic data and pretrained foundation weights are deliberately excluded.

## Integrity

`CHECKSUMS.sha256` lists all final tracked files except itself. Read each line as SHA-256, two spaces, repository-relative filename. Verify against the extracted archive rather than platform-converted text. `.gitattributes` fixes scientific text files to LF.

## Single-file source bundle

```bash
python HIERA_VEP_COMPLETE_CODE.py --list
python HIERA_VEP_COMPLETE_CODE.py --extract extracted_source
```

The source bundle embeds every public Python file except itself, preserving paths. Model assets, examples, dependencies, and figures remain separate. Extraction into a new directory refuses to overwrite files.
