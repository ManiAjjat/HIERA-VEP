# Batch scoring example

The 12 input rows and reference predictions are copied from the existing Step 56B public release, without selecting rows based on performance.

```bash
python src/hiera_vep_score.py --bundle . --input examples/example_scoring_input.tsv --output examples/example_predictions.tsv
```

`../tests/smoke_expected.json` is the original reference; `example_predictions.tsv` is the verified public scorer output. See [input/output schemas](../docs/DATA_DICTIONARY.md). Required annotations must be supplied explicitly. Exactly one model, vectorizer and preprocessing specification must exist under the bundle path; use a clean checkout as the bundle.
