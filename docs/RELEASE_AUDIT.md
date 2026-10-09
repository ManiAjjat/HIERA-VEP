# Local release audit

Tests: 29 passed using Python 3.11.16, NumPy 2.4.6, pandas 2.3.3 and scikit-learn 1.9.1. Serialization version checks for both joblib assets are 1.9.1. Vocabulary: 5,440; model features: 27,212; Ridge alpha: 1000. Saved model, vectorizer and preprocessing specification match the original authoritative asset bytes. No training or scaling fit was performed.

Full locked equivalence was independently rerun on all 4,408 original locked rows, checking row and element alignment. Maximum difference against saved predictions: **5.277252190794002e-08**. Maximum exact additive reconstruction error: **5.103647708892822e-07**. The example scorer processed 12 original release rows, with maximum error **2.468004822731018e-07**. Original reference IDs/predictions are retained as a compact independent fixture instead of duplicating the example TSV.

All staged JSON files parse strictly (undefined original NaN statistics are null); all TSV files load with pandas; all Python sources are parsed and compiled. The secret scan examines token patterns, assigned passwords/API keys, private keys, emails, personal directory names and absolute paths. No matches were detected in public text assets after distinguishing URL schemes from filesystem paths. No file exceeds 25 MB. Binary model assets are unchanged authoritative files; PNGs are original manuscript exports. This is a targeted audit, not a guarantee against every possible hidden secret.

The source inventory and SHA-256 duplicate audit were recorded outside the repository before copying. Duplicate model-summary copies were reduced to the canonical model summary. Empty package initializer duplicates are intentional. Historical copies change runtime roots/imports and optional font paths only. Source provenance records final hashes. Final file count, byte count and checksum verification are recorded in the local publication report; CHECKSUMS.sha256 covers the frozen tracked set except itself.

The largest file is the 1,247,132-byte model coefficient TSV after LF line-ending normalization; its numeric content is unchanged. Other large assets are selected manuscript PNGs around 0.64–1.06 MB. Model, vectorizer, saved preprocessing and original smoke-report bytes are preserved. Excluded data categories are listed in data/README.md. Five main and four selected supplementary original PNGs are included. Full figure reconstruction needs excluded upstream panels and variant tables; full raw-to-result reproduction needs original upstream intermediates. No complete-data reconstruction claim is made.

GitHub authentication, ownership, remote completeness and Actions success must be checked independently before a public release is declared. Final manuscript citation metadata remain placeholders.

## Asset SHA-256

- `context_selected_ridge.joblib`: `9aa3fcfbca85360455cddf13997e97e5eca7d22d099811f2c90d326bc818f9df`
- `context_model_vectorizer.joblib`: `7a0c6539f38acc75f806b2bee3b2473ba5ca97f5bf12f9432b98c40570f6032f`
