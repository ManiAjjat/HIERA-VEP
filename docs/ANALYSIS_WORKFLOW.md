# Analysis workflow

Run historical analyses only in a separate reconstruction project, after retrieving the original public data and upstream intermediate files. Some scripts execute analysis immediately on invocation or import and can overwrite results. They are not inference commands. Do not execute training scripts against the released model assets.

The copied scripts accept their existing `--project` arguments where present. Scripts with a global root use the repository root by default or the `HIERA_VEP_PROJECT` environment variable pointing to a reconstruction directory. Preserve original intermediate subdirectory names under that directory. The public result exports are flattened into evidence categories; historical analysis scripts expect their original Step-specific paths, so those intermediate outputs must be regenerated in the reconstruction project.

| Steps | Purpose |
|---|---|
| 27–30 | Paired bootstrap comparisons, k-mer interpretation and motif robustness |
| 32–37 | External data inventory, harmonization, shared k-mer and frozen NT transfer, padding audit |
| 38–42 | Secondary assay adaptation, few-shot learning, condition and fungal motif analyses |
| 43–47 | Context architecture, unified table, historical neural candidate, grouped selection and external holdouts |
| 49A–49C | Coordinate audit, ENCODE/promoter/TSS annotation and conservation |
| 50–55 | Final context-gated Ridge, fair NT comparison, exact attribution, motif/GTEx concordance, ablation and failure diagnostics |
| 56 / 56B | Historical release builder and saved preprocessing/inference validation |

The released model is the Step 50 context-gated Ridge, not the earlier Step 45 neural candidate. Step 56B is archived for provenance; its original purpose included deriving preprocessing, so do not rerun it to replace the validated release specification.

Final curated composites use Step 57F and existing source panel images; those large upstream panel archives are excluded. Step 59 regenerates a different publication figure set from original tables, including variant-level inputs that are excluded here. Step 60 exports individual panels. The included composites are original selected manuscript PNGs, and no figures or scientific results were regenerated during packaging. Full figure reconstruction requires upstream panels/tables; inference reproduction is self-contained.
