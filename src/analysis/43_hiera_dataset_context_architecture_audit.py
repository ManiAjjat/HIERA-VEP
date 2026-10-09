import os
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
DEVELOPMENT_FILE = ROOT / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_within_element_splits.tsv"
EXTERNAL_FILE = ROOT / "results/external_validation_harmonized/external_model_ready_501bp.tsv"
OUTPUT = ROOT / "results/hiera_dataset_context_architecture_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVELOPMENT_TARGET = "consensus_score_z"
EXTERNAL_TARGET = "robust_z"


def first_present(frame, candidates, default="unknown"):
    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


def stable_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]


def numeric_summary(values):
    x = pd.to_numeric(values, errors="coerce")
    return {
        "target_nonmissing": int(x.notna().sum()),
        "target_mean": float(x.mean()), "target_sd": float(x.std(ddof=1)),
        "target_median": float(x.median()),
        "target_q05": float(x.quantile(0.05)), "target_q95": float(x.quantile(0.95)),
    }


def infer_species(source, tier, element):
    text = f"{source} {tier} {element}".lower()
    if any(x in text for x in ["sul1", "ltv1", "cross_species"]):
        return "Saccharomyces_cerevisiae"
    if any(x in text for x in ["splice_ac", "splice_cg", "primary_human", "aldob", "ecr11"]):
        return "Homo_sapiens"
    if source == "development":
        return "Homo_sapiens_or_mixed_requires_metadata_review"
    return "unknown"


def infer_assay_family(tier, element):
    text = f"{tier} {element}".lower()
    if "splice" in text:
        return "splicing_reporter"
    if "sul1" in text or "ltv1" in text:
        return "yeast_regulatory_reporter"
    if "aldob" in text or "ecr11" in text or "primary_human" in text:
        return "human_regulatory_reporter"
    return "regulatory_MAVE"


def normalize(frame, source):
    out = pd.DataFrame(index=frame.index)
    out["source_partition"] = source
    if source == "development":
        out["dataset_id"] = first_present(frame, ["score_set_urn", "score_set_urns", "element_group"])
        out["element_id"] = first_present(frame, ["element_group", "element_name"])
        out["validation_tier"] = "development_panel"
        out["original_split"] = first_present(frame, ["within_element_split", "unseen_element_split"])
        out["target"] = pd.to_numeric(frame[DEVELOPMENT_TARGET], errors="coerce")
    else:
        out["dataset_id"] = first_present(frame, ["score_set_urn", "dataset_id", "element_name"])
        out["element_id"] = first_present(frame, ["element_name", "score_set_urn"])
        out["validation_tier"] = first_present(frame, ["validation_tier"])
        out["original_split"] = "external"
        out["target"] = pd.to_numeric(frame[EXTERNAL_TARGET], errors="coerce")
    out["condition"] = first_present(frame, ["condition", "conditions", "assay_condition", "element_name"])
    out["cell_type"] = first_present(frame, ["cell_type", "cell_line", "host_cell_type"])
    out["reference_allele"] = first_present(frame, ["reference_allele"]).str.upper()
    out["alternative_allele"] = first_present(frame, ["alternative_allele"]).str.upper()
    out["reference_sequence"] = first_present(frame, ["reference_sequence"]).str.upper()
    out["alternative_sequence"] = first_present(frame, ["alternative_sequence"]).str.upper()
    if "position_1based" in frame:
        out["position"] = pd.to_numeric(frame.position_1based, errors="coerce")
    elif "variant_start" in frame:
        out["position"] = pd.to_numeric(frame.variant_start, errors="coerce")
    else:
        out["position"] = np.arange(len(frame))
    out["species"] = [infer_species(source, t, e)
                      for t, e in zip(out.validation_tier, out.element_id)]
    out["assay_family"] = [infer_assay_family(t, e)
                           for t, e in zip(out.validation_tier, out.element_id)]
    out["mutation_class"] = out.reference_allele + ">" + out.alternative_allele
    out["sequence_length"] = out.reference_sequence.str.len()
    out["ambiguous_fraction"] = out.reference_sequence.map(
        lambda x: np.mean([base not in "ACGT" for base in x]) if x else np.nan)
    out["sequence_pair_hash"] = [stable_hash(r + "|" + a)
                                 for r, a in zip(out.reference_sequence, out.alternative_sequence)]
    out["variant_group"] = (out.dataset_id + "|" + out.element_id + "|" +
                            out.position.fillna(-1).astype(str))
    out["context_id"] = (out.species + "|" + out.assay_family + "|" +
                         out.cell_type + "|" + out.condition)
    return out


print("Loading development and external harmonized datasets...")
development_raw = pd.read_csv(DEVELOPMENT_FILE, sep="\t", low_memory=False)
external_raw = pd.read_csv(EXTERNAL_FILE, sep="\t", low_memory=False)
for frame, target, label in [(development_raw, DEVELOPMENT_TARGET, "development"),
                             (external_raw, EXTERNAL_TARGET, "external")]:
    required = {target, "reference_sequence", "alternative_sequence",
                "reference_allele", "alternative_allele"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{label} data missing required columns: {sorted(missing)}")

development = normalize(development_raw, "development")
external = normalize(external_raw, "external")
unified = pd.concat([development, external], ignore_index=True)

print("Auditing sequence identity and split leakage...")
development_hashes = set(development.sequence_pair_hash)
external_hashes = set(external.sequence_pair_hash)
hash_overlap = development_hashes & external_hashes

leakage_rows = [{
    "audit": "exact_reference_alternative_sequence_pair_overlap",
    "development_unique": len(development_hashes),
    "external_unique": len(external_hashes),
    "overlap": len(hash_overlap),
    "status": "PASS" if len(hash_overlap) == 0 else "REQUIRES_GROUPED_EXCLUSION",
}]

if "within_element_split" in development_raw.columns:
    temp = development.copy()
    temp["split"] = development_raw.within_element_split.fillna("missing").astype(str).str.lower().to_numpy()
    split_counts = temp.groupby("sequence_pair_hash").split.nunique()
    cross_split = int(split_counts.gt(1).sum())
    leakage_rows.append({
        "audit": "development_exact_sequence_pair_crosses_within_element_splits",
        "development_unique": int(temp.sequence_pair_hash.nunique()),
        "external_unique": np.nan, "overlap": cross_split,
        "status": "PASS" if cross_split == 0 else "REQUIRES_REGROUPING",
    })

print("Building dataset and context manifests...")
manifest_rows = []
for (source, dataset, element, tier, species, family), x in unified.groupby(
        ["source_partition", "dataset_id", "element_id", "validation_tier", "species", "assay_family"],
        dropna=False):
    summary = numeric_summary(x.target)
    manifest_rows.append({
        "source_partition": source, "dataset_id": dataset, "element_id": element,
        "validation_tier": tier, "species": species, "assay_family": family,
        "variants": len(x), "positions": int(x.variant_group.nunique()),
        "contexts": int(x.context_id.nunique()),
        "sequence_length_min": int(x.sequence_length.min()),
        "sequence_length_median": float(x.sequence_length.median()),
        "sequence_length_max": int(x.sequence_length.max()),
        "mean_ambiguous_fraction": float(x.ambiguous_fraction.mean()),
        **summary,
    })
manifest = pd.DataFrame(manifest_rows).sort_values(
    ["source_partition", "validation_tier", "element_id"])

context_registry = (unified.groupby(
    ["context_id", "species", "assay_family", "cell_type", "condition"], dropna=False)
    .agg(variants=("target", "size"), datasets=("dataset_id", "nunique"),
         elements=("element_id", "nunique"), positions=("variant_group", "nunique"),
         target_sd=("target", "std"))
    .reset_index().sort_values("variants", ascending=False))

candidate_features = {
    "species": ["species", "organism"], "cell_type": ["cell_type", "cell_line"],
    "condition": ["condition", "conditions", "assay_condition"],
    "assay_identity": ["score_set_urn", "score_set_urns", "dataset_id"],
    "element_identity": ["element_group", "element_name"],
    "cCRE_class": ["ccre_class", "cCRE_class", "encode_ccre_class"],
    "chromatin_accessibility": ["atac_signal", "dnase_signal", "chromatin_accessibility"],
    "H3K27ac": ["h3k27ac", "H3K27ac"], "H3K4me3": ["h3k4me3", "H3K4me3"],
    "conservation": ["phylop", "phyloP", "phastcons", "phastCons"],
    "eQTL": ["eqtl", "eQTL", "gtex_eqtl"], "TSS_distance": ["tss_distance", "distance_to_tss"],
    "motif_features": ["motif_disruption", "jaspar_disruption"],
}
feature_rows = []
for feature, aliases in candidate_features.items():
    for source, raw in [("development", development_raw), ("external", external_raw)]:
        found = [x for x in aliases if x in raw.columns]
        if found:
            nonmissing = int(raw[found[0]].notna().sum())
            fraction = nonmissing / len(raw)
        else:
            nonmissing, fraction = 0, 0.0
        feature_rows.append({"feature": feature, "source_partition": source,
                             "matched_column": found[0] if found else "",
                             "nonmissing_rows": nonmissing,
                             "available_fraction": fraction,
                             "status": "AVAILABLE" if fraction >= 0.95 else
                                       "PARTIAL" if fraction > 0 else "MISSING"})
feature_availability = pd.DataFrame(feature_rows)

# Architecture gates prevent unsupported model complexity.
n_datasets = int(unified.dataset_id.nunique())
n_elements = int(unified.element_id.nunique())
n_contexts = int(unified.context_id.nunique())
architecture = {
    "shared_sparse_sequence_component": "SUPPORTED",
    "dataset_random_intercepts": "SUPPORTED" if n_datasets >= 5 else "INSUFFICIENT_LEVELS",
    "element_random_intercepts": "SUPPORTED" if n_elements >= 10 else "INSUFFICIENT_LEVELS",
    "context_interaction_component": "PROVISIONAL" if n_contexts >= 5 else "INSUFFICIENT_LEVELS",
    "cell_type_generalization": "BLOCKED_UNTIL_CELL_TYPE_METADATA_ADDED",
    "chromatin_context_component": "BLOCKED_UNTIL_ANNOTATION_STEP",
    "human_primary_model": "SUPPORTED_ONLY_AFTER_SPECIES_METADATA_REVIEW",
    "strict_leave_one_dataset_out": "SUPPORTED" if n_datasets >= 5 else "INSUFFICIENT_DATASETS",
    "strict_leave_one_element_out": "SUPPORTED" if n_elements >= 10 else "INSUFFICIENT_ELEMENTS",
}

unified.drop(columns=["reference_sequence", "alternative_sequence"]).to_csv(
    OUTPUT / "hiera_context_registry_rows.tsv", sep="\t", index=False)
manifest.to_csv(OUTPUT / "hiera_dataset_manifest.tsv", sep="\t", index=False)
context_registry.to_csv(OUTPUT / "hiera_context_manifest.tsv", sep="\t", index=False)
feature_availability.to_csv(OUTPUT / "hiera_feature_availability.tsv", sep="\t", index=False)
pd.DataFrame(leakage_rows).to_csv(OUTPUT / "hiera_sequence_leakage_audit.tsv", sep="\t", index=False)

summary = {
    "development_variants": int(len(development)), "external_variants": int(len(external)),
    "total_variants": int(len(unified)), "datasets": n_datasets,
    "elements": n_elements, "contexts": n_contexts,
    "species_counts": unified.species.value_counts().to_dict(),
    "assay_family_counts": unified.assay_family.value_counts().to_dict(),
    "exact_development_external_sequence_pair_overlap": int(len(hash_overlap)),
    "architecture_gates": architecture,
    "next_action": (
        "Resolve species/study/cell-type metadata and add genomic context annotations before "
        "training HIERA-VEP; enforce sequence-pair and position grouping in every split."),
}
(OUTPUT / "hiera_architecture_audit_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")

print("\nHIERA-VEP DATASET/CONTEXT ARCHITECTURE AUDIT")
print(json.dumps(summary, indent=2))
print("\nDATASET MANIFEST")
print(manifest[["source_partition", "element_id", "validation_tier", "species",
                "assay_family", "variants", "positions", "target_sd"]].to_string(index=False))
print("\nFEATURE AVAILABILITY")
print(feature_availability.to_string(index=False))
print("\nStep 43 HIERA-VEP dataset/context architecture audit: COMPLETED")
print("Saved:", OUTPUT)
