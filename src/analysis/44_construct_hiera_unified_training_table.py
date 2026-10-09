import os
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
DEV_FILE = ROOT / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_within_element_splits.tsv"
EXT_FILE = ROOT / "results/external_validation_harmonized/external_model_ready_501bp.tsv"
OUTPUT = ROOT / "results/hiera_unified_training_table"
OUTPUT.mkdir(parents=True, exist_ok=True)

HUMAN_DEVELOPMENT_ELEMENTS = {
    "BCL11A", "F9", "FOXE1", "GP1BB", "HBB", "HBG1", "HNF4A", "IRF4",
    "IRF6", "LDLR", "MSMB", "MYC_rs11986220", "MYC_rs6983267", "PKLR",
    "RET", "SORT1", "TCF7L2", "TERT", "UC88", "ZFAND3", "ZRS",
}
YEAST_EXTERNAL = {"LTV1_consensus", "SUL1_glucose", "SUL1_sulfate"}
SPLICING_EXTERNAL = {"Splice_AC", "Splice_CG"}


def first(frame, candidates, default="unknown"):
    for column in candidates:
        if column in frame:
            return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def normalize_sequence(x):
    return str(x).upper().replace("U", "T")


def make_development(raw):
    x = pd.DataFrame(index=raw.index)
    x["source_partition"] = "development"
    x["element_id"] = first(raw, ["element_group", "element_name"])
    unknown = sorted(set(x.element_id) - HUMAN_DEVELOPMENT_ELEMENTS)
    if unknown:
        raise ValueError(f"Unreviewed development elements require species curation: {unknown}")
    x["assay_id"] = first(raw, ["score_set_urns", "score_set_urn"])
    x["condition"] = first(raw, ["conditions", "condition"], "primary")
    x["cell_context"] = np.where(x.element_id.eq("TERT"), x.condition, "unknown")
    x["species"] = "Homo_sapiens"
    x["assay_family"] = "human_regulatory_MAVE"
    x["validation_tier"] = "development_human"
    x["original_split"] = first(raw, ["within_element_split"])
    x["unseen_element_split"] = first(raw, ["unseen_element_split"])
    x["target_raw"] = pd.to_numeric(raw.consensus_score_z, errors="coerce")
    x["target_definition"] = "consensus_score_z"
    x["position"] = pd.to_numeric(raw["variant_start"], errors="coerce") if "variant_start" in raw else np.arange(len(raw))
    for column in ["reference_allele", "alternative_allele", "reference_sequence", "alternative_sequence"]:
        x[column] = raw[column].map(normalize_sequence)
    return x


def make_external(raw):
    x = pd.DataFrame(index=raw.index)
    x["source_partition"] = "external"
    x["element_id"] = first(raw, ["element_name"])
    x["assay_id"] = first(raw, ["score_set_urn", "dataset_id"])
    x["condition"] = x.element_id
    x["cell_context"] = "unknown"
    x["species"] = np.where(x.element_id.isin(YEAST_EXTERNAL),
                            "Saccharomyces_cerevisiae", "Homo_sapiens")
    x["assay_family"] = np.select(
        [x.element_id.isin(YEAST_EXTERNAL), x.element_id.isin(SPLICING_EXTERNAL)],
        ["yeast_regulatory_MAVE", "human_splicing_MAVE"],
        default="human_regulatory_MAVE")
    x["validation_tier"] = first(raw, ["validation_tier"])
    x["original_split"] = "external_locked"
    x["unseen_element_split"] = "external_locked"
    x["target_raw"] = pd.to_numeric(raw.robust_z, errors="coerce")
    x["target_definition"] = "robust_z"
    x["position"] = pd.to_numeric(raw.position_1based, errors="coerce")
    for column in ["reference_allele", "alternative_allele", "reference_sequence", "alternative_sequence"]:
        x[column] = raw[column].map(normalize_sequence)
    return x


print("Loading source tables...")
dev_raw = pd.read_csv(DEV_FILE, sep="\t", low_memory=False)
ext_raw = pd.read_csv(EXT_FILE, sep="\t", low_memory=False)
dev = make_development(dev_raw)
ext = make_external(ext_raw)
data = pd.concat([dev, ext], ignore_index=True)

print("Constructing hierarchy and leakage-safe group identifiers...")
data["mutation_class"] = data.reference_allele + ">" + data.alternative_allele
data["sequence_length"] = data.reference_sequence.str.len()
data["mutation_index"] = data.sequence_length // 2
data["ambiguous_fraction"] = data.reference_sequence.map(
    lambda s: sum(base not in "ACGT" for base in s) / len(s) if len(s) else np.nan)
data["sequence_pair_hash"] = [digest(r + "|" + a) for r, a in zip(
    data.reference_sequence, data.alternative_sequence)]
data["position_group"] = [digest(f"{source}|{assay}|{element}|{position}")
                          for source, assay, element, position in zip(
                              data.source_partition, data.assay_id,
                              data.element_id, data.position)]
data["element_group"] = [digest(f"{species}|{element}") for species, element in zip(
    data.species, data.element_id)]
data["context_group"] = [digest(f"{species}|{family}|{cell}|{condition}")
                          for species, family, cell, condition in zip(
                              data.species, data.assay_family,
                              data.cell_context, data.condition)]
data["row_id"] = [f"HIERA_{i:08d}" for i in range(len(data))]
data["primary_human_regulatory"] = (
    data.species.eq("Homo_sapiens") & data.assay_family.eq("human_regulatory_MAVE"))
data["secondary_transfer_only"] = ~data.primary_human_regulatory

# Do not use a held-out outcome to estimate a new target transformation here.
# target_raw remains the assay-provided standardized score. Calibration is learned
# inside each training fold in later steps.
data["target_model"] = data.target_raw

bad_alignment = []
for row in data.itertuples():
    i = int(row.mutation_index)
    if (len(row.reference_sequence) != len(row.alternative_sequence)
            or i >= len(row.reference_sequence)
            or row.reference_sequence[i] != row.reference_allele
            or row.alternative_sequence[i] != row.alternative_allele):
        bad_alignment.append(row.row_id)
if bad_alignment:
    raise ValueError(f"Sequence alignment failed for {len(bad_alignment)} rows: {bad_alignment[:10]}")

if data.target_model.isna().any():
    raise ValueError(f"Missing model targets: {int(data.target_model.isna().sum())}")

# Registry used by all future splitters. Exact sequence pairs and positions must
# never cross training/evaluation within the same evaluation task.
registry = data[["row_id", "source_partition", "validation_tier", "assay_id",
                 "element_id", "species", "assay_family", "condition",
                 "cell_context", "original_split", "unseen_element_split",
                 "sequence_pair_hash", "position_group", "element_group",
                 "context_group", "primary_human_regulatory",
                 "secondary_transfer_only"]].copy()

dataset_manifest = (data.groupby(
    ["source_partition", "validation_tier", "assay_id", "element_id", "species",
     "assay_family", "condition", "cell_context"], dropna=False)
    .agg(variants=("row_id", "size"), positions=("position_group", "nunique"),
         target_mean=("target_model", "mean"), target_sd=("target_model", "std"),
         sequence_length=("sequence_length", "median"),
         ambiguous_fraction=("ambiguous_fraction", "mean"))
    .reset_index())

data.to_csv(OUTPUT / "hiera_unified_variant_table.tsv", sep="\t", index=False)
registry.to_csv(OUTPUT / "hiera_group_registry.tsv", sep="\t", index=False)
dataset_manifest.to_csv(OUTPUT / "hiera_unified_dataset_manifest.tsv", sep="\t", index=False)

summary = {
    "rows": int(len(data)), "development_rows": int(data.source_partition.eq("development").sum()),
    "external_rows": int(data.source_partition.eq("external").sum()),
    "assay_groups": int(data.assay_id.nunique()), "elements": int(data.element_group.nunique()),
    "position_groups": int(data.position_group.nunique()),
    "context_groups": int(data.context_group.nunique()),
    "primary_human_regulatory_rows": int(data.primary_human_regulatory.sum()),
    "secondary_transfer_rows": int(data.secondary_transfer_only.sum()),
    "exact_sequence_pairs": int(data.sequence_pair_hash.nunique()),
    "duplicate_sequence_pair_rows": int(data.sequence_pair_hash.duplicated(keep=False).sum()),
    "alignment_failures": 0, "missing_targets": 0,
    "target_policy": "Use assay-provided standardized target; learn calibration within training folds only.",
    "split_policy": "Group by position and exact sequence pair; hold out entire assays/elements/contexts when specified.",
    "metadata_policy": "Unknown cell contexts remain explicit unknowns; no biological metadata is inferred without evidence.",
}
(OUTPUT / "hiera_unified_table_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")

print("\nHIERA-VEP UNIFIED TRAINING TABLE SUMMARY")
print(json.dumps(summary, indent=2))
print("\nUNIFIED DATASET MANIFEST")
print(dataset_manifest[["source_partition", "element_id", "species", "assay_family",
                        "variants", "positions", "target_sd"]].to_string(index=False))
print("\nStep 44 unified hierarchical training table: COMPLETED")
print("Saved:", OUTPUT)
