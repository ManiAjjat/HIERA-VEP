#!/usr/bin/env python3
"""Step 53B: exact allele-matched GTEx eQTL concordance on the locked test set.

This is a confirmatory annotation analysis. It queries precomputed significant
GTEx v8 single-tissue eQTL records using exact GRCh38 variant identifiers,
collapses variants to independent genomic positions, and tests a prespecified
top-10% absolute-effect contrast. Absence from GTEx is not evidence of no
regulatory function.
"""

from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats


API = "https://gtexportal.org/api/v2/association/singleTissueEqtl"


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2])))
    p.add_argument("--dataset", default="gtex_v8")
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--top-fraction", type=float, default=0.10)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=5302)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected inputs exists:\n" + "\n".join(map(str, paths)))


def find_tsv_with_columns(folder, required):
    """Locate a TSV by schema so minor output-filename changes do not break the step."""
    if not folder.exists():
        raise FileNotFoundError(f"Required results folder does not exist: {folder}")
    examined = []
    for path in sorted(folder.glob("*.tsv")):
        try:
            cols = set(pd.read_csv(path, sep="\t", nrows=0).columns)
        except Exception:
            continue
        examined.append((path.name, sorted(cols)))
        if set(required).issubset(cols):
            return path
    raise FileNotFoundError(
        f"No TSV in {folder} contains required columns {required}. "
        f"Examined files: {[x[0] for x in examined]}"
    )


def choose(df, names, label, required=True):
    exact = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in exact:
            return exact[name.lower()]
    if required:
        raise KeyError(f"No {label} column. Available: {list(df.columns)}")
    return None


def normalize_merge_id(values):
    """Create a stable string key from integer-, float-, or text-encoded row IDs."""
    s = values.astype("string").str.strip()
    # Pandas/TSV round-trips sometimes turn integer IDs into strings such as 42.0.
    s = s.str.replace(r"^([+-]?\d+)\.0$", r"\1", regex=True)
    return s


def norm_chr(x):
    s = str(x).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    if s == "23": s = "X"
    if s == "24": s = "Y"
    return "chr" + s.upper().replace("MT", "M")


def flatten_records(obj):
    """Extract GTEx result records despite small API envelope changes."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    for key in ("data", "singleTissueEqtl", "results", "items"):
        val = obj.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
        if isinstance(val, dict):
            found = flatten_records(val)
            if found:
                return found
    return []


def get_field(d, names, default=np.nan):
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return default


def request_batch(ids, dataset, cache_dir, refresh, session):
    token = hashlib.sha256((dataset + "|" + "|".join(ids)).encode()).hexdigest()[:20]
    cache = cache_dir / f"{token}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    # FastAPI arrays are represented by repeated variantId parameters.
    params = [("datasetId", dataset), ("itemsPerPage", "100000")]
    params += [("variantId", x) for x in ids]
    last = None
    for attempt in range(1, 7):
        try:
            r = session.get(API, params=params, timeout=(20, 120))
            r.raise_for_status()
            payload = r.json()
            cache.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == 6:
                break
            wait = min(60, 2 ** attempt) + random.random()
            print(f"  API retry {attempt}/6 after {type(exc).__name__}; waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"GTEx request failed after 6 attempts: {last}")


def bh(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if not len(vals): return out
    order = np.argsort(vals)
    ranked = vals[order]
    adj = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked)+1))[::-1])[::-1]
    tmp = np.empty(len(vals)); tmp[order] = np.minimum(adj, 1)
    out[ok] = tmp
    return out


def bootstrap_risk_difference(pos, iterations, seed):
    rng = np.random.default_rng(seed)
    elements = pos["element_id"].astype(str).unique()
    values = []
    for _ in range(iterations):
        chunks = []
        for element in elements:
            sub = pos[pos.element_id.astype(str) == element]
            if len(sub):
                chunks.append(sub.iloc[rng.integers(0, len(sub), len(sub))])
        b = pd.concat(chunks, ignore_index=True)
        a = b.loc[b.top_effect, "gtex_eqtl"].mean()
        c = b.loc[~b.top_effect, "gtex_eqtl"].mean()
        values.append(a-c)
    return np.quantile(values, [0.025, 0.975]), np.asarray(values)


def main():
    a = args_parser(); root = a.project
    out = root / "results" / "exact_gtex_eqtl_concordance"
    cache = out / "api_cache"; cache.mkdir(parents=True, exist_ok=True)

    attr_path = first_existing([
        root/"results"/"exact_attribution_counterfactual_audit"/"locked_variant_exact_attributions.tsv",
        root/"locked_variant_exact_attributions.tsv",
    ])
    unified_path = find_tsv_with_columns(
        root/"results"/"hiera_unified_training_table",
        ["row_id", "reference_allele", "alternative_allele"],
    )
    context_path = first_existing([
        root/"results"/"human_conservation_annotation"/"development_variant_context_conservation_annotations.tsv",
        root/"results"/"human_encode_promoter_annotation"/"development_variant_encode_promoter_annotations.tsv",
    ])
    attr = pd.read_csv(attr_path, sep="\t", low_memory=False)
    unified = pd.read_csv(unified_path, sep="\t", low_memory=False)
    context = pd.read_csv(context_path, sep="\t", low_memory=False)

    # Prefer row_id only when the two files demonstrably share the same ID namespace.
    # Some pipeline versions regenerated HIERA IDs; in that case use the stable
    # biological key (element, position group, reference allele, alternative allele).
    attr = attr.copy(); unified = unified.copy()
    attr["__merge_row_id"] = normalize_merge_id(attr["row_id"])
    unified["__merge_row_id"] = normalize_merge_id(unified["row_id"])
    shared_ids = set(attr["__merge_row_id"].dropna()) & set(unified["__merge_row_id"].dropna())
    if len(shared_ids) == len(attr):
        merge_keys = ["__merge_row_id"]
        merge_label = "HIERA row_id"
    else:
        aliases = {
            "element": ["element_id", "element_group", "element_name"],
            "position_group": ["position_group", "position_group_id", "position_key"],
            "ref": ["reference_allele", "ref"],
            "alt": ["alternative_allele", "alt"],
        }
        for label, names in aliases.items():
            ac = choose(attr, names, f"attribution {label}")
            uc = choose(unified, names, f"unified {label}")
            attr[f"__key_{label}"] = attr[ac].astype("string").str.strip().str.upper()
            unified[f"__key_{label}"] = unified[uc].astype("string").str.strip().str.upper()
        merge_keys = ["__key_element", "__key_position_group", "__key_ref", "__key_alt"]
        merge_label = "element + position_group + ref + alt"
        print(f"HIERA row-ID overlap was {len(shared_ids):,}/{len(attr):,}; using stable biological composite key")

    if unified.duplicated(merge_keys).any():
        dup = int(unified.duplicated(merge_keys, keep=False).sum())
        raise ValueError(f"Step 44 unified table contains {dup} rows duplicated on {merge_label}.")
    coordinate_columns = [c for c in unified.columns if c not in attr.columns or c == "__merge_row_id"]
    # Keep authoritative coordinate fields even if an identically named partial field exists.
    for c in ["chromosome", "chromosome_accession", "chromosome_grch38", "genomic_chromosome",
              "variant_start", "variant_position", "position", "position_1based", "start",
              "reference_allele", "alternative_allele"]:
        if c in unified.columns and c not in coordinate_columns:
            coordinate_columns.append(c)
    for key in merge_keys:
        if key not in coordinate_columns:
            coordinate_columns.append(key)
    data = attr.merge(unified[coordinate_columns], on=merge_keys, how="left",
                      suffixes=("", "_unified"), validate="one_to_one")
    position_sentinel = choose(data, ["variant_start_unified", "variant_start", "variant_position_unified",
                                      "variant_position", "position_unified", "position",
                                      "position_1based_unified", "position_1based"],
                               "post-merge genomic position", False)
    matched = data[position_sentinel].notna() if position_sentinel else pd.Series(False, index=data.index)
    if not matched.all():
        examples = data.loc[~matched, "__merge_row_id"].head(10).tolist()
        raise ValueError(
            f"Step 44 coordinate mapping failed for {(~matched).sum():,}/{len(data):,} locked rows. "
            f"Example HIERA IDs: {examples}"
        )
    print(f"Step 44 coordinate mapping by {merge_label}: {matched.sum():,}/{len(data):,} locked rows matched")

    # Step 44 intentionally has position but no chromosome. Step 49 provides the
    # confirmed GRCh38 chromosome; chromosome is constant within each assayed element.
    existing_chrom = choose(data, ["chromosome_unified", "chromosome", "chromosome_accession_unified",
                                   "chromosome_accession", "chromosome_grch38", "genomic_chromosome"],
                            "existing chromosome", False)
    if existing_chrom is None:
        ce = choose(context, ["element_id", "element_group", "element_name"], "Step 49 element")
        cc = choose(context, ["chromosome", "chromosome_accession", "chromosome_grch38", "chr"], "Step 49 chromosome")
        element_chrom = context[[ce, cc]].dropna().copy()
        element_chrom[ce] = element_chrom[ce].astype(str)
        conflicts = element_chrom.groupby(ce)[cc].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            raise ValueError(f"Step 49 contains multiple chromosomes for elements: {conflicts.index.tolist()}")
        element_chrom = element_chrom.drop_duplicates(ce).rename(columns={ce:"__chrom_element", cc:"chromosome_step49"})
        data["__chrom_element"] = data[choose(data, ["element_id", "element_group", "element_name"], "element")].astype(str)
        data = data.merge(element_chrom, on="__chrom_element", how="left", validate="many_to_one")
        if data["chromosome_step49"].isna().any():
            missing = sorted(data.loc[data.chromosome_step49.isna(), "__chrom_element"].unique())
            raise ValueError(f"No Step 49 GRCh38 chromosome mapping for elements: {missing}")
        print(f"Step 49 chromosome mapping: {data['chromosome_step49'].notna().sum():,}/{len(data):,} rows matched")

    chrom = choose(data, ["chromosome_step49", "chromosome_unified", "chromosome", "chromosome_accession_unified",
                          "chromosome_accession", "chromosome_grch38", "genomic_chromosome", "chr"], "chromosome")
    poscol = choose(data, ["variant_start_unified", "variant_start", "variant_position_unified",
                           "variant_position", "position_1based_unified", "position_1based",
                           "position_unified", "position", "start_unified", "start"], "1-based position")
    ref = choose(data, ["reference_allele_unified", "reference_allele", "ref"], "reference allele")
    alt = choose(data, ["alternative_allele_unified", "alternative_allele", "alt"], "alternative allele")
    element = choose(data, ["element_id", "element_group", "element_name"], "element")
    observed = choose(data, ["target_model", "observed", "observed_effect", "target", "y_true",
                             "consensus_score_z", "target_raw"], "observed effect")
    print(f"Coordinate fields selected from {unified_path.name}: chromosome={chrom}, position={poscol}, ref={ref}, alt={alt}")

    data["chromosome_grch38"] = data[chrom].map(norm_chr)
    data["position_grch38"] = pd.to_numeric(data[poscol], errors="coerce").astype("Int64")
    data["ref"] = data[ref].astype(str).str.upper()
    data["alt"] = data[alt].astype(str).str.upper()
    data["element_id"] = data[element].astype(str)
    data["observed_effect"] = pd.to_numeric(data[observed], errors="coerce")
    valid = data.position_grch38.notna() & data.ref.str.fullmatch("[ACGT]") & data.alt.str.fullmatch("[ACGT]")
    if not valid.all():
        raise ValueError(f"Locked table contains {(~valid).sum()} rows without an exact GRCh38 SNV representation.")
    data["gtex_variant_id"] = (data.chromosome_grch38 + "_" + data.position_grch38.astype(str)
                               + "_" + data.ref + "_" + data.alt + "_b38")
    ids = sorted(data.gtex_variant_id.unique())
    print(f"Locked variants: {len(data):,}; exact GTEx IDs: {len(ids):,}; dataset: {a.dataset}")

    session = requests.Session()
    session.headers.update({"User-Agent": "HIERA-VEP-GTEx-validation/1.0"})
    records = []
    for start in range(0, len(ids), a.batch_size):
        batch = ids[start:start+a.batch_size]
        payload = request_batch(batch, a.dataset, cache, a.refresh, session)
        records.extend(flatten_records(payload))
        done = min(start+a.batch_size, len(ids))
        if done % (a.batch_size*10) == 0 or done == len(ids):
            print(f"  Queried {done:,}/{len(ids):,} variant IDs; returned records so far: {len(records):,}")

    clean = []
    for r in records:
        vid = str(get_field(r, ["variantId", "variant_id"], ""))
        if vid not in set(ids): continue
        clean.append({
            "gtex_variant_id": vid,
            "tissue_id": get_field(r, ["tissueSiteDetailId", "tissue_id", "tissueSiteDetail"]),
            "gencode_id": get_field(r, ["gencodeId", "geneId", "gene_id"]),
            "gene_symbol": get_field(r, ["geneSymbol", "gene_symbol"]),
            "nes": pd.to_numeric(get_field(r, ["nes", "normalizedEffectSize"]), errors="coerce"),
            "p_value": pd.to_numeric(get_field(r, ["pValue", "pvalue", "p_value"]), errors="coerce"),
        })
    rec = pd.DataFrame(clean).drop_duplicates() if clean else pd.DataFrame(columns=[
        "gtex_variant_id","tissue_id","gencode_id","gene_symbol","nes","p_value"])
    rec.to_csv(out/"gtex_significant_eqtl_records.tsv", sep="\t", index=False)

    if len(rec):
        ann = rec.groupby("gtex_variant_id").agg(
            significant_tissue_count=("tissue_id", "nunique"),
            significant_gene_count=("gencode_id", "nunique"),
            min_p_value=("p_value", "min"),
            maximum_absolute_nes=("nes", lambda x: np.nanmax(np.abs(x)) if np.isfinite(x).any() else np.nan),
            tissues=("tissue_id", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
            genes=("gencode_id", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
        ).reset_index()
    else:
        ann = pd.DataFrame(columns=["gtex_variant_id","significant_tissue_count","significant_gene_count",
                                    "min_p_value","maximum_absolute_nes","tissues","genes"])
    data = data.merge(ann, on="gtex_variant_id", how="left")
    data["gtex_eqtl"] = data.significant_tissue_count.fillna(0).gt(0)
    data[["significant_tissue_count","significant_gene_count"]] = data[["significant_tissue_count","significant_gene_count"]].fillna(0).astype(int)
    data.to_csv(out/"locked_variant_gtex_eqtl_annotations.tsv", sep="\t", index=False)

    # Position is the independent unit; take strongest observed allele at each tested position.
    data["position_key"] = data.chromosome_grch38 + ":" + data.position_grch38.astype(str)
    idx = data.groupby("position_key")["observed_effect"].apply(lambda x: x.abs().idxmax())
    pos = data.loc[idx].copy().reset_index(drop=True)
    pos["absolute_observed_effect"] = pos.observed_effect.abs()
    cutoff = pos.absolute_observed_effect.quantile(1-a.top_fraction)
    pos["top_effect"] = pos.absolute_observed_effect >= cutoff
    pos.to_csv(out/"position_level_gtex_validation_table.tsv", sep="\t", index=False)

    tab = pd.crosstab(pos.top_effect, pos.gtex_eqtl).reindex(index=[True,False], columns=[True,False], fill_value=0)
    odds, fisher = stats.fisher_exact(tab.to_numpy(), alternative="greater")
    top_rate = pos.loc[pos.top_effect, "gtex_eqtl"].mean()
    other_rate = pos.loc[~pos.top_effect, "gtex_eqtl"].mean()
    ci, boot = bootstrap_risk_difference(pos, a.bootstrap, a.seed)

    tests = [{"analysis":"top_effect_eqtl_enrichment", "odds_ratio":odds, "p_value":fisher}]
    # Secondary continuous test is explicitly labelled exploratory.
    if pos.gtex_eqtl.any() and (~pos.gtex_eqtl).any():
        u = stats.mannwhitneyu(pos.loc[pos.gtex_eqtl,"absolute_observed_effect"],
                              pos.loc[~pos.gtex_eqtl,"absolute_observed_effect"], alternative="greater")
        tests.append({"analysis":"absolute_effect_eqtl_vs_non_eqtl_exploratory", "odds_ratio":np.nan, "p_value":u.pvalue})
    tests_df = pd.DataFrame(tests); tests_df["fdr_bh"] = bh(tests_df.p_value)
    tests_df.to_csv(out/"gtex_concordance_tests.tsv", sep="\t", index=False)
    pd.DataFrame({"bootstrap_risk_difference":boot}).to_csv(out/"position_clustered_gtex_bootstrap.tsv", sep="\t", index=False)

    summary = {
        "gtex_dataset": a.dataset,
        "locked_variants": int(len(data)), "independent_positions": int(len(pos)),
        "exact_variants_with_significant_gtex_eqtl": int(data.gtex_eqtl.sum()),
        "positions_with_significant_gtex_eqtl": int(pos.gtex_eqtl.sum()),
        "top_fraction": a.top_fraction, "top_effect_cutoff": float(cutoff),
        "top_effect_positions": int(pos.top_effect.sum()),
        "top_effect_eqtl_fraction": float(top_rate), "other_eqtl_fraction": float(other_rate),
        "risk_difference": float(top_rate-other_rate),
        "risk_difference_ci_95": [float(ci[0]), float(ci[1])],
        "fisher_odds_ratio": float(odds), "fisher_p_one_sided": float(fisher),
        "interpretation_limits": [
            "GTEx records are observational associations, not causal validation.",
            "Only exact GRCh38 chromosome-position-reference-alternative matches are counted.",
            "No tissue was selected after inspecting results; the annotation is pan-tissue.",
            "Absence of a significant GTEx record is not evidence that a variant lacks regulatory function.",
            "Reporter-assay context can differ substantially from adult bulk GTEx tissue context."
        ]
    }
    (out/"gtex_eqtl_concordance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nEXACT GTEx eQTL CONCORDANCE")
    print(json.dumps(summary, indent=2))
    print("\nStep 53B exact GTEx eQTL concordance: COMPLETED")
    print("Saved:", out)


if __name__ == "__main__":
    main()
