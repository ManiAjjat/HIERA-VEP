import os
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from Bio import motifs


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/condition_differential_regulatory_model/paired_sul1_condition_effects.tsv"
JASPAR = ROOT / "results/fungal_motif_context_interpretation/JASPAR2026_CORE_fungi_nonredundant.jaspar"
OUTPUT = ROOT / "results/position_clustered_motif_robustness"
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (0.70, 0.80, 0.90)
EVENT_TYPES = ("any", "gain", "loss")
TOP_FRACTION = 0.10
PSEUDOCOUNT = 0.5
PERMUTATIONS = 5000
SEED = 20261007
JACCARD_CUTOFF = 0.80
MIN_TOP_HITS = 5
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def bh(p_values):
    p = np.asarray(p_values, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0, 1)
    out[ok] = restored
    return out


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def prepare(record):
    counts = np.vstack([np.asarray(record.counts[b], float) for b in "ACGT"])
    probability = (counts + PSEUDOCOUNT) / (counts.sum(axis=0) + 4 * PSEUDOCOUNT)
    lod = np.log2(probability / 0.25)
    minimum = float(lod.min(axis=0).sum())
    maximum = float(lod.max(axis=0).sum())
    return {
        "motif_id": str(record.matrix_id or record.name),
        "motif_name": str(record.name or record.matrix_id),
        "length": counts.shape[1], "lod": lod,
        "minimum": minimum, "maximum": maximum,
    }


def affinity(sequence, motif):
    if any(x not in BASE_INDEX for x in sequence):
        return np.nan
    ix = np.fromiter((BASE_INDEX[x] for x in sequence), dtype=int)
    score = float(motif["lod"][ix, np.arange(motif["length"])].sum())
    denominator = motif["maximum"] - motif["minimum"]
    return (score - motif["minimum"]) / denominator if denominator > 0 else np.nan


def best_overlap(reference, alternative, center, motif):
    length = motif["length"]
    best = {threshold: None for threshold in THRESHOLDS}
    for start in range(center - length + 1, center + 1):
        if start < 0 or start + length > len(reference):
            continue
        rw = reference[start:start + length]
        aw = alternative[start:start + length]
        for strand, rs, als in (("+", rw, aw),
                                ("-", reverse_complement(rw), reverse_complement(aw))):
            ra, aa = affinity(rs, motif), affinity(als, motif)
            if not np.isfinite(ra) or not np.isfinite(aa):
                continue
            candidate = (abs(aa - ra), aa - ra, ra, aa, start, strand)
            for threshold in THRESHOLDS:
                if max(ra, aa) < threshold:
                    continue
                if best[threshold] is None or candidate[0] > best[threshold][0]:
                    best[threshold] = candidate
    return best


def odds_ratio(a, b, c, d):
    # Haldane-Anscombe correction makes zero cells finite and reproducible.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def make_block_permutations(position, n_perm, rng):
    groups = [np.asarray(x, int) for x in pd.Series(np.arange(len(position))).groupby(position).apply(list)]
    by_size = {}
    for g in groups:
        by_size.setdefault(len(g), []).append(g)
    for _ in range(n_perm):
        mapping = np.empty(len(position), dtype=int)
        for same_size in by_size.values():
            destination_order = rng.permutation(len(same_size))
            for source, destination_number in zip(same_size, destination_order):
                destination = same_size[destination_number]
                mapping[destination] = source
        yield mapping


def union_find_clusters(binary, motif_ids, cutoff):
    parent = list(range(len(motif_ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sizes = binary.sum(axis=1)
    for i in range(len(motif_ids)):
        for j in range(i + 1, len(motif_ids)):
            intersection = np.logical_and(binary[i], binary[j]).sum()
            union_size = sizes[i] + sizes[j] - intersection
            if union_size and intersection / union_size >= cutoff:
                union(i, j)
    roots = {}
    labels = []
    for i in range(len(motif_ids)):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots) + 1
        labels.append(roots[root])
    return np.asarray(labels, int)


print("Loading SUL1 paired-condition variants...")
d = pd.read_csv(INPUT, sep="\t", low_memory=False)
d = d.reset_index(drop=True)
required = {"position_1based", "reference_allele", "alternative_allele",
            "reference_sequence", "alternative_sequence", "condition_difference"}
missing = required.difference(d.columns)
if missing:
    raise KeyError(f"Missing columns: {sorted(missing)}")
d["reference_sequence"] = d.reference_sequence.astype(str).str.upper()
d["alternative_sequence"] = d.alternative_sequence.astype(str).str.upper()
d["mutation_index"] = d.reference_sequence.str.len() // 2
d["absolute_condition_difference"] = d.condition_difference.abs()
cutoff = float(d.absolute_condition_difference.quantile(1 - TOP_FRACTION))
top = d.absolute_condition_difference.ge(cutoff).to_numpy(bool)

print("Loading fungal PWMs...")
with JASPAR.open(encoding="utf-8") as handle:
    pwm = [prepare(x) for x in motifs.parse(handle, "jaspar")]
motif_ids = [x["motif_id"] for x in pwm]
motif_names = [x["motif_name"] for x in pwm]
print(f"Variants: {len(d):,}; positions: {d.position_1based.nunique():,}; motifs: {len(pwm):,}")

# Scan once without a threshold; all requested thresholds are then derived identically.
shape = (len(pwm), len(d))
ref_aff = {threshold: np.full(shape, np.nan, dtype=np.float32)
           for threshold in THRESHOLDS}
alt_aff = {threshold: np.full(shape, np.nan, dtype=np.float32)
           for threshold in THRESHOLDS}
started = time.time()
print("Scanning best mutation-overlapping site for every motif and variant...")
for m, motif in enumerate(pwm):
    for row in d.itertuples():
        threshold_results = best_overlap(
            row.reference_sequence, row.alternative_sequence,
            row.mutation_index, motif)
        for threshold, result in threshold_results.items():
            if result is not None:
                _, _, ra, aa, _, _ = result
                ref_aff[threshold][m, row.Index] = ra
                alt_aff[threshold][m, row.Index] = aa
    if (m + 1) % 25 == 0 or m + 1 == len(pwm):
        print(f"  Scanned {m + 1}/{len(pwm)} motifs | {(time.time()-started)/60:.1f} min")

matrices = {}
for threshold in THRESHOLDS:
    qualifying = np.isfinite(ref_aff[threshold]) & np.isfinite(alt_aff[threshold])
    matrices[(threshold, "any")] = qualifying
    matrices[(threshold, "gain")] = (qualifying & (ref_aff[threshold] < threshold)
                                       & (alt_aff[threshold] >= threshold))
    matrices[(threshold, "loss")] = (qualifying & (ref_aff[threshold] >= threshold)
                                       & (alt_aff[threshold] < threshold))

print(f"Running {PERMUTATIONS:,} position-block permutations...")
rng = np.random.default_rng(SEED)
keys = list(matrices)
observed = {}
observed_z = {}
exceed = {key: np.zeros(len(pwm), dtype=np.int64) for key in keys}
exceed_global = {key: np.zeros(len(pwm), dtype=np.int64) for key in keys}
top_n, other_n = int(top.sum()), int((~top).sum())
for key, binary in matrices.items():
    top_hits = binary[:, top].sum(axis=1)
    other_hits = binary[:, ~top].sum(axis=1)
    observed[key] = top_hits / top_n - other_hits / other_n
    pooled = (top_hits + other_hits) / (top_n + other_n)
    se = np.sqrt(pooled * (1 - pooled) * (1 / top_n + 1 / other_n))
    observed_z[key] = np.divide(observed[key], se, out=np.zeros_like(observed[key]),
                                where=se > 0)

for iteration, mapping in enumerate(
        make_block_permutations(d.position_1based.to_numpy(), PERMUTATIONS, rng), start=1):
    permuted_top = top[mapping]
    pn, on = int(permuted_top.sum()), int((~permuted_top).sum())
    permutation_z = {}
    for key, binary in matrices.items():
        perm_top_hits = binary[:, permuted_top].sum(axis=1)
        perm_other_hits = binary[:, ~permuted_top].sum(axis=1)
        statistic = perm_top_hits / pn - perm_other_hits / on
        exceed[key] += statistic >= (observed[key] - 1e-15)
        pooled = (perm_top_hits + perm_other_hits) / (pn + on)
        se = np.sqrt(pooled * (1 - pooled) * (1 / pn + 1 / on))
        permutation_z[key] = np.divide(statistic, se, out=np.zeros_like(statistic),
                                       where=se > 0)
    # Westfall-Young single-step maxT controls all motifs, thresholds and event types.
    global_max_z = max(float(z.max()) for z in permutation_z.values())
    for key in keys:
        exceed_global[key] += global_max_z >= (observed_z[key] - 1e-15)
    if iteration % 500 == 0:
        print(f"  Permutations {iteration:,}/{PERMUTATIONS:,}")

rows = []
for threshold, event_type in keys:
    binary = matrices[(threshold, event_type)]
    a = binary[:, top].sum(axis=1)
    c = binary[:, ~top].sum(axis=1)
    for m in range(len(pwm)):
        rows.append({
            "affinity_threshold": threshold, "event_type": event_type,
            "motif_id": motif_ids[m], "motif_name": motif_names[m],
            "top_effect_with_event": int(a[m]),
            "top_effect_without_event": int(top_n - a[m]),
            "other_with_event": int(c[m]),
            "other_without_event": int(other_n - c[m]),
            "prevalence_difference": float(observed[(threshold, event_type)][m]),
            "odds_ratio_corrected": float(odds_ratio(a[m], top_n-a[m], c[m], other_n-c[m])),
            "empirical_p": float((exceed[(threshold, event_type)][m] + 1) / (PERMUTATIONS + 1)),
            "global_fwer_maxT": float((exceed_global[(threshold, event_type)][m] + 1)
                                      / (PERMUTATIONS + 1)),
        })
results = pd.DataFrame(rows)
# BH is reported as a secondary global FDR measure; maxT is the conservative primary test.
results["global_fdr_bh"] = bh(results.empirical_p)

# Collapse highly overlapping central-threshold annotations for interpretation.
central_any = matrices[(0.80, "any")]
cluster_labels = union_find_clusters(central_any, motif_ids, JACCARD_CUTOFF)
cluster_map = dict(zip(motif_ids, cluster_labels))
results["annotation_cluster"] = results.motif_id.map(cluster_map)

sig_counts = (results.assign(significant=results.global_fwer_maxT.lt(0.05))
              .groupby(["motif_id", "event_type"]).significant.sum()
              .rename("significant_threshold_count").reset_index())
results = results.merge(sig_counts, on=["motif_id", "event_type"], how="left")

central = results[(results.affinity_threshold == 0.80) & (results.event_type == "any")].copy()
central = central.sort_values(["annotation_cluster", "global_fdr_bh", "empirical_p"])
central["cluster_representative"] = ~central.annotation_cluster.duplicated()
shortlist = central[
    central.global_fwer_maxT.lt(0.05)
    & central.significant_threshold_count.ge(2)
    & central.top_effect_with_event.ge(MIN_TOP_HITS)
    & central.cluster_representative
].sort_values(["global_fwer_maxT", "empirical_p", "odds_ratio_corrected"],
              ascending=[True, True, False])

results.to_csv(OUTPUT / "position_clustered_motif_permutation_results.tsv", sep="\t", index=False)
shortlist.to_csv(OUTPUT / "conservative_motif_shortlist.tsv", sep="\t", index=False)
pd.DataFrame({"motif_id": motif_ids, "motif_name": motif_names,
              "annotation_cluster": cluster_labels}).to_csv(
    OUTPUT / "motif_annotation_clusters.tsv", sep="\t", index=False)

summary = {
    "variants": int(len(d)), "independent_position_blocks": int(d.position_1based.nunique()),
    "motifs": int(len(pwm)), "thresholds": list(THRESHOLDS),
    "event_types": list(EVENT_TYPES), "permutations": PERMUTATIONS,
    "top_effect_cutoff": cutoff, "global_tests": int(len(results)),
    "globally_fdr_significant_tests": int(results.global_fdr_bh.lt(0.05).sum()),
    "globally_maxT_fwer_significant_tests": int(results.global_fwer_maxT.lt(0.05).sum()),
    "annotation_clusters_at_0.80": int(len(set(cluster_labels))),
    "conservative_shortlist_motifs": int(len(shortlist)),
    "shortlist_rule": (
        "threshold 0.80 any-event global maxT-FWER <0.05; significant at >=2 thresholds; "
        f">={MIN_TOP_HITS} top-effect hits; one representative per Jaccard annotation cluster"),
    "warning": "PWM associations are computational and do not establish direct TF binding or causality.",
}
(OUTPUT / "position_clustered_motif_robustness_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")

print("\nPOSITION-CLUSTERED MOTIF ROBUSTNESS SUMMARY")
print(json.dumps(summary, indent=2))
print("\nCONSERVATIVE MOTIF SHORTLIST")
if shortlist.empty:
    print("No motif passed the complete conservative rule.")
else:
    print(shortlist[["motif_id", "motif_name", "annotation_cluster",
                     "top_effect_with_event", "odds_ratio_corrected",
                     "empirical_p", "global_fdr_bh", "global_fwer_maxT",
                     "significant_threshold_count"]].to_string(index=False))
print("\nStep 42 position-clustered motif robustness: COMPLETED")
print("Saved:", OUTPUT)
