import os
from pathlib import Path
import json, time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
PANEL = ROOT / "data/processed/mavedb/regulatory_panel/mavedb_regulatory_within_element_splits.tsv"
OUT = ROOT / "results/external_validation_inventory"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.mavedb.org/api/v1"
TERMS = ["regulatory", "MPRA", "promoter", "enhancer", "5' UTR", "expression"]
REGULATORY_WORDS = ("regulatory", "mpra", "promoter", "enhancer", "expression", "utr")
MIN_VARIANTS = 100
METADATA_CACHE = OUT / "mavedb_candidate_metadata_checkpoint.json"
FAILURE_LOG = OUT / "api_request_failures.tsv"

retry_policy = Retry(
    total=8, connect=8, read=8, status=8,
    backoff_factor=1.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(("GET", "POST")),
    raise_on_status=False,
)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry_policy, pool_connections=4, pool_maxsize=4))
session.headers.update({"User-Agent": "CausalRegNet-MaveDB-inventory/1.1"})


def request(method, url, required=True, **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            r = session.request(method, url, timeout=(30, 120), **kwargs)
            r.raise_for_status()
            time.sleep(0.20)
            return r
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
    if required:
        raise last_error
    return None


def records(payload):
    if isinstance(payload, list): return payload
    for key in ("items", "results", "data", "scoreSets"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list): return payload[key]
    return []


def flatten_text(obj):
    if obj is None: return ""
    if isinstance(obj, str): return obj
    if isinstance(obj, (int, float, bool)): return str(obj)
    if isinstance(obj, list): return " ".join(flatten_text(x) for x in obj)
    if isinstance(obj, dict): return " ".join(flatten_text(v) for v in obj.values())
    return str(obj)


def first(meta, *keys):
    for key in keys:
        value = meta.get(key)
        if value not in (None, "", [], {}): return value
    return None


if not PANEL.exists():
    raise FileNotFoundError(f"Existing regulatory panel not found: {PANEL}")
panel = pd.read_csv(PANEL, sep="\t", low_memory=False)
existing = set()
if "score_set_urns" in panel:
    for value in panel.score_set_urns.dropna().astype(str):
        for urn in value.replace(";", ",").split(","):
            urn = urn.strip()
            if urn.startswith("urn:mavedb:"): existing.add(urn)
print("Existing score sets excluded:", len(existing))

found = {}
for term in TERMS:
    offset = 0
    while True:
        response = request("POST", f"{API}/score-sets/search",
                           json={"text": term, "limit": 100, "offset": offset})
        payload = response.json()
        batch = records(payload)
        if not batch: break
        for item in batch:
            urn = item.get("urn") or item.get("accession")
            if urn: found[urn] = item
        if len(batch) < 100: break
        offset += len(batch)
    print(f"Search '{term}': cumulative unique score sets = {len(found)}")

rows=[]
if METADATA_CACHE.exists():
    raw_metadata = json.loads(METADATA_CACHE.read_text(encoding="utf-8"))
    print("Cached metadata records resumed:", len(raw_metadata))
else:
    raw_metadata = {}
failures=[]
for i, urn in enumerate(sorted(found), 1):
    if urn in raw_metadata:
        meta = raw_metadata[urn]
    else:
        response = request("GET", f"{API}/score-sets/{urn}", required=False)
        if response is None:
            failures.append({"score_set_urn":urn,"stage":"metadata","status":"temporarily_unavailable"})
            print(f"  WARNING: skipped unavailable metadata {urn}")
            continue
        meta = response.json()
        raw_metadata[urn] = meta
        METADATA_CACHE.write_text(json.dumps(raw_metadata, indent=2), encoding="utf-8")
    text = flatten_text(meta).lower()
    title = str(first(meta, "title", "name") or "")
    n = first(meta, "numVariants", "num_variants", "variantCount")
    try: n = int(n)
    except (TypeError, ValueError): n = None
    target_text = flatten_text(first(meta, "targetGenes", "targets", "target", "targetSequence"))
    method_text = flatten_text(first(meta, "methodText", "methods", "assay", "assayFacts"))
    publication = flatten_text(first(meta, "primaryPublicationIdentifiers", "publicationIdentifiers", "publication"))
    is_regulatory = any(word in text for word in REGULATORY_WORDS)
    independent = urn not in existing
    enough_variants = n is not None and n >= MIN_VARIANTS
    mapped_status="not_checked_not_candidate"
    if is_regulatory and independent and enough_variants:
        mapped=request("GET",f"{API}/score-sets/{urn}/mapped-variants",required=False)
        if mapped is None:
            mapped_status="temporarily_unavailable"
            failures.append({"score_set_urn":urn,"stage":"mapped_variants","status":"temporarily_unavailable"})
        else:
            mapped_status="available" if len(mapped.content)>20 else "empty"
    rows.append({
        "score_set_urn":urn, "title":title, "num_variants":n,
        "target_summary":target_text[:1000], "assay_summary":method_text[:1500],
        "publication":publication[:500], "regulatory_keyword_match":is_regulatory,
        "already_in_current_panel":not independent, "mapped_variants":mapped_status,
        "passes_initial_screen":bool(is_regulatory and independent and enough_variants),
        "review_status":"pending_manual_review" if is_regulatory and independent and enough_variants else "screened_out",
        "exclusion_reason":"" if is_regulatory and independent and enough_variants else
            ("already_used" if not independent else "not_regulatory_or_too_small"),
    })
    pd.DataFrame(rows).to_csv(OUT/"mavedb_regulatory_dataset_inventory_checkpoint.tsv",sep="\t",index=False)
    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_LOG,sep="\t",index=False)
    if i % 25 == 0: print(f"Metadata checked: {i}/{len(found)}")

inventory=pd.DataFrame(rows).sort_values(
    ["passes_initial_screen","num_variants"], ascending=[False,False], na_position="last")
eligible=inventory[inventory.passes_initial_screen].copy()
inventory.to_csv(OUT/"mavedb_regulatory_dataset_inventory.tsv",sep="\t",index=False)
eligible.to_csv(OUT/"external_validation_candidates.tsv",sep="\t",index=False)
(OUT/"mavedb_candidate_metadata.json").write_text(json.dumps(raw_metadata,indent=2),encoding="utf-8")
summary={"search_terms":TERMS,"unique_score_sets_found":len(inventory),
         "search_hits_before_metadata_failures":len(found),
         "temporarily_unavailable_requests":len(failures),
         "existing_score_sets_excluded":len(existing),"initial_external_candidates":len(eligible),
         "minimum_variants":MIN_VARIANTS,
         "warning":"Initial keyword screening is not final eligibility. Every candidate requires manual assay, target, score-direction, sequence, and study-independence review."}
(OUT/"external_inventory_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\nEXTERNAL VALIDATION INVENTORY SUMMARY")
print(json.dumps(summary,indent=2))
if len(eligible):
    print("\nTOP CANDIDATES")
    print(eligible[["score_set_urn","title","num_variants","mapped_variants"]].head(30).to_string(index=False))
print("\nStep 32 external regulatory dataset inventory: COMPLETED")
print("Saved:",OUT)
