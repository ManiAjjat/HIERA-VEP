from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_model_checksums():
    manifest=pd.read_csv(ROOT/"models/MODEL_ASSET_MANIFEST.tsv",sep="\t")
    for row in manifest.itertuples():
        p=ROOT/"models"/row.filename
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row.public_sha256

def test_recorded_locked_smoke():
    report=json.loads((ROOT/"models/locked_inference_smoke_test.json").read_text())
    assert report["status"]=="PASS" and report["rows"]==4408
    assert report["prediction_max_absolute_difference"]<1e-5
    assert report["additivity_max_absolute_error"]<1e-5
