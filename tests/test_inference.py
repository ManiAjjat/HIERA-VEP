from pathlib import Path
import numpy as np
import pandas as pd
import json
import pytest
from src.hiera_vep_score import score_frame, build_design

ROOT=Path(__file__).resolve().parents[1]

def test_model_dimensions(example, assets):
    model,vec,spec=assets
    assert len(vec.vocabulary_)==5440
    assert model.n_features_in_==27212
    assert model.alpha==1000
    assert build_design(example,vec,spec)[0].shape==(12,27212)

def test_stable_smoke_predictions(example, assets):
    expected=json.loads((ROOT/"tests/smoke_expected.json").read_text())
    actual=score_frame(example,*assets)
    assert actual.row_id.astype(str).tolist()==expected["row_ids"]
    np.testing.assert_allclose(actual.hiera_vep_prediction,expected["predictions"],rtol=0,atol=1e-5)

def test_wrong_feature_order(example, assets):
    model,vec,spec=assets; bad=dict(spec);bad["gating_columns"]=list(reversed(spec["gating_columns"]))
    with pytest.raises(ValueError,match="order"): score_frame(example,model,vec,bad)

def test_dimension_mismatch(example,assets):
    import copy
    model,vec,spec=assets;bad=copy.copy(model);bad.n_features_in_=1
    with pytest.raises(ValueError,match="Feature dimension"): score_frame(example,bad,vec,spec)
