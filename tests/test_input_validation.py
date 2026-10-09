import numpy as np
import pytest
from src.hiera_vep_score import score_frame, find_asset

def test_valid_example(example, assets):
    assert len(score_frame(example, *assets)) == 12

@pytest.mark.parametrize("column", ["reference_sequence", "alternative_sequence", "phyloP100way", "primary_ccre_class"])
def test_missing_column(example, assets, column):
    with pytest.raises(KeyError): score_frame(example.drop(columns=column), *assets)

def test_unsupported_class(example, assets):
    example.loc[0,"primary_ccre_class"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported"): score_frame(example, *assets)

@pytest.mark.parametrize("case", ["unequal", "multiple", "none", "invalid"])
def test_invalid_sequence(example, assets, case):
    ref=example.loc[0,"reference_sequence"]
    if case=="unequal": alt=ref+"A"
    elif case=="multiple": alt=''.join(('C' if c=='A' else 'A') if i<2 else c for i,c in enumerate(ref))
    elif case=="invalid": alt="X"+ref[1:]
    else: alt=ref
    example.loc[0,"alternative_sequence"]=alt
    with pytest.raises(ValueError): score_frame(example, *assets)

@pytest.mark.parametrize("column", ["ccre_overlap", "promoter_overlap", "absolute_distance_to_nearest_tss", "phyloP100way", "phastCons100way", "phyloP_conserved", "phyloP_accelerated", "phastCons_high", "primary_ccre_class"])
def test_missing_context(example, assets, column):
    example[column]=example[column].astype(object)
    example.loc[0,column]=None
    with pytest.raises(ValueError): score_frame(example, *assets)

@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_nonfinite_context(example, assets, value):
    example.loc[0,"phyloP100way"]=value
    with pytest.raises(ValueError): score_frame(example,*assets)

def test_assets_missing_or_ambiguous(tmp_path):
    with pytest.raises(FileNotFoundError): find_asset(tmp_path,"model.joblib")
    for d in [tmp_path/"a",tmp_path/"b"]:
        d.mkdir();(d/"model.joblib").touch()
    with pytest.raises(FileNotFoundError): find_asset(tmp_path,"model.joblib")
