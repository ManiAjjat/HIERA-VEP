from pathlib import Path
import json
import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def assets():
    return (joblib.load(ROOT/"models/context_selected_ridge.joblib"),
            joblib.load(ROOT/"models/context_model_vectorizer.joblib"),
            json.loads((ROOT/"models/preprocessing_spec.json").read_text()))

@pytest.fixture
def example():
    return pd.read_csv(ROOT/"examples/example_scoring_input.tsv", sep="\t")
