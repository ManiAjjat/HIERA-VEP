# Installation

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Conda users may run `conda env create -f environment.yml`. The model and vectorizer were serialized with scikit-learn 1.9.1. Do not suppress a version warning or substitute a different scikit-learn version.

Core inference needs a CPU and no GPU; memory scales with batch size and sparse sequence features. No precise hardware benchmark is claimed. Optional analyses use `requirements-analysis.txt`; the frozen Nucleotide Transformer comparisons require externally downloaded weights and may require a GPU. PyTorch 2.5.1 GPU installation depends on the local driver and platform. Model assets are repository files and are not automatically included in a pip wheel; use `--bundle` pointing to the checkout.
