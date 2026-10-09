import os
from pathlib import Path
import copy
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse, stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(os.environ.get("HIERA_VEP_PROJECT", Path(__file__).resolve().parents[2]))
INPUT = ROOT / "results/hiera_unified_training_table/hiera_unified_variant_table.tsv"
OUTPUT = ROOT / "results/hiera_vep_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 20261007
NGRAM_RANGE = (3, 6)
RANK = 8
LEARNING_RATE = 0.03
MAX_EPOCHS = 2000
PATIENCE = 120
MIN_DELTA = 1e-5
RIDGE_ALPHA = 100.0
LAMBDA_SHARED = 1e-4
LAMBDA_INTERACTION = 5e-4
LAMBDA_CONTEXT = 2e-3
LAMBDA_GROUP = 2e-3
LAMBDA_SCALE = 5e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean(values):
    return values.fillna("").astype(str).str.upper().tolist()


def delta_kmers(vectorizer, frame):
    ref = vectorizer.transform(clean(frame.reference_sequence))
    alt = vectorizer.transform(clean(frame.alternative_sequence))
    return (alt - ref).astype(np.float32).tocsr()


def scipy_to_torch(matrix):
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=DEVICE)
    values = torch.tensor(coo.data, dtype=torch.float32, device=DEVICE)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=DEVICE).coalesce()


def category_map(training_values):
    # Zero is permanently reserved for an unseen category.
    return {value: i + 1 for i, value in enumerate(sorted(set(training_values.astype(str))))}


def encode(values, mapping):
    return torch.tensor([mapping.get(str(x), 0) for x in values],
                        dtype=torch.long, device=DEVICE)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    pearson = stats.pearsonr(y, p).statistic if np.std(p) > 0 else np.nan
    spearman = stats.spearmanr(y, p).statistic if np.std(p) > 0 else np.nan
    return {
        "n": len(y), "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "r2": r2_score(y, p), "pearson_r": pearson,
        "spearman_rho": spearman,
        "direction_accuracy": np.mean(np.sign(y) == np.sign(p)),
        "observed_sd": np.std(y, ddof=1), "predicted_sd": np.std(p, ddof=1),
    }


class HieraVEP(torch.nn.Module):
    """Transparent hierarchical sequence model with low-rank context modulation."""

    def __init__(self, n_features, n_context, n_assay, n_element, rank):
        super().__init__()
        self.global_bias = torch.nn.Parameter(torch.zeros(()))
        self.shared_beta = torch.nn.Parameter(torch.zeros(n_features))
        self.interaction_basis = torch.nn.Parameter(torch.randn(n_features, rank) * 0.005)
        self.context_embedding = torch.nn.Embedding(n_context + 1, rank, padding_idx=0)
        self.assay_bias = torch.nn.Embedding(n_assay + 1, 1, padding_idx=0)
        self.element_bias = torch.nn.Embedding(n_element + 1, 1, padding_idx=0)
        self.assay_log_scale = torch.nn.Embedding(n_assay + 1, 1, padding_idx=0)
        torch.nn.init.zeros_(self.context_embedding.weight)
        torch.nn.init.zeros_(self.assay_bias.weight)
        torch.nn.init.zeros_(self.element_bias.weight)
        torch.nn.init.zeros_(self.assay_log_scale.weight)

    def forward(self, x, context, assay, element, return_parts=False):
        shared = torch.sparse.mm(x, self.shared_beta[:, None]).squeeze(1)
        latent_sequence = torch.sparse.mm(x, self.interaction_basis)
        interaction = (latent_sequence * self.context_embedding(context)).sum(1)
        sequence_effect = shared + interaction
        scale = torch.exp(torch.clamp(self.assay_log_scale(assay).squeeze(1), -2, 2))
        assay_part = self.assay_bias(assay).squeeze(1)
        element_part = self.element_bias(element).squeeze(1)
        prediction = self.global_bias + assay_part + element_part + scale * sequence_effect
        if return_parts:
            return prediction, shared, interaction, scale, assay_part, element_part
        return prediction

    def penalty(self):
        return (LAMBDA_SHARED * self.shared_beta.square().mean()
                + LAMBDA_INTERACTION * self.interaction_basis.square().mean()
                + LAMBDA_CONTEXT * self.context_embedding.weight[1:].square().mean()
                + LAMBDA_GROUP * (self.assay_bias.weight[1:].square().mean()
                                  + self.element_bias.weight[1:].square().mean())
                + LAMBDA_SCALE * self.assay_log_scale.weight[1:].square().mean())


def predict_parts(model, x, context, assay, element):
    model.eval()
    with torch.no_grad():
        parts = model(x, context, assay, element, return_parts=True)
    return [z.detach().cpu().numpy() for z in parts]


print("Loading Step 44 unified table...")
data = pd.read_csv(INPUT, sep="\t", low_memory=False)
development = data[data.source_partition.eq("development")].copy().reset_index(drop=True)
if development.empty:
    raise RuntimeError("Development partition is empty")
split = development.original_split.astype(str).str.lower()
train_idx = np.flatnonzero(split.eq("train"))
val_idx = np.flatnonzero(split.isin(["validation", "val"]))
test_idx = np.flatnonzero(split.eq("test"))
if min(map(len, [train_idx, val_idx, test_idx])) == 0:
    raise ValueError(f"Invalid development split counts: {split.value_counts().to_dict()}")

print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Train / validation / locked test:", len(train_idx), len(val_idx), len(test_idx))

print("Fitting interpretable k-mer vocabulary on training variants only...")
vectorizer = CountVectorizer(analyzer="char", ngram_range=NGRAM_RANGE,
                             lowercase=False, dtype=np.float32)
vectorizer.fit(clean(development.iloc[train_idx].reference_sequence) +
               clean(development.iloc[train_idx].alternative_sequence))
X = delta_kmers(vectorizer, development)
print("Sequence features:", X.shape[1], "nonzero delta entries:", X.nnz)

maps = {
    "context": category_map(development.iloc[train_idx].context_group),
    "assay": category_map(development.iloc[train_idx].assay_id),
    "element": category_map(development.iloc[train_idx].element_group),
}
context = encode(development.context_group, maps["context"])
assay = encode(development.assay_id, maps["assay"])
element = encode(development.element_group, maps["element"])
y = development.target_model.to_numpy(np.float32)
y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

X_train = scipy_to_torch(X[train_idx])
X_val = scipy_to_torch(X[val_idx])
X_test = scipy_to_torch(X[test_idx])
ti = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
vi = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)
tei = torch.tensor(test_idx, dtype=torch.long, device=DEVICE)

model = HieraVEP(X.shape[1], len(maps["context"]), len(maps["assay"]),
                 len(maps["element"]), RANK).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-5)

best_state, best_epoch, best_val = None, 0, np.inf
wait = 0
history = []
print("Training HIERA-VEP v1 with validation-based early stopping...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    prediction = model(X_train, context[ti], assay[ti], element[ti])
    mse = torch.mean((prediction - y_t[ti]) ** 2)
    loss = mse + model.penalty()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prediction = model(X_val, context[vi], assay[vi], element[vi])
        val_mse = torch.mean((val_prediction - y_t[vi]) ** 2).item()
    scheduler.step(val_mse)
    history.append({"epoch": epoch, "training_mse": float(mse.item()),
                    "training_objective": float(loss.item()), "validation_mse": val_mse,
                    "learning_rate": optimizer.param_groups[0]["lr"]})
    if val_mse < best_val - MIN_DELTA:
        best_val, best_epoch = val_mse, epoch
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
    if epoch % 50 == 0:
        print(f"  epoch={epoch:4d} train_mse={mse.item():.4f} val_mse={val_mse:.4f} best={best_val:.4f}")
    if wait >= PATIENCE:
        print("Early stopping at epoch", epoch)
        break

if best_state is None:
    raise RuntimeError("No valid model state was produced")
model.load_state_dict(best_state)

print("Fitting shared Ridge baseline on identical training features...")
ridge = Ridge(alpha=RIDGE_ALPHA)
ridge.fit(X[train_idx], y[train_idx])

records, prediction_tables = [], []
for split_name, indices, x_torch, index_tensor in [
        ("validation", val_idx, X_val, vi), ("locked_internal_test", test_idx, X_test, tei)]:
    total, shared, interaction, scale, assay_part, element_part = predict_parts(
        model, x_torch, context[index_tensor], assay[index_tensor], element[index_tensor])
    ridge_prediction = ridge.predict(X[indices])
    zero_prediction = np.zeros(len(indices))
    for name, values in [("hiera_vep_v1", total), ("shared_kmer_ridge", ridge_prediction),
                         ("zero_baseline", zero_prediction)]:
        records.append({"split": split_name, "model": name, **metrics(y[indices], values)})
    table = development.iloc[indices][["row_id", "assay_id", "element_id", "position_group",
                                       "mutation_class", "target_model"]].copy()
    table["split"] = split_name
    table["hiera_prediction"] = total
    table["shared_sequence_component"] = shared * scale
    table["context_interaction_component"] = interaction * scale
    table["assay_intercept_component"] = assay_part
    table["element_intercept_component"] = element_part
    table["assay_sequence_scale"] = scale
    table["shared_ridge_prediction"] = ridge_prediction
    prediction_tables.append(table)

metric_table = pd.DataFrame(records)
predictions = pd.concat(prediction_tables, ignore_index=True)
history_table = pd.DataFrame(history)

feature_names = np.asarray(vectorizer.get_feature_names_out())
shared_beta = model.shared_beta.detach().cpu().numpy()
feature_table = pd.DataFrame({"kmer": feature_names, "shared_coefficient": shared_beta})
feature_table["absolute_shared_coefficient"] = feature_table.shared_coefficient.abs()
feature_table = feature_table.sort_values("absolute_shared_coefficient", ascending=False)

context_rows = []
inverse_context = {v: k for k, v in maps["context"].items()}
basis = model.interaction_basis.detach().cpu().numpy()
embeddings = model.context_embedding.weight.detach().cpu().numpy()
for code, context_name in inverse_context.items():
    coefficients = shared_beta + basis @ embeddings[code]
    order = np.argsort(np.abs(coefficients))[::-1][:100]
    for rank_number, j in enumerate(order, 1):
        context_rows.append({"context_group": context_name, "rank": rank_number,
                             "kmer": feature_names[j], "effective_coefficient": coefficients[j],
                             "shared_coefficient": shared_beta[j],
                             "context_deviation": coefficients[j] - shared_beta[j]})
context_features = pd.DataFrame(context_rows)

torch.save({"state_dict": best_state, "n_features": X.shape[1], "rank": RANK,
            "mappings": maps, "settings": {"ngram_range": NGRAM_RANGE,
            "best_epoch": best_epoch}}, OUTPUT / "hiera_vep_v1_model.pt")
joblib.dump(vectorizer, OUTPUT / "hiera_vep_v1_vectorizer.joblib")
joblib.dump(ridge, OUTPUT / "shared_ridge_comparator.joblib")
metric_table.to_csv(OUTPUT / "hiera_v1_point_estimates.tsv", sep="\t", index=False)
predictions.to_csv(OUTPUT / "hiera_v1_predictions.tsv", sep="\t", index=False)
history_table.to_csv(OUTPUT / "hiera_v1_training_history.tsv", sep="\t", index=False)
feature_table.to_csv(OUTPUT / "hiera_v1_shared_kmer_coefficients.tsv", sep="\t", index=False)
context_features.to_csv(OUTPUT / "hiera_v1_context_kmer_coefficients.tsv", sep="\t", index=False)

settings = {
    "model": "HIERA-VEP-v1", "device": str(DEVICE), "seed": SEED,
    "training_rows": len(train_idx), "validation_rows": len(val_idx),
    "locked_test_rows": len(test_idx), "sequence_features": X.shape[1],
    "interaction_rank": RANK, "best_epoch": best_epoch,
    "best_validation_mse": best_val, "learning_rate": LEARNING_RATE,
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "regularization": {"shared": LAMBDA_SHARED, "interaction": LAMBDA_INTERACTION,
                       "context": LAMBDA_CONTEXT, "group": LAMBDA_GROUP,
                       "scale": LAMBDA_SCALE},
    "interpretation": (
        "Prediction decomposes into shared sequence, sequence-context interaction, "
        "assay calibration, and element intercept components."),
    "scope": "Development-only v1 feasibility; hyperparameters are not yet optimized.",
}
(OUTPUT / "hiera_v1_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

print("\nHIERA-VEP V1 POINT ESTIMATES")
print(metric_table.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
print("\nBest epoch:", best_epoch, "validation MSE:", round(best_val, 6))
print("\nStep 45 HIERA-VEP version 1: COMPLETED")
print("Saved:", OUTPUT)
