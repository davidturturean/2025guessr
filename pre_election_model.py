# Pre-election runoff prediction pipeline
# Implements data loading, preprocessing, model training, and prediction

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import joblib


AGE_BRACKETS = [(18, 29), (30, 44), (45, 59), (60, 120)]
SEXES = ["M", "F"]


def aggregate_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate single-age columns into broader age brackets."""
    for sex in SEXES:
        for low, high in AGE_BRACKETS:
            cols = [f"{sex}_{age}" for age in range(low, high + 1) if f"{sex}_{age}" in df.columns]
            if cols:
                df[f"{sex}_{low}_{high}"] = df[cols].sum(axis=1)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Compute turnout and vote share features and aggregate demographics."""
    df = aggregate_age_groups(df)

    df["turnout_pct_1st_round"] = df["votes_first_round_total"] / df["registered_voters"].clip(lower=1)
    df["candA_share_1st_round"] = df["candA_votes_1st_round"] / df["votes_first_round_total"].clip(lower=1)
    df["candB_share_1st_round"] = df["candB_votes_1st_round"] / df["votes_first_round_total"].clip(lower=1)
    df["other_share_1st_round"] = 1.0 - (
        df["candA_share_1st_round"] + df["candB_share_1st_round"]
    )
    df["urban_flag"] = df["urban_rural"].map({"Urban": 1, "Rural": 0}).fillna(0)

    for sex in SEXES:
        for low, high in AGE_BRACKETS:
            col = f"{sex}_{low}_{high}"
            if col in df.columns:
                df[col] = df[col] / df["registered_voters"].clip(lower=1)

    df = add_neighbor_features(df)
    return df


def add_neighbor_features(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Add features derived from nearby precincts using coordinates."""
    if "lat" not in df.columns or "lon" not in df.columns:
        return df

    coords = df[["lat", "lon"]].to_numpy()
    if len(df) <= 1:
        df["neighbor_candA_share"] = df["candA_share_1st_round"]
        df["neighbor_candB_share"] = df["candB_share_1st_round"]
        return df

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(df)))
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    neighbor_a = []
    neighbor_b = []
    for row_idx, neigh in enumerate(indices):
        neigh = neigh[1:]
        neighbor_a.append(df.iloc[neigh]["candA_share_1st_round"].mean())
        neighbor_b.append(df.iloc[neigh]["candB_share_1st_round"].mean())

    df["neighbor_candA_share"] = neighbor_a
    df["neighbor_candB_share"] = neighbor_b
    return df


def build_features(df: pd.DataFrame, numeric_cols, cat_cols, scaler, encoder):
    num = scaler.transform(df[numeric_cols])
    cat = encoder.transform(df[cat_cols])
    return np.hstack([num, cat])


class VoteCountNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(X_train, y_train, weights, X_val, y_val, epochs=50, lr=0.001):
    model = VoteCountNN(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            mse = ((preds - yb) ** 2).mean(dim=1)
            loss = (mse * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pv = model(X_val_t)
            mae_a = mean_absolute_error(y_val[:, 0], pv[:, 0].numpy())
            mae_b = mean_absolute_error(y_val[:, 1], pv[:, 1].numpy())
            r2_a = r2_score(y_val[:, 0], pv[:, 0].numpy())
            r2_b = r2_score(y_val[:, 1], pv[:, 1].numpy())
            print(
                f"Epoch {epoch:3d} loss {epoch_loss/len(train_ds):.2f} "
                f"MAE A {mae_a:.1f} B {mae_b:.1f} "
                f"R2 A {r2_a:.3f} B {r2_b:.3f}"
            )
    return model


def main():
    # Paths to CSV files (update these to actual locations)
    data_2019 = "data/2019_precincts.csv"
    data_2024 = "data/2024_precincts.csv"
    data_2025 = "data/2025_first_round.csv"

    df19 = preprocess(pd.read_csv(data_2019))
    df24 = preprocess(pd.read_csv(data_2024))
    df25 = preprocess(pd.read_csv(data_2025))

    df19["sample_weight"] = 1.0
    df24["sample_weight"] = 2.0

    train_df = pd.concat([df19, df24], ignore_index=True)

    numeric_cols = [
        "turnout_pct_1st_round",
        "candA_share_1st_round",
        "candB_share_1st_round",
        "other_share_1st_round",
        "registered_voters",
    ]
    for sex in SEXES:
        for low, high in AGE_BRACKETS:
            col = f"{sex}_{low}_{high}"
            if col in train_df.columns:
                numeric_cols.append(col)
    numeric_cols.append("urban_flag")
    if "neighbor_candA_share" in train_df.columns:
        numeric_cols.append("neighbor_candA_share")
    if "neighbor_candB_share" in train_df.columns:
        numeric_cols.append("neighbor_candB_share")

    cat_cols = ["county", "uat_name"]

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scaler.fit(train_df[numeric_cols])
    encoder.fit(train_df[cat_cols])

    X_train = build_features(train_df, numeric_cols, cat_cols, scaler, encoder)
    y_train = train_df[["candA_votes_2nd_round", "candB_votes_2nd_round"]].values
    w_train = train_df["sample_weight"].values

    # simple split by county for validation
    val_counties = set(train_df["county"].drop_duplicates().sample(frac=0.2, random_state=42))
    val_mask = train_df["county"].isin(val_counties)
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    w_val = w_train[val_mask]
    X_train = X_train[~val_mask]
    y_train = y_train[~val_mask]
    w_train = w_train[~val_mask]

    model = train_model(X_train, y_train, w_train, X_val, y_val)

    # persist model and preprocessing objects for live updates
    torch.save(model.state_dict(), "runoff_model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")
    meta = {
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
    }
    joblib.dump(meta, "preprocess_meta.pkl")

    # predict 2025
    X_25 = build_features(df25, numeric_cols, cat_cols, scaler, encoder)
    with torch.no_grad():
        preds = model(torch.tensor(X_25, dtype=torch.float32)).numpy()
    df25["pred_candA_votes_2nd_round"] = preds[:, 0].round().astype(int)
    df25["pred_candB_votes_2nd_round"] = preds[:, 1].round().astype(int)

    df25[[
        "precinct_id",
        "pred_candA_votes_2nd_round",
        "pred_candB_votes_2nd_round",
    ]].to_csv("2025_runoff_predictions.csv", index=False)
    print("Saved predictions to 2025_runoff_predictions.csv")


if __name__ == "__main__":
    main()
