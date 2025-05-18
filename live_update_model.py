"""Real-time election-night prediction updater."""

import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import Ridge

from pre_election_model import VoteCountNN, preprocess, build_features


def load_artifacts():
    meta = joblib.load("preprocess_meta.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    model = VoteCountNN(len(meta["numeric_cols"]) + encoder.transform([["dummy", "dummy"]]).shape[1])
    model.load_state_dict(torch.load("runoff_model.pth"))
    model.eval()
    return model, scaler, encoder, meta


def update_predictions(partial_results_csv: str, baseline_csv: str = "2025_runoff_predictions.csv", features_csv: str = "data/2025_first_round.csv"):
    model, scaler, encoder, meta = load_artifacts()
    baseline = pd.read_csv(baseline_csv)
    features = preprocess(pd.read_csv(features_csv))
    df = baseline.merge(features, on="precinct_id")

    # Merge partial results if available
    partial = pd.read_csv(partial_results_csv)
    df = df.merge(partial, on="precinct_id", how="left", suffixes=("", "_reported"))
    reported = df["candA_votes_reported"].notna()

    X_all = build_features(df, meta["numeric_cols"], meta["cat_cols"], scaler, encoder)
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    with torch.no_grad():
        baseline_pred = model(X_tensor).numpy()

    df["baseline_A"] = baseline_pred[:, 0]
    df["baseline_B"] = baseline_pred[:, 1]

    # Residuals for reported precincts
    residual_A = df.loc[reported, "candA_votes_reported"] - df.loc[reported, "baseline_A"]
    residual_B = df.loc[reported, "candB_votes_reported"] - df.loc[reported, "baseline_B"]
    X_rep = X_all[reported]

    if len(residual_A) >= 5:
        model_a = Ridge(alpha=1.0).fit(X_rep, residual_A)
        model_b = Ridge(alpha=1.0).fit(X_rep, residual_B)
        adjust_A = model_a.predict(X_all)
        adjust_B = model_b.predict(X_all)
    else:
        adjust_A = adjust_B = 0.0

    df["adj_A"] = df["baseline_A"] + adjust_A
    df["adj_B"] = df["baseline_B"] + adjust_B
    df.loc[reported, "adj_A"] = df.loc[reported, "candA_votes_reported"]
    df.loc[reported, "adj_B"] = df.loc[reported, "candB_votes_reported"]

    df[["precinct_id", "adj_A", "adj_B"]].to_csv("runoff_updated_predictions.csv", index=False)

    total_A = df["adj_A"].sum()
    total_B = df["adj_B"].sum()
    print(f"Updated totals: candidate A {total_A:.0f} votes, candidate B {total_B:.0f} votes")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python live_update_model.py partial_results.csv")
    else:
        update_predictions(sys.argv[1])
