import argparse
from pathlib import Path

import pandas as pd
import torch

from runoff_model import RunoffPredictor, dataframe_to_tensor, ElectionForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live forecast")
    parser.add_argument("--features", required=True, help="CSV with 2025 first round features")
    parser.add_argument("--model", default="model.pt", help="Trained model weights")
    parser.add_argument("--id-cols", nargs="*", default=["Judet", "UAT", "Siruta", "Nr sectie"], help="Columns identifying a precinct")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    X = dataframe_to_tensor(features.drop(columns=args.id_cols))
    model = RunoffPredictor(X.shape[1])
    state = torch.load(args.model, map_location=torch.device("cpu"))
    model.load_state_dict(state)

    forecaster = ElectionForecaster(model, features, id_cols=args.id_cols)
    (totals_sim, totals_dan), (std_sim, std_dan) = forecaster.forecast_totals()
    print("Initial forecast: Simion={:.0f} Dan={:.0f}".format(totals_sim, totals_dan))
    print("Std dev: Simion={:.1f} Dan={:.1f}".format(std_sim, std_dan))


if __name__ == "__main__":
    main()
