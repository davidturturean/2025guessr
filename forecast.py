import argparse
from pathlib import Path

import torch

from runoff_model import (
    RunoffPredictor,
    dataframe_to_tensor,
    ElectionForecaster,
    load_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live forecast")
    parser.add_argument(
        "--features",
        action="append",
        required=True,
        help="Path to first round results CSV. May be passed multiple times.",
    )
    parser.add_argument("--demo", help="Optional demographics CSV")
    parser.add_argument("--model", default="model.pt", help="Trained model weights")
    parser.add_argument("--id-cols", nargs="*", default=["Judet", "UAT", "Siruta", "Nr sectie"], help="Columns identifying a precinct")
    parser.add_argument(
        "--no-error-adjustment",
        action="store_true",
        help="Disable precinct error adjustment",
    )
    args = parser.parse_args()

    feature_paths = [Path(p) for p in args.features]
    demo_path = Path(args.demo) if args.demo else None
    features = load_features(feature_paths, demo_path, args.id_cols)
    X = dataframe_to_tensor(features.drop(columns=args.id_cols))
    model = RunoffPredictor(X.shape[1])
    state = torch.load(args.model, map_location=torch.device("cpu"))
    model.load_state_dict(state)

    forecaster = ElectionForecaster(
        model,
        features,
        id_cols=args.id_cols,
        adjust_errors=not args.no_error_adjustment,
    )
    (totals_sim, totals_dan), (std_sim, std_dan) = forecaster.forecast_totals()
    print("Initial forecast: Simion={:.0f} Dan={:.0f}".format(totals_sim, totals_dan))
    print("Std dev: Simion={:.1f} Dan={:.1f}".format(std_sim, std_dan))


if __name__ == "__main__":
    main()
