import argparse
from pathlib import Path

import torch

from runoff_model import prepare_training_data, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train runoff prediction model")
    parser.add_argument("--demo", required=True, help="Path to demographics CSV")
    parser.add_argument("--first", required=True, help="Path to first round results CSV")
    parser.add_argument("--second", required=True, help="Path to second round results CSV")
    parser.add_argument("--simion-col", required=True, help="Column name of Simion votes in second round")
    parser.add_argument("--dan-col", required=True, help="Column name of Dan votes in second round")
    parser.add_argument("--output", default="model.pt", help="Where to save trained model")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data used for validation",
    )
    args = parser.parse_args()

    id_cols = ["Judet", "UAT", "Siruta", "Nr sectie"]
    data = prepare_training_data(
        Path(args.demo), Path(args.first), Path(args.second), id_cols, args.simion_col, args.dan_col
    )
    model = train_model(data, val_split=args.val_split)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
