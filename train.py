import argparse
from pathlib import Path

import torch

from runoff_model import load_multi_election_data, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train runoff prediction model")
    parser.add_argument("--demo", required=True, nargs="+", help="Paths to demographics CSVs")
    parser.add_argument("--first", required=True, nargs="+", help="Paths to first round results CSVs")
    parser.add_argument("--second", required=True, nargs="+", help="Paths to second round results CSVs")
    parser.add_argument("--simion-col", required=True, help="Column name of Simion votes in second round")
    parser.add_argument("--dan-col", required=True, help="Column name of Dan votes in second round")
    parser.add_argument("--output", default="model.pt", help="Where to save trained model")
    args = parser.parse_args()

    id_cols = ["Judet", "UAT", "Siruta", "Nr sectie"]
    demo_paths = [Path(p) for p in args.demo]
    first_paths = [Path(p) for p in args.first]
    second_paths = [Path(p) for p in args.second]
    data = load_multi_election_data(
        demo_paths,
        first_paths,
        second_paths,
        id_cols,
        args.simion_col,
        args.dan_col,
    )
    model = train_model(data)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
