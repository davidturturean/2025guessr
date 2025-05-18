import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class RunoffPredictor(nn.Module):
    """Simple feed-forward network predicting runoff turnout and candidate share."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.turnout = nn.Linear(64, 1)
        self.share = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        turnout = F.relu(self.turnout(x))
        share = torch.sigmoid(self.share(x))
        simion = turnout * share
        dan = turnout * (1 - share)
        return simion, dan


@dataclass
class ElectionData:
    """Container for features and targets used in training or prediction."""

    features: pd.DataFrame
    target_simion: Optional[pd.Series] = None
    target_dan: Optional[pd.Series] = None


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with UTF-8 encoding and handle potential BOM."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return pd.read_csv(f)


def prepare_training_data(
    demo_path: Path,
    first_round_path: Path,
    second_round_path: Path,
    id_cols: List[str],
    simion_col: str,
    dan_col: str,
) -> ElectionData:
    """Load and merge data for model training."""
    demo = load_csv(demo_path)
    first_round = load_csv(first_round_path)
    second_round = load_csv(second_round_path)

    df = first_round.merge(second_round[id_cols + [simion_col, dan_col]], on=id_cols, suffixes=("", "_y"))
    df = df.merge(demo, on=id_cols, how="left")

    # Drop possible duplicate columns from merge
    df = df.loc[:, ~df.columns.duplicated()]

    # Targets
    target_simion = df[simion_col]
    target_dan = df[dan_col]

    # Remove target columns from features
    features = df.drop(columns=[simion_col, dan_col])

    return ElectionData(features=features, target_simion=target_simion, target_dan=target_dan)


def load_multi_election_data(
    demo_paths: List[Path],
    first_round_paths: List[Path],
    second_round_paths: List[Path],
    id_cols: List[str],
    simion_col: str,
    dan_col: str,
) -> ElectionData:
    """Load and concatenate training data from several elections."""
    datasets = []
    for d_path, f_path, s_path in zip(demo_paths, first_round_paths, second_round_paths):
        datasets.append(
            prepare_training_data(d_path, f_path, s_path, id_cols, simion_col, dan_col)
        )
    features = pd.concat([d.features for d in datasets], ignore_index=True)
    target_simion = pd.concat([d.target_simion for d in datasets], ignore_index=True)
    target_dan = pd.concat([d.target_dan for d in datasets], ignore_index=True)
    return ElectionData(features=features, target_simion=target_simion, target_dan=target_dan)


def dataframe_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Convert numeric dataframe to a float32 tensor."""
    numeric = df.select_dtypes(include=[np.number]).fillna(0)
    return torch.tensor(numeric.values, dtype=torch.float32)


def train_model(data: ElectionData, epochs: int = 100, batch_size: int = 64) -> RunoffPredictor:
    X = dataframe_to_tensor(data.features)
    y_sim = torch.tensor(data.target_simion.values, dtype=torch.float32).unsqueeze(1)
    y_dan = torch.tensor(data.target_dan.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y_sim, y_dan)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RunoffPredictor(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    patience = 10
    wait = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y_sim, batch_y_dan in loader:
            opt.zero_grad()
            pred_sim, pred_dan = model(batch_X)
            loss = F.mse_loss(pred_sim, batch_y_sim) + F.mse_loss(pred_dan, batch_y_dan)
            loss.backward()
            opt.step()
        # simple early stopping based on loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def predict_with_uncertainty(model: RunoffPredictor, X: torch.Tensor, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Predict vote totals with Monte Carlo dropout to estimate uncertainty."""
    model.train()  # enable dropout during inference
    sims = []
    dans = []
    for _ in range(n_samples):
        sim, dan = model(X)
        sims.append(sim.detach().numpy())
        dans.append(dan.detach().numpy())
    model.eval()
    sim_arr = np.stack(sims, axis=0)
    dan_arr = np.stack(dans, axis=0)
    sim_mean = sim_arr.mean(axis=0)
    dan_mean = dan_arr.mean(axis=0)
    sim_std = sim_arr.std(axis=0)
    dan_std = dan_arr.std(axis=0)
    return (sim_mean.squeeze(), dan_mean.squeeze()), (sim_std.squeeze(), dan_std.squeeze())


class ElectionForecaster:
    """Manage live precinct updates and national forecast."""

    def __init__(self, model: RunoffPredictor, features: pd.DataFrame, id_cols: List[str]):
        self.model = model
        self.id_cols = id_cols
        self.features = features.set_index(id_cols)
        self.unreported = self.features.index.tolist()
        self.reported_simion = 0.0
        self.reported_dan = 0.0
        self.feature_tensor = dataframe_to_tensor(self.features.reset_index(drop=True))
        self.pred_sim, self.pred_dan = model(self.feature_tensor)
        self.pred_sim = self.pred_sim.detach().numpy().squeeze()
        self.pred_dan = self.pred_dan.detach().numpy().squeeze()

    def update_precinct(self, precinct_id: Tuple, simion_votes: float, dan_votes: float):
        if precinct_id not in self.unreported:
            return
        idx = self.unreported.index(precinct_id)
        self.reported_simion += simion_votes
        self.reported_dan += dan_votes
        self.unreported.pop(idx)
        self.pred_sim = np.delete(self.pred_sim, idx)
        self.pred_dan = np.delete(self.pred_dan, idx)
        self.feature_tensor = torch.cat([self.feature_tensor[:idx], self.feature_tensor[idx+1:]], dim=0)

    def forecast_totals(self, n_samples: int = 100) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if len(self.unreported) == 0:
            return (self.reported_simion, self.reported_dan), (0.0, 0.0)
        (pred_sim_mean, pred_dan_mean), (pred_sim_std, pred_dan_std) = predict_with_uncertainty(
            self.model, self.feature_tensor, n_samples=n_samples
        )
        total_sim_mean = self.reported_simion + pred_sim_mean.sum()
        total_dan_mean = self.reported_dan + pred_dan_mean.sum()
        total_sim_std = np.sqrt((pred_sim_std ** 2).sum())
        total_dan_std = np.sqrt((pred_dan_std ** 2).sum())
        return (total_sim_mean, total_dan_mean), (total_sim_std, total_dan_std)
