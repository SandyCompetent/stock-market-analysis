"""Deep learning time-series models (LSTM and GRU) implemented in PyTorch."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..sentiment.finbert import get_optimal_device

logger = logging.getLogger(__name__)


class StockLSTM(nn.Module):
    """Multi-layer Recurrent Neural Network (LSTM or GRU) for time-series forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_gru: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gru = use_gru

        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.rnn(x)
        # Take the output of the final time step
        last_step_out = out[:, -1, :]
        prediction = self.fc(last_step_out)
        return prediction.squeeze(-1)


class LSTMTrainer:
    """Trainer and predictor wrapper for StockLSTM models with early stopping."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_gru: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_optimal_device()
        self.model = StockLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_gru=use_gru,
        ).to(self.device)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 8,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Trains the model with early stopping."""
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_loss = float("inf")
        best_weights = None
        patience_counter = 0

        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            batch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            train_epoch_loss = float(np.mean(batch_losses))
            self.history["train_loss"].append(train_epoch_loss)

            val_epoch_loss = train_epoch_loss
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_preds = self.model(vx)
                        v_loss = criterion(v_preds, vy)
                        val_losses.append(v_loss.item())
                val_epoch_loss = float(np.mean(val_losses))
                self.history["val_loss"].append(val_epoch_loss)

            scheduler.step(val_epoch_loss)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f}")

            # Early stopping check
            monitored_loss = val_epoch_loss if val_loader else train_epoch_loss
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Generates predictions for sequence array X."""
        self.model.eval()
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                out = self.model(bx)
                preds.append(out.cpu().numpy())

        return np.concatenate(preds, axis=0)
