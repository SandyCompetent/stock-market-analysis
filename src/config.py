"""Configuration and hyperparameter management for Stock Market Analysis."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List


class TargetMode(str, Enum):
    """Target prediction modes."""
    PRICE = "Close"
    PERCENTAGE = "Pct_Change"


@dataclass
class ModelConfig:
    """Hyperparameters and architecture configuration for forecasting models."""
    sequence_length: int = 60
    test_size: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    early_stopping_patience: int = 8


@dataclass
class PipelineConfig:
    """Global configuration for the data, sentiment, and modeling pipelines."""
    # Target stock symbol
    symbol: str = "NVDA"
    target_mode: TargetMode = TargetMode.PRICE

    # Data paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    sample_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "sample")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "output")

    # Feature definitions
    baseline_features: List[str] = field(default_factory=lambda: [
        "Close",
        "SMA_50",
        "RSI",
        "MACD_line",
        "BB_width",
        "OBV",
        "Volume",
    ])

    enhanced_features: List[str] = field(default_factory=lambda: [
        "Close",
        "SMA_50",
        "RSI",
        "MACD_line",
        "BB_width",
        "OBV",
        "Volume",
        "Avg_Sentiment",
        "Sentiment_Ratio",
        "News_Count",
    ])

    model_config: ModelConfig = field(default_factory=ModelConfig)

    def get_target_column(self) -> str:
        """Returns the target column name according to current target_mode."""
        return self.target_mode.value

    def get_features(self, enhanced: bool = False) -> List[str]:
        """Returns feature columns depending on whether sentiment is included."""
        base = list(self.enhanced_features if enhanced else self.baseline_features)
        target = self.get_target_column()
        if target not in base:
            base.append(target)
        return base


# Default singleton instance for convenience
default_config = PipelineConfig()
