# config.py
from __future__ import annotations

import torch
from dataclasses import dataclass, field, asdict
from typing import List
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """BLIP-2 Model Configuration"""
    # Model selection
    model_name: str = "Salesforce/blip2-itm-vit-g"

    # Device configuration
    device: str = "cpu"   # use cpu for now, if change to gpu, make it gpu, true, float16
    force_cpu: bool = False
    dtype: str = "float32"

    # Feature extraction toggles
    extract_image_embeds: bool = True
    extract_text_embeds: bool = True
    extract_similarity_score: bool = True
    extract_itm_score: bool = True

    # Dimensions (BLIP-2 ITC head exposes 256-d proj by default)
    image_embed_dim: int = 256
    text_embed_dim: int = 256
    max_text_length: int = 128

    def __post_init__(self):
        if self.force_cpu:
            self.device = "cpu"
            self.dtype = "float32"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = "float16" if self.device == "cuda" else "float32"
        print(f"[CONFIG] Device: {self.device}, Dtype: {self.dtype}")


@dataclass
class DataConfig:
    """Data Configuration"""
    # Dataset paths
    data_dir: Path = Path("./data/fakeddit")
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"

    # Column names
    image_column: str = "image_path"
    text_column: str = "clean_title"
    label_column: str = "2_way_label"

    # Data split
    val_split_ratio: float = 0.2  # Split val into val/test
    random_seed: int = 42

    # Processing
    image_size: int = 224
    num_classes: int = 2


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Training parameters, could be changed later
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    # Optimizer
    optimizer: str = "adamw"

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # Regularization
    max_grad_norm: float = 1.0
    dropout_rate: float = 0.3

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 3

    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints")
    save_best_only: bool = True


@dataclass
class ClassifierConfig:
    """Classifier Head Configuration"""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    activation: str = "relu"  # "relu" or "gelu"
    dropout_rate: float = 0.3
    num_classes: int = 2


@dataclass
class Config:
    """Complete Configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Experiment settings
    experiment_name: str = "blip2_fake_news"
    log_dir: Path = Path("./logs")
    random_seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        if self.model.device == "cpu":
            self.num_workers = 0
            print("[CONFIG] CPU mode: num_workers set to 0")


def save_config(config: Config, path: Path):
    """Save configuration to YAML"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    print(f"[CONFIG] Saved to {path}")


def load_config(path: Path) -> Config:
    """Load configuration from YAML"""
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)