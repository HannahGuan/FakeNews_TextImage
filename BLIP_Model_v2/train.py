from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys

from config import DataConfig, Config
from model import FakeNewsClassifier, FocalLoss

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FakeNewsDataset(Dataset):
    """Dataset that loads PIL images and raw text for BLIP-2"""

    def __init__(self, df: pd.DataFrame, data_config: DataConfig):
        self.df = df.reset_index(drop=True)
        self.config = data_config
        self.data_root = Path(self.config.data_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # construct image path based on id_column and image directory
        image_id = str(row[self.config.id_column])

        # determine which split this is (train or val) based on data root
        # try to detect from config which image directory to use
        if hasattr(self.config, 'train_image_dir') and hasattr(self.config, 'val_image_dir'):
            # we'll need to check if this is train or val split
            # for now, try both directories
            possible_paths = [
                self.data_root / self.config.train_image_dir / f"{image_id}.jpg",
                self.data_root / self.config.val_image_dir / f"{image_id}.jpg",
                self.data_root / self.config.train_image_dir / f"{image_id}.png",
                self.data_root / self.config.val_image_dir / f"{image_id}.png",
            ]
        else:
            # fallback to generic images directory
            possible_paths = [
                self.data_root / "images" / f"{image_id}.jpg",
                self.data_root / "images" / f"{image_id}.png",
            ]

        # try to load from one of the possible paths
        image = None
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                try:
                    image = Image.open(path).convert("RGB")
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load existing image {path}: {e}")
                    continue

        # if no image found, print warning and create blank image
        if image is None:
            print(f"[ERROR] Image not found for ID '{image_id}'. Tried paths:")
            for path in possible_paths:
                print(f"  - {path} (exists: {path.exists()})")
            print(f"[WARNING] Using blank white image as fallback")
            image = Image.new("RGB", (224, 224), color="white")

        # get text and label
        text = str(row[self.config.text_column])
        label = int(row[self.config.label_column])

        return {"image": image, "text": text, "label": label}


def compute_class_weights(train_df: pd.DataFrame, label_column: str, device: str) -> torch.Tensor:
    """
    Compute balanced class weights to handle class imbalance
    
    This is CRITICAL for fake news detection where:
    - "Real" news is often abundant
    - Rare types like "Imposter", "Manipulated" are few
    
    Without class weights, the model will ignore minority classes!
    
    Args:
        train_df: Training dataframe
        label_column: Name of label column
        device: Device to put weights on
    
    Returns:
        class_weights: Tensor of weights for each class
    """
    all_labels = train_df[label_column].values
    unique_classes = np.unique(all_labels)
    
    # compute balanced weights: inversely proportional to class frequency
    # formula: n_samples / (n_classes * n_samples_per_class)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_labels
    )
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # print class distribution for debugging
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION & WEIGHTS")
    print("="*70)
    for cls in unique_classes:
        count = np.sum(all_labels == cls)
        percentage = 100 * count / len(all_labels)
        weight = class_weights[cls].item()
        print(f"Class {cls}: {count:5d} samples ({percentage:5.2f}%) | Weight: {weight:.4f}")
    print("="*70 + "\n")
    
    return class_weights


def create_dataloaders(config: Config, processor) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train/val/test dataloaders with class weights
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    train_df = pd.read_csv(config.data.data_dir / config.data.train_csv)
    val_df = pd.read_csv(config.data.data_dir / config.data.val_csv)

    # compute class weights before any splitting
    class_weights = compute_class_weights(train_df, config.data.label_column, config.model.device)

    # split validation set into val/test
    # Check if stratification is possible (each class needs at least 2 samples)
    label_counts = val_df[config.data.label_column].value_counts()
    min_class_count = label_counts.min()
    
    # Try stratified split first
    try:
        val_df, test_df = train_test_split(
            val_df,
            test_size=config.data.val_split_ratio,
            random_state=config.data.random_seed,
            stratify=val_df[config.data.label_column]  # Maintain class distribution
        )
        print(f"[DATA] Using stratified split for val/test")
    except ValueError as e:
        # Stratified split failed - use non-stratified split
        print(f"[WARNING] Stratified split failed: {e}")
        print(f"[WARNING] Class distribution in validation set: {label_counts.to_dict()}")
        print(f"[WARNING] Minimum class count: {min_class_count}")
        print(f"[WARNING] Falling back to non-stratified split (class distribution may be imbalanced)")
        print(f"[WARNING] Consider collecting more data or using fewer classes for better evaluation.")
        val_df, test_df = train_test_split(
            val_df,
            test_size=config.data.val_split_ratio,
            random_state=config.data.random_seed,
            stratify=None  # No stratification
        )
    print(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # create datasets
    train_dataset = FakeNewsDataset(train_df, config.data)
    val_dataset = FakeNewsDataset(val_df, config.data)
    test_dataset = FakeNewsDataset(test_df, config.data)

    # collate function: convert batch to format BLIP-2 expects
    def collate_fn(batch):
        images = [item["image"] for item in batch]  # List of PIL Images
        texts = [item["text"] for item in batch]    # List of strings
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {"images": images, "texts": texts, "labels": labels}

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, class_weights


class Trainer:
    """
    Enhanced Training Manager with:
    - Weighted loss / Focal loss for class imbalance
    - Differential learning rates (lower for pretrained BLIP-2)
    - AdamW optimizer with weight decay
    - Adaptive learning rate scheduling
    - Better early stopping
    """

    def __init__(
        self, 
        model: FakeNewsClassifier, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        config: Config,
        class_weights: torch.Tensor,
        use_focal_loss: bool = False,
        pooling_strategy: str = "max"
    ):
        """
        Args:
            model: The FakeNewsClassifier model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration object
            class_weights: Computed class weights for handling imbalance
            use_focal_loss: If True, use Focal Loss instead of CrossEntropy
            pooling_strategy: 'max', 'mean', or 'attention' for feature pooling
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.model.device
        self.pooling_strategy = pooling_strategy

        self.model.to(self.device)

        # Loss Function Selection
        if use_focal_loss:
            print("[TRAINING] Using Focal Loss (gamma=2.0) for class imbalance")
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            print("[TRAINING] Using Weighted Cross-Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # separate parameters into BLIP-2 and classifier
        blip_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'blip2' in name:
                    blip_params.append(param)
                else:
                    classifier_params.append(param)
        
        print(f"[OPTIMIZER] Classifier parameters: {sum(p.numel() for p in classifier_params):,}")
        print(f"[OPTIMIZER] BLIP-2 parameters: {sum(p.numel() for p in blip_params):,}")
        
        # use AdamW
        optimizer_params = [
            {'params': classifier_params, 'lr': config.training.learning_rate}
        ]
        
        # if BLIP-2 has trainable parameters, use 10x smaller learning rate
        if len(blip_params) > 0:
            optimizer_params.append({
                'params': blip_params, 
                'lr': config.training.learning_rate * 0.1
            })
            print(f"[OPTIMIZER] Using differential LR:")
            print(f"  - Classifier: {config.training.learning_rate}")
            print(f"  - BLIP-2: {config.training.learning_rate * 0.1}")
        else:
            print(f"[OPTIMIZER] Single LR: {config.training.learning_rate}")
        
        self.optimizer = AdamW(
            optimizer_params,
            weight_decay=config.training.weight_decay
        )

        if config.training.use_scheduler:
            # reduce LR when validation loss plateaus
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min',           # minimize validation loss
                factor=0.5,           # reduce LR by half
                patience=3,           # wait 3 epochs before reducing
                min_lr=1e-7          # don't go below this
            )
            print("[SCHEDULER] Using ReduceLROnPlateau (factor=0.5, patience=3)")
        else:
            self.scheduler = None

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

        # create checkpoint directory
        config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Disable tqdm if output is redirected to avoid excessive logging
        use_tqdm = sys.stdout.isatty()

        if use_tqdm:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1} [Train]",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            iterator = pbar
        else:
            iterator = self.train_loader
            num_batches = len(self.train_loader)
            print(f"Epoch {epoch+1} [Train]: Starting {num_batches} batches...")

        for batch_idx, batch in enumerate(iterator):
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(self.device)

            # forward pass
            logits = self.model(images, texts, pooling_strategy=self.pooling_strategy)
            loss = self.criterion(logits, labels)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

            self.optimizer.step()

            # calculate metrics
            total_loss += loss.item()
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            # update progress
            if use_tqdm:
                # update tqdm progress bar every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                    current_acc = 100.0 * correct / total
                    current_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "acc": f"{current_acc:.2f}%"
                    })
            else:
                # print progress every 200 batches to reduce logging
                if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(self.train_loader):
                    current_acc = 100.0 * correct / total
                    current_loss = total_loss / (batch_idx + 1)
                    print(f"  [{batch_idx+1}/{len(self.train_loader)}] loss: {current_loss:.4f} | acc: {current_acc:.2f}%")

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Disable tqdm if output is redirected to avoid excessive logging
        use_tqdm = sys.stdout.isatty()

        with torch.no_grad():
            if use_tqdm:
                pbar = tqdm(
                    self.val_loader,
                    desc="Validating",
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )
                iterator = pbar
            else:
                iterator = self.val_loader
                print(f"Validating: Starting {len(self.val_loader)} batches...")

            for batch_idx, batch in enumerate(iterator):
                images = batch["images"]
                texts = batch["texts"]
                labels = batch["labels"].to(self.device)

                # forward pass
                logits = self.model(images, texts, pooling_strategy=self.pooling_strategy)
                loss = self.criterion(logits, labels)

                # calculate metrics
                total_loss += loss.item()
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

                # update progress
                if use_tqdm:
                    # update tqdm progress bar every 5 batches
                    if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(self.val_loader):
                        current_acc = 100.0 * correct / total
                        current_loss = total_loss / (batch_idx + 1)
                        pbar.set_postfix({
                            "loss": f"{current_loss:.4f}",
                            "acc": f"{current_acc:.2f}%"
                        })
                else:
                    # print progress every 50 batches to reduce logging
                    if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.val_loader):
                        current_acc = 100.0 * correct / total
                        current_loss = total_loss / (batch_idx + 1)
                        print(f"  [{batch_idx+1}/{len(self.val_loader)}] loss: {current_loss:.4f} | acc: {current_acc:.2f}%")

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100.0 * correct / total
        
        return avg_loss, avg_acc

    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        for epoch in range(self.config.training.num_epochs):
            # train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current LR: {current_lr:.2e}")

            # print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            # save best model based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"[BEST] New best model! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epoch(s)")

            # early stopping
            if self.config.training.use_early_stopping and \
               self.patience_counter >= self.config.training.patience:
                print(f"\n[STOP] Early stopping triggered after {epoch+1} epochs")
                print(f"Best Val Acc: {self.best_val_acc:.2f}%")
                break

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*70 + "\n")
        
        return self.train_losses, self.val_losses, self.val_accuracies

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "config": self.config
        }
        
        if is_best:
            path = self.config.training.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            print(f"[SAVE] Best checkpoint saved to {path}")