from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFile
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import DataConfig, Config
from model import FakeNewsClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FakeNewsDataset(Dataset):
    """Dataset that loads PIL images and raw text for BLIP-2"""

    def __init__(self, df: pd.DataFrame, data_config: DataConfig, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.config = data_config
        self.data_root = Path(self.config.data_dir)
        self.image_dir = image_dir  # e.g., "train_images" or "dev_images"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load image using id column: Data/{image_dir}/{id}.jpg
        image_id = str(row[self.config.id_column])
        image_path = self.data_root / self.image_dir / f"{image_id}.jpg"
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # fallback to a blank white image
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")

        # text & label
        text = str(row[self.config.text_column])
        label = int(row[self.config.label_column])

        return {"image": image, "text": text, "label": label}


def create_dataloaders(config: Config, processor) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders using CSVs"""
    train_df = pd.read_csv(config.data.data_dir / config.data.train_csv)
    val_df = pd.read_csv(config.data.data_dir / config.data.val_csv)

    # split val into val/test
    val_df, test_df = train_test_split(
        val_df,
        test_size=config.data.val_split_ratio,
        random_state=config.data.random_seed,
        stratify=val_df[config.data.label_column]
    )
    print(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # datasets with appropriate image directories
    train_dataset = FakeNewsDataset(train_df, config.data, config.data.train_image_dir)
    val_dataset = FakeNewsDataset(val_df, config.data, config.data.val_image_dir)
    test_dataset = FakeNewsDataset(test_df, config.data, config.data.val_image_dir)

    # collate -> lists for BLIP-2, tensor for labels
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {"images": images, "texts": texts, "labels": labels}

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader


class Trainer:
    """Training Manager"""

    def __init__(self, model: FakeNewsClassifier, train_loader: DataLoader, val_loader: DataLoader, config: Config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.model.device

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.training.num_epochs) \
            if config.training.use_scheduler else None

        # Automatic Mixed Precision for stable float16 training
        self.use_amp = (config.model.dtype == "float16" and config.model.device == "cuda")
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("[TRAINER] Using Automatic Mixed Precision (AMP)")

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

        config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Use AMP if enabled
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(images, texts)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images, texts)
                loss = self.criterion(logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0*correct/total:.2f}%"})

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["images"]
                texts = batch["texts"]
                labels = batch["labels"].to(self.device)

                # Use AMP if enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images, texts)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images, texts)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        return total_loss / len(self.val_loader), 100.0 * correct / total

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if self.scheduler:
                self.scheduler.step()

            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            if self.config.training.use_early_stopping and self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

        print("\n" + "="*70 + "\nTRAINING COMPLETE\n" + "="*70)
        return self.train_losses, self.val_losses, self.val_accuracies

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        if is_best:
            path = self.config.training.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            print(f"[SAVE] Checkpoint saved to {path}")
