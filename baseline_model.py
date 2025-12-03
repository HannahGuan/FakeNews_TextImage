"""
Baseline Multimodal Model: BERT + ResNet-50 Concatenation
Binary classification combining text and image features.

Author: Xinru Pan
Time: 2025-11-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import ssl
import certifi
from typing import Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

ssl_context = ssl.create_default_context(cafile=certifi.where())


class MultiModalDataset(Dataset):
    """loading text and image pairs with labels."""

    def __init__(self, csv_path: str, image_dir: str, tokenizer, image_transform, max_length: int = 128, label_type: str = '2_way'):
        """
        Args:
            csv_path: Path to CSV file
            image_dir: Path to image directory
            tokenizer: BERT tokenizer
            image_transform: Image preprocessing transforms
            max_length: Maximum sequence length for BERT
            label_type: Type of label to use ('2_way', '3_way', or '6_way')
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.label_type = label_type

        #filter out rows with missing images or text
        self.data = self.data[self.data['hasImage'] == True].reset_index(drop=True)
        label_column = f'{label_type}_label'
        self.data = self.data.dropna(subset=['clean_title', 'id', label_column]).reset_index(drop=True)

        print(f"Loaded {len(self.data)} samples from {csv_path} with {label_type} labels")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        #process text
        title = str(row['clean_title'])
        encoded = self.tokenizer(
            title,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        #process image using id column
        image_id = str(row['id'])
        image_path = self.image_dir / f"{image_id}.jpg"
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            #create a dummy image in case of error
            image = torch.zeros(3, 224, 224)

        # get label based on label_type
        label_column = f'{self.label_type}_label'
        label = int(row[label_column])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)  # long for CrossEntropyLoss
        }


class MultiModalClassifier(nn.Module):
    """Baseline model: BERT + ResNet-50 with MLP classifier."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 512, dropout: float = 0.3):
        """
        Args:
            num_classes: Number of output classes (2, 3, or 6)
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
        """
        super(MultiModalClassifier, self).__init__()
        self.num_classes = num_classes

        # text encoder: BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dim = 768  # BERT base hidden size

        # image encoder: ResNet-50
        try:
            # try the new API first (torchvision >= 0.13)
            from torchvision.models import ResNet50_Weights
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            # old API
            resnet = models.resnet50(pretrained=True)
        # remove the final classification layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_dim = 2048  # ResNet-50 output dimension

        # freeze pretrained models initially
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

        # MLP Classifier with 2 ReLU layers
        concat_dim = self.bert_dim + self.resnet_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # output logits for each class
        )

    def forward(self, input_ids, attention_mask, image):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            image: [batch_size, 3, 224, 224]

        Returns:
            logits: [batch_size, num_classes]
        """
        # extract text features (CLS token embedding)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :]

        # extract image features (global average pooling)
        image_features = self.resnet(image)
        image_features = image_features.view(image_features.size(0), -1)

        # concatenate features
        combined_features = torch.cat([text_features, image_features], dim=1) 

        # classification
        logits = self.classifier(combined_features) 

        return logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)

        # compute loss (CrossEntropyLoss expects raw logits)
        loss = criterion(logits, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # forward pass
            logits = model(input_ids, attention_mask, images)

            # compute loss (CrossEntropyLoss expects raw logits)
            loss = criterion(logits, labels)

            # calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def plot_training_history(train_losses, train_accs, val_losses, val_accs, log_dir, label_type):
    """Plot and save training history."""
    plt.figure(figsize=(14, 5))

    # loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    # accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Accuracy", marker='o')
    plt.plot(val_accs, label="Val Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.grid(True, alpha=0.3)

    # val accuracy zoom
    plt.subplot(1, 3, 3)
    plt.plot(val_accs, label="Val Accuracy", marker='s', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy (Detail)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = log_dir / f"training_history_{label_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] Training history saved to {save_path}")
    plt.close()


def main(args):
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    HIDDEN_DIM = 512
    DROPOUT = 0.3
    MAX_LENGTH = 128
    LABEL_TYPE = args.classification_type

    # Set number of classes based on label type
    NUM_CLASSES = {'2_way': 2, '3_way': 3, '6_way': 6}[LABEL_TYPE]

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")
    print(f"Label type: {LABEL_TYPE} ({NUM_CLASSES} classes)")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, Learning rate: {LEARNING_RATE}")

    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # image transformations (ResNet-50 preprocessing)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load and split dev into val/test (same as BLIP model)
    dev_df = pd.read_csv('../dev_sampled_with_images.csv')
    label_column = f'{LABEL_TYPE}_label'
    val_df, test_df = train_test_split(
        dev_df,
        test_size=0.2,  # 20% for test, 80% for validation
        random_state=42,
        stratify=dev_df[label_column]
    )

    # save split CSVs for later use
    val_csv_path = f'../val_split_{LABEL_TYPE}.csv'
    test_csv_path = f'../test_split_{LABEL_TYPE}.csv'
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    print(f"[DATA] Val split saved to {val_csv_path}")
    print(f"[DATA] Test split saved to {test_csv_path}")

    # create datasets
    train_dataset = MultiModalDataset(
        '../train_sampled_with_images.csv',
        '../train_images',
        tokenizer,
        image_transform,
        max_length=MAX_LENGTH,
        label_type=LABEL_TYPE
    )

    val_dataset = MultiModalDataset(
        val_csv_path,
        '../dev_images',
        tokenizer,
        image_transform,
        max_length=MAX_LENGTH,
        label_type=LABEL_TYPE
    )

    test_dataset = MultiModalDataset(
        test_csv_path,
        '../dev_images',
        tokenizer,
        image_transform,
        max_length=MAX_LENGTH,
        label_type=LABEL_TYPE
    )

    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # initialize model
    model = MultiModalClassifier(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # only optimize MLP classifier parameters (BERT and ResNet are frozen)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # create directories for checkpoints and logs
    checkpoint_dir = Path(f'checkpoints/{LABEL_TYPE}')
    log_dir = Path(f'logs/{LABEL_TYPE}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # training loop
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_type': LABEL_TYPE,
            }, checkpoint_path)
            print(f"[SAVE] Checkpoint saved to {checkpoint_path}")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # evaluate on test set
    print(f"\n{'='*70}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'='*70}")

    # load best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs, log_dir, LABEL_TYPE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Baseline Multimodal Fake News Detection (BERT + ResNet-50)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--classification-type',
        type=str,
        default='2_way',
        choices=['2_way', '3_way', '6_way'],
        help='Classification type: 2_way, 3_way, or 6_way'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )

    args = parser.parse_args()
    main(args)
