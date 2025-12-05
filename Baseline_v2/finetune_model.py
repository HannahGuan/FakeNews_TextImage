"""
Fine-tuning Script with Multiple Loss Functions and Hyperparameter Grid Search

Author: Xinru Pan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from itertools import product
from datetime import datetime
import ssl
import certifi

from baseline_model import MultiModalDataset
from sklearn.model_selection import train_test_split
from transformers import BertModel
from torchvision import models

ssl_context = ssl.create_default_context(cafile=certifi.where())


# Enhanced Model with Pooling Strategy Support

class EnhancedMultiModalClassifier(nn.Module):
    """Enhanced model: BERT + ResNet-50 with configurable pooling strategies."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 512, dropout: float = 0.3, pooling_strategy: str = 'mean'):
        """
        Args:
            num_classes: Number of output classes (2, 3, or 6)
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
            pooling_strategy: 'mean', 'max', or 'cls' for text pooling
        """
        super(EnhancedMultiModalClassifier, self).__init__()
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy

        # text encoder: BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dim = 768  # BERT base hidden size

        # image encoder: ResNet-50
        try:
            from torchvision.models import ResNet50_Weights
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
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
            nn.Linear(hidden_dim, num_classes)
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
        # extract text features with pooling strategy
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling_strategy == 'cls':
            # Use CLS token (first token)
            text_features = bert_output.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == 'mean':
            # Mean pooling over sequence
            token_embeddings = bert_output.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            text_features = sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence
            token_embeddings = bert_output.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings = token_embeddings.masked_fill(attention_mask_expanded == 0, -1e9)
            text_features = torch.max(token_embeddings, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # extract image features (global average pooling)
        image_features = self.resnet(image)
        image_features = image_features.view(image_features.size(0), -1)

        # concatenate features
        combined_features = torch.cat([text_features, image_features], dim=1)

        # classification
        logits = self.classifier(combined_features)

        return logits


# Loss Functions

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (list or tensor). If None, all classes weighted equally.
            gamma: Focusing parameter. Higher gamma = more focus on hard examples.
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    Prevents overconfident predictions by smoothing hard labels.

    Hard label: [0, 0, 1, 0]
    Soft label (smoothing=0.1): [0.025, 0.025, 0.925, 0.025]
    """
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 is common)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)

        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * log_probs, dim=1)
        return loss.mean()


def get_loss_function(loss_type, num_classes=None, class_weights=None, **kwargs):
    """
    Factory function to get loss function by name.

    Args:
        loss_type: 'ce', 'weighted_ce', 'focal', or 'label_smoothing'
        num_classes: Number of classes (unused, kept for API compatibility)
        class_weights: Optional class weights for weighted CE or Focal Loss
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Loss function (nn.Module)
    """
    _ = num_classes  # Unused, kept for API compatibility
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce':
        if class_weights is None:
            print("[WARNING] Weighted CE requested but no class weights provided. Using standard CE.")
            return nn.CrossEntropyLoss()
        weights = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == 'focal':
        gamma = kwargs.get('focal_gamma', 2.0)
        alpha = class_weights if class_weights is not None else None
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('label_smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(train_df, label_type, num_classes):
    """
    Compute inverse frequency class weights for handling class imbalance.

    Args:
        train_df: Training dataframe
        label_type: Label type (e.g., '2_way', '3_way', '6_way')
        num_classes: Number of classes

    Returns:
        List of class weights
    """
    # Count samples per class
    label_column = f'{label_type}_label'
    class_counts = np.zeros(num_classes)

    for label in train_df[label_column]:
        class_counts[int(label)] += 1

    # Compute inverse frequency weights
    total_samples = len(train_df)
    class_weights = total_samples / (num_classes * class_counts)

    print(f"\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {int(count)} samples (weight: {class_weights[i]:.3f})")

    return class_weights.tolist()


# Training Functions

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)

        # compute loss
        loss = criterion(logits, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # forward pass
            logits = model(input_ids, attention_mask, images)

            # compute loss
            loss = criterion(logits, labels)

            # calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_single_config(config, train_loader, dev_loader, device, num_classes, class_weights=None):
    """
    Train model with a single hyperparameter configuration.

    Args:
        config: Dictionary with hyperparameters
        train_loader: Training data loader
        dev_loader: Validation data loader
        device: Device to train on
        num_classes: Number of classes
        class_weights: Optional class weights for loss function

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Training with config: {config}")
    print(f"{'='*80}")

    # Initialize model with pooling strategy
    model = EnhancedMultiModalClassifier(
        num_classes=num_classes,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        pooling_strategy=config.get('pooling_strategy', 'mean')
    )
    model = model.to(device)

    # Get loss function
    criterion = get_loss_function(
        config['loss_type'],
        num_classes,
        class_weights=class_weights,
        focal_gamma=config.get('focal_gamma', 2.0),
        label_smoothing=config.get('label_smoothing', 0.1)
    )
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Learning rate scheduler (optional)
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

    # Training loop
    best_dev_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }

    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Evaluate
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.2f}%")

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(dev_acc)

        # Track best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch

    results = {
        'config': config,
        'best_dev_acc': best_dev_acc,
        'best_epoch': best_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_dev_acc': history['dev_acc'][-1],
        'history': history
    }

    return results, model


# Grid Search

def grid_search(
    param_grid,
    train_loader,
    dev_loader,
    device,
    num_classes,
    class_weights=None,
    save_dir='grid_search_results'
):
    """
    Perform grid search over hyperparameters.

    Args:
        param_grid: Dictionary of parameter lists
        train_loader: Training data loader
        dev_loader: Validation data loader
        device: Device to train on
        num_classes: Number of classes
        class_weights: Optional class weights
        save_dir: Directory to save results

    Returns:
        List of results for all configurations
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f"\n{'='*80}")
    print(f"GRID SEARCH: Testing {len(combinations)} configurations")
    print(f"{'='*80}")

    all_results = []
    best_overall_acc = 0.0
    best_config = None

    for i, config in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        try:
            results, model = train_single_config(
                config, train_loader, dev_loader, device, num_classes, class_weights
            )
            all_results.append(results)

            # Track best model
            if results['best_dev_acc'] > best_overall_acc:
                best_overall_acc = results['best_dev_acc']
                best_config = config

                # Save best model
                torch.save({
                    'config': config,
                    'model_state_dict': model.state_dict(),
                    'dev_acc': best_overall_acc,
                }, save_path / 'best_model_grid_search.pth')
                print(f"\n*** NEW BEST MODEL: {best_overall_acc:.2f}% ***")

        except Exception as e:
            print(f"[ERROR] Configuration failed: {e}")
            results = {
                'config': config,
                'error': str(e),
                'best_dev_acc': 0.0
            }
            all_results.append(results)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_path / f'grid_search_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"Best Dev Accuracy: {best_overall_acc:.2f}%")

    # Generate summary report
    generate_summary_report(all_results, save_path / f'summary_{timestamp}.txt')

    return all_results, best_model, best_config


def generate_summary_report(results, save_path):
    """Generate a summary report of grid search results."""
    # Sort by dev accuracy
    sorted_results = sorted(results, key=lambda x: x.get('best_dev_acc', 0), reverse=True)

    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRID SEARCH SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total configurations tested: {len(results)}\n")
        f.write(f"Successful runs: {sum(1 for r in results if 'error' not in r)}\n")
        f.write(f"Failed runs: {sum(1 for r in results if 'error' in r)}\n\n")

        f.write("TOP 10 CONFIGURATIONS:\n")
        f.write("-"*80 + "\n")

        for i, result in enumerate(sorted_results[:10], 1):
            if 'error' in result:
                continue

            f.write(f"\n#{i} - Dev Acc: {result['best_dev_acc']:.2f}%\n")
            f.write(f"  Config:\n")
            for key, value in result['config'].items():
                f.write(f"    {key}: {value}\n")
            f.write(f"  Best Epoch: {result['best_epoch']}\n")
            f.write(f"  Final Train Acc: {result['final_train_acc']:.2f}%\n")
            f.write(f"  Final Dev Acc: {result['final_dev_acc']:.2f}%\n")

    print(f"\nSummary report saved to: {save_path}")


# Main Function

def main(args):
    # Configuration
    LABEL_TYPE = args.classification_type
    NUM_CLASSES = {'2_way': 2, '3_way': 3, '6_way': 6}[LABEL_TYPE]
    MAX_LENGTH = 128
    COMPUTE_CLASS_WEIGHTS = args.use_class_weights

    # Get the base data directory (parent of Baseline directory)
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'Data'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")
    print(f"Label type: {LABEL_TYPE} ({NUM_CLASSES} classes)")
    print(f"Class weights: {'Enabled' if COMPUTE_CLASS_WEIGHTS else 'Disabled'}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and split dev into val/test (same as baseline model)
    print("\nLoading and splitting datasets...")
    train_df = pd.read_csv(data_dir / 'train_sampled_with_images.csv')
    dev_df = pd.read_csv(data_dir / 'dev_sampled_with_images.csv')

    label_column = f'{LABEL_TYPE}_label'
    val_df, test_df = train_test_split(
        dev_df,
        test_size=0.2,  # 20% for test, 80% for validation
        random_state=42,
        stratify=dev_df[label_column]
    )

    # Save split CSVs for reproducibility
    val_csv_path = data_dir / f'val_split_{LABEL_TYPE}_finetune.csv'
    test_csv_path = data_dir / f'test_split_{LABEL_TYPE}_finetune.csv'
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    print(f"[DATA] Val split saved to {val_csv_path}")
    print(f"[DATA] Test split saved to {test_csv_path}")

    # Create datasets with new API (csv_path, image_dir)
    train_dataset = MultiModalDataset(
        str(data_dir / 'train_sampled_with_images.csv'),
        str(data_dir / 'train_images'),
        tokenizer,
        image_transform,
        max_length=MAX_LENGTH,
        label_type=LABEL_TYPE
    )

    # Compute class weights if enabled
    class_weights = None
    if COMPUTE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_df, LABEL_TYPE, NUM_CLASSES)

    # Define Parameter Grid 

    # Configure loss functions based on classification type
    if LABEL_TYPE == '2_way':
        # For 2-way: weighted CrossEntropy and label smoothing
        loss_types = ['weighted_ce', 'label_smoothing']
        print("\n[INFO] 2-way classification: Using Weighted CrossEntropy and Label Smoothing")
    else:
        # For 3-way and 6-way: focal loss and weighted CrossEntropy
        loss_types = ['weighted_ce', 'focal']
        print(f"\n[INFO] {LABEL_TYPE} classification: Using Focal Loss and Weighted CrossEntropy")

    param_grid = {
        'loss_type': loss_types,
        'learning_rate': [1e-4, 5e-4, 5e-5],
        'batch_size': [8, 16],
        'dropout': [0.3, 0.5],
        'pooling_strategy': ['max', 'mean'],
        'hidden_dim': [512],  # MLP hidden dimension
        'num_epochs': [10],  # Number of training epochs per configuration
        'focal_gamma': [2.0],  # For focal loss
        'label_smoothing': [0.1]  # For label smoothing
    }

    print("\n" + "="*80)
    print("PARAMETER GRID")
    print("="*80)
    for key, values in param_grid.items():
        print(f"{key}: {values}")

    total_configs = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal configurations to test: {total_configs}")

    # Warning for large grid searches
    if total_configs > 50:
        print("\n[WARNING] Large number of configurations!")
        print("Consider reducing the parameter grid or using random search.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return

    # Run Grid Search

    all_results = []
    best_overall_acc = 0.0
    best_config = None
    save_path = Path('grid_search_results')
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f"\n{'='*80}")
    print(f"GRID SEARCH: Testing {len(combinations)} configurations")
    print(f"{'='*80}")

    for i, config in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing configuration:")

        # Create validation dataset with current config
        val_dataset = MultiModalDataset(
            str(val_csv_path),
            str(data_dir / 'dev_images'),
            tokenizer,
            image_transform,
            max_length=MAX_LENGTH,
            label_type=LABEL_TYPE
        )

        # Create dataloaders with current batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        dev_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        try:
            results, model = train_single_config(
                config, train_loader, dev_loader, device, NUM_CLASSES, class_weights
            )
            all_results.append(results)

            # Track best model
            if results['best_dev_acc'] > best_overall_acc:
                best_overall_acc = results['best_dev_acc']
                best_config = config

                # Save best model
                torch.save({
                    'config': config,
                    'model_state_dict': model.state_dict(),
                    'dev_acc': best_overall_acc,
                }, save_path / 'best_model_grid_search.pth')
                print(f"\n*** NEW BEST MODEL: {best_overall_acc:.2f}% ***")

        except Exception as e:
            print(f"[ERROR] Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            results = {
                'config': config,
                'error': str(e),
                'best_dev_acc': 0.0
            }
            all_results.append(results)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = save_path / f'grid_search_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"Best Dev Accuracy: {best_overall_acc:.2f}%")

    # Generate summary report
    generate_summary_report(all_results, save_path / f'summary_{timestamp}.txt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning Multimodal Model with Grid Search",
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
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs for each configuration'
    )
    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        default=True,
        help='Use class weights to handle imbalance'
    )
    parser.add_argument(
        '--no-class-weights',
        dest='use_class_weights',
        action='store_false',
        help='Disable class weights'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )

    args = parser.parse_args()
    main(args)
