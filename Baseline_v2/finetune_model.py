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

from baseline_model import MultiModalDataset
from sklearn.model_selection import train_test_split
from transformers import BertModel
from torchvision import models


# Enhanced Model with Pooling Strategy Support

class EnhancedMultiModalClassifier(nn.Module):
    """Enhanced model: BERT + ResNet-50 with configurable pooling strategies."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 512, dropout: float = 0.3, pooling_strategy: str = 'mean'):
        super(EnhancedMultiModalClassifier, self).__init__()
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy

        # text encoder: BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dim = 768 

        # image encoder: ResNet-50
        from torchvision.models import ResNet50_Weights
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # remove the final classification layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_dim = 2048  

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
        # Get BERT outputs -- this gives us token-level representations
        bert_stuff = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Three pooling strategies
        if self.pooling_strategy == 'cls':
            text_vec = bert_stuff.last_hidden_state[:, 0, :]

        elif self.pooling_strategy == 'mean':
            all_tokens = bert_stuff.last_hidden_state
            attn_exp = attention_mask.unsqueeze(-1).expand(all_tokens.size()).float()
            summed = torch.sum(all_tokens * attn_exp, dim=1)
            summed_mask = torch.clamp(attn_exp.sum(1), min=1e-9) 
            text_vec = summed / summed_mask

        elif self.pooling_strategy == 'max':
            all_tokens = bert_stuff.last_hidden_state
            attn_exp = attention_mask.unsqueeze(-1).expand(all_tokens.size()).float()
            all_tokens = all_tokens.masked_fill(attn_exp == 0, -1e9)
            text_vec = torch.max(all_tokens, dim=1)[0] 
        else:
            raise ValueError(f"Pooling strategy '{self.pooling_strategy}' is not recognized")

        img_vec = self.resnet(image)
        img_vec = img_vec.view(img_vec.size(0), -1)  # flatten to 2D

        merged = torch.cat([text_vec, img_vec], dim=1)

        outputs = self.classifier(merged)

        return outputs

# Loss Functions

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights. If None, all classes weighted equally.
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

    """
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Label smoothing factor
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

        # create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * log_probs, dim=1)
        return loss.mean()


def get_loss_function(loss_type, class_weights=None, focal_gamma=2.0, label_smoothing=0.1):
    """
    Returns a loss module based on loss_type.
    Supported: 'ce', 'weighted_ce', 'focal', 'label_smoothing'
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32)
        ) if class_weights is not None else nn.CrossEntropyLoss()

    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)

    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)

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
    """
    print(f"Training with config: {config}")

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
        print(f" Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Evaluate
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f" Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.2f}%")

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

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

    print(f"GRID SEARCH: Testing {len(combinations)} configurations")

    all_results = []
    best_overall_acc = 0.0
    best_config = None

    for i, config in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        results, model = train_single_config(config, train_loader, dev_loader, device, num_classes, class_weights)
        all_results.append(results)

        if results['best_dev_acc'] > best_overall_acc:
            best_overall_acc = results['best_dev_acc']
            best_config = config
            best_model = model

            # save best model
            torch.save({
                'config': config,
                'model_state_dict': model.state_dict(),
                'dev_acc': best_overall_acc,
            }, save_path / 'best_model_grid_search.pth')

    # Save all results
    results_file = save_path / f'grid_search_results.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"Best Dev Accuracy: {best_overall_acc:.2f}%")

    return all_results, best_model, best_config


# Main Function

def main(args):
    # Configuration
    LABEL_TYPE = args.classification_type
    NUM_CLASSES = {'2_way': 2, '3_way': 3, '6_way': 6}[LABEL_TYPE]
    MAX_LENGTH = 128
    COMPUTE_CLASS_WEIGHTS = args.use_class_weights

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'Data'

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Label type: {LABEL_TYPE} ({NUM_CLASSES} classes)")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and split dev into val/test
    train_df = pd.read_csv(data_dir / 'train_sampled_with_images.csv')
    dev_df = pd.read_csv(data_dir / 'dev_sampled_with_images.csv')

    label_column = f'{LABEL_TYPE}_label'
    val_df, test_df = train_test_split(
        dev_df,
        test_size=0.2,
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

    # Configure loss functions based on classification type
    if LABEL_TYPE == '2_way':
        # For 2-way: weighted CrossEntropy and label smoothing
        loss_types = ['weighted_ce', 'label_smoothing']
    else:
        # For 3-way and 6-way: focal loss and weighted CrossEntropy
        loss_types = ['weighted_ce', 'focal']

    param_grid = {
        'loss_type': loss_types,
        'learning_rate': [1e-4, 5e-4, 5e-5],
        'batch_size': [8, 16],
        'dropout': [0.3, 0.5],
        'pooling_strategy': ['max', 'mean'],
        'hidden_dim': [512], 
        'num_epochs': [10],  
        'focal_gamma': [2.0],  
        'label_smoothing': [0.1]
    }

    for key, values in param_grid.items():
        print(f"{key}: {values}")

    total_configs = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal configurations to test: {total_configs}")

    # Store original batch_size values
    batch_sizes = param_grid.pop('batch_size')

    all_results = []
    best_overall_acc = 0.0
    best_config = None
    save_path = Path('grid_search_results')
    save_path.mkdir(parents=True, exist_ok=True)

    # Iterate over each batch size and run grid search
    for batch_size in batch_sizes:
        # Create validation dataset
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
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        dev_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Add batch_size to param_grid temporarily
        current_param_grid = {**param_grid, 'batch_size': [batch_size]}

        # Run grid search for this batch size
        results, _, _ = grid_search(
            current_param_grid,
            train_loader,
            dev_loader,
            device,
            NUM_CLASSES,
            class_weights=class_weights,
            save_dir='grid_search_results'
        )

        all_results.extend(results)

        # Track overall best config across all batch sizes
        for result in results:
            if result.get('best_dev_acc', 0) > best_overall_acc:
                best_overall_acc = result['best_dev_acc']
                best_config = result['config']

    # Final summary
    final_results_file = save_path / f'final_grid_search_results.json'

    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nOverall Best Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"Overall Best Dev Accuracy: {best_overall_acc:.2f}%")


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
