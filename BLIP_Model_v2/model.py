from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from blip2_extractor import BLIP2FeatureExtractor


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard-to-classify examples
    
    Args:
        alpha: Class weights (same as CrossEntropyLoss weight parameter)
        gamma: Focusing parameter (default 2.0, higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # compute standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # compute pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # compute focal loss (gives more weight to hard examples)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FakeNewsClassifier(nn.Module):
    """
    Enhanced Fake News Classifier with:
    - Deeper architecture
    - BatchNormalization for stable training
    - Skip connections (residual connections) for gradient flow
    - Dropout for regularization
    - Better initialization
    
    Architecture:
        BLIP-2 Features [frozen/partially trainable]
        |
        [image_embeds | text_embeds | similarity_score | itm_score]
        |
        FC1 (input_dim → hidden[0]) + BatchNorm + ReLU + Dropout
        |
        FC2 (hidden[0] → hidden[1]) + BatchNorm + ReLU + Dropout
        |
        ... (more layers if specified)
        |
        Skip Connection (input_dim → hidden[-1])
        |
        Output (hidden[-1] → num_classes)
    """

    def __init__(self, blip2_extractor: BLIP2FeatureExtractor, config: Config):
        super().__init__()
        self.blip2 = blip2_extractor
        self.config = config

        # compute input dimension from BLIP-2 features
        input_dim = 0
        if config.model.extract_image_embeds:
            input_dim += config.model.image_embed_dim  # 256
        if config.model.extract_text_embeds:
            input_dim += config.model.text_embed_dim   # 256
        if config.model.extract_similarity_score:
            input_dim += 1  # scalar similarity score
        if config.model.extract_itm_score:
            input_dim += 1  # scalar ITM score
        
        print(f"[MODEL] Input feature dimension: {input_dim}")

        # get configuration
        hidden_dims = config.classifier.hidden_dims
        dropout = config.classifier.dropout_rate
        num_classes = config.classifier.num_classes

        # first projection layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout)

        # build intermediate layers with BatchNorm
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)

        # skip connection: project input directly to last hidden dimension
        self.skip = nn.Linear(input_dim, hidden_dims[-1])
        
        # output layer
        self.output = nn.Linear(hidden_dims[-1], num_classes)

        # initialize weights for better convergence
        self._initialize_weights()
        
        # print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] Total parameters: {total_params:,}")
        print(f"[MODEL] Trainable parameters: {trainable_params:,}")
        print(f"[MODEL] Architecture: {input_dim} → {' → '.join(map(str, hidden_dims))} → {num_classes}")

    def _initialize_weights(self):
        """
        Kaiming initialization for ReLU networks
        Better than default initialization for deep networks
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # he initialization for layers followed by ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, texts, pooling_strategy="max"):
        """
        Forward pass through the network
        
        Args:
            images: List[PIL.Image] - Batch of images
            texts: List[str] - Batch of text strings
            pooling_strategy: str - 'max', 'mean', or 'attention'
                - 'max': Keep strongest signals (best for detecting artifacts/manipulation)
                - 'mean': Average all tokens (original approach)
                - 'attention': Weighted average based on feature magnitude
        
        Returns:
            logits: Tensor [batch_size, num_classes] - Raw logits (no softmax)
        """
        # extract features from BLIP-2 with specified pooling
        feats = self.blip2.extract_features(images, texts, pooling_strategy=pooling_strategy)
        
        # concatenate all enabled features
        parts = []
        if self.config.model.extract_image_embeds:
            parts.append(feats["image_embeds"])  # [batch, 256]
        if self.config.model.extract_text_embeds:
            parts.append(feats["text_embeds"])   # [batch, 256]
        if self.config.model.extract_similarity_score:
            parts.append(feats["similarity_score"].unsqueeze(1))  # [batch, 1]
        if self.config.model.extract_itm_score:
            parts.append(feats["itm_score"].unsqueeze(1))  # [batch, 1]
        
        # concatenate: [batch, input_dim]
        x = torch.cat(parts, dim=1)
        
        # save input for skip connection
        identity = self.skip(x)  # [batch, hidden[-1]]
        
        # first layer with activation and dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # pass through intermediate layers
        for layer in self.layers:
            x = layer(x)  # each layer has Linear + BN + ReLU + Dropout
        
        # add skip connection (residual)
        x = x + identity  # [batch, hidden[-1]]
        
        # final output layer (no activation - raw logits)
        logits = self.output(x)  # [batch, num_classes]
        
        return logits