from __future__ import annotations
import torch
import torch.nn as nn
from config import Config
from blip2_extractor import BLIP2FeatureExtractor


class FakeNewsClassifier(nn.Module):
    """
    Architecture:
        BLIP-2 Feature Extractor (frozen)
        ↓
        [image_embeds | text_embeds | similarity | itm_score]
        ↓
        MLP (trainable)
        ↓
        Binary Classification (fake/real)
    """

    def __init__(self, blip2_extractor: BLIP2FeatureExtractor, config: Config):
        super().__init__()
        self.blip2 = blip2_extractor
        self.config = config

        # Compute fused feature dimension
        input_dim = 0
        if config.model.extract_image_embeds:
            input_dim += config.model.image_embed_dim
        if config.model.extract_text_embeds:
            input_dim += config.model.text_embed_dim
        if config.model.extract_similarity_score:
            input_dim += 1
        if config.model.extract_itm_score:
            input_dim += 1

        # Build MLP
        act = nn.ReLU if config.classifier.activation.lower() == "relu" else nn.GELU
        layers = []
        prev = input_dim
        for h in config.classifier.hidden_dims:
            layers.extend([nn.Linear(prev, h), act(), nn.Dropout(config.classifier.dropdown_rate if hasattr(config.classifier, 'dropdown_rate') else config.classifier.dropout_rate)])
            prev = h
        layers.append(nn.Linear(prev, config.classifier.num_classes))
        self.classifier = nn.Sequential(*layers)


    def forward(self, images, texts):
        """
        images: List[PIL.Image]
        texts:  List[str]
        """
        feats = self.blip2.extract_features(images, texts)
        parts = []
        if self.config.model.extract_image_embeds:
            parts.append(feats["image_embeds"])
        if self.config.model.extract_text_embeds:
            parts.append(feats["text_embeds"])
        if self.config.model.extract_similarity_score:
            parts.append(feats["similarity_score"].unsqueeze(1))
        if self.config.model.extract_itm_score:
            parts.append(feats["itm_score"].unsqueeze(1))
        x = torch.cat(parts, dim=1)
        logits = self.classifier(x)
        return logits
