from __future__ import annotations

import torch
from transformers import Blip2ForImageTextRetrieval, AutoProcessor
from PIL import Image
from typing import Dict, List, Union
from config import ModelConfig


class BLIP2FeatureExtractor:
    """Extract multimodal features using BLIP-2 with improved pooling strategies"""

    def __init__(self, model_config: ModelConfig):
        """Initialize BLIP-2 model"""
        self.config = model_config
        self.device = model_config.device

        # dtype selection
        if model_config.dtype == "float16" and self.device == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        print(f"[BLIP-2] Loading model: {model_config.model_name}")
        print(f"[BLIP-2] Device: {self.device}, Dtype: {self.torch_dtype}")

        # load model and processor
        try:
            self.model = Blip2ForImageTextRetrieval.from_pretrained(
                model_config.model_name,
                torch_dtype=self.torch_dtype
            )
            self.processor = AutoProcessor.from_pretrained(model_config.model_name)
        except Exception as e:
            print(f"[BLIP-2] Error with torch_dtype, retrying without dtype: {e}")
            self.model = Blip2ForImageTextRetrieval.from_pretrained(model_config.model_name)
            self.processor = AutoProcessor.from_pretrained(model_config.model_name)

        self.model.to(self.device)
        self.model.eval()

        # selective freezing strategy
        self._setup_freezing_strategy()

        print("[BLIP-2] Model loaded with selective freezing")

    def _setup_freezing_strategy(self):
        """
        Freeze vision encoder but unfreeze Q-Former for task-specific adaptation
        """
        for name, param in self.model.named_parameters():
            # freeze the vision encoder (ViT)
            if "vision_model" in name:
                param.requires_grad = False
            # Freeze the text encoder
            elif "text_model" in name:
                param.requires_grad = False
            #unfreeze Q-Former (how image and text are connected for the specific task)
            elif "qformer" in name:
                param.requires_grad = True
            else:
                #  freeze other components
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[BLIP-2] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def unfreeze_vision_encoder_partially(self, num_layers: int = 2):
        """
        Optionally unfreeze last N layers of vision encoder for fine-grained adaptation
        Use this if you have enough GPU memory
        """
        if hasattr(self.model, 'vision_model'):
            vision_layers = list(self.model.vision_model.encoder.layers)
            for layer in vision_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"[BLIP-2] Unfroze last {num_layers} layers of vision encoder")

    def extract_features(
        self,
        images: Union[List[Image.Image], Image.Image],
        texts: Union[List[str], str],
        pooling_strategy: str = "max"  # NEW: 'max', 'mean', or 'attention'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features with improved pooling strategies
        
        Args:
            pooling_strategy: 
                - 'max': Keep strongest signals (best for artifacts/manipulation)
                - 'mean': Average all tokens (original approach)
                - 'attention': Weighted average based on importance
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]

        # add news-specific prompting to help model focus
        texts = self._add_task_prompts(texts)

        # tokenize/process
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # move to device and cast if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.torch_dtype == torch.float16:
            for k, v in list(inputs.items()):
                if v.dtype == torch.float32:
                    inputs[k] = v.half()

        with torch.no_grad():
            # ITC features
            itc_outputs = self.model(**inputs, use_image_text_matching_head=False)
            image_embeds = itc_outputs.image_embeds
            text_embeds = itc_outputs.text_embeds
            sim_matrix = itc_outputs.logits_per_image
            similarity_scores = torch.diagonal(sim_matrix, dim1=-2, dim2=-1).contiguous()

            # ITM (probability of match)
            if self.config.extract_itm_score:
                itm_outputs = self.model(**inputs, use_image_text_matching_head=True)
                itm_logits = itm_outputs.logits_per_image
                itm_probs = torch.nn.functional.softmax(itm_logits, dim=1)
                itm_scores = itm_probs[:, 1]
            else:
                itm_scores = None

        # smart pooling strategy
        def pool_features(t: torch.Tensor, strategy: str) -> torch.Tensor:
            if t is None:
                return None
            
            # if already 2D, return as is
            if t.dim() == 2:
                return t
            
            # for 3D tensors [batch, seq_len, dim]
            if t.dim() == 3:
                if strategy == "max":
                    # keep strongest signals
                    t, _ = t.max(dim=1)
                elif strategy == "mean":
                    # average all tokens
                    t = t.mean(dim=1)
                elif strategy == "attention":
                    # compute attention weights based on feature magnitude
                    weights = torch.softmax(t.norm(dim=-1), dim=-1).unsqueeze(-1)
                    t = (t * weights).sum(dim=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {strategy}")
            
            # ensure 1D scores stay as [batch]
            if t.dim() == 1:
                return t
            
            return t
        
        # apply pooling
        if self.config.extract_image_embeds and image_embeds is not None:
            image_embeds = pool_features(image_embeds, pooling_strategy)
        
        if self.config.extract_text_embeds and text_embeds is not None:
            text_embeds = pool_features(text_embeds, pooling_strategy)
        
        # ensure scores are 1D
        if similarity_scores is not None and similarity_scores.dim() != 1:
            similarity_scores = similarity_scores.view(similarity_scores.size(0))
        
        if itm_scores is not None and itm_scores.dim() != 1:
            itm_scores = itm_scores.view(itm_scores.size(0))
        
        features = {}
        if self.config.extract_image_embeds:
            features['image_embeds'] = image_embeds
        if self.config.extract_text_embeds:
            features['text_embeds'] = text_embeds
        if self.config.extract_similarity_score:
            features['similarity_score'] = similarity_scores
        if self.config.extract_itm_score and itm_scores is not None:
            features['itm_score'] = itm_scores
        
        return features

    def _add_task_prompts(self, texts: List[str]) -> List[str]:
        """
        Add context to help BLIP-2 focus on news analysis
        This helps with classes like Satire where the image is real but context is wrong
        """
        prompted_texts = []
        for text in texts:
            # add news-specific framing
            prompted_text = f"News headline: {text}"
            prompted_texts.append(prompted_text)
        return prompted_texts
