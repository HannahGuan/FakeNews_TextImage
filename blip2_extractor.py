from __future__ import annotations

import torch
from transformers import Blip2ForImageTextRetrieval, AutoProcessor
from PIL import Image
from typing import Dict, List, Union
from config import ModelConfig


class BLIP2FeatureExtractor:
    """Extract multimodal features using BLIP-2"""

    def __init__(self, model_config: ModelConfig):
        """
        Initialize BLIP-2 model
        """
        self.config = model_config
        self.device = model_config.device

        # dtype selection
        if model_config.dtype == "float16" and self.device == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        print(f"[BLIP-2] Loading model: {model_config.model_name}")
        print(f"[BLIP-2] Device: {self.device}, Dtype: {self.torch_dtype}")

        # Load model and processor
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

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        print("[BLIP-2] Model loaded and frozen successfully!")

    def extract_features(
        self,
        images: Union[List[Image.Image], Image.Image],
        texts: Union[List[str], str]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          dict containing: image_embeds [B,256], text_embeds [B,256], similarity_score [B], itm_score [B]
        """
        # Ensure lists
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize/process
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device and cast if needed
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

        def to_2d(t: torch.Tensor) -> torch.Tensor:
            if t is None:
                return None
            # If [batch, seq_len, dim] → mean-pool over seq_len
            if t.dim() == 3:
                t = t.mean(dim=1)
            # If [batch] is expected for scores, keep it; otherwise ensure 2-D
            if t.dim() == 1:
                t = t.unsqueeze(1)
            return t
        
        # embeddings should be 2-D
        if self.config.extract_image_embeds and image_embeds is not None:
            image_embeds = to_2d(image_embeds) 
        
        if self.config.extract_text_embeds and text_embeds is not None:
            text_embeds = to_2d(text_embeds)
        
        # scores should be [batch]
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