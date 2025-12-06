# Fake News Detection with Text-Image Mismatch Score

**Stanford CS230 Deep Learning Final Project**
Course Website: https://cs230.stanford.edu

## Overview

The rapid spread of misinformation and fake news on online platforms poses a significant threat to public trust, social cohesion, and informed decision-making. As multimodal posts combining text and images become increasingly common, they often leverage striking visuals and emotionally charged headlines to boost engagement, making them more persuasive and challenging to detect with text-only models.

Traditional fake news detection systems focus primarily on textual analysis, overlooking the visual cues and cross-modal inconsistencies often present in misinformation. Recent studies demonstrate that combining text and image signals substantially improves detection accuracy.

This project aims to design and evaluate a neural network model that can determine whether a given multimodal post (image + text) represents fake news. Each input consists of an image paired with an associated text caption or title. We leverage the **BLIP-2 Vision-Language Model (VLM)** to explicitly model semantic consistency between text and image, enabling deep cross-modal understanding rather than treating the two modalities separately.

---

## Dataset

**Original Data Repository:** https://github.com/entitize/Fakeddit

### Download Links

- **Train Images:** [Google Drive Link](https://drive.google.com/file/d/1EaSgAheEGBHWQsysokz-Pw6-k9NQUaHr/view?usp=share_link)
  Mapped via the 'id' column in the corresponding CSV file

- **Dev Images:** [Google Drive Link](https://drive.google.com/file/d/1FSvK1CPIt6CqUXR-g4ndp7TbPUazXwpR/view?usp=share_link)

- **Baseline Model Checkpoints:** (to large to upload to Github) [Google Drive Link](https://drive.google.com/file/d/1QTVnnN_0c6h0-CWGrijmKHsKOrh3XrH-/view?usp=drive_link)

### Label Structure

- **[2-way]** 0: True | 1: False
- **[3-way]** 0: True | 1: Fake with true text | 2: Fake with false text
- **[6-way]** 0: True | 1: Satire/Parody | 2: Misleading Content | 3: Imposter Content | 4: False Connection | 5: Manipulated Content

---

## Repository Structure

### [Data/](Data/)
Contains the preprocessed dataset used for training and evaluation:
- Training and validation CSV files with image IDs and labels
- Split datasets for 2-way, 3-way, and 6-way classification tasks
- `train_images/` and `dev_images/` folders containing the actual image files
- Grid search subset for hyperparameter tuning

### [Data_processing/](Data_processing/)
Scripts and notebooks for data preprocessing and preparation:
- `data_processing.ipynb`: Jupyter notebook for data exploration and preprocessing pipeline
- `dataProcess_util.py`: Utility functions for data loading, cleaning, and transformation

### [Baseline_v2/](Baseline_v2/)
Baseline model implementation using traditional multimodal approaches:
- `baseline_model.py`: Core baseline model architecture
- `finetune_model.py`: Fine-tuning scripts for the baseline model
- `evaluate_model.py`: Evaluation scripts with comprehensive metrics
- `checkpoints/`: Saved model checkpoints from training
- `evaluation_results/`: Model performance results and metrics
- `grid_search_results/`: Hyperparameter tuning results
- `logs/`: Training logs and TensorBoard files
- Detailed documentation in `README_BASELINE.md`

### [BLIP_Model_v2/](BLIP_Model_v2/)
BLIP-2 Vision-Language Model implementation:
- `model.py`: BLIP-2 based multimodal architecture
- `blip2_extractor.py`: Feature extraction using BLIP-2 embeddings
- `train.py`: Training pipeline for the BLIP-2 model
- `evaluate.py`: Comprehensive evaluation with confusion matrices and metrics
- `grid_search.py`: Automated hyperparameter search
- `config.py`: Centralized configuration management
- `checkpoints/`: Saved BLIP-2 model checkpoints
- `logs/`: Training logs and experiment tracking
- Batch scripts for running experiments (`train_all_full_data.bat`, `run_grid_search_all.bat`)

### Root Directory Files
- `analysis.ipynb`: Data analysis and result visualization notebook
- `confusion_matrix_*_comparison.png`: Confusion matrices comparing model performance
- `model_comparison_3metrics.png`: Comparative visualization of model metrics
- `training_validation_accuracy.png`: Training curves and validation performance 
