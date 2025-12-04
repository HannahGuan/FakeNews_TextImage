"""
Evaluate trained model on the FULL dev set (without splitting into val/test)
"""
import argparse
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from config import Config
from blip2_extractor import BLIP2FeatureExtractor
from model import FakeNewsClassifier
from train import FakeNewsDataset
from evaluate import Evaluator, save_predictions_json


def collate_fn(batch):
    """Collate function for dataloader"""
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"images": images, "texts": texts, "labels": labels}


def main(args):
    print("\n" + "="*70)
    print("EVALUATE ON FULL DEV SET")
    print("="*70 + "\n")

    # Load config
    config = Config()
    config.data.classification_type = args.classification_type
    config.data.__post_init__()
    config.classifier.num_classes = config.data.num_classes

    if args.force_cpu:
        config.model.force_cpu = True
        config.model.device = "cpu"
        config.model.dtype = "float32"

    print(f"Classification Type: {args.classification_type}")
    print(f"Number of Classes: {config.data.num_classes}")
    print(f"Device: {config.model.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Load full dev set WITHOUT splitting
    print("[1/4] Loading FULL dev set...")
    dev_df = pd.read_csv(config.data.data_dir / config.data.val_csv)
    print(f"Total dev samples: {len(dev_df)}")

    # Print class distribution
    print(f"\nClass distribution:")
    print(dev_df[config.data.label_column].value_counts().sort_index())
    print()

    # Create dataset
    dev_dataset = FakeNewsDataset(dev_df, config.data)

    # Create dataloader
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize BLIP-2
    print("[2/4] Initializing BLIP-2...")
    blip2_extractor = BLIP2FeatureExtractor(config.model)

    # Build model
    print("[3/4] Building model...")
    model = FakeNewsClassifier(blip2_extractor, config)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"[LOAD] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.model.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("[LOAD] Checkpoint loaded successfully")

    if 'best_val_acc' in checkpoint:
        print(f"[INFO] Checkpoint best val accuracy: {checkpoint['best_val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"[INFO] Checkpoint from epoch: {checkpoint['epoch']}")
    print()

    # Evaluate
    print("[4/4] Evaluating on FULL dev set...")
    evaluator = Evaluator(model, config)
    metrics, labels, predictions, probabilities, logits = evaluator.evaluate(dev_loader)

    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "predictions_full_dev.json"
    save_predictions_json(
        labels,
        predictions,
        probabilities,
        logits,
        pred_path,
        config.data.num_classes,
        evaluator.class_names
    )

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Total Samples: {len(dev_df)}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Results saved to: {pred_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate on full dev set without train/test split"
    )
    parser.add_argument(
        "--classification-type",
        type=str,
        required=True,
        choices=["2_way", "3_way", "6_way"],
        help="Classification type"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_full_dev",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage"
    )

    args = parser.parse_args()
    main(args)
