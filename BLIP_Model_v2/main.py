from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from config import Config
from blip2_extractor import BLIP2FeatureExtractor
from model import FakeNewsClassifier
from train import create_dataloaders, Trainer
from evaluate import Evaluator, plot_training_history, save_predictions_json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Random seed set to {seed}")


def main(args):
    print("\n" + "="*70)
    print("BLIP-2 FAKE NEWS DETECTION")
    print("="*70 + "\n")

    # build config
    config = Config()

    # overrides
    if args.classification_type:
        config.data.classification_type = args.classification_type
        config.data.__post_init__()
        config.classifier.num_classes = config.data.num_classes
        
    if args.force_cpu:
        config.model.force_cpu = True
        config.model.device = "cpu"
        config.model.dtype = "float32"
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay

    config.experiment_name = f"blip2_fake_news_{config.data.classification_type}"
    config.log_dir = config.log_dir / config.data.classification_type
    config.training.checkpoint_dir = config.training.checkpoint_dir / config.data.classification_type

    # seed
    set_seed(config.random_seed)

    # init BLIP-2
    print("\n[1/4] Initializing BLIP-2...")
    blip2_extractor = BLIP2FeatureExtractor(config.model)

    # data - FIXED: Now expecting 4 return values
    print("\n[2/4] Loading data...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        config, 
        blip2_extractor.processor
    )

    # model
    print("\n[3/4] Building model...")
    model = FakeNewsClassifier(blip2_extractor, config)

    if args.mode == "train":
        print("\n[4/4] Training...")
        trainer = Trainer(
            model, 
            train_loader, 
            val_loader, 
            config,
            class_weights,
            use_focal_loss=False,
            pooling_strategy="max"
        )
        train_losses, val_losses, val_accuracies = trainer.train()
        plot_training_history(train_losses, val_losses, val_accuracies, config)

    elif args.mode == "eval":
        print("\n[4/4] Evaluating...")

        # load best checkpoint if available
        ckpt_path = config.training.checkpoint_dir / "best_model.pt"
        if ckpt_path.exists():
            print(f"[LOAD] Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=config.model.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("[LOAD] Checkpoint loaded successfully")
        else:
            print(f"[WARNING] No checkpoint found at {ckpt_path}")
            print("[WARNING] Using untrained model")

        evaluator = Evaluator(model, config)
        metrics, labels, predictions, probabilities, logits = evaluator.evaluate(test_loader)

        # save predictions to JSON with all details
        pred_path = config.log_dir / "predictions.json"
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
    print("Done")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BLIP-2 Fake News Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--classification-type", type=str, default=None, choices=["2_way", "3_way", "6_way"], 
                       help="Classification type: 2_way, 3_way, or 6_way")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay (default: 0.01)")
    args = parser.parse_args()
    main(args)