from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from config import Config
from blip2_extractor import BLIP2FeatureExtractor
from model import FakeNewsClassifier
from train import create_dataloaders, Trainer
from evaluate import Evaluator, plot_training_history, save_predictions_json


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Random seed set to {seed}")


def print_config_summary(config: Config, args):
    """Print configuration summary"""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Classification Type: {config.data.classification_type} ({config.data.num_classes} classes)")
    print(f"Device: {config.model.device}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Pooling Strategy: {args.pooling}")
    print(f"Use Focal Loss: {args.use_focal_loss}")
    print(f"Use Scheduler: {config.training.use_scheduler}")
    print(f"Early Stopping: {config.training.use_early_stopping} (patience={config.training.patience})")
    print(f"Data Dir: {config.data.data_dir}")
    print(f"Checkpoint Dir: {config.training.checkpoint_dir}")
    print("="*70 + "\n")


def main(args):
    print("\n" + "="*70)
    print("BLIP-2 FAKE NEWS DETECTION - IMPROVED VERSION")
    print("="*70 + "\n")

    config = Config()

    if args.classification_type:
        config.data.classification_type = args.classification_type
        config.data.__post_init__()
        config.classifier.num_classes = config.data.num_classes
    
    if args.data_dir:
        config.data.data_dir = Path(args.data_dir)
    if args.train_csv:
        config.data.train_csv = args.train_csv
    if args.val_csv:
        config.data.val_csv = args.val_csv
        
    if args.force_cpu:
        config.model.force_cpu = True
        config.model.device = "cpu"
        config.model.dtype = "float32"
        config.num_workers = 0
    
    # training hyperparameters
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay
    
    # dropout
    if args.dropout:
        config.classifier.dropout_rate = args.dropout
        config.training.dropout_rate = args.dropout
    
    # early stopping
    if args.no_early_stopping:
        config.training.use_early_stopping = False
    if args.patience:
        config.training.patience = args.patience
    
    # experiment naming and directories
    config.experiment_name = f"blip2_fake_news_{config.data.classification_type}"
    if args.use_focal_loss:
        config.experiment_name += "_focal"
    if args.pooling != "max":
        config.experiment_name += f"_{args.pooling}"
    
    config.log_dir = config.log_dir / config.data.classification_type
    config.training.checkpoint_dir = config.training.checkpoint_dir / config.data.classification_type
    
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_config_summary(config, args)

    set_seed(config.random_seed)

    print("[1/4] nIitializing BLIP-2...")
    blip2_extractor = BLIP2FeatureExtractor(config.model)

    print("\n[2/4] Loading data...")
    try:
        train_loader, val_loader, test_loader, class_weights = create_dataloaders(
            config, 
            blip2_extractor.processor
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find data files: {e}")
        print(f"[ERROR] Please check your data paths:")
        print(f"  - Data directory: {config.data.data_dir}")
        print(f"  - Train CSV: {config.data.train_csv}")
        print(f"  - Val CSV: {config.data.val_csv}")
        return
    except KeyError as e:
        print(f"\n[ERROR] Column not found in CSV: {e}")
        print(f"[ERROR] Expected label column: {config.data.label_column}")
        print(f"[ERROR] Please check your CSV files have the correct columns")
        return

    print("\n[3/4] Building model...")
    model = FakeNewsClassifier(blip2_extractor, config)

    if args.mode == "train":
        print("\n[4/4] Training...")
        print(f"Training with:")
        print(f"  - Pooling: {args.pooling}")
        print(f"  - Loss: {'Focal Loss' if args.use_focal_loss else 'Weighted CrossEntropy'}")
        print(f"  - Optimizer: AdamW (weight_decay={config.training.weight_decay})")
        print(f"  - Scheduler: {'ReduceLROnPlateau' if config.training.use_scheduler else 'None'}")
        
        trainer = Trainer(
            model, 
            train_loader, 
            val_loader, 
            config,
            class_weights,
            use_focal_loss=args.use_focal_loss,
            pooling_strategy=args.pooling
        )
        
        train_losses, val_losses, val_accuracies = trainer.train()
        
        # plot training history
        plot_training_history(train_losses, val_losses, val_accuracies, config)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
        print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
        print(f"Model saved to: {config.training.checkpoint_dir / 'best_model.pt'}")
        print("="*70)
        
        # optionally evaluate on test set after training
        if args.eval_after_train:
            print("\n[BONUS] Evaluating on test set...")
            
            # load best checkpoint
            ckpt_path = config.training.checkpoint_dir / "best_model.pt"
            checkpoint = torch.load(ckpt_path, map_location=config.model.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            evaluator = Evaluator(model, config)
            metrics, labels, predictions, probabilities, logits = evaluator.evaluate(test_loader)
            
            # save predictions
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

    elif args.mode == "eval":
        print("\n[4/4] Evaluating...")

        # load best checkpoint if available
        ckpt_path = config.training.checkpoint_dir / "best_model.pt"
        if ckpt_path.exists():
            print(f"[LOAD] Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=config.model.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("[LOAD] Checkpoint loaded successfully")
            
            # Print checkpoint info
            if 'best_val_acc' in checkpoint:
                print(f"[INFO] Checkpoint best val accuracy: {checkpoint['best_val_acc']:.2f}%")
            if 'epoch' in checkpoint:
                print(f"[INFO] Checkpoint from epoch: {checkpoint['epoch']}")
        else:
            print(f"[WARNING] No checkpoint found at {ckpt_path}")
            print("[WARNING] Using untrained model (results will be random)")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Evaluation cancelled.")
                return

        # evaluate on test set
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
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Results saved to: {config.log_dir}")
        print("="*70)

    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BLIP-2 Fake News Detection - Improved Version",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "eval"], 
        help="Mode: train or eval"
    )
    
    parser.add_argument(
        "--classification-type", 
        type=str, 
        default=None, 
        choices=["2_way", "3_way", "6_way"], 
        help="Classification type: 2_way, 3_way, or 6_way"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ./data/fakeddit)"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Training CSV filename (default: train.csv)"
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="Validation CSV filename (default: val.csv)"
    )
    
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force CPU usage even if GPU is available"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None, 
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None, 
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate"
    )
    
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "mean", "attention"],
        help="Pooling strategy for BLIP-2 features"
    )
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use Focal Loss instead of Weighted CrossEntropy"
    )
    
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 3)"
    )
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help="Automatically evaluate on test set after training"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training/Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()