from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import json
import itertools
from pathlib import Path
from datetime import datetime

from config import Config
from blip2_extractor import BLIP2FeatureExtractor
from model import FakeNewsClassifier
from train import create_dataloaders, Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GridSearchCV:
    """Grid Search for hyperparameter optimization"""
    
    def __init__(self, base_config: Config, param_grid: dict, save_dir: str = "grid_search_results"):
        self.base_config = base_config
        self.param_grid = param_grid
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def _generate_configs(self):
        """Generate all combinations of hyperparameters"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def _apply_params(self, config: Config, params: dict) -> Config:
        """Apply parameter combination to config"""
        import copy
        config = copy.deepcopy(config)
        
        # map parameter names to config attributes
        param_mapping = {
            'learning_rate': ('training', 'learning_rate'),
            'batch_size': ('training', 'batch_size'),
            'dropout_rate': ('classifier', 'dropout_rate'),
            'hidden_dims': ('classifier', 'hidden_dims'),
            'weight_decay': ('training', 'weight_decay'),
            'pooling_strategy': ('model', 'pooling_strategy'),
            'use_focal_loss': ('training', 'use_focal_loss'),
            'num_epochs': ('training', 'num_epochs'),
        }
        
        for param_name, param_value in params.items():
            if param_name in param_mapping:
                section, attr = param_mapping[param_name]
                setattr(getattr(config, section), attr, param_value)
        
        return config
    
    def search(self, blip2_extractor: BLIP2FeatureExtractor, train_loader, val_loader, class_weights):
        """Run grid search"""
        print("\n" + "="*70)
        print("STARTING GRID SEARCH")
        print("="*70)
        
        total_combinations = sum(1 for _ in self._generate_configs())
        print(f"Total combinations to test: {total_combinations}\n")
        
        for idx, params in enumerate(self._generate_configs(), 1):
            print(f"\n{'='*70}")
            print(f"Configuration {idx}/{total_combinations}")
            print(f"{'='*70}")
            print("Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            # apply parameters to config
            config = self._apply_params(self.base_config, params)
            
            # extract special parameters
            pooling_strategy = params.get('pooling_strategy', 'max')
            use_focal_loss = params.get('use_focal_loss', False)
            
            try:
                # build model with current config
                model = FakeNewsClassifier(blip2_extractor, config)
                
                # train
                trainer = Trainer(
                    model, 
                    train_loader, 
                    val_loader, 
                    config,
                    class_weights,
                    use_focal_loss=use_focal_loss,
                    pooling_strategy=pooling_strategy
                )
                
                train_losses, val_losses, val_accs = trainer.train()
                
                # record results
                result = {
                    'params': params,
                    'best_val_acc': max(val_accs),
                    'best_val_loss': min(val_losses),
                    'final_val_acc': val_accs[-1],
                    'final_train_acc': 100.0 * np.mean([1 for _ in train_losses]),  # Placeholder
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accs,
                    'converged_epoch': val_accs.index(max(val_accs)) + 1,
                }
                
                self.results.append(result)
                
                print(f"\nBest Val Accuracy: {result['best_val_acc']:.2f}%")
                print(f"Best Val Loss: {result['best_val_loss']:.4f}")
                print(f"Converged at epoch: {result['converged_epoch']}")
                
            except Exception as e:
                print(f"\nConfiguration failed with error: {e}")
                self.results.append({
                    'params': params,
                    'error': str(e),
                    'best_val_acc': 0.0,
                })
            
            # save intermediate results
            self._save_results()
        
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        
        return self._get_best_params()
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"grid_search_results_{timestamp}.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n[SAVE] Results saved to {save_path}")
    
    def _get_best_params(self):
        """Get best parameter combination"""
        if not self.results:
            return None
        
        # sort by best validation accuracy
        valid_results = [r for r in self.results if 'error' not in r]
        if not valid_results:
            print("No valid results found!")
            return None
        
        best_result = max(valid_results, key=lambda x: x['best_val_acc'])
        
        print("\n" + "="*70)
        print("BEST CONFIGURATION")
        print("="*70)
        print(f"Best Val Accuracy: {best_result['best_val_acc']:.2f}%")
        print(f"Best Val Loss: {best_result['best_val_loss']:.4f}")
        print(f"Converged at epoch: {best_result['converged_epoch']}")
        print("\nBest Parameters:")
        for k, v in best_result['params'].items():
            print(f"  {k}: {v}")
        
        return best_result['params']


def main(args):
    print("\n" + "="*70)
    print("HYPERPARAMETER GRID SEARCH - FAKE NEWS DETECTION")
    print("="*70 + "\n")
    
    # base config
    config = Config()

    # use smaller grid search dataset
    config.data.train_csv = "train_grid_search.csv"

    if args.classification_type:
        config.data.classification_type = args.classification_type
        config.data.__post_init__()
        config.classifier.num_classes = config.data.num_classes
    
    if args.force_cpu:
        config.model.force_cpu = True
        config.model.device = "cpu"
        config.model.dtype = "float32"
    
    config.experiment_name = f"grid_search_{config.data.classification_type}"
    
    # set seed
    set_seed(config.random_seed)
    
    # define parameter grid
    if args.quick_search:
        # quick search with fewer combinations (for testing)
        param_grid = {
            'learning_rate': [1e-4, 5e-5],
            'batch_size': [8, 16],
            'dropout_rate': [0.3, 0.5],
            'pooling_strategy': ['max', 'mean'],
            'use_focal_loss': [False, True],
        }
    else:
        # comprehensive search
        param_grid = {
            'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            'batch_size': [4, 8, 16, 32],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'hidden_dims': [
                [1024, 512],
                [1024, 512, 256],
                [512, 256],
            ],
            'weight_decay': [1e-4, 1e-3, 1e-2],
            'pooling_strategy': ['max', 'mean', 'attention'],
            'use_focal_loss': [False, True],
        }
    
    print("Parameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    print(f"\nTotal combinations: {total_combos}")
    print(f"Estimated time (10 epochs each, ~2 min/epoch): {total_combos * 20} minutes")
    
    if not args.auto_confirm:
        response = input("\nProceed with grid search? (y/n): ")
        if response.lower() != 'y':
            print("Grid search cancelled.")
            return
    
    # initialize BLIP-2
    print("\n[1/3] Initializing BLIP-2...")
    blip2_extractor = BLIP2FeatureExtractor(config.model)
    
    # load data
    print("\n[2/3] Loading data...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        config, 
        blip2_extractor.processor
    )
    
    # run grid search
    print("\n[3/3] Running grid search...")
    grid_search = GridSearchCV(config, param_grid, save_dir=args.save_dir)
    best_params = grid_search.search(blip2_extractor, train_loader, val_loader, class_weights)
    
    # save best params separately
    if best_params:
        best_params_path = Path(args.save_dir) / "best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n[SAVE] Best parameters saved to {best_params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid Search for Hyperparameter Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--classification-type", 
        type=str, 
        default="6_way", 
        choices=["2_way", "3_way", "6_way"],
        help="Classification type"
    )
    parser.add_argument(
        "--quick-search", 
        action="store_true",
        help="Run quick search with fewer combinations (for testing)"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="grid_search_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true",
        help="Force CPU usage"
    )
    parser.add_argument(
        "--auto-confirm", 
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    main(args)