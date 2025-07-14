import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, cast, Literal
import json
import logging
from collections import defaultdict
import random
import shutil

from .hierarchical_trading_transformer import ModelConfig
from .risk_adjusted_loss import LossConfig
from .adaptive_learning_system import AdaptiveLearningSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingDataset(torch.utils.data.Dataset):
    """Dataset for trading data with multiple timeframes"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        timeframes: Optional[List[str]] = None,
        seq_length: int = 100,
        target_length: int = 10,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing the processed data
            split: One of 'train', 'val', 'test'
            timeframes: List of timeframes to include
            seq_length: Length of input sequences
            target_length: Length of target sequences
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.timeframes = timeframes or ['5m', '15m', '30m', '1h', '4h', '1d']
        self.seq_length = seq_length
        self.target_length = target_length
        self.random_seed = random_seed
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data()
        
        # Split data
        self.indices = self._split_data(train_ratio, val_ratio, test_ratio)
        
        logger.info(f"Initialized {split} dataset with {len(self.indices)} samples")
    
    def _load_and_preprocess_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load and preprocess data for all timeframes"""
        data = {}
        
        for tf in self.timeframes:
            # Load features
            features_path = self.data_dir / 'features' / f'features_{tf}.parquet'
            if not features_path.exists():
                logger.warning(f"Features not found for {tf}, skipping...")
                continue
                
            features_df = pd.read_parquet(features_path)
            
            # Load labels
            labels_path = self.data_dir / 'labels' / f'labels_{tf}.parquet'
            if not labels_path.exists():
                logger.warning(f"Labels not found for {tf}, skipping...")
                continue
                
            labels_df = pd.read_parquet(labels_path)
            
            # Align features and labels
            common_index = features_df.index.intersection(labels_df.index)
            if len(common_index) == 0:
                logger.warning(f"No common index found for {tf}, skipping...")
                continue
            
            features_df = features_df.loc[common_index]
            labels_df = labels_df.loc[common_index]
            
            # Convert to numpy
            features = features_df.values.astype(np.float32)
            labels = labels_df[['direction', 'magnitude', 'duration', 'confidence']].values.astype(np.float32)
            
            # Store data
            data[tf] = {
                'features': features,
                'labels': labels,
                'timestamps': common_index.values,
                'returns': labels_df.get('returns', np.zeros(len(common_index))).astype(np.float32)
            }
            
            logger.info(f"Loaded {len(common_index)} samples for {tf}")
        
        if not data:
            raise ValueError("No valid data found for any timeframe")
            
        return data
    
    def _split_data(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> List[Tuple[int, int, int]]:
        """Split data into train/val/test sets"""
        # Find common time range across all timeframes
        all_timestamps = []
        for tf_data in self.data.values():
            all_timestamps.extend(tf_data['timestamps'])
        
        # Sort and get unique timestamps
        all_timestamps = sorted(list(set(all_timestamps)))
        num_samples = len(all_timestamps)
        
        # Calculate split points
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)
        
        # Create index ranges for each split
        if self.split == 'train':
            start_idx = 0
            end_idx = train_end - self.seq_length - self.target_length
        elif self.split == 'val':
            start_idx = train_end - self.seq_length  # Include some overlap
            end_idx = val_end - self.seq_length - self.target_length
        else:  # test
            start_idx = val_end - self.seq_length  # Include some overlap
            end_idx = num_samples - self.seq_length - self.target_length
        
        # Create list of (timestamp_idx, seq_start, seq_end) tuples
        indices = []
        for i in range(start_idx, end_idx):
            # Skip if any timeframe doesn't have enough data for this sequence
            valid = True
            for tf_data in self.data.values():
                timestamps = tf_data['timestamps']
                # Find the first index where timestamp >= all_timestamps[i]
                start_idx = np.searchsorted(timestamps, all_timestamps[i], side='left')
                end_idx = start_idx + self.seq_length + self.target_length
                
                if end_idx > len(timestamps):
                    valid = False
                    break
            
            if valid:
                indices.append((i, start_idx, start_idx + self.seq_length))
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        random.shuffle(indices)
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Get a sample from the dataset
        
        Returns:
            Tuple of (x, y, returns) where:
            - x: Dictionary of input sequences for each timeframe
            - y: Dictionary of target values
            - returns: Optional tensor of returns for the target period
        """
        timestamp_idx, seq_start, seq_end = self.indices[idx]
        
        x = {}
        y = {}
        returns = []
        
        for tf, tf_data in self.data.items():
            # Get sequence of features
            features = tf_data['features'][seq_start:seq_end]
            
            # Get target (next step after sequence)
            target_start = seq_end
            target_end = target_start + self.target_length
            
            if target_end > len(tf_data['labels']):
                # Skip this timeframe if not enough data for target
                continue
            
            target = tf_data['labels'][target_start:target_end]
            
            # Get returns for the target period
            tf_returns = tf_data['returns'][target_start:target_end]
            
            # Store data
            x[tf] = torch.from_numpy(features)
            
            # For now, just use the first target step
            y[tf] = {
                'direction': torch.tensor(target[0, 0], dtype=torch.long),
                'magnitude': torch.tensor(target[0, 1], dtype=torch.float32),
                'duration': torch.tensor(target[0, 2], dtype=torch.float32),
                'confidence': torch.tensor(target[0, 3], dtype=torch.float32)
            }
            
            returns.append(tf_returns[0])
        
        # Average returns across timeframes
        avg_returns = np.mean(returns) if returns else 0.0
        
        return x, y, torch.tensor(avg_returns, dtype=torch.float32)

def collate_fn(batch):
    """Collate function for DataLoader"""
    # Separate x, y, returns
    x_list, y_list, returns_list = zip(*batch)
    
    # Get all timeframes across all samples
    all_timeframes = set()
    for x in x_list:
        all_timeframes.update(x.keys())
    
    # Initialize batched data
    batched_x_list = {tf: [] for tf in all_timeframes}
    batched_y = {
        'direction': [],
        'magnitude': [],
        'duration': [],
        'confidence': []
    }
    batched_returns = []
    
    # Process each sample
    for x, y, returns in zip(x_list, y_list, returns_list):
        # Get common timeframes
        set(x.keys())
        
        # Add data for each timeframe
        for tf in all_timeframes:
            if tf in x:
                batched_x_list[tf].append(x[tf])
            else:
                # If timeframe is missing, use zeros with the same shape as other samples
                # This should be handled by the model if needed
                pass
        
        # Add targets (same for all timeframes)
        if y:
            # Just use the first available timeframe's targets
            first_tf = next(iter(y.keys()))
            batched_y['direction'].append(y[first_tf]['direction'])
            batched_y['magnitude'].append(y[first_tf]['magnitude'])
            batched_y['duration'].append(y[first_tf]['duration'])
            batched_y['confidence'].append(y[first_tf]['confidence'])
        
        # Add returns
        batched_returns.append(returns)
    
    # Stack tensors
    batched_x = {}
    for tf in batched_x_list:
        if batched_x_list[tf]:
            batched_x[tf] = torch.stack(batched_x_list[tf])
    
    # Stack targets
    batched_y = {k: torch.stack(v) for k, v in batched_y.items() if v}
    
    # Stack returns
    batched_returns = torch.stack(batched_returns) if batched_returns else None
    
    return batched_x, batched_y, batched_returns

class TrainingPipeline:
    """Training pipeline for the trading model"""
    
    def __init__(
        self,
        data_dir: str,
        model_dir: str = 'models',
        results_dir: str = 'results',
        device: Optional[str] = None,
        timeframes: Optional[List[str]] = None,
        seq_length: int = 100,
        target_length: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
        random_seed: int = 42
    ):
        """Initialize the training pipeline"""
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device: Literal['cuda', 'cpu']
        if device in ('cuda', 'cpu'):
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.timeframes = timeframes or ['5m', '15m', '30m', '1h', '4h', '1d']
        self.seq_length = seq_length
        self.target_length = target_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize model and optimizer
        self.model_config = ModelConfig(
            feature_dims={tf: 40 for tf in self.timeframes},  # Assuming 40 features for all timeframes
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=512,
            dropout=0.1,
            timeframes=self.timeframes,
            max_seq_len=seq_length,
        )
        
        self.loss_config = LossConfig(
            direction_weight=0.3,
            magnitude_weight=0.2,
            duration_weight=0.1,
            confidence_weight=0.2,
            risk_weight=0.2,
            max_drawdown_penalty=1.0,
            sharpe_ratio_weight=0.5,
            sortino_ratio_weight=0.5,
            margin=0.1,
            label_smoothing=0.1
        )
        
        # Initialize adaptive learning system
        self.adaptive_system = AdaptiveLearningSystem(
            model_config=self.model_config,
            loss_config=self.loss_config,
            device=cast(Any, self.device),
            model_dir=str(self.model_dir / 'adaptive')
        )
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.startswith('cuda'))
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def prepare_data(self):
        """Prepare data loaders"""
        logger.info("Preparing data loaders...")
        
        # Create datasets
        train_dataset = TradingDataset(
            data_dir=str(self.data_dir),
            split='train',
            timeframes=self.timeframes,
            seq_length=self.seq_length,
            target_length=self.target_length,
            random_seed=self.random_seed
        )
        
        val_dataset = TradingDataset(
            data_dir=str(self.data_dir),
            split='val',
            timeframes=self.timeframes,
            seq_length=self.seq_length,
            target_length=self.target_length,
            random_seed=self.random_seed
        )
        
        test_dataset = TradingDataset(
            data_dir=str(self.data_dir),
            split='test',
            timeframes=self.timeframes,
            seq_length=self.seq_length,
            target_length=self.target_length,
            random_seed=self.random_seed
        )
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Test batches: {len(self.test_loader)}")
    
    def train(self, epochs: int = 100):
        """Train the model"""
        if self.train_loader is None or self.val_loader is None:
            self.prepare_data()
        
        logger.info("Starting training...")
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Check for improvement
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(is_best=False)
        
        # Load best model
        self._load_best_model()
        
        # Test
        test_metrics = self.test()
        logger.info(f"Test metrics: {test_metrics}")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        assert self.train_loader is not None
        for model in self.adaptive_system.ensemble.models.values():
            model.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (x, y, returns) in enumerate(self.train_loader):
            # Move data to device
            x_device = {}
            for tf, data in x.items():
                if isinstance(data, torch.Tensor):
                    x_device[tf] = data.to(self.device, non_blocking=True)
                else:
                    x_device[tf] = data
            
            y_device = {}
            for k, v in y.items():
                if isinstance(v, torch.Tensor):
                    y_device[k] = v.to(self.device, non_blocking=True)
                else:
                    y_device[k] = v
            
            if returns is not None:
                returns = returns.to(self.device, non_blocking=True)
            
            # Detect market regime for this batch
            # For now, just use a simple approach - in practice, you'd use OHLCV data
            # to detect the regime more accurately
            regime = random.choice(['trending_up', 'trending_down', 'ranging', 'volatile', 'low_volatility'])
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.device.startswith('cuda')):
                # Get model outputs
                outputs = self.adaptive_system.ensemble.models[regime](x_device)
                
                # Calculate loss
                loss_dict = self.adaptive_system.ensemble.loss_fn(
                    outputs, y_device, returns
                )
            
            # Backward pass
            self.adaptive_system.ensemble.optimizers[regime].zero_grad()
            
            if self.device.startswith('cuda'):
                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.step(self.adaptive_system.ensemble.optimizers[regime])
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                self.adaptive_system.ensemble.optimizers[regime].step()
            
            # Update metrics
            batch_size = len(next(iter(x.values())))
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_metrics[k] += v.item() * batch_size
            
            num_batches += batch_size
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {self.current_epoch + 1} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {loss_dict['total'].item():.4f} | "
                    f"Regime: {regime}"
                )
        
        # Average metrics
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model"""
        assert self.val_loader is not None
        for model in self.adaptive_system.ensemble.models.values():
            model.eval()
        
        val_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for x, y, returns in self.val_loader:
                # Move data to device
                x_device = {}
                for tf, data in x.items():
                    if isinstance(data, torch.Tensor):
                        x_device[tf] = data.to(self.device, non_blocking=True)
                    else:
                        x_device[tf] = data
                
                y_device = {}
                for k, v in y.items():
                    if isinstance(v, torch.Tensor):
                        y_device[k] = v.to(self.device, non_blocking=True)
                    else:
                        y_device[k] = v
                
                if returns is not None:
                    returns = returns.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.device.startswith('cuda')):
                    # Get predictions from all models
                    all_outputs = {}
                    for regime in self.adaptive_system.ensemble.models:
                        outputs = self.adaptive_system.ensemble.models[regime](x_device)
                        all_outputs[regime] = outputs
                    
                    # For validation, just use the first model's outputs
                    # In practice, you might want to ensemble the predictions
                    outputs = all_outputs[next(iter(all_outputs.keys()))]
                    
                    # Calculate loss
                    loss_dict = self.adaptive_system.ensemble.loss_fn(
                        outputs, y_device, returns
                    )
                
                # Update metrics
                batch_size = len(next(iter(x.values())))
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        val_metrics[k] += v.item() * batch_size
                
                num_batches += batch_size
        
        # Average metrics
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        return val_metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model"""
        if self.test_loader is None:
            self.prepare_data()
        
        assert self.test_loader is not None
        for model in self.adaptive_system.ensemble.models.values():
            model.eval()
        
        test_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for x, y, returns in self.test_loader:
                # Move data to device
                x_device = {}
                for tf, data in x.items():
                    if isinstance(data, torch.Tensor):
                        x_device[tf] = data.to(self.device, non_blocking=True)
                    else:
                        x_device[tf] = data
                
                y_device = {}
                for k, v in y.items():
                    if isinstance(v, torch.Tensor):
                        y_device[k] = v.to(self.device, non_blocking=True)
                    else:
                        y_device[k] = v
                
                if returns is not None:
                    returns = returns.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.device.startswith('cuda')):
                    # Get predictions from all models
                    all_outputs = {}
                    for regime in self.adaptive_system.ensemble.models:
                        outputs = self.adaptive_system.ensemble.models[regime](x_device)
                        all_outputs[regime] = outputs
                    
                    # For testing, just use the first model's outputs
                    # In practice, you might want to ensemble the predictions
                    outputs = all_outputs[next(iter(all_outputs.keys()))]
                    
                    # Calculate loss
                    loss_dict = self.adaptive_system.ensemble.loss_fn(
                        outputs, y_device, returns
                    )
                
                # Update metrics
                batch_size = len(next(iter(x.values())))
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        test_metrics[k] += v.item() * batch_size
                
                num_batches += batch_size
        
        # Average metrics
        test_metrics = {k: v / num_batches for k, v in test_metrics.items()}
        
        # Save test metrics
        with open(self.results_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log training and validation metrics"""
        # Log to console
        log_str = f"Epoch {epoch + 1} | "
        log_str += f"Train Loss: {train_metrics['total']:.4f} | "
        log_str += f"Val Loss: {val_metrics['total']:.4f} | "
        
        # Add other metrics
        for k in ['direction', 'magnitude', 'duration', 'confidence', 'risk']:
            if k in train_metrics:
                log_str += f"{k.capitalize()}: {train_metrics[k]:.4f} (train) / {val_metrics.get(k, 0):.4f} (val) | "
        
        logger.info(log_str)
        
        # Log to file
        with open(self.results_dir / 'training_log.csv', 'a') as f:
            if epoch == 0:
                # Write header
                header = 'epoch,' + ','.join([f'train_{k},val_{k}' for k in train_metrics.keys()])
                f.write(header + '\n')
            
            # Write metrics
            row = [str(epoch + 1)]
            for k in train_metrics:
                row.append(f"{train_metrics[k]:.6f}")
                row.append(f"{val_metrics.get(k, 0):.6f}")
            
            f.write(','.join(row) + '\n')
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'config': {
                'model': self.model_config.__dict__,
                'loss': self.loss_config.__dict__
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.model_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = self.model_dir / 'best_model.pth'
            shutil.copyfile(checkpoint_path, best_model_path)
            
            # Also save the adaptive system
            self.adaptive_system.save()
    
    def _load_best_model(self):
        """Load the best model"""
        best_model_path = self.model_dir / 'best_model.pth'
        
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            
            # Load adaptive system
            adaptive_system_path = self.model_dir / 'adaptive'
            if adaptive_system_path.exists():
                self.adaptive_system = AdaptiveLearningSystem.load(
                    adaptive_system_path,
                    device=cast(Any, self.device)
                )
            
            # Update training state
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint['patience_counter']
            
            # Restore random states
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            
            if torch.cuda.is_available() and checkpoint['torch_cuda_random_state'] is not None:
                torch.cuda.set_rng_state_all(checkpoint['torch_cuda_random_state'])
            
            logger.info(f"Loaded best model from epoch {self.current_epoch} with val loss {self.best_val_loss:.6f}")
        else:
            logger.warning("No best model found, training from scratch")

def main():
    """Main function for training the model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the trading model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing the processed data')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save the model')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (e.g., cuda, cpu)')
    parser.add_argument('--timeframes', type=str, nargs='+',
                       default=['5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Timeframes to use for training')
    parser.add_argument('--seq-length', type=int, default=100,
                       help='Length of input sequences')
    parser.add_argument('--target-length', type=int, default=10,
                       help='Length of target sequences')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train for')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        device=args.device,
        timeframes=args.timeframes,
        seq_length=args.seq_length,
        target_length=args.target_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed
    )
    
    # Train the model
    pipeline.train(epochs=args.epochs)

if __name__ == '__main__':
    main()
