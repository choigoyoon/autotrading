import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Deque
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime

from .hierarchical_trading_transformer import HierarchicalTradingTransformer, ModelConfig
from .risk_adjusted_loss import RiskAdjustedLoss, LossConfig

class MarketRegimeDetector:
    """Detects different market regimes using technical indicators"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.regimes = ['trending_up', 'trending_down', 'ranging', 'volatile', 'low_volatility']
    
    def detect_regime(self, ohlcv: Dict[str, np.ndarray]) -> str:
        """
        Detect current market regime based on OHLCV data
        
        Args:
            ohlcv: Dictionary with 'open', 'high', 'low', 'close', 'volume' keys
            
        Returns:
            str: Detected market regime
        """
        close = ohlcv['close'][-self.lookback:]
        high = ohlcv['high'][-self.lookback:]
        low = ohlcv['low'][-self.lookback:]
        volume = ohlcv['volume'][-self.lookback:]
        
        # Calculate technical indicators
        returns = np.diff(np.log(close))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Simple moving averages
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:])
        
        # ADX for trend strength
        adx = self._calculate_adx(high, low, close, timeperiod=14)
        
        # ATR for volatility
        atr = self._calculate_atr(high, low, close, timeperiod=14)
        atr_pct = atr / close[-1]
        
        # Determine regime
        if adx > 25:
            if sma_20 > sma_50 * 1.01:  # 1% above
                return 'trending_up'
            elif sma_20 < sma_50 * 0.99:  # 1% below
                return 'trending_down'
        
        if atr_pct > 0.02:  # High volatility
            return 'volatile'
        elif atr_pct < 0.005:  # Low volatility
            return 'low_volatility'
        
        return 'ranging'  # Default to ranging
    
    def _calculate_adx(self, high, low, close, timeperiod=14):
        """Calculate Average Directional Index (ADX)"""
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(low)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        tr = self._calculate_true_range(high, low, close)
        
        plus_di = 100 * self._ema(plus_dm, timeperiod) / self._ema(tr, timeperiod)
        minus_di = 100 * self._ema(minus_dm, timeperiod) / self._ema(tr, timeperiod)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, timeperiod)
        
        return adx[-1] if len(adx) > 0 else 0.0
    
    def _calculate_atr(self, high, low, close, timeperiod=14):
        """Calculate Average True Range (ATR)"""
        tr = self._calculate_true_range(high, low, close)
        atr = self._ema(tr, timeperiod)
        return atr[-1] if len(atr) > 0 else 0.0
    
    def _calculate_true_range(self, high, low, close):
        """Calculate True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return tr[1:]  # Skip first NaN
    
    def _ema(self, values, period):
        """Exponential Moving Average"""
        if len(values) < period:
            return np.array([])
            
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        
        ema = np.convolve(values, weights, mode='full')[:len(values)]
        ema[:period] = ema[period]
        return ema

class ModelEnsemble:
    """Manages an ensemble of models for different market regimes"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        device: torch.device,
        model_dir: str = 'models/ensemble'
    ):
        self.model_config = model_config
        self.loss_config = loss_config
        self.device = device
        self.model_dir = Path(model_dir)
        _ = self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models for each regime
        self.models = {
            regime: HierarchicalTradingTransformer(model_config).to(device)
            for regime in ['trending_up', 'trending_down', 'ranging', 'volatile', 'low_volatility']
        }
        
        # Initialize optimizers and schedulers
        self.optimizers = {
            regime: optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )
            for regime, model in self.models.items()
        }
        
        # Loss function
        self.loss_fn = RiskAdjustedLoss(loss_config)
        
        # Training history
        self.history = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
    
    def predict(
        self,
        x: Dict[str, torch.Tensor],
        regime: str,
        use_ema: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Make prediction using the model for the given regime"""
        model = self.models[regime]
        _ = model.eval()
        
        with torch.no_grad():
            # Move inputs to device
            x_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in x.items()
            }
            
            # Get predictions
            outputs = model(x_device)
            
            # Convert to CPU and numpy for easier handling
            predictions = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
            
            return predictions
    
    def train_step(
        self,
        x: Dict[str, torch.Tensor],
        y: Dict[str, torch.Tensor],
        regime: str,
        returns: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Perform a single training step"""
        model = self.models[regime]
        optimizer = self.optimizers[regime]
        
        _ = model.train()
        
        # Move data to device
        x_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in x.items()
        }
        y_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in y.items()
        }
        
        if returns is not None:
            returns = returns.to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x_device)
        
        # Calculate loss
        loss_dict = self.loss_fn(outputs, y_device, returns)
        
        # Backward pass
        _ = loss_dict['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        
        return loss_dict
    
    def save_models(self, suffix: str = ''):
        """Save all models and optimizers"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = self.model_dir / f"{timestamp}{f'_{suffix}' if suffix else ''}"
        _ = save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model and optimizer
        for regime, model in self.models.items():
            _ = torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[regime].state_dict(),
                'config': {
                    'model': self.model_config.__dict__,
                    'loss': self.loss_config.__dict__
                }
            }, save_dir / f"{regime}_model.pth")
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Models saved to {save_dir}")
        return str(save_dir)
    
    def load_models(self, model_dir: Path):
        """Load all models and optimizers"""
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            return
            
        for regime, model in self.models.items():
            model_path = model_dir / f"{regime}_model.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizers[regime].load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded model for {regime} from {model_path}")

class AdaptiveLearningSystem:
    """Manages the overall adaptive training and prediction process"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        device: torch.device = torch.device("cpu"),
        model_dir: str = 'models/adaptive'
    ):
        self.device = device
        self.model_dir = Path(model_dir)
        _ = self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config = model_config
        self.loss_config = loss_config
        
        self.ensemble = ModelEnsemble(model_config, loss_config, self.device, str(self.model_dir))
        self.regime_detector = MarketRegimeDetector()
        
        self.current_regime: Optional[str] = None
        self.regime_history: Deque[str] = deque(maxlen=100)
        self.performance_metrics: Dict[str, float] = {}
        self.performance_history: Deque[float] = deque(maxlen=100)
        self.performance_log = pd.DataFrame()

    def update(self, metrics: Dict[str, float]):
        """Update system based on recent performance"""
        self.performance_history.append(metrics.get('sharpe_ratio', 0.0))
        
        # Example: adjust loss parameters if Sharpe ratio is consistently low
        if len(self.performance_history) == self.performance_history.maxlen:
            if np.mean(list(self.performance_history)) < 0.5:
                # Make loss function more risk-averse
                self.ensemble.loss_fn.config.risk_aversion *= 1.1
                print(f"Low Sharpe ratio detected. Increasing risk aversion to {self.ensemble.loss_fn.config.risk_aversion:.2f}")

    def process_batch(
        self,
        batch,
        is_training: bool
    ) -> tuple[dict[str, float], dict[str, np.ndarray] | None, str]:
        """Process a single batch of data"""
        x, y, returns, ohlcv = batch
        
        # Detect regime
        regime = self.regime_detector.detect_regime(ohlcv)
        
        if is_training:
            metrics = self.ensemble.train_step(x, y, regime, returns)
            return metrics, None, regime
        else:
            predictions = self.ensemble.predict(x, regime)
            
            # Since validation loss is needed, calculate it here
            # This requires moving data to device again inside loss_fn
            y_device = {k: v.to(self.device) for k, v in y.items()}
            returns_device = returns.to(self.device)
            
            # Re-create model outputs as tensors on the correct device
            outputs_device = {k: torch.tensor(v, device=self.device) for k, v in predictions.items()}
            
            metrics = self.ensemble.loss_fn(outputs_device, y_device, returns_device)
            metrics_items = {k: v.item() for k, v in metrics.items()}
            
            return metrics_items, predictions, regime

    def save(self, path: Optional[Path] = None):
        """Save the entire adaptive system state"""
        if path is None:
            path = self.model_dir / 'adaptive_system_latest.pth'
            
        _ = torch.save({
            'ensemble_state': {regime: model.state_dict() for regime, model in self.ensemble.models.items()},
            'optimizer_states': {regime: opt.state_dict() for regime, opt in self.ensemble.optimizers.items()},
            'model_config': self.model_config.to_dict(),
            'loss_config': self.loss_config.to_dict(),
            'performance_log': self.performance_log,
        }, path)
        print(f"Adaptive system saved to {path}")

    @classmethod
    def load(cls, path: Path, device: torch.device = None):
        """Load the adaptive system state"""
        if device is None:
            device = torch.device("cpu")
            
        checkpoint = torch.load(path, map_location=device)
        
        system = cls(
            model_config=ModelConfig.from_dict(checkpoint['model_config']),
            loss_config=LossConfig.from_dict(checkpoint['loss_config']),
            device=device
        )
        
        for regime, state_dict in checkpoint['ensemble_state'].items():
            system.ensemble.models[regime].load_state_dict(state_dict)
            
        for regime, state_dict in checkpoint['optimizer_states'].items():
            system.ensemble.optimizers[regime].load_state_dict(state_dict)
            
        system.performance_log = checkpoint['performance_log']
        
        print(f"Adaptive system loaded from {path}")
        return system
