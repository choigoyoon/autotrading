import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, cast
import time
import logging
from datetime import datetime
import threading
import queue

from .adaptive_learning_system import AdaptiveLearningSystem, MarketRegimeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataBuffer:
    """Thread-safe buffer for storing and retrieving market data"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the data buffer
        
        Args:
            max_size: Maximum number of data points to store
        """
        self.max_size = max_size
        self.buffer = {}
        self.lock = threading.Lock()
        self.data_ready = threading.Event()
    
    def add_data(self, timeframe: str, data: Dict[str, Any]):
        """
        Add data to the buffer
        
        Args:
            timeframe: Timeframe identifier (e.g., '1m', '5m', '15m')
            data: Dictionary containing market data (OHLCV, features, etc.)
        """
        with self.lock:
            if timeframe not in self.buffer:
                self.buffer[timeframe] = []
            
            self.buffer[timeframe].append(data)
            
            # Trim buffer if it exceeds max size
            if len(self.buffer[timeframe]) > self.max_size:
                self.buffer[timeframe] = self.buffer[timeframe][-self.max_size:]
            
            # Signal that new data is available
            self.data_ready.set()
    
    def get_latest(self, timeframe: str, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get the most recent n data points for a timeframe
        
        Args:
            timeframe: Timeframe identifier
            n: Number of data points to retrieve
            
        Returns:
            List of the most recent n data points (or fewer if not enough data)
        """
        with self.lock:
            if timeframe not in self.buffer or not self.buffer[timeframe]:
                return []
            
            return self.buffer[timeframe][-n:]
    
    def get_all(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get all data for a timeframe
        
        Args:
            timeframe: Timeframe identifier
            
        Returns:
            List of all data points for the timeframe
        """
        with self.lock:
            return self.buffer.get(timeframe, []).copy()
    
    def clear(self, timeframe: Optional[str] = None):
        """
        Clear data from the buffer
        
        Args:
            timeframe: If provided, clear only this timeframe. Otherwise, clear all.
        """
        with self.lock:
            if timeframe is not None:
                if timeframe in self.buffer:
                    self.buffer[timeframe].clear()
            else:
                self.buffer.clear()
    
    def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for new data to be available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if data is available, False if timeout occurred
        """
        return self.data_ready.wait(timeout=timeout)
    
    def clear_event(self):
        """Clear the data ready event"""
        self.data_ready.clear()

class OnlineAdaptationSystem:
    """System for online adaptation of trading models"""
    
    def __init__(
        self,
        model_dir: str = 'models/adaptive',
        adaptation_interval: int = 3600,  # 1 hour in seconds
        max_adaptation_steps: int = 100,
        device: Optional[Literal['cuda', 'cpu']] = None,
        enable_adaptation: bool = True,
        enable_inference: bool = True,
        inference_interval: float = 60.0  # 60 seconds
    ):
        """
        Initialize the online adaptation system
        
        Args:
            model_dir: Directory containing the pre-trained models
            adaptation_interval: Time between model adaptations in seconds
            max_adaptation_steps: Maximum number of adaptation steps per interval
            device: Device to run models on ('cuda' or 'cpu')
            enable_adaptation: Whether to enable model adaptation
            enable_inference: Whether to enable inference mode
            inference_interval: Time between inference runs in seconds
        """
        self.model_dir = Path(model_dir)
        self.adaptation_interval = adaptation_interval
        self.max_adaptation_steps = max_adaptation_steps
        self.device: Literal['cuda', 'cpu'] = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_adaptation = enable_adaptation
        self.enable_inference = enable_inference
        self.inference_interval = inference_interval
        
        # Data buffers
        self.market_data_buffer = DataBuffer(max_size=10000)
        self.inference_queue = queue.Queue()
        
        # Model and regime detector
        self.adaptive_system = None
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = None
        
        # Thread control
        self._stop_event = threading.Event()
        self._threads = []
        
        # Performance metrics
        self.metrics = {
            'inference_count': 0,
            'adaptation_count': 0,
            'last_adaptation_time': None,
            'performance_metrics': {}
        }
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            self.adaptive_system = AdaptiveLearningSystem.load(
                self.model_dir,
                device=cast(Any, self.device)
            )
            logger.info("Successfully loaded pre-trained models")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def start(self):
        """Start the online adaptation system"""
        if self.enable_adaptation:
            adaptation_thread = threading.Thread(
                target=self._adaptation_loop,
                daemon=True
            )
            self._threads.append(adaptation_thread)
            adaptation_thread.start()
            logger.info("Started model adaptation thread")
        
        if self.enable_inference:
            inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True
            )
            self._threads.append(inference_thread)
            inference_thread.start()
            logger.info("Started inference thread")
        
        logger.info("Online adaptation system started")
    
    def stop(self):
        """Stop the online adaptation system"""
        self._stop_event.set()
        
        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=5.0)
        
        logger.info("Online adaptation system stopped")
    
    def add_market_data(self, timeframe: str, data: Dict[str, Any]):
        """
        Add market data to the buffer
        
        Args:
            timeframe: Timeframe identifier (e.g., '1m', '5m', '15m')
            data: Dictionary containing market data (OHLCV, features, etc.)
        """
        self.market_data_buffer.add_data(timeframe, data)
    
    def get_predictions(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get predictions from the inference queue
        
        Args:
            timeout: Maximum time to wait for predictions
            
        Returns:
            List of prediction dictionaries, or empty list if timeout
        """
        try:
            if self.inference_queue.empty():
                return []
            
            predictions = []
            while True:
                try:
                    pred = self.inference_queue.get_nowait()
                    predictions.append(pred)
                except queue.Empty:
                    break
            
            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def _adaptation_loop(self):
        """Main loop for model adaptation"""
        last_adaptation_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                time_since_last_adaptation = current_time - last_adaptation_time
                
                # Check if it's time to adapt
                if time_since_last_adaptation >= self.adaptation_interval:
                    logger.info("Starting model adaptation...")
                    
                    # Get recent market data for adaptation
                    adaptation_data = self._prepare_adaptation_data()
                    
                    if adaptation_data:
                        # Perform adaptation
                        self._adapt_models(adaptation_data)
                        
                        # Update metrics
                        self.metrics['adaptation_count'] += 1
                        self.metrics['last_adaptation_time'] = datetime.now().isoformat()
                        
                        logger.info("Model adaptation completed")
                    else:
                        logger.warning("Insufficient data for adaptation")
                    
                    # Update last adaptation time
                    last_adaptation_time = current_time
                
                # Sleep for a short time to prevent busy waiting
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                time.sleep(5.0)  # Prevent tight loop on error
    
    def _inference_loop(self):
        """Main loop for inference"""
        last_inference_time = 0.0
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                time_since_last_inference = current_time - last_inference_time
                
                # Check if it's time to run inference
                if time_since_last_inference >= self.inference_interval:
                    # Get latest market data
                    inference_data = self._prepare_inference_data()
                    
                    if inference_data:
                        # Run inference
                        predictions = self._run_inference(inference_data)
                        
                        # Add predictions to queue
                        if predictions:
                            self.inference_queue.put(predictions)
                            self.metrics['inference_count'] += 1
                    
                    # Update last inference time
                    last_inference_time = current_time
                
                # Sleep for a short time to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                time.sleep(1.0)  # Prevent tight loop on error
    
    def _prepare_adaptation_data(self) -> Dict[str, Any]:
        """
        Prepare data for model adaptation
        
        Returns:
            Dictionary containing data for adaptation
        """
        # Get recent market data for different timeframes
        adaptation_data = {}
        
        # Example: Get 1-hour of 1-minute data for adaptation
        ohlcv_data = self.market_data_buffer.get_latest('1m', n=60)
        
        if not ohlcv_data:
            return {}
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(ohlcv_data)
        
        # Add basic technical indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Add regime information
        current_regime = self.regime_detector.detect_regime({
            'open': df['open'].to_numpy(),
            'high': df['high'].to_numpy(),
            'low': df['low'].to_numpy(),
            'close': df['close'].to_numpy(),
            'volume': df['volume'].to_numpy()
        })
        
        # Store data for adaptation
        adaptation_data = {
            'timestamps': df['timestamp'].values,
            'prices': df['close'].values,
            'returns': df['returns'].values,
            'volatility': df['volatility'].values,
            'regime': current_regime,
            'features': {}
        }
        
        # Add any additional features
        # ...
        
        return adaptation_data
    
    def _prepare_inference_data(self) -> Dict[str, Any]:
        """
        Prepare data for inference
        
        Returns:
            Dictionary containing data for inference
        """
        # Get latest market data for different timeframes
        inference_data = {}
        
        # Example: Get the most recent data point for each timeframe
        for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
            data = self.market_data_buffer.get_latest(tf, n=1)
            if data:
                inference_data[tf] = data[0]
        
        return inference_data
    
    def _adapt_models(self, adaptation_data: Dict[str, Any]):
        """
        Adapt models using the provided data
        
        Args:
            adaptation_data: Dictionary containing data for adaptation
        """
        if not self.adaptive_system:
            logger.warning("No adaptive system available for adaptation")
            return
        
        try:
            # Get current regime
            current_regime = adaptation_data.get('regime', 'ranging')
            
            # Convert data to tensors
            returns = torch.tensor(
                adaptation_data['returns'][-100:],  # Use last 100 returns
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(-1)  # Add batch dimension
            
            # Create dummy targets for self-supervised learning
            # In practice, you'd use a more sophisticated approach
            targets = {
                'direction': torch.zeros(1, dtype=torch.long, device=self.device),
                'magnitude': torch.zeros(1, dtype=torch.float32, device=self.device),
                'duration': torch.zeros(1, dtype=torch.float32, device=self.device),
                'confidence': torch.ones(1, dtype=torch.float32, device=self.device)
            }
            
            # Perform a few adaptation steps
            for step in range(self.max_adaptation_steps):
                # Create dummy input (in practice, use actual features)
                x = {
                    'features': torch.randn(1, 100, 40, device=self.device),  # [batch, seq_len, features]
                    'returns': returns.unsqueeze(0)  # [batch, seq_len, 1]
                }
                
                # Update model
                self.adaptive_system.ensemble.models[current_regime].train()
                self.adaptive_system.ensemble.optimizers[current_regime].zero_grad()
                
                # Forward pass
                outputs = self.adaptive_system.ensemble.models[current_regime](x)
                
                # Calculate loss (simplified)
                loss = torch.nn.functional.cross_entropy(
                    outputs['direction'],
                    targets['direction']
                )
                
                # Backward pass
                loss.backward()
                self.adaptive_system.ensemble.optimizers[current_regime].step()
                
                logger.debug(f"Adaptation step {step + 1}/{self.max_adaptation_steps} - Loss: {loss.item():.4f}")
            
            logger.info(f"Adapted {current_regime} model with {self.max_adaptation_steps} steps")
            
            # Update current regime
            self.current_regime = current_regime
            
        except Exception as e:
            logger.error(f"Error during model adaptation: {e}")
    
    def _run_inference(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference using the current models
        
        Args:
            inference_data: Dictionary containing data for inference
            
        Returns:
            Dictionary containing predictions
        """
        if not self.adaptive_system:
            logger.warning("No adaptive system available for inference")
            return {}
        
        try:
            # Detect current regime
            ohlcv = {}
            for tf, data in inference_data.items():
                if 'open' in data and 'high' in data and 'low' in data and 'close' in data and 'volume' in data:
                    ohlcv[tf] = data
            
            if not ohlcv:
                logger.warning("No OHLCV data available for regime detection")
                return {}
            
            # Use the highest timeframe available for regime detection
            max_tf = max(ohlcv.keys(), key=lambda x: int(x[:-1]) if x[-1] in ['m', 'h'] else float('inf'))
            data = ohlcv[max_tf]
            
            # Detect regime
            current_regime = self.regime_detector.detect_regime({
                'open': np.array([data['open']]),
                'high': np.array([data['high']]),
                'low': np.array([data['low']]),
                'close': np.array([data['close']]),
                'volume': np.array([data['volume']])
            })
            
            # Prepare input for the model
            # In practice, you'd preprocess the data to match the model's expected input format
            x = {
                tf: torch.tensor([
                    [data['open'], data['high'], data['low'], data['close'], data['volume']]
                    for _ in range(100)  # Dummy sequence
                ], dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dimension
                for tf, data in inference_data.items()
            }
            
            # Run inference
            self.adaptive_system.ensemble.models[current_regime].eval()
            with torch.no_grad():
                outputs = self.adaptive_system.ensemble.models[current_regime](x)
            
            # Process outputs
            direction = int(torch.argmax(outputs['direction'], dim=-1).item())
            direction_str = {0: 'LONG', 1: 'NEUTRAL', 2: 'SHORT'}.get(direction, 'UNKNOWN')
            
            # Create prediction dictionary
            prediction = {
                'timestamp': datetime.utcnow().isoformat(),
                'regime': current_regime,
                'direction': direction_str,
                'confidence': outputs['confidence'].item(),
                'risk': outputs['risk'].item(),
                'signals': {
                    'magnitude': outputs['magnitude'].item(),
                    'duration': outputs['duration'].item()
                },
                'source': 'online_adaptation'
            }
            
            logger.debug(f"Inference result: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the online adaptation system
        
        Returns:
            Dictionary containing status information
        """
        return {
            'status': 'running' if not self._stop_event.is_set() else 'stopped',
            'current_regime': self.current_regime,
            'metrics': self.metrics,
            'model_status': {
                'regimes': list(self.adaptive_system.ensemble.models.keys()) if self.adaptive_system else [],
                'device': str(self.device)
            },
            'timestamps': {
                'current': datetime.utcnow().isoformat(),
                'last_adaptation': self.metrics.get('last_adaptation_time')
            }
        }

# Example usage
if __name__ == "__main__":
    import time
    import random
    
    # Initialize the system
    online_system = OnlineAdaptationSystem(
        model_dir='models/adaptive',
        adaptation_interval=300,  # 5 minutes
        inference_interval=30,    # 30 seconds
        enable_adaptation=True,
        enable_inference=True
    )
    
    # Start the system
    online_system.start()
    
    try:
        # Simulate adding market data
        print("Simulating market data...")
        for i in range(100):
            # Generate random OHLCV data
            ohlcv = {
                'timestamp': int(time.time() * 1000),
                'open': 50000 + random.uniform(-100, 100),
                'high': 50000 + random.uniform(0, 200),
                'low': 50000 - random.uniform(0, 200),
                'close': 50000 + random.uniform(-100, 100),
                'volume': random.uniform(0, 1000)
            }
            
            # Add to buffer
            online_system.add_market_data('1m', ohlcv)
            
            # Get predictions
            predictions = online_system.get_predictions()
            for pred in predictions:
                print(f"Prediction: {pred}")
            
            # Get status
            if i % 10 == 0:
                status = online_system.get_status()
                print(f"Status: {status}")
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop the system
        online_system.stop()
        print("System stopped")
