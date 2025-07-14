import pandas as pd
from pathlib import Path
import talib # type: ignore
import logging
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class ContextFeatureExtractor:
    """Extracts contextual features for trading signals"""
    
    def __init__(self, data_dir: str):
        """
        Initialize the feature extractor
        
        Args:
            data_dir: Directory containing OHLCV data
        """
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load OHLCV data for a specific timeframe"""
        file_path = self.data_dir / f"{symbol}_{timeframe}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"OHLCV data not found at {file_path}")
        return pd.read_parquet(file_path)
    
    def extract_downtrend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract downtrend related features"""
        features = {}
        
        # Type conversion for talib functions
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # Moving Averages
        df['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
        df['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        df['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
        
        # Price below moving averages
        features['below_sma20'] = (df['close'] < df['SMA_20']).astype(int)
        features['below_sma50'] = (df['close'] < df['SMA_50']).astype(int)
        features['below_sma200'] = (df['close'] < df['SMA_200']).astype(int)
        
        # RSI
        df['rsi'] = talib.RSI(close_prices, timeperiod=14)
        features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # MACD
        macd, signal, _ = talib.MACD(close_prices)
        features['macd_below_signal'] = (macd < signal).astype(int)
        
        # Volume features
        volume_ma = df['volume'].rolling(window=20).mean()
        features['volume_above_avg'] = (df['volume'] > volume_ma).astype(int)
        
        # ATR for volatility
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Price rate of change
        features['roc'] = talib.ROC(close_prices, timeperiod=10) # type: ignore
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
        features['bb_width'] = (upper - lower) / middle
        features['bb_pct'] = (df['close'] - lower) / (upper - lower)
        
        # ADX for trend strength
        features['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14) # type: ignore
        
        # OBV for volume flow
        features['obv'] = talib.OBV(close_prices, volume)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices) # type: ignore
        features['stoch_oversold'] = ((slowk < 20) & (slowd < 20)).astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_uptrend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract uptrend related features"""
        features = {}

        # Type conversion for talib functions
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        # Moving Averages
        features['above_sma20'] = (df['close'] > df['SMA_20']).astype(int)
        features['above_sma50'] = (df['close'] > df['SMA_50']).astype(int)
        features['above_sma200'] = (df['close'] > df['SMA_200']).astype(int)
        
        # RSI
        features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd, signal, _ = talib.MACD(close_prices)
        features['macd_above_signal'] = (macd > signal).astype(int)
        
        # Volume features
        features['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)
        
        # Price momentum
        features['momentum'] = talib.MOM(close_prices, timeperiod=10) # type: ignore
        
        # CCI
        features['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20) # type: ignore
        
        # Aroon
        aroon_up, aroon_down = talib.AROON(high_prices, low_prices, timeperiod=14) # type: ignore
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        
        # MFI
        features['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14) # type: ignore
        
        # Stochastic
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices) # type: ignore
        features['stoch_overbought'] = ((slowk > 80) & (slowd > 80)).astype(int)
        
        # Williams %R
        features['willr'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14) # type: ignore
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract market consistency features"""
        features = {}
        
        # Volatility measures
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
        features['range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Trend consistency
        price_diff = df['close'].diff()
        features['trend_consistency'] = price_diff.rolling(window=10).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # Volume consistency
        volume_diff = df['volume'].diff()
        features['volume_consistency'] = volume_diff.rolling(window=10).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # Price-Volume correlation
        price_returns = df['close'].pct_change()
        volume_returns = df['volume'].pct_change()
        features['price_volume_corr'] = price_returns.rolling(window=20).corr(volume_returns)
        
        # Gap analysis
        features['gap_pct'] = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
        
        # Price position in range
        features['price_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-6)
        
        # ADX for trend strength
        features['adx_trend'] = talib.ADX(df['high'].values.astype(float), df['low'].values.astype(float), df['close'].values.astype(float), timeperiod=14) # type: ignore
        
        # Average true range percentage
        features['atr_pct'] = df['atr'] / df['close']
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_all_features(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Extract all 40 features"""
        self.logger.info(f"Extracting features for {symbol} {timeframe}")
        
        # Load OHLCV data
        df = self.load_ohlcv(symbol, timeframe)
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # Extract different feature groups
        downtrend_features = self.extract_downtrend_features(df.copy())
        uptrend_features = self.extract_uptrend_features(df.copy())
        consistency_features = self.extract_consistency_features(df.copy())
        
        # Combine all features
        features = pd.concat([
            features,
            downtrend_features,
            uptrend_features,
            consistency_features
        ], axis=1)
        
        # Forward fill and drop remaining NaN values
        features = features.ffill().dropna()
        
        # Ensure we have exactly 40 features
        assert len(features.columns) == 40, f"Expected 40 features, got {len(features.columns)}"
        
        return features
    
    def save_features(self, features: pd.DataFrame, output_dir: str, symbol: str, timeframe: str):
        """Save extracted features to disk"""
        output_path = Path(output_dir) / symbol / f"features_{timeframe}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)
        self.logger.info(f"Saved features to {output_path}")

def extract_features(data_dir: str, output_dir: str, symbol: str, timeframe: str):
    """Extract and save features for a given symbol and timeframe"""
    extractor = ContextFeatureExtractor(data_dir)
    features = extractor.extract_all_features(symbol, timeframe)
    extractor.save_features(features, output_dir, symbol, timeframe)
    return features

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract trading features from OHLCV data')
    parser.add_argument('--data-dir', type=str, default='data/processed/btc_usdt_kst/resampled_ohlcv',
                       help='Directory containing OHLCV data')
    parser.add_argument('--output-dir', type=str, default='data/processed/features',
                       help='Directory to save extracted features')
    parser.add_argument('--symbol', type=str, default='btc_usdt',
                       help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='1d',
                       help='Timeframe for feature extraction')
    
    args = parser.parse_args()
    extract_features(args.data_dir, args.output_dir, args.symbol, args.timeframe)
