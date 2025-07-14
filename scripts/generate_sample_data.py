import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 랜덤 시드 설정
np.random.seed(42)

# 디렉토리 생성
data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

# 시간대 생성
end_time = datetime.now()
start_time = end_time - timedelta(days=365)  # 1년치 데이터

def generate_ohlcv(tf_minutes, start, end):
    """OHLCV 데이터 생성"""
    delta = timedelta(minutes=tf_minutes)
    timestamps = pd.date_range(start=start, end=end, freq=f"{tf_minutes}min")
    
    # 랜덤 워크로 가격 생성
    n = len(timestamps)
    log_returns = np.random.normal(0.0001, 0.01, n)
    log_prices = np.cumsum(log_returns) + np.log(50000)  # 시작 가격: 50,000
    prices = np.exp(log_prices)
    
    # OHLC 생성
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = prices * (1 + np.random.normal(0, 0.002, n))
    close = prices * (1 + np.random.normal(0, 0.002, n))
    
    # 볼륨 생성
    volume = np.random.lognormal(mean=10, sigma=1, size=n)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=timestamps)
    
    return df

# 라벨 생성 함수
def generate_labels(ohlcv_df, forward_bars=10):
    """가격 변동 기반 라벨 생성"""
    close = ohlcv_df['close']
    future_returns = close.pct_change(forward_bars).shift(-forward_bars)
    
    # 방향성 라벨 (상승: 1, 하락: 0, 보합: 2)
    direction = np.where(
        future_returns > 0.002,  # 0.2% 이상 상승
        1,  # 매수
        np.where(
            future_returns < -0.002,  # 0.2% 이상 하락
            0,  # 매도
            2   # 보유
        )
    )
    
    # 신호 강도 (변동성 기반)
    volatility = ohlcv_df['high'] / ohlcv_df['low'] - 1
    signal_strength = np.minimum(volatility * 100, 1.0)  # 0~1 사이로 정규화
    
    # 지속 기간 (랜덤)
    duration = np.random.randint(1, 24*6, size=len(ohlcv_df))  # 1~6시간
    
    # 신뢰도 (랜덤)
    confidence = np.random.uniform(0.5, 1.0, size=len(ohlcv_df))
    
    # 데이터프레임 생성
    labels = pd.DataFrame({
        'direction': direction,
        'magnitude': signal_strength,
        'duration': duration,
        'confidence': confidence,
        'returns': future_returns.fillna(0)
    }, index=ohlcv_df.index)
    
    return labels

# 피처 생성 함수
def generate_features(ohlcv_df):
    """기술적 지표 기반 피처 생성"""
    df = ohlcv_df.copy()
    
    # 단순 이동평균
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    df['bb_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 거래량 가중 이동평균
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # 변동성
    df['atr'] = df['high'] - df['low']
    df['atr'] = df['atr'].rolling(window=14).mean()
    
    # 전일 대비 변동률
    df['returns'] = df['close'].pct_change()
    
    # NaN 값 제거
    df = df.dropna()
    
    return df

# 메인 함수
def main():
    print("Generating sample data...")
    
    # 5분봉 데이터 생성
    print("Generating 5m OHLCV data...")
    ohlcv_5m = generate_ohlcv(5, start_time, end_time)
    
    # 피처 생성
    print("Generating features...")
    features_5m = generate_features(ohlcv_5m)
    
    # 라벨 생성
    print("Generating labels...")
    labels_5m = generate_labels(ohlcv_5m)
    
    # 데이터 저장
    print("Saving data...")
    features_5m.to_parquet(data_dir / "features_5m.parquet")
    labels_5m.to_parquet(data_dir / "labels_5m.parquet")
    
    # 다른 시간대 생성 (예: 15분, 1시간, 4시간, 1일)
    timeframes = [15, 60, 240, 1440]  # 분 단위
    
    for tf in timeframes:
        tf_str = f"{tf}m" if tf < 1440 else "1d"
        print(f"Generating {tf_str} OHLCV data...")
        
        # OHLCV 생성
        ohlcv = generate_ohlcv(tf, start_time, end_time)
        
        # 피처 생성
        features = generate_features(ohlcv)
        
        # 라벨 생성
        labels = generate_labels(ohlcv)
        
        # 데이터 저장
        features.to_parquet(data_dir / f"features_{tf_str}.parquet")
        labels.to_parquet(data_dir / f"labels_{tf_str}.parquet")
    
    print("Sample data generation completed!")

if __name__ == "__main__":
    main()
