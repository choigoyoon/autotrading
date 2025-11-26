"""
기술적 지표 계산 모듈
다양한 기술적 지표를 계산하여 나우캐스트 모델의 특징으로 사용합니다.
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""

    def __init__(self, df):
        """
        Args:
            df: OHLCV 데이터프레임 (columns: open, high, low, close, volume)
        """
        self.df = df.copy()

    def add_all_indicators(self):
        """모든 기술적 지표를 추가"""
        self.add_trend_indicators()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_price_features()
        return self.df

    def add_trend_indicators(self):
        """추세 지표 추가"""
        # 이동평균선
        for period in [7, 14, 21, 50, 100, 200]:
            sma = SMAIndicator(close=self.df['close'], window=period)
            self.df[f'sma_{period}'] = sma.sma_indicator()

            ema = EMAIndicator(close=self.df['close'], window=period)
            self.df[f'ema_{period}'] = ema.ema_indicator()

        # MACD
        macd = MACD(close=self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()

    def add_momentum_indicators(self):
        """모멘텀 지표 추가"""
        # RSI
        for period in [6, 12, 24]:
            rsi = RSIIndicator(close=self.df['close'], window=period)
            self.df[f'rsi_{period}'] = rsi.rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close']
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()

        # ROC (Rate of Change)
        for period in [6, 12, 24]:
            self.df[f'roc_{period}'] = self.df['close'].pct_change(period) * 100

    def add_volatility_indicators(self):
        """변동성 지표 추가"""
        # Bollinger Bands
        bb = BollingerBands(close=self.df['close'])
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_mid'] = bb.bollinger_mavg()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_width'] = bb.bollinger_wband()
        self.df['bb_pct'] = bb.bollinger_pband()

        # ATR (Average True Range)
        atr = AverageTrueRange(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close']
        )
        self.df['atr'] = atr.average_true_range()

        # Historical Volatility
        for period in [6, 12, 24]:
            returns = np.log(self.df['close'] / self.df['close'].shift(1))
            self.df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)

    def add_volume_indicators(self):
        """거래량 지표 추가"""
        # OBV (On-Balance Volume)
        obv = OnBalanceVolumeIndicator(
            close=self.df['close'],
            volume=self.df['volume']
        )
        self.df['obv'] = obv.on_balance_volume()

        # CMF (Chaikin Money Flow)
        cmf = ChaikinMoneyFlowIndicator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            volume=self.df['volume']
        )
        self.df['cmf'] = cmf.chaikin_money_flow()

        # Volume MA
        for period in [7, 14, 21]:
            self.df[f'volume_ma_{period}'] = self.df['volume'].rolling(window=period).mean()

        # Volume Ratio
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume'].rolling(window=20).mean()

    def add_price_features(self):
        """가격 기반 특징 추가"""
        # 가격 변화율
        for period in [1, 3, 6, 12, 24]:
            self.df[f'price_change_{period}'] = self.df['close'].pct_change(period)

        # High-Low 범위
        self.df['hl_ratio'] = (self.df['high'] - self.df['low']) / self.df['close']

        # Close 위치 (High-Low 범위 내)
        self.df['close_position'] = (self.df['close'] - self.df['low']) / (self.df['high'] - self.df['low'])

        # 갭
        self.df['gap'] = (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1)

        # 캔들 실체
        self.df['body'] = (self.df['close'] - self.df['open']) / self.df['open']

        # 상승/하락 여부
        self.df['is_green'] = (self.df['close'] > self.df['open']).astype(int)

    def get_feature_columns(self):
        """특징 컬럼 목록 반환 (OHLCV 제외)"""
        exclude_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in exclude_cols]


def calculate_indicators(df):
    """
    편의 함수: DataFrame에 모든 기술적 지표를 추가

    Args:
        df: OHLCV DataFrame

    Returns:
        지표가 추가된 DataFrame
    """
    ti = TechnicalIndicators(df)
    return ti.add_all_indicators()


if __name__ == "__main__":
    # 테스트
    import os

    data_dir = "data"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"테스트용 파일: {csv_files[0]}")
            df = pd.read_csv(os.path.join(data_dir, csv_files[0]))

            print(f"\n원본 데이터 shape: {df.shape}")
            print(f"원본 컬럼: {df.columns.tolist()}")

            # 지표 추가
            df_with_indicators = calculate_indicators(df)

            print(f"\n지표 추가 후 shape: {df_with_indicators.shape}")
            print(f"\n추가된 지표 수: {df_with_indicators.shape[1] - df.shape[1]}")

            # 일부 지표 출력
            indicator_cols = TechnicalIndicators(df).get_feature_columns()
            print(f"\n지표 컬럼 샘플 (처음 10개):")
            for col in indicator_cols[:10]:
                print(f"  - {col}")

            print(f"\n마지막 5행:")
            print(df_with_indicators[['close'] + indicator_cols[:5]].tail())
        else:
            print("CSV 파일이 없습니다. 먼저 데이터를 수집하세요.")
    else:
        print("data 디렉토리가 없습니다.")
