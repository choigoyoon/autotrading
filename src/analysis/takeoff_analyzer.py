# src/analysis/takeoff_analyzer.py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, cast
import numpy as np
from scipy.signal import find_peaks

class TakeoffAnalyzer:
    """
    "수익 이륙 후 추가 상승 vs 되돌림"을 분석하는 시스템.

    수익률이 특정 임계점을 돌파한 '이륙 시점(takeoff_point)'을 기준으로,
    이후의 움직임이 추가 상승(Non-Reversion)인지 하락 반전(Reversion)인지를
    분석하고 예측하는 것을 목표로 합니다.

    핵심 질문: "왜 어떤 수익은 계속되고, 어떤 수익은 되돌아오는가?"
    """

    def __init__(self, config: Dict[str, Any]):
        """
        분석기 초기화
        """
        self.config = config
        raw_data_path = config.get("raw_data_path")
        if not raw_data_path or not isinstance(raw_data_path, (str, Path)):
            raise ValueError("'raw_data_path'가 config에 올바르게 지정되어 있지 않습니다.")
        self.raw_data_path = Path(raw_data_path)
        try:
            self.ohlcv_data = self._load_ohlcv_data()
            if self.ohlcv_data.empty:
                raise ValueError("OHLCV 데이터가 비어 있습니다. 파일 경로를 확인하세요.")
        except Exception as e:
            print(f"OHLCV 데이터 로드 중 오류 발생: {e}")
            self.ohlcv_data = pd.DataFrame()  # 빈 데이터프레임으로 초기화
            raise

    def _load_ohlcv_data(self) -> pd.DataFrame:
        """분석에 사용할 전체 OHLCV 데이터를 로드합니다."""
        print(f"Loading raw OHLCV data from {self.raw_data_path}...")
        df = pd.read_parquet(self.raw_data_path)
        df.index = pd.to_datetime(df.index)
        print("OHLCV data loaded successfully.")
        return df

    def load_trade_logs(self) -> pd.DataFrame:
        """Holding 및 Breakeven 거래 로그를 로드하고 통합합니다."""
        holding_path_str = self.config.get("holding_trades_path")
        breakeven_path_str = self.config.get("breakeven_trades_path")

        if not holding_path_str or not breakeven_path_str:
            raise ValueError("설정 파일에 'holding_trades_path' 또는 'breakeven_trades_path'가 필요합니다.")

        holding_path = Path(holding_path_str)
        breakeven_path = Path(breakeven_path_str)

        if not holding_path.exists() or not breakeven_path.exists():
            raise FileNotFoundError("거래 로그 파일 중 하나 이상을 찾을 수 없습니다.")

        holding_df = pd.read_csv(holding_path)
        breakeven_df = pd.read_csv(breakeven_path)
        
        holding_df['reversion_type'] = 'Holding'
        breakeven_df['reversion_type'] = 'Breakeven'
        
        combined_df = pd.concat([holding_df, breakeven_df], ignore_index=True)
        combined_df['진입시점'] = pd.to_datetime(combined_df['진입시점'])
        return combined_df

    def find_takeoff_points(self, trade_logs: pd.DataFrame, profit_threshold: float) -> pd.DataFrame:
        """
        각 거래에 대해 수익률이 처음으로 임계값을 돌파하는 '이륙 시점'을 찾습니다.

        Args:
            trade_logs (pd.DataFrame): 분석할 거래 로그
            profit_threshold (float): 수익률 임계값 (예: 0.2 for 20%)

        Returns:
            pd.DataFrame: 'takeoff_point'와 'takeoff_price'가 추가된 데이터프레임
        """
        print(f"Finding takeoff points with threshold {profit_threshold*100}%...")
        takeoff_points = []
        for _, trade in trade_logs.iterrows():
            entry_price = trade['진입가격']
            entry_time = trade['진입시점']
            signal_type = trade['신호타입']

            future_data = self.ohlcv_data[self.ohlcv_data.index > entry_time]

            target_price = 0
            if signal_type == 'Buy':
                target_price = entry_price * (1 + profit_threshold)
            elif signal_type == 'Sell':
                target_price = entry_price * (1 - profit_threshold)
            else:
                continue

            takeoff_candle = None
            for idx, candle in future_data.iterrows():
                if signal_type == 'Buy' and candle['high'] >= target_price:
                    takeoff_candle = candle
                    break
                elif signal_type == 'Sell' and candle['low'] <= target_price:
                    takeoff_candle = candle
                    break
            
            if takeoff_candle is not None:
                takeoff_time = takeoff_candle.name
                
                trade_info = trade.to_dict()
                trade_info['takeoff_timestamp'] = takeoff_time
                trade_info['takeoff_price'] = target_price 
                trade_info['takeoff_candle_high'] = takeoff_candle['high']
                trade_info['takeoff_candle_low'] = takeoff_candle['low']
                takeoff_points.append(trade_info)
        
        print(f"Found {len(takeoff_points)} takeoff points.")
        return pd.DataFrame(takeoff_points)

    def label_post_takeoff_movement(self, takeoff_data: pd.DataFrame) -> pd.DataFrame:
        """
        '이륙 시점' 이후의 움직임을 'Non-Reversion' 또는 'Reversion'으로 레이블링합니다.
        
        - Holding 거래 -> Non-Reversion (추가 상승)
        - Breakeven 거래 -> Reversion (하락 반전)

        Args:
            takeoff_data (pd.DataFrame): 'takeoff_point' 정보가 포함된 데이터

        Returns:
            pd.DataFrame: 'post_takeoff_label'이 추가된 데이터프레임
        """
        print("Labeling post-takeoff movements...")
        if 'reversion_type' not in takeoff_data.columns:
            print("Warning: 'reversion_type' column not found. Cannot label post-takeoff movement.")
            return takeoff_data

        def assign_label(row):
            if row['reversion_type'] == 'Holding':
                return 'Non-Reversion'
            elif row['reversion_type'] == 'Breakeven':
                return 'Reversion'
            return 'Unknown'

        takeoff_data['post_takeoff_label'] = takeoff_data.apply(assign_label, axis=1)
        print("Post-takeoff labeling complete.")
        return takeoff_data

    def _analyze_volume_trend(self, volume_series: pd.Series, lookback: int = 10) -> str:
        """거래량의 최근 추세를 분석합니다."""
        if len(volume_series) < lookback:
            return 'stable'
        
        recent_volumes = volume_series.tail(lookback)
        x = np.arange(len(recent_volumes))
        y = cast(np.ndarray, recent_volumes.values)
        slope = np.polyfit(x, y, 1)[0]
        
        normalized_slope = slope / (y.mean() + 1e-9)
        
        if normalized_slope > 0.1:
            return 'increasing'
        elif normalized_slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """주어진 데이터의 마지막 시점의 거래량 패턴을 분석합니다."""
        volume_window = self.config.get('volume_window', 20)
        
        data['volume_ma'] = data['volume'].rolling(window=volume_window).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_ma'] + 1e-9)
        
        spike_threshold = self.config.get('volume_spike_threshold', 2.0)
        dry_up_threshold = self.config.get('volume_dry_up_threshold', 0.5)
        data['volume_spike'] = data['volume_ratio'] > spike_threshold
        data['volume_dry_up'] = data['volume_ratio'] < dry_up_threshold

        data['volume_trend'] = data['volume'].rolling(window=10).apply(
            lambda x: self._analyze_volume_trend(x), raw=False
        ).fillna('stable')

        price_change = data['close'].pct_change().abs()
        data['smart_money_ratio'] = price_change / (data['volume_ratio'] + 1e-9)
        
        last_row = data.iloc[-1]
        score = 0
        if last_row['volume_spike']: score += 40
        if last_row['volume_trend'] == 'increasing': score += 30
        score += min(last_row['volume_ratio'] * 10, 30)
        
        return {
            'volume_score': score,
            'volume_ratio': last_row['volume_ratio'],
            'volume_spike': last_row['volume_spike'],
            'volume_trend': last_row['volume_trend'],
        }

    def _quantify_divergence_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """주어진 데이터의 마지막 시점의 MACD 다이버전스 강도를 정량화합니다."""
        close = data['close']
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        lookback = min(self.config.get('divergence_lookback', 50), len(data))
        if lookback < 15: return {}

        subset_hist = macd_hist.tail(lookback)
        subset_price = data['close'].tail(lookback)
        
        macd_peaks_indices, _ = find_peaks(subset_hist, distance=5)
        macd_valleys_indices, _ = find_peaks(-subset_hist, distance=5)

        macd_peaks = cast(np.ndarray, macd_peaks_indices)
        macd_valleys = cast(np.ndarray, macd_valleys_indices)

        best_score, best_duration, best_type = 0, 0, 'none'
        
        # Bearish Divergence
        if len(macd_peaks) >= 2:
            for i in range(len(macd_peaks) - 1):
                for j in range(i + 1, len(macd_peaks)):
                    p1_idx, p2_idx = macd_peaks[i], macd_peaks[j]
                    
                    price_p1 = subset_price.iloc[p1_idx]
                    price_p2 = subset_price.iloc[p2_idx]
                    macd_p1 = subset_hist.iloc[p1_idx]
                    macd_p2 = subset_hist.iloc[p2_idx]

                    if price_p2 > price_p1 and macd_p2 < macd_p1:
                        duration = p2_idx - p1_idx
                        price_change_pct = (price_p2 / price_p1 - 1) * 100
                        macd_change_pct = abs(macd_p2 / macd_p1 - 1) * 100
                        
                        score = (price_change_pct * 0.4 + macd_change_pct * 0.4 + duration * 0.2)
                        if score > best_score:
                            best_score, best_duration, best_type = score, duration, 'bearish'
        
        # Bullish Divergence
        if len(macd_valleys) >= 2:
            for i in range(len(macd_valleys) - 1):
                for j in range(i + 1, len(macd_valleys)):
                    v1_idx, v2_idx = macd_valleys[i], macd_valleys[j]

                    price_v1 = subset_price.iloc[v1_idx]
                    price_v2 = subset_price.iloc[v2_idx]
                    macd_v1 = subset_hist.iloc[v1_idx]
                    macd_v2 = subset_hist.iloc[v2_idx]

                    if price_v2 < price_v1 and macd_v2 > macd_v1:
                        duration = v2_idx - v1_idx
                        price_change_pct = (price_v1 / price_v2 - 1) * 100
                        macd_change_pct = abs(macd_v2 / macd_v1 - 1) * 100
                        
                        score = (price_change_pct * 0.4 + macd_change_pct * 0.4 + duration * 0.2)
                        if score > best_score:
                            best_score, best_duration, best_type = score, duration, 'bullish'

        strength_level = 'none'
        if best_score >= 90: strength_level = 'very_strong'
        elif best_score >= 70: strength_level = 'strong'
        elif best_score >= 50: strength_level = 'medium'
        elif best_score > 0: strength_level = 'weak'

        return {
            'divergence_score': best_score,
            'divergence_duration': best_duration,
            'divergence_type': best_type,
            'divergence_strength_level': strength_level
        }

    def analyze_features_at_takeoff(self, labeled_takeoff_data: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
        """
        레이블링된 각 '이륙 시점'의 시장 특징을 분석합니다.
        """
        print("Analyzing market features at each takeoff point...")
        
        feature_results = []
        for _, row in labeled_takeoff_data.iterrows():
            takeoff_time = row['takeoff_timestamp']
            
            # 이륙 시점 이전 데이터 추출
            # searchsorted가 스칼라 입력에 대해 int를 반환해야 하지만, pyright는 더 넓은 타입을 추론할 수 있습니다.
            # 반환값을 int로 명시적으로 캐스팅하여 타입 오류를 해결합니다.
            context_data_end_index = cast(int, self.ohlcv_data.index.searchsorted(takeoff_time, side='right'))
            context_data_start_index = np.maximum(0, context_data_end_index - window_size)
            context_data = self.ohlcv_data.iloc[int(context_data_start_index):int(context_data_end_index)].copy()

            if len(context_data) < 15: continue
            
            # 특징 분석
            volume_features = self._analyze_volume_patterns(context_data)
            divergence_features = self._quantify_divergence_strength(context_data)

            # 결과 취합
            combined_features = row.to_dict()
            combined_features.update(volume_features)
            combined_features.update(divergence_features)
            feature_results.append(combined_features)
            
        return pd.DataFrame(feature_results)
        
    def run_analysis(self, profit_threshold: float = 0.2, window_size: int = 100):
        """
        전체 분석 파이프라인을 실행합니다.
        """
        # 1. 거래 로그 로드
        trade_logs = self.load_trade_logs()

        # 2. 이륙 시점 탐색
        takeoff_data = self.find_takeoff_points(trade_logs, profit_threshold)
        if takeoff_data.empty:
            print("No takeoff points found. Aborting analysis.")
            return pd.DataFrame()
        
        # 3. 이륙 후 움직임 레이블링
        labeled_data = self.label_post_takeoff_movement(takeoff_data)

        # 4. 이륙 시점 특징 분석
        featured_data = self.analyze_features_at_takeoff(labeled_data, window_size)

        print("\n--- Takeoff Analysis Summary ---")
        if not featured_data.empty:
            print(featured_data.groupby('post_takeoff_label').size())

            print("\n--- Volume Analysis at Takeoff ---")
            if 'volume_score' in featured_data.columns:
                 print(featured_data.groupby('post_takeoff_label')['volume_score'].describe())
            
            print("\n--- Divergence Analysis at Takeoff ---")
            if 'divergence_score' in featured_data.columns:
                 print(featured_data.groupby('post_takeoff_label')['divergence_score'].describe())
        
        return featured_data

if __name__ == '__main__':
    # Force re-analysis by adding a comment
    config = {
        "raw_data_path": "data/rwa/parquet_converted/btc_kst_1min.parquet",
        "holding_trades_path": "results/top_trades/top_100_holding_trades.csv",
        "breakeven_trades_path": "results/top_trades/top_100_breakeven_trades.csv",
        "volume_window": 20,
        "volume_spike_threshold": 2.0,
        "volume_dry_up_threshold": 0.5,
        "divergence_lookback": 50,
    }

    analyzer = TakeoffAnalyzer(config)
    analysis_results_df = analyzer.run_analysis(profit_threshold=0.05, window_size=100) # 임계값을 5%로 낮춰서 테스트
    
    print("\n--- Analysis Results (Sample) ---")
    print(analysis_results_df.head()) 