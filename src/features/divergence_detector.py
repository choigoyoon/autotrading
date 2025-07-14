from __future__ import annotations
import pandas as pd
import pandas_ta as ta # type: ignore
from typing import List, Dict, Optional, Any, Tuple
import logging
from scipy.signal import find_peaks
import numpy as np

class DivergenceDetector:
    """
    가격과 MACD 히스토그램 사이의 다이버전스를 감지합니다.
    """

    def __init__(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        self.df = df.copy()
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if 'macd_hist' not in self.df.columns:
            self._calculate_macd()

    def _calculate_macd(self):
        """MACD 지표를 계산하고 데이터프레임에 추가합니다."""
        try:
            macd = self.df.ta.macd(fast=self.fast, slow=self.slow, signal=self.signal)
            if macd is not None:
                self.df['macd'] = macd[f'MACD_{self.fast}_{self.slow}_{self.signal}']
                self.df['macd_hist'] = macd[f'MACDh_{self.fast}_{self.slow}_{self.signal}']
                self.df['macd_signal'] = macd[f'MACDs_{self.fast}_{self.slow}_{self.signal}']
            else:
                raise ValueError("Pandas TA MACD calculation returned None.")
        except Exception as e:
            self.logger.error(f"MACD 계산 중 오류 발생: {e}")
            for col in ['macd', 'macd_hist', 'macd_signal']:
                if col not in self.df.columns:
                    self.df[col] = pd.NA

    def find_all_divergences(self) -> List[Dict[str, Any]]:
        return self.find_bullish_divergences() + self.find_bearish_divergences()

    def find_bullish_divergences(self, lookback: int = 60, prominence: float = 0.1, width: int = 5) -> List[Dict[str, Any]]:
        """
        Scipy를 사용하여 강세 다이버전스를 찾습니다.
        가격은 저점을 낮추고, MACD 히스토그램은 저점을 높이는 패턴을 찾습니다.
        """
        divergences = []
        price_lows = self.df['low'].values
        macd_hist = self.df['macd_hist'].fillna(0).values

        # 가격의 저점과 MACD 히스토그램의 저점(음수 극값)을 찾습니다.
        price_peaks, _ = find_peaks(-np.asarray(price_lows), prominence=prominence, width=(width,))
        macd_peaks, _ = find_peaks(-np.asarray(macd_hist), prominence=prominence, width=(width,))

        for i in range(len(price_peaks)):
            for j in range(i + 1, len(price_peaks)):
                p1_idx, p2_idx = price_peaks[i], price_peaks[j]

                # lookback 기간 내에 있는지 확인
                if p2_idx - p1_idx > lookback:
                    continue
                
                # 강세 다이버전스 조건 확인
                # 1. 가격 저점은 낮아짐
                # 2. MACD 저점은 높아짐
                if price_lows[p2_idx] < price_lows[p1_idx] and macd_hist[p2_idx] > macd_hist[p1_idx]:
                    # 두 가격 최저점 사이에서 MACD 최저점을 찾음
                    macd_low_points_in_range = [p for p in macd_peaks if p1_idx <= p <= p2_idx]
                    if len(macd_low_points_in_range) < 2:
                        continue

                    # 가장 가까운 MACD 저점을 찾아 페어로 만듬
                    m1_idx = min(macd_low_points_in_range, key=lambda p: abs(p - p1_idx))
                    m2_idx = min(macd_low_points_in_range, key=lambda p: abs(p - p2_idx))

                    if m1_idx >= m2_idx:
                        continue

                    if macd_hist[m2_idx] > macd_hist[m1_idx]:
                        divergences.append({
                            "type": "bullish",
                            "price_points": (p1_idx, p2_idx),
                            "macd_points": (m1_idx, m2_idx),
                            "price_values": (price_lows[p1_idx], price_lows[p2_idx]),
                            "macd_values": (macd_hist[m1_idx], macd_hist[m2_idx]),
                            "start_time": self.df.index[p1_idx],
                            "end_time": self.df.index[p2_idx]
                        })
        return divergences

    def find_bearish_divergences(self, lookback: int = 60, prominence: float = 0.1, width: int = 5) -> List[Dict[str, Any]]:
        """
        Scipy를 사용하여 약세 다이버전스를 찾습니다.
        가격은 고점을 높이고, MACD 히스토그램은 고점을 낮추는 패턴을 찾습니다.
        """
        divergences = []
        price_highs = self.df['high'].values
        macd_hist = self.df['macd_hist'].fillna(0).values

        # 가격의 고점과 MACD 히스토그램의 고점(양수 극값)을 찾습니다.
        price_peaks, _ = find_peaks(np.asarray(price_highs), prominence=prominence, width=(width,))
        macd_peaks, _ = find_peaks(np.asarray(macd_hist), prominence=prominence, width=(width,))

        for i in range(len(price_peaks)):
            for j in range(i + 1, len(price_peaks)):
                p1_idx, p2_idx = price_peaks[i], price_peaks[j]
                
                if p2_idx - p1_idx > lookback:
                    continue

                if price_highs[p2_idx] > price_highs[p1_idx] and macd_hist[p2_idx] < macd_hist[p1_idx]:
                    macd_high_points_in_range = [p for p in macd_peaks if p1_idx <= p <= p2_idx]
                    if len(macd_high_points_in_range) < 2:
                        continue

                    m1_idx = min(macd_high_points_in_range, key=lambda p: abs(p - p1_idx))
                    m2_idx = min(macd_high_points_in_range, key=lambda p: abs(p - p2_idx))
                
                    if m1_idx >= m2_idx:
                        continue

                    if macd_hist[m2_idx] < macd_hist[m1_idx]:
                        divergences.append({
                            "type": "bearish",
                            "price_points": (p1_idx, p2_idx),
                            "macd_points": (m1_idx, m2_idx),
                            "price_values": (price_highs[p1_idx], price_highs[p2_idx]),
                            "macd_values": (macd_hist[m1_idx], macd_hist[m2_idx]),
                            "start_time": self.df.index[p1_idx],
                            "end_time": self.df.index[p2_idx]
                        })
        return divergences


class DivergenceQualityAnalyzer:
    """
    발견된 MACD 다이버전스의 품질을 정량화하여 점수로 변환합니다.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        self.df = df
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "strength_weight": 0.4,
            "duration_weight": 0.3,
            "uniqueness_weight": 0.3,
            "min_duration_candles": 5,
            "max_duration_candles": 100
        }

    def _calculate_strength(self, divergence_info: Dict[str, Any]) -> float:
        """다이버전스 강도를 계산합니다."""
        price_change = (divergence_info['price_values'][1] - divergence_info['price_values'][0]) / divergence_info['price_values'][0]
        macd_change = (divergence_info['macd_values'][1] - divergence_info['macd_values'][0]) / abs(divergence_info['macd_values'][0]) if divergence_info['macd_values'][0] != 0 else 0
        
        if price_change >= 0: # 강세 다이버전스는 가격 하락이 전제
            return 0.0

        strength = abs(macd_change / price_change) if price_change != 0 else 0
        return min(strength / 10, 1.0) # 0과 1 사이로 정규화

    def _calculate_duration(self, divergence_info: Dict[str, Any]) -> float:
        """다이버전스 기간의 적절성을 계산합니다."""
        duration = divergence_info['price_points'][1] - divergence_info['price_points'][0]
        min_duration = self.config.get("min_duration_candles", 5)
        max_duration = self.config.get("max_duration_candles", 100)
        
        if min_duration <= duration <= max_duration:
            return 1.0
        elif duration < min_duration:
            return duration / min_duration
        else:
            return max(0, 1 - (duration - max_duration) / max_duration)

    def _calculate_uniqueness(self, divergence_info: Dict[str, Any]) -> float:
        """다이버전스 구간 내 저점의 특이성/차별성을 계산합니다."""
        p1_idx, p2_idx = divergence_info['price_points']
        m1_idx, m2_idx = divergence_info['macd_points']
        
        # 가격 저점 특이성
        price_segment = self.df['low'].iloc[p1_idx:p2_idx+1]
        if not price_segment.iloc[np.array([0, -1])].equals(price_segment.nsmallest(2)):
            return 0.0 # 두 저점이 가장 낮은 점이 아니면 점수 없음

        # MACD 저점 특이성
        macd_segment = self.df['macd_hist'].iloc[m1_idx:m2_idx+1]
        if not (macd_segment.iloc[-1] > macd_segment.iloc[0]):
             return 0.0
        
        # 두 번째 MACD 저점이 첫 번째 저점 이후 가장 높은 저점인지 확인
        if macd_segment.iloc[-1] < macd_segment.iloc[1:-1].min():
            return 0.2
            
        return 1.0

    def analyze_quality(self, divergence_info: Dict[str, Any]) -> float:
        price_points = divergence_info.get("price_points")
        if not price_points or not isinstance(price_points, tuple) or len(price_points) != 2:
            return 0.0

        strength = self._calculate_strength(divergence_info)
        duration = self._calculate_duration(divergence_info)
        uniqueness = self._calculate_uniqueness(divergence_info)

        score = (
            strength * self.config["strength_weight"] +
            duration * self.config["duration_weight"] +
            uniqueness * self.config["uniqueness_weight"]
        )
        return score


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    data = {
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100)),
        'close': [100] * 50 + [98] * 50, # 가격 하락
    }
    mock_df = pd.DataFrame(data)
    mock_df['low'] = mock_df['close']
    mock_df.set_index('date', inplace=True)
    
    detector = DivergenceDetector(mock_df, fast=12, slow=26, signal=9)
    # MACD 히스토그램을 인위적으로 만들어 다이버전스 조건 충족
    detector.df['macd_hist'] = [-0.5] * 50 + [-0.2] * 50 # 히스토그램 저점 상승

    bullish_signals = detector.find_bullish_divergences()
    
    print("="*50)
    print("Testing DivergenceDetector")
    print("="*50)
    if bullish_signals:
        print(f"\nFound {len(bullish_signals)} Bullish Divergences:")
        for signal in bullish_signals:
            print(signal)
    else:
        print("\nNo bullish divergences found.")

    print("\n" + "="*50)
    print("Testing DivergenceQualityAnalyzer")
    print("="*50)
    
    if bullish_signals:
        quality_analyzer = DivergenceQualityAnalyzer(detector.df)
        first_divergence = bullish_signals[0]
        quality_score = quality_analyzer.analyze_quality(first_divergence)

        print(f"\nAnalyzing first detected divergence:\n{first_divergence}")
        print(f"\nCalculated Quality Score: {quality_score:.4f}")
    else:
        print("\nNo divergences found to analyze quality.") 