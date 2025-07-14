import pandas as pd
from typing import List, Dict
import logging

class BullishDivergenceDetector:
    """
    '진짜 스윙 저점' (`swing_low`)을 기반으로 강세 다이버전스를 탐지하는 최적화된 클래스.
    """
    def __init__(self, 
                 min_price_decline_pct: float = 0.01,
                 min_macd_improvement_pct: float = 0.05,
                 max_lookback_candles: int = 100):
        """
        Args:
            min_price_decline_pct (float): 두 저점 사이의 최소 가격 하락률.
            min_macd_improvement_pct (float): 두 저점 사이의 최소 MACD 상승률.
            max_lookback_candles (int): 다이버전스를 찾기 위해 두 저점 사이의 최대 캔들 수.
        """
        self.min_price_decline_pct = min_price_decline_pct
        self.min_macd_improvement_pct = min_macd_improvement_pct
        self.max_lookback_candles = max_lookback_candles
        self.logger = logging.getLogger(__name__)

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        swing_low 컬럼을 사용하여 강세 다이버전스를 탐지합니다.

        Args:
            df (pd.DataFrame): 'close', 'macd_line', 'swing_low' 컬럼을 포함해야 함.

        Returns:
            List[Dict]: 감지된 다이버전스 신호 목록.
        """
        if not all(col in df.columns for col in ['close', 'macd_line', 'swing_low']):
            raise ValueError("입력 DataFrame에 'close', 'macd_line', 'swing_low' 컬럼이 필요합니다.")
            
        signals = []
        
        # '진짜 스윙 저점'의 인덱스와 위치를 미리 추출
        swing_low_points = df[df['swing_low'] == 1]
        
        if len(swing_low_points) < 2:
            self.logger.info("분석에 필요한 스윙 저점이 2개 미만입니다.")
            return signals

        self.logger.info(f"총 {len(swing_low_points)}개의 스윙 저점을 기반으로 다이버전스 분석 시작...")
        
        # 스윙 저점 리스트를 순회하며 모든 가능한 쌍을 비교
        for i in range(len(swing_low_points)):
            for j in range(i + 1, len(swing_low_points)):
                l1 = swing_low_points.iloc[i]
                l2 = swing_low_points.iloc[j]

                # 원본 DataFrame에서의 캔들 위치 차이 계산
                pos1 = df.index.get_loc(l1.name)
                pos2 = df.index.get_loc(l2.name)
                
                if not (isinstance(pos1, int) and isinstance(pos2, int)):
                     # get_loc이 슬라이스를 반환하는 경우를 방지 (중복 인덱스 등)
                    continue

                if (pos2 - pos1) > self.max_lookback_candles:
                    continue

                price_l1, price_l2 = l1['close'], l2['close']
                macd_l1, macd_l2 = l1['macd_line'], l2['macd_line']
                
                # 1. 가격 조건: l2의 가격이 l1보다 낮아야 함
                price_decline_pct = (price_l2 - price_l1) / price_l1
                if price_decline_pct >= -self.min_price_decline_pct:
                    continue

                # 2. MACD 조건: l2의 MACD가 l1보다 높아야 함
                epsilon = 1e-9
                macd_improvement_pct = (macd_l2 - macd_l1) / (abs(macd_l1) + epsilon)
                if macd_improvement_pct < self.min_macd_improvement_pct:
                    continue
                    
                strength = min(abs(price_decline_pct) + macd_improvement_pct, 1.0)
                
                signals.append({
                    'datetime': l2.name,
                    'type': 'bullish_divergence',
                    'price_lows': ((l1.name, price_l1), (l2.name, price_l2)),
                    'macd_lows': ((l1.name, macd_l1), (l2.name, macd_l2)),
                    'strength': strength
                })
        
        self.logger.info(f"✅ 총 {len(signals)}개의 강세 다이버전스 신호를 감지했습니다.")
        return signals 