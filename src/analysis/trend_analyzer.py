import pandas as pd
from typing import Dict, List, Any
import pandas_ta as ta # type: ignore

class TrendAnalyzer:
    """
    "흐름 우선" 6단계 매매 결정 프로세스의 첫 단계를 수행합니다.
    다중 타임프레임에 걸쳐 시장 추세를 분석하여 전체적인 시장의 '흐름'을 파악하고,
    이를 정량적인 점수로 변환합니다.
    """

    def __init__(self, ohlcv_data: Dict[str, pd.DataFrame], config: Dict[str, Any]):
        """
        TrendAnalyzer를 초기화합니다.

        Args:
            ohlcv_data (Dict[str, pd.DataFrame]): 키는 타임프레임(예: '1d', '4h'),
                                                  값은 OHLCV 데이터프레임인 딕셔너리.
            config (Dict[str, Any]): 추세 분석에 사용될 설정값.
                                     예: {'timeframes': {'long': '1d', 'mid': '4h', 'short': '1h'},
                                          'ma_periods': {'long': 50, 'mid': 20, 'short': 10},
                                          'sideways_threshold': 0.005}
        """
        self.ohlcv_data = ohlcv_data
        self.config = config
        self.analysis_results: Dict[str, Any] = {}

    def _analyze_single_timeframe_trend(self, df: pd.DataFrame, period: int) -> str:
        """
        단일 타임프레임의 추세를 이동평균선(EMA)을 기준으로 분석합니다.
        가격이 EMA 위에 있으면 'uptrend', 아래에 있으면 'downtrend'로 판단합니다.

        Args:
            df (pd.DataFrame): 분석할 OHLCV 데이터프레임.
            period (int): EMA 계산에 사용할 기간.

        Returns:
            str: 'uptrend', 'downtrend', 또는 'sideways'.
        """
        if df.empty or len(df) < period:
            return "unknown"

        # EMA 계산
        ema = df.ta.ema(length=period)
        if ema is None or ema.dropna().empty:
            return "unknown"

        last_close = df['close'].iloc[-1]
        last_ema = ema.iloc[-1]

        # 추세 판단
        sideways_threshold = self.config.get('sideways_threshold', 0.005)
        if abs(last_close - last_ema) / last_ema <= sideways_threshold:
            return "sideways"
        elif last_close > last_ema:
            return "uptrend"
        else:
            return "downtrend"

    def analyze_trends(self) -> Dict[str, str]:
        """
        설정된 장기/중기/단기 타임프레임 각각의 추세를 분석합니다.

        Returns:
            Dict[str, str]: 타임프레임별 추세 방향 ('uptrend', 'downtrend', 'sideways')
        """
        trends = {}
        timeframe_mapping = self.config.get('timeframes', {})
        ma_periods = self.config.get('ma_periods', {})

        for alias, timeframe_str in timeframe_mapping.items():
            if timeframe_str in self.ohlcv_data and alias in ma_periods:
                df = self.ohlcv_data[timeframe_str]
                period = ma_periods[alias]
                trends[alias] = self._analyze_single_timeframe_trend(df, period)
            else:
                trends[alias] = "data_missing"

        self.analysis_results['trends'] = trends
        return trends

    def detect_trend_conflicts(self) -> List[str]:
        """
        장기, 중기, 단기 추세 간의 충돌을 감지합니다.

        예: 장기 추세는 'uptrend'인데 단기 추세가 'downtrend'인 경우.

        Returns:
            List[str]: 감지된 충돌에 대한 설명 문자열 리스트.
        """
        conflicts = []
        trends = self.analysis_results.get('trends', {})
        
        long_trend = trends.get('long')
        mid_trend = trends.get('mid')
        short_trend = trends.get('short')

        if long_trend == 'uptrend' and short_trend == 'downtrend':
            conflicts.append("Long-term uptrend vs. Short-term downtrend")
        
        if long_trend == 'downtrend' and short_trend == 'uptrend':
            conflicts.append("Long-term downtrend vs. Short-term uptrend (potential dead cat bounce)")

        # 중기-단기 충돌도 확인할 수 있습니다.
        if mid_trend == 'uptrend' and short_trend == 'downtrend':
            conflicts.append("Mid-term uptrend vs. Short-term downtrend")

        if mid_trend == 'downtrend' and short_trend == 'uptrend':
            conflicts.append("Mid-term downtrend vs. Short-term uptrend")

        self.analysis_results['conflicts'] = conflicts
        return conflicts

    def calculate_overall_flow_score(self) -> float:
        """
        전체 시장 흐름을 나타내는 종합 점수를 계산합니다.
        장기 추세에 더 높은 가중치를 부여합니다.

        Returns:
            float: -1.0 (강한 하락 흐름) ~ +1.0 (강한 상승 흐름) 사이의 점수.
        """
        score = 0.0
        
        weights = self.config.get('weights', {'long': 0.5, 'mid': 0.3, 'short': 0.2})
        trend_scores = {'uptrend': 1, 'sideways': 0, 'downtrend': -1}
        
        trends = self.analysis_results.get('trends', {})
        for timeframe_alias, trend in trends.items():
            score += trend_scores.get(trend, 0) * weights.get(timeframe_alias, 0)

        self.analysis_results['overall_flow_score'] = score
        return score

    def run_analysis(self) -> Dict[str, Any]:
        """
        전체 흐름 분석 파이프라인을 실행합니다.

        Returns:
            Dict[str, Any]: 분석 결과 요약 (타임프레임별 추세, 충돌, 종합 점수 등)
        """
        self.analyze_trends()
        self.detect_trend_conflicts()
        self.calculate_overall_flow_score()
        return self.analysis_results

if __name__ == '__main__':
    # 이 모듈을 테스트하기 위한 예시 코드

    # 1. 가상 데이터 생성
    # 1d: 완만한 상승 추세
    # 4h: 횡보
    # 1h: 급격한 하락 추세
    base_data = pd.Series(range(100, 200))
    dummy_data = {
        '1d': pd.DataFrame({'close': base_data + pd.Series(range(100)) * 0.5}),
        '4h': pd.DataFrame({'close': base_data.iloc[-50:].reset_index(drop=True)}),
        '1h': pd.DataFrame({'close': base_data.iloc[-30:].reset_index(drop=True) - pd.Series(range(30)) * 2})
    }

    # 2. 설정 정의
    dummy_config = {
        'timeframes': {
            'long': '1d',
            'mid': '4h',
            'short': '1h'
        },
        'ma_periods': {
            'long': 50,
            'mid': 20,
            'short': 10
        },
        'weights': { # 점수 계산 시 가중치
            'long': 0.5,
            'mid': 0.3,
            'short': 0.2
        },
        'sideways_threshold': 0.02 # 테스트를 위해 횡보 기준을 2%로 설정
    }

    # 3. 분석기 생성 및 실행
    analyzer = TrendAnalyzer(ohlcv_data=dummy_data, config=dummy_config)
    analysis_summary = analyzer.run_analysis()

    # 4. 결과 출력
    import json
    print("Trend Analysis Summary:")
    print(json.dumps(analysis_summary, indent=2))
    
    # 예상 결과:
    # {
    #   "trends": {
    #     "long": "uptrend",
    #     "mid": "sideways",
    #     "short": "downtrend"
    #   },
    #   "conflicts": [
    #     "Long-term uptrend vs. Short-term downtrend",
    #     "Mid-term uptrend vs. Short-term downtrend"  <-- 이 부분은 mid가 sideways면 안나올 수 있음
    #   ],
    #   "overall_flow_score": 0.3
    # } 