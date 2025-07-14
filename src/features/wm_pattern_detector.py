import pandas as pd
from scipy.signal import find_peaks
from typing import List, Dict
import numpy as np

class WMPatternDetector:
    """
    MACD 데이터에서 W (Inverse Head & Shoulders) 및 M (Head & Shoulders) 패턴을 감지하는 클래스.
    """
    def __init__(self, 
                 min_peak_distance: int = 5, 
                 prominence_ratio: float = 0.02, 
                 symmetry_tolerance: float = 0.20):
        """
        WMPatternDetector를 초기화합니다.

        Args:
            min_peak_distance (int): 피크 사이의 최소 캔들 수.
            prominence_ratio (float): 전체 MACD 범위에 대한 피크의 최소 돌출 비율.
            symmetry_tolerance (float): 두 어깨 사이의 높이 비대칭 허용 오차.
        """
        self.min_peak_distance = min_peak_distance
        self.prominence_ratio = prominence_ratio
        self.symmetry_tolerance = symmetry_tolerance

    def _find_extrema(self, series: pd.Series) -> np.ndarray:
        """주어진 Series에서 극값(최대/최소)을 찾습니다."""
        prominence = (series.max() - series.min()) * self.prominence_ratio
        peaks, _ = find_peaks(series, distance=self.min_peak_distance, prominence=prominence)
        return peaks

    def _validate_w_structure(self, l1: float, l2: float, l3: float) -> bool:
        """W 패턴의 구조적 유효성을 검사합니다. (양수화된 값을 기준)"""
        # l1, l2, l3는 음수 MACD 값을 양수로 변환한 것이므로, 머리에 해당하는 l2가 가장 커야 합니다.
        head_is_deepest = l2 > l1 and l2 > l3
        
        # 머리의 깊이가 어깨보다 최소 10% 이상 깊은지 확인합니다.
        # l2가 0보다 클 때만 계산하여 0으로 나누는 오류를 방지합니다.
        min_head_depth = ((l2 - l1) / l2 > 0.10 and (l2 - l3) / l2 > 0.10) if l2 > 0 else False
        
        # 어깨의 대칭성을 확인합니다.
        shoulder_symmetry = abs(l1 - l3) / max(l1, l3) <= self.symmetry_tolerance if max(l1, l3) > 0 else True
        
        return head_is_deepest and min_head_depth and shoulder_symmetry

    def _calculate_bullish_divergence(self, df: pd.DataFrame, l1_idx: int, l2_idx: int, l3_idx: int) -> float:
        """가격과 MACD 간의 강세 다이버전스 점수를 계산합니다."""
        price_low_at_l1 = df['close'].iloc[l1_idx]
        price_low_at_l3 = df['close'].iloc[l3_idx]
        macd_low_at_l1 = df['macd_histogram'].iloc[l1_idx]
        macd_low_at_l3 = df['macd_histogram'].iloc[l3_idx]

        price_declining = price_low_at_l3 < price_low_at_l1
        macd_improving = macd_low_at_l3 > macd_low_at_l1
        
        if price_declining and macd_improving:
            price_change_pct = (price_low_at_l1 - price_low_at_l3) / price_low_at_l1
            macd_change_pct = (macd_low_at_l3 - macd_low_at_l1) / abs(macd_low_at_l1) if macd_low_at_l1 != 0 else 0
            return (price_change_pct + macd_change_pct) / 2
        return 0.0
        
    def _calculate_neckline_level(self, df: pd.DataFrame, p1_idx: int, l2_idx: int, p2_idx: int) -> float:
        """W 패턴의 넥라인 레벨을 계산합니다."""
        return max(df['macd_histogram'].iloc[p1_idx], df['macd_histogram'].iloc[p2_idx])


    def _calculate_pattern_strength(self, l1: float, l2: float, l3: float, div_score: float) -> float:
        """패턴의 전체적인 강도를 계산합니다. (양수화된 값 기준)"""
        # 머리의 깊이를 점수화합니다. l2가 가장 큰 값이므로, (l2 - 어깨높이)/l2 로 계산합니다.
        head_depth_score = (l2 - max(l1, l3)) / l2 if l2 > 0 else 0
        
        # 깊이 점수 70%, 다이버전스 점수 30%로 강도를 종합합니다.
        return (head_depth_score * 0.7) + (div_score * 0.3)

    def detect_w_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """데이터프레임에서 W 패턴을 감지합니다."""
        patterns = []
        
        macd_hist_series = df['macd_histogram']
        if not isinstance(macd_hist_series, pd.Series):
            macd_hist_series = pd.Series(macd_hist_series, name='macd_histogram')

        lows_indices = self._find_extrema(-macd_hist_series)
        highs_indices = self._find_extrema(macd_hist_series)
        
        if len(lows_indices) < 3:
            return []

        for i in range(len(lows_indices) - 2):
            l1_idx, l2_idx, l3_idx = lows_indices[i:i+3]
            
            l1_val, l2_val, l3_val = -macd_hist_series.iloc[[l1_idx, l2_idx, l3_idx]]

            if not self._validate_w_structure(l1_val, l2_val, l3_val):
                continue
            
            div_score = self._calculate_bullish_divergence(df, l1_idx, l2_idx, l3_idx)
            if div_score == 0:
                continue

            # 넥라인 계산을 위한 피크 찾기
            peaks_between_l1_l2 = highs_indices[(highs_indices > l1_idx) & (highs_indices < l2_idx)]
            peaks_between_l2_l3 = highs_indices[(highs_indices > l2_idx) & (highs_indices < l3_idx)]

            if len(peaks_between_l1_l2) == 0 or len(peaks_between_l2_l3) == 0:
                continue

            p1_idx = peaks_between_l1_l2[0]
            p2_idx = peaks_between_l2_l3[0]

            neckline_level = self._calculate_neckline_level(df, p1_idx, l2_idx, p2_idx)
            pattern_strength = self._calculate_pattern_strength(l1_val, l2_val, l3_val, div_score)

            patterns.append({
                "type": "W_pattern",
                "left_shoulder": (df.index[l1_idx], l1_val),
                "head": (df.index[l2_idx], l2_val),
                "right_shoulder": (df.index[l3_idx], l3_val),
                "neckline_level": neckline_level,
                "divergence_score": div_score,
                "pattern_strength": pattern_strength,
                "datetime": df.index[l3_idx]
            })
            
        return patterns

    def detect_m_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """M 패턴 감지 로직 (향후 구현)"""
        # W 패턴과 유사하게, 반대 로직으로 구현될 예정
        return []

if __name__ == '__main__':
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt

    # --- 실제 데이터로 W 패턴 탐지 및 검증 ---

    # 1. 프로젝트 경로 설정 및 모듈 임포트
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))

    from src.data.ohlcv_loader import load_ohlcv, resample_ohlcv
    from src.features.indicator_calculator import add_indicators

    # 2. 데이터 로드 및 전처리
    try:
        data_file = project_root / 'data' / 'rwa' / 'parquet_converted' / 'btc_kst_1min.parquet'
        df_1min = load_ohlcv(data_file, start_date='2017-01-01')
        
        # 4시간봉으로 리샘플링 및 MACD 지표 추가
        df_4h = resample_ohlcv(df_1min, '4H')
        # 'macd'와 'macd_histogram'이 모두 필요합니다.
        df_4h_indicators = add_indicators(df_4h, indicators=['macd']) 
        
        if 'macd_histogram' not in df_4h_indicators.columns:
            raise ValueError("MACD 히스토그램이 데이터에 없습니다. indicator_calculator를 확인하세요.")

    except FileNotFoundError:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {data_file}")
        sys.exit(1)
    except Exception as e:
        print(f"데이터 준비 중 오류 발생: {e}")
        sys.exit(1)

    # 3. W 패턴 탐지기 실행
    # 실제 데이터에 맞게 파라미터 조정 (대칭성 허용 오차를 늘려 더 많은 패턴 후보를 찾음)
    detector = WMPatternDetector(
        min_peak_distance=5, 
        prominence_ratio=0.01, 
        symmetry_tolerance=0.50 
    )
    w_patterns = detector.detect_w_pattern(df_4h_indicators)

    # 4. 결과 출력
    print(f"--- W 패턴 감지 결과 (2017년부터, 4시간봉) ---")
    if w_patterns:
        print(f"총 {len(w_patterns)}개의 W 패턴을 감지했습니다.")
        # 최근 5개 패턴 정보 출력
        print("\n최근 5개 감지 패턴:")
        for p in w_patterns[-5:]:
            print(f"  - 감지 시간: {p['datetime']}, 패턴 강도: {p['pattern_strength']:.3f}, 다이버전스 점수: {p['divergence_score']:.3f}")
    else:
        print("감지된 W 패턴이 없습니다.")

    # 시각화로 간단히 확인
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # 가격 차트
        ax1.plot(df_4h_indicators.index, df_4h_indicators['close'], label='Close Price', color='blue')
        ax1.set_title('Price and MACD Histogram with W-Pattern Detection')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        # MACD 히스토그램 차트
        colors = ['g' if val >= 0 else 'r' for val in df_4h_indicators['macd_histogram']]
        ax2.bar(df_4h_indicators.index, df_4h_indicators['macd_histogram'], label='MACD Histogram', color=colors, width=0.03)
        ax2.set_ylabel('MACD Histogram')
        ax2.grid(True)
        
        if w_patterns:
            for p in w_patterns:
                # 패턴 구간을 가격 차트에 표시
                ax1.axvline(p['left_shoulder'][0], color='gray', linestyle='--', alpha=0.7)
                ax1.axvline(p['head'][0], color='red', linestyle='--', alpha=0.7, label=f"Pattern Head ({p['head'][0].strftime('%Y-%m-%d')})")
                ax1.axvline(p['right_shoulder'][0], color='gray', linestyle='--', alpha=0.7)
                
                # 넥라인을 MACD 차트에 표시
                ax2.axhline(p['neckline_level'], color='purple', linestyle='--', alpha=0.8, label=f'Neckline: {p["neckline_level"]:.2f}')

        # 범례를 하나로 합치기
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib이 설치되지 않아 시각화 결과를 표시할 수 없습니다.")
        print("pip install matplotlib 로 설치해주세요.")