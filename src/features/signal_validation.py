import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore
from scipy.stats import linregress
from scipy.signal import find_peaks
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def calculate_momentum_exhaustion_score(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    하락 모멘텀이 소진되었는지 0-3점으로 평가합니다.
    점수가 높을수록 모멘텀 소진 가능성이 높습니다.

    - 1점: 하락 속도 둔화 (가격의 ROC 기울기 증가)
    - 1점: 거래량 감소 (매도 압력 약화)
    - 1점: MACD 히스토그램 0 수렴

    Args:
        df (pd.DataFrame): OHLCV 및 MACD('macd_h') 컬럼을 포함해야 함.
        lookback (int): 분석 기간(window).

    Returns:
        pd.Series: 각 시점의 모멘텀 소진 점수 (0-3).
    """
    scores = pd.Series(0, index=df.index, dtype=int)
    
    # 1. 하락 속도 둔화 (Rate of Change 기울기)
    roc = ta.roc(df['close'], length=1)
    if roc is None: roc = pd.Series(np.nan, index=df.index)
    roc_slope = roc.rolling(window=lookback).apply(lambda x: linregress(np.arange(len(x)), x)[0] if not np.isnan(x).any() and len(x) > 1 else np.nan, raw=False)
    
    # 2. 거래량 감소 (거래량 기울기)
    volume_slope = df['volume'].rolling(window=lookback).apply(lambda x: linregress(np.arange(len(x)), x)[0] if not np.isnan(x).any() and len(x) > 1 else np.nan, raw=False)

    # 3. MACD 히스토그램 0 수렴
    # MACD 히스토그램이 음수에서 양수 방향으로 증가하고, 0에 가까워지는 경향
    macd_hist = df['macd_h']
    is_converging = (macd_hist.rolling(window=lookback).apply(lambda x: linregress(np.arange(len(x)), x)[0] if not np.isnan(x).any() and len(x) > 1 else np.nan, raw=False) > 0) & (macd_hist < 0)

    # 점수 계산
    scores += (roc_slope > 0).astype(int)
    scores += (volume_slope < 0).astype(int)
    scores += is_converging.astype(int)
    
    return scores.fillna(0).astype(int)


def detect_dual_divergence_confirmation(df: pd.DataFrame, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14, distance=5) -> pd.Series:
    """
    가격 저점은 낮아지지만 MACD와 RSI의 저점은 높아지는 강세 다이버전스가
    동시에 발생하는지 확인합니다.

    Args:
        df (pd.DataFrame): 'close', 'high', 'low' 컬럼을 포함해야 합니다.
        macd_fast (int): MACD 단기 EMA.
        macd_slow (int): MACD 장기 EMA.
        macd_signal (int): MACD 신호선 EMA.
        rsi_period (int): RSI 기간.
        distance (int): 다이버전스 저점을 찾기 위한 최소 간격.

    Returns:
        pd.Series: 이중 다이버전스가 확인된 지점을 True로 표시하는 boolean 시리즈.
    """
    # MACD 및 RSI 계산
    if 'macd_line' not in df.columns:
        macd = df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df['macd_line'] = macd[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'] if macd is not None else np.nan
    if 'rsi' not in df.columns:
        df['rsi'] = df.ta.rsi(length=rsi_period)

    df.dropna(subset=['low', 'macd_line', 'rsi'], inplace=True)
    if df.empty:
        return pd.Series(False, index=df.index)
        
    # 저점 찾기 (음수 값에 대해 find_peaks 사용, .to_numpy()로 타입 안정성 확보)
    price_lows_idx, _ = find_peaks(df['low'].mul(-1).to_numpy(), distance=distance)
    macd_lows_idx, _ = find_peaks(df['macd_line'].mul(-1).to_numpy(), distance=distance)
    rsi_lows_idx, _ = find_peaks(df['rsi'].mul(-1).to_numpy(), distance=distance)
    
    bullish_divergence = pd.Series(False, index=df.index)

    # 잠재적 다이버전스 쌍 찾기
    for i in range(1, len(price_lows_idx)):
        prev_price_low_idx_loc = price_lows_idx[i-1]
        curr_price_low_idx_loc = price_lows_idx[i]

        prev_price_low_val = df['low'].iloc[prev_price_low_idx_loc]
        curr_price_low_val = df['low'].iloc[curr_price_low_idx_loc]

        # 가격이 새로운 저점을 형성했는지 확인
        if curr_price_low_val < prev_price_low_val:
            
            def check_indicator_divergence(indicator_name, indicator_lows_idx):
                # 현재 가격 저점과 가장 가까운 지표 저점 찾기
                prev_ind_lows = indicator_lows_idx[(indicator_lows_idx >= prev_price_low_idx_loc - distance) & (indicator_lows_idx <= prev_price_low_idx_loc + distance)]
                curr_ind_lows = indicator_lows_idx[(indicator_lows_idx >= curr_price_low_idx_loc - distance) & (indicator_lows_idx <= curr_price_low_idx_loc + distance)]

                if len(prev_ind_lows) > 0 and len(curr_ind_lows) > 0:
                    # 각 구간에서 가장 낮은 지점을 선택
                    prev_ind_low_loc = prev_ind_lows[df[indicator_name].iloc[prev_ind_lows].argmin()]
                    curr_ind_low_loc = curr_ind_lows[df[indicator_name].iloc[curr_ind_lows].argmin()]
                    
                    if df[indicator_name].iloc[curr_ind_low_loc] > df[indicator_name].iloc[prev_ind_low_loc]:
                        return True
                return False

            macd_div = check_indicator_divergence('macd_line', macd_lows_idx)
            rsi_div = check_indicator_divergence('rsi', rsi_lows_idx)
            
            if macd_div and rsi_div:
                bullish_divergence.iloc[curr_price_low_idx_loc] = True
                
    return bullish_divergence


def analyze_orderly_decline_pattern(df: pd.DataFrame, signal_indices: pd.Index, lookback: int = 20) -> pd.Series:
    """
    주어진 신호 지점 이전의 하락이 '질서 정연한' 패턴인지 평가합니다.
    선형 회귀의 R-squared 값을 사용하여 하락 추세의 일관성을 측정합니다.
    값이 1에 가까울수록 변동성 없이 꾸준한 하락을 의미합니다.

    Args:
        df (pd.DataFrame): 'close' 컬럼이 포함된 데이터프레임.
        signal_indices (pd.Index): 분석할 신호가 발생한 인덱스.
        lookback (int): 신호 지점 이전의 분석 기간.

    Returns:
        pd.Series: 각 신호 지점에 대한 하락 품질 점수 (R-squared 값).
    """
    decline_quality = pd.Series(np.nan, index=df.index)
    
    # get_loc은 레이블이 중복될 경우 슬라이스나 마스크를 반환할 수 있으므로 to_list로 처리
    valid_signal_indices = df.index.intersection(signal_indices)
    
    for idx in valid_signal_indices:
        try:
            loc = df.index.get_loc(idx)
            # 단일 위치가 아닐 경우 첫 번째 위치 사용
            if isinstance(loc, (slice, np.ndarray)):
                loc = loc.start if isinstance(loc, slice) else loc[0]
        except KeyError:
            continue

        start_loc = loc - lookback + 1
        end_loc = loc + 1
        
        if start_loc < 0:
            continue
            
        window_df = df.iloc[start_loc:end_loc]
        
        if len(window_df) < lookback:
            continue
            
        y = window_df['close']
        x = np.arange(len(y))
        
        # 데이터에 NaN이 없는지 확인
        if y.isnull().any():
            continue

        lin_reg_result = linregress(x, y)
        slope = lin_reg_result[0]
        r_value = lin_reg_result[2]
        
        # 하락 추세일 경우에만 품질 점수 계산 (기울기가 음수)
        if isinstance(slope, (int, float)) and slope < 0 and isinstance(r_value, (int, float)):
            decline_quality.loc[idx] = float(r_value**2)
            
    return decline_quality.reindex(signal_indices).fillna(0)


class SignalAuthenticator:
    """
    다이버전스 신호의 '진정성'을 여러 보조 지표를 통해 검증하고 점수화합니다.
    """
    def __init__(self, df: pd.DataFrame, divergence_info: dict | None = None, config: dict | None = None):
        self.df = df
        self.divergence = divergence_info if divergence_info is not None else {}
        self.config = config or self._get_default_config()
        
        price_points = self.divergence.get('price_points')
        if isinstance(price_points, tuple) and len(price_points) == 2:
            self.p1_idx = price_points[0]
            self.p2_idx = price_points[1]
        else:
            self.p1_idx, self.p2_idx = None, None

    def _get_default_config(self) -> dict:
        return {
            "volume_spike_window": 5,
            "volume_spike_multiplier": 2.0,
            "weights": {
                "volume": 0.4,
                "rsi": 0.3,
                "pattern": 0.3
            }
        }

    def _check_volume_spike(self) -> float:
        """두 번째 저점에서 거래량 급증을 확인합니다."""
        if self.p1_idx is None or self.p2_idx is None:
            return 0.0

        window = self.config['volume_spike_window']
        multiplier = self.config['volume_spike_multiplier']
        
        # 두 번째 저점 이전의 평균 거래량 계산
        lookback_start = max(0, self.p2_idx - window)
        if lookback_start >= self.p2_idx:
            return 0.0

        avg_volume = self.df['volume'].iloc[lookback_start:self.p2_idx].mean()
        volume_at_p2 = self.df['volume'].iloc[self.p2_idx]

        if avg_volume > 0 and volume_at_p2 > avg_volume * multiplier:
            return 1.0
        
        # 점진적 점수
        if avg_volume > 0:
            score = volume_at_p2 / (avg_volume * multiplier)
            return min(score, 1.0)
            
        return 0.0

    def _check_rsi_concordance(self) -> float:
        """RSI 동조성을 확인합니다. MACD와 RSI 모두에서 다이버전스가 관찰되는지 확인합니다."""
        if self.p1_idx is None or self.p2_idx is None or self.divergence is None:
            return 0.0

        # RSI 계산
        if 'rsi' not in self.df.columns:
            self.df['rsi'] = ta.rsi(self.df['close'])

        rsi = self.df['rsi'].values
        
        # 다이버전스 타입에 따라 RSI 저점/고점 확인
        divergence_type = self.divergence.get('type', 'bullish')
        
        if divergence_type == 'bullish':
            # RSI 저점도 높아지는지 확인
            if rsi[self.p2_idx] > rsi[self.p1_idx]:
                return 1.0
        elif divergence_type == 'bearish':
            # RSI 고점도 낮아지는지 확인
            if rsi[self.p2_idx] < rsi[self.p1_idx]:
                return 1.0
                
        return 0.0

    def _check_reversal_pattern(self) -> float:
        """주요 반전 캔들 패턴(상승장악형, 망치형 등)이 p2 지점에서 발생하는지 확인합니다."""
        if self.p2_idx is None:
            return 0.0

        # pandas-ta를 사용하여 주요 반전 캔들 패턴 탐지
        # 상승장악형(100), 망치형(100), 상승반격형(100) 등
        engulfing = self.df.ta.cdl_engulfing().iloc[self.p2_idx]
        hammer = self.df.ta.cdl_hammer().iloc[self.p2_idx]
        
        # 다른 주요 강세 반전 패턴 추가 가능
        # cdl_morningstar, cdl_piercing, cdl_bullish_harami 등

        if (engulfing is not None and engulfing > 0) or \
           (hammer is not None and hammer > 0):
            return 1.0
            
        return 0.0

    def authenticate(self) -> dict:
        """모든 검증을 수행하고 최종 점수를 반환합니다."""
        volume_score = self._check_volume_spike()
        rsi_score = self._check_rsi_concordance()
        pattern_score = self._check_reversal_pattern()
        
        weights = self.config['weights']
        final_score = (
            volume_score * weights['volume'] +
            rsi_score * weights['rsi'] +
            pattern_score * weights['pattern']
        )
        
        return {
            "final_authenticity_score": round(final_score, 4),
            "volume_score": round(volume_score, 4),
            "rsi_score": round(rsi_score, 4),
            "pattern_score": round(pattern_score, 4)
        }


# --- 예시 및 테스트 코드 ---
def _create_test_data():
    """테스트용 샘플 데이터 생성"""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
    price = 100 - np.log(np.arange(1, 101)) * 10
    price_noise = price + np.random.randn(100) * 0.5
    
    # 모멘텀 소진 시나리오
    price_momentum_end = price[-20:] + np.linspace(0, 5, 20)
    price_noise[-20:] = price_momentum_end
    
    volume = np.random.randint(100, 500, 100).astype(float)
    volume[-20:] = np.linspace(300, 100, 20) # 거래량 감소

    df = pd.DataFrame({
        'open': price_noise,
        'high': price_noise + 0.5,
        'low': price_noise - 0.5,
        'close': price_noise,
        'volume': volume
    }, index=dates)

    macd = df.ta.macd()
    df['macd_h'] = macd['MACDh_12_26_9'] if macd is not None else np.nan
    df['macd_line'] = macd['MACD_12_26_9'] if macd is not None else np.nan
    df['rsi'] = df.ta.rsi()
    
    # 다이버전스 시나리오 (가격을 더 낮추고 지표는 높임)
    price_low_1 = pd.Timestamp("2023-02-15")
    price_low_2 = pd.Timestamp("2023-03-15")
    df.at[price_low_1, 'low'] = df.at[price_low_1, 'low'] - 3
    df.at[price_low_2, 'low'] = df.at[price_low_2, 'low'] - 5  # 더 낮은 저점

    # 지표 저점은 높게
    df.at[price_low_1, 'macd_line'] = -1.5
    df.loc[price_low_2, 'macd_line'] = -0.5
    df.loc[price_low_1, 'rsi'] = 25
    df.loc[price_low_2, 'rsi'] = 35

    return df.dropna().copy()

if __name__ == '__main__':
    test_df = _create_test_data()

    print("--- 1. 모멘텀 소진 점수 테스트 ---")
    momentum_scores = calculate_momentum_exhaustion_score(test_df, lookback=10)
    print("Momentum Exhaustion Scores (last 15):")
    print(momentum_scores.tail(15))
    print(f"점수가 1 이상인 경우: {len(momentum_scores[momentum_scores > 0])}개")
    print(f"최고 점수: {momentum_scores.max()}")
    print("\n" + "="*50 + "\n")


    print("--- 2. 이중 다이버전스 탐지 테스트 ---")
    dual_divs = detect_dual_divergence_confirmation(test_df.copy(), distance=5)
    print("Dual Divergence Detections:")
    
    detected_points = dual_divs[dual_divs]
    if not detected_points.empty: # type: ignore
        print(detected_points.index) # type: ignore
        print("이중 다이버전스 탐지 성공!")
    else:
        print("No detections")
        print("이중 다이버전스 탐지 실패.")
    print("\n" + "="*50 + "\n")

    
    print("--- 3. 질서있는 하락 패턴 분석 테스트 ---")
    # 질서있는 하락이 예상되는 구간의 끝을 신호 지점으로 가정
    signal_indices = test_df.index[test_df['close'] < 90]
    
    # 일부러 노이즈가 많은 구간 추가
    noisy_decline_start = "2023-01-20"
    noisy_decline_end = "2023-01-30"
    noise = np.random.normal(0, 2, len(test_df.loc[noisy_decline_start:noisy_decline_end]))
    test_df.loc[noisy_decline_start:noisy_decline_end, 'close'] += noise
    
    decline_quality_scores = analyze_orderly_decline_pattern(test_df, signal_indices, lookback=15)
    print("Decline Quality Scores (R-squared):")
    print(decline_quality_scores.head())
    print("\n평균 품질 점수:", decline_quality_scores.mean())
    print("최고 품질 점수:", decline_quality_scores.max())
    print("최저 품질 점수:", decline_quality_scores.min())

    # 시각화로 확인 (선택 사항)
    try:
        import matplotlib.pyplot as plt
        
        high_quality_signal = None
        low_quality_signal = None
        if not decline_quality_scores.empty:
            high_quality_signal = decline_quality_scores.idxmax()
            low_quality_signal = decline_quality_scores.idxmin()

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle("Signal Validation Analysis", fontsize=16)

        # Plot 1: Momentum Score
        axes[0].plot(test_df.index, test_df['close'], label='Close Price', color='k')
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(momentum_scores.index, momentum_scores, label='Momentum Score', color='orange', alpha=0.7, drawstyle="steps-post") # type: ignore
        axes[0].set_title('Momentum Exhaustion Score')
        axes[0].legend(loc='upper left')
        ax0_twin.legend(loc='upper right')

        # Plot 2: Dual Divergence
        axes[1].plot(test_df.index, test_df['close'], label='Close Price', color='k')
        divergence_points = dual_divs[dual_divs].index # type: ignore
        if not divergence_points.empty:
            axes[1].plot(divergence_points, test_df['close'].loc[divergence_points], 'g^', markersize=10, label='Dual Divergence Signal')
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(test_df.index, test_df['macd_line'], label='MACD', color='blue', alpha=0.6)
        ax1_twin.plot(test_df.index, test_df['rsi'], label='RSI', color='purple', alpha=0.6)
        axes[1].set_title('Dual Divergence Confirmation')
        axes[1].legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # Plot 3: Orderly Decline
        axes[2].plot(test_df.index, test_df['close'], label='Close Price', color='k', alpha=0.3)
        
        def plot_lookback_window(signal_idx, color, label):
            if signal_idx is None: return
            loc_val = test_df.index.get_loc(signal_idx)
            # get_loc이 슬라이스나 배열을 반환할 경우를 처리
            if isinstance(loc_val, (slice, np.ndarray)):
                loc = loc_val.start if isinstance(loc_val, slice) else loc_val[0]
            else:
                loc = loc_val

            if loc is None: return
                
            start_loc = loc - 15 + 1
            window = test_df.iloc[start_loc : loc + 1]
            axes[2].plot(window.index, window['close'], color=color, linewidth=2, label=f"{label} (R2: {decline_quality_scores.loc[signal_idx]:.2f})")
            axes[2].axvline(signal_idx, color=color, linestyle='--', alpha=0.7)

        if high_quality_signal is not None:
            plot_lookback_window(high_quality_signal, 'green', 'High Quality Decline')
        if low_quality_signal is not None:
            plot_lookback_window(low_quality_signal, 'red', 'Low Quality Decline')
        
        axes[2].set_title('Orderly Decline Pattern Quality')
        axes[2].legend()

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    except ImportError:
        print("\nMatplotlib is not installed. Skipping visualization.")
    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}") 