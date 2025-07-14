# src/labeling/enhanced_validator.py

import pandas as pd
import numpy as np
import talib # type: ignore
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 4단계 검증 함수 (V1 시스템) ---

def calculate_divergence_quality_score(df: pd.DataFrame, signal_idx: pd.Timestamp) -> float:
    """
    1단계: 다이버전스의 품질을 강도, 지속성, 차별성을 기반으로 평가합니다.
    """
    try:
        signal_loc_val = df.index.get_loc(signal_idx)
        if isinstance(signal_loc_val, slice) or isinstance(signal_loc_val, np.ndarray):
             # get_loc이 슬라이스나 배열을 반환하는 경우, 첫 번째 위치를 사용하거나 처리하지 않음
             # 여기서는 단순화를 위해 처리하지 않고 0.0을 반환
            return 0.0
        signal_loc = int(signal_loc_val)
    except KeyError:
        return 0.0 # signal_idx가 df에 없는 경우

    signal_type = df.iloc[signal_loc]['signal']
    search_window = 60
    start_loc = max(0, signal_loc - search_window)
    
    df_window = df.iloc[start_loc:signal_loc+1]

    current_price = df.iloc[signal_loc]['close']
    current_macd_hist = df.iloc[signal_loc]['macd_hist']

    prev_peak_idx = None
    if signal_type == 1:
        prev_peak_idx = df_window['macd_hist'].idxmin()
    elif signal_type == -1:
        prev_peak_idx = df_window['macd_hist'].idxmax()
    
    if prev_peak_idx is None or prev_peak_idx == signal_idx:
        return 0.0

    # 타입 안정성 확보
    prev_peak_ts = pd.to_datetime(prev_peak_idx)
    signal_ts = pd.to_datetime(signal_idx)

    prev_price = df.loc[prev_peak_ts, 'close']
    prev_macd_hist = df.loc[prev_peak_ts, 'macd_hist']
    
    # 값들이 숫자인지 확인
    if not all(pd.notna(v) and isinstance(v, (int, float)) for v in [current_price, prev_price, current_macd_hist, prev_macd_hist]):
        return 0.0

    duration = (signal_ts - prev_peak_ts).total_seconds() / 3600
    duration_score = min(float(duration) / 24.0, 1.0)
    
    price_change_pct = (current_price - prev_price) / prev_price if prev_price != 0 else 0.0
    macd_change_pct = (current_macd_hist - prev_macd_hist) / prev_macd_hist if prev_macd_hist != 0 else 0.0

    strength_score = 0.0
    if price_change_pct != 0:
        if signal_type == 1 and price_change_pct < 0 and macd_change_pct > 0:
            strength_score = min(abs(macd_change_pct / price_change_pct) / 10.0, 1.0)
        elif signal_type == -1 and price_change_pct > 0 and macd_change_pct < 0:
            strength_score = min(abs(macd_change_pct / price_change_pct) / 10.0, 1.0)
        
    return (strength_score * 0.7) + (duration_score * 0.3)


def verify_inflection_point_authenticity(df: pd.DataFrame, signal_idx: pd.Timestamp) -> float:
    """
    2단계: 신호 지점이 진정한 변곡점인지 모멘텀, 거래량, 캔들 패턴으로 검증합니다.
    """
    try:
        loc_val = df.index.get_loc(signal_idx)
        if isinstance(loc_val, (slice, np.ndarray)):
            return 0.0
        loc = int(loc_val)
    except KeyError:
        return 0.0

    signal_type = df.iloc[loc]['signal']
    window = df.iloc[max(0, loc-5):loc+1]

    if window.empty or len(window) < 2:
        return 0.0
        
    hist_slope = np.polyfit(range(len(window)), window['macd_hist'], 1)[0]
    momentum_score = 0.0
    if (signal_type == 1 and hist_slope > 0) or \
       (signal_type == -1 and hist_slope < 0):
        momentum_score = 1.0
        
    avg_volume_window = df['volume'].iloc[max(0, loc-20):loc]
    avg_volume = avg_volume_window.mean() if not avg_volume_window.empty else 0.0
    volume_score = 1.0 if pd.notna(avg_volume) and df.iloc[loc]['volume'] > avg_volume * 1.5 else 0.0
    
    candle_patterns = {
        'CDLENGULFING': 1.0, 'CDLHAMMER': 0.8, 'CDLMORNINGSTAR': 1.0, 
        'CDLINVERTEDHAMMER': 0.8, 'CDLPIERCING': 0.9
    }
    pattern_score = 0.0
    for pattern, score in candle_patterns.items():
        # talib 함수가 전체 데이터프레임을 필요로 함
        full_result = getattr(talib, pattern)(df['open'], df['high'], df['low'], df['close'])
        result_at_loc = full_result.iloc[loc]
        
        if pd.notna(result_at_loc):
            if (signal_type == 1 and result_at_loc > 0) or \
               (signal_type == -1 and result_at_loc < 0):
                pattern_score = max(pattern_score, score)

    return (momentum_score * 0.4) + (volume_score * 0.3) + (pattern_score * 0.3)


def validate_with_lower_timeframe_zigzag(df: pd.DataFrame, signal_idx: pd.Timestamp) -> float:
    """
    3단계: 하위 타임프레임에서 의미있는 지그재그 변곡점인지 확인합니다.
    """
    try:
        loc_val = df.index.get_loc(signal_idx)
        if isinstance(loc_val, (slice, np.ndarray)):
            return 0.0
        loc = int(loc_val)
    except KeyError:
        return 0.0
        
    signal_type = df.iloc[loc]['signal']
    window = df.iloc[max(0, loc - 10) : loc + 11]

    if window.empty:
        return 0.0
    
    is_significant_low = bool(df.iloc[loc]['low'] == window['low'].min())
    is_significant_high = bool(df.iloc[loc]['high'] == window['high'].max())
    
    if (signal_type == 1 and is_significant_low) or \
       (signal_type == -1 and is_significant_high):
        return 1.0
    return 0.0
    

def comprehensive_divergence_confirmation(df: pd.DataFrame, signal_idx: pd.Timestamp) -> float:
    """
    4단계: RSI 동조성과 "매매의 느낌"을 종합하여 최종 확인합니다.
    """
    try:
        loc_val = df.index.get_loc(signal_idx)
        if isinstance(loc_val, (slice, np.ndarray)):
            return 0.0
        loc = int(loc_val)
    except KeyError:
        return 0.0
        
    signal_type = df.iloc[loc]['signal']
    
    future_window = df.iloc[loc+1 : loc+4]
    if len(future_window) < 3: return 0.0

    rsi_after = future_window['rsi_14'].mean()
    rsi_now = df.iloc[loc]['rsi_14']
    
    if not (pd.notna(rsi_after) and pd.notna(rsi_now) and isinstance(signal_type, (int, float))):
        return 0.0

    rsi_score = 0.0
    if (signal_type == 1 and rsi_after > rsi_now) or \
       (signal_type == -1 and rsi_after < rsi_now):
        rsi_score = 1.0
        
    sense_score = 1.0
    stop_loss_pct = 0.02
    current_close = df.iloc[loc]['close']
    if not pd.notna(current_close):
        return 0.0

    if signal_type == 1:
        stop_price = current_close * (1 - stop_loss_pct)
        if future_window['low'].min() < stop_price:
            sense_score = 0.0
    elif signal_type == -1:
        stop_price = current_close * (1 + stop_loss_pct)
        if future_window['high'].max() > stop_price:
            sense_score = 0.0
            
    return (rsi_score * 0.5) + (sense_score * 0.5)

def calculate_realistic_pnl(df: pd.DataFrame, signal_idx: pd.Timestamp) -> float:
    """
    현실적인 PnL을 계산합니다.
    """
    try:
        loc_val = df.index.get_loc(signal_idx)
        if isinstance(loc_val, (slice, np.ndarray)):
            return 0.0
        loc = int(loc_val)
    except KeyError:
        return 0.0

    signal_type_val = df.iloc[loc]['signal']
    entry_price_val = df.iloc[loc]['close']

    if not (pd.notna(signal_type_val) and pd.notna(entry_price_val)):
        return 0.0
    
    signal_type = int(signal_type_val)
    entry_price = float(entry_price_val)

    if signal_type == 1:
        take_profit_price = entry_price * 1.03
        stop_loss_price = entry_price * 0.985
    else:
        take_profit_price = entry_price * 0.97
        stop_loss_price = entry_price * 1.015

    for i in range(loc + 1, min(loc + 24, len(df))):
        future_high = df.iloc[i]['high']
        future_low = df.iloc[i]['low']

        if not (pd.notna(future_high) and pd.notna(future_low)):
            continue

        if signal_type == 1:
            if future_high >= take_profit_price: return 3.0
            if future_low <= stop_loss_price: return -1.5
        else:
            if future_low <= take_profit_price: return 3.0
            if future_high >= stop_loss_price: return -1.5
            
    final_price_loc = min(loc + 23, len(df) - 1)
    final_price = df.iloc[final_price_loc]['close']
    
    if not pd.notna(final_price) or entry_price == 0:
        return 0.0
        
    pnl = (float(final_price) - entry_price) / entry_price * 100 * signal_type
    return pnl 