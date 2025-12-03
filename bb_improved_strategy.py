#!/usr/bin/env python3
"""
볼린저밴드 개선 전략 - 실제 수익나는 방법 적용
1. RSI 필터 추가 (30/70)
2. 손절 넓히기 (밴드폭 100%)
3. 익절 목표 높이기 (반대 밴드)
4. Mean Reversion (역추세) 전략
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 개선 전략 백테스트")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
print(f"\n15M 데이터: {len(df):,}개")

# BB 계산
period = 20
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = df['bb_upper'] - df['bb_lower']
df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# RSI 계산
rsi_period = 14
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# MACD 계산
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# NaN 제거
df = df.iloc[30:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# numpy 배열
close = df['close'].values
high = df['high'].values
low = df['low'].values
bb_mid = df['bb_mid'].values
bb_upper = df['bb_upper'].values
bb_lower = df['bb_lower'].values
bb_width = df['bb_width'].values
rsi = df['rsi'].values
macd_hist = df['macd_hist'].values
datetimes = df['datetime'].values

# 거래 비용
COST = 0.18  # 수수료 + 슬리피지

# ============================================================
# 전략 1: Mean Reversion (BB 터치 + RSI 필터)
# ============================================================
print("\n" + "=" * 80)
print("전략 1: Mean Reversion (BB 터치 + RSI 30/70)")
print("=" * 80)

def simulate_mean_reversion(idx, direction, sl_mult=1.0, tp_target='opposite'):
    """
    Mean Reversion 전략 시뮬레이션
    - 손절: 밴드폭의 sl_mult 배
    - 익절: 'opposite' = 반대 밴드, 'mid' = 중간선
    """
    entry = close[idx]
    width = bb_width[idx]
    max_hold = 160  # 40시간
    
    if direction == 'LONG':
        sl = entry - width * sl_mult
        if tp_target == 'opposite':
            tp = bb_upper[idx]
        else:
            tp = bb_mid[idx]
    else:
        sl = entry + width * sl_mult
        if tp_target == 'opposite':
            tp = bb_lower[idx]
        else:
            tp = bb_mid[idx]
    
    mfe = 0
    for j in range(idx + 1, min(idx + max_hold + 1, len(close))):
        if direction == 'LONG':
            mfe = max(mfe, (high[j] - entry) / entry * 100)
            # 손절
            if low[j] <= sl:
                pnl = (sl - entry) / entry * 100
                return {'pnl': pnl, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe}
            # 익절 (반대 밴드 도달)
            if high[j] >= tp:
                pnl = (tp - entry) / entry * 100
                return {'pnl': pnl, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
        else:
            mfe = max(mfe, (entry - low[j]) / entry * 100)
            if high[j] >= sl:
                pnl = (entry - sl) / entry * 100
                return {'pnl': pnl, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe}
            if low[j] <= tp:
                pnl = (entry - tp) / entry * 100
                return {'pnl': pnl, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
    
    # 시간 만료
    final = close[min(idx + max_hold, len(close) - 1)]
    if direction == 'LONG':
        pnl = (final - entry) / entry * 100
    else:
        pnl = (entry - final) / entry * 100
    return {'pnl': pnl, 'reason': 'TIMEOUT', 'bars': max_hold, 'mfe': mfe}


def run_mean_reversion(rsi_low=30, rsi_high=70, sl_mult=1.0, tp_target='opposite'):
    """Mean Reversion 전략 실행"""
    trades = []
    last_trade_idx = 0
    
    for i in range(50, len(close) - 200):
        if i < last_trade_idx + 4:  # 최소 1시간 간격
            continue
        
        # LONG: BB 하단 터치 + RSI < rsi_low
        if close[i] <= bb_lower[i] and rsi[i] < rsi_low:
            result = simulate_mean_reversion(i, 'LONG', sl_mult, tp_target)
            result['direction'] = 'LONG'
            result['rsi'] = rsi[i]
            result['time'] = datetimes[i]
            trades.append(result)
            last_trade_idx = i
        
        # SHORT: BB 상단 터치 + RSI > rsi_high
        elif close[i] >= bb_upper[i] and rsi[i] > rsi_high:
            result = simulate_mean_reversion(i, 'SHORT', sl_mult, tp_target)
            result['direction'] = 'SHORT'
            result['rsi'] = rsi[i]
            result['time'] = datetimes[i]
            trades.append(result)
            last_trade_idx = i
    
    return pd.DataFrame(trades)


# 파라미터 테스트
print(f"\n{'RSI':>8} {'SL배수':>8} {'익절':>10} {'건수':>6} {'승률':>8} {'평균PnL':>10} {'실제PnL':>10} {'기대값':>10}")
print("-" * 90)

best_result = None
best_expected = -999

for rsi_th in [20, 25, 30]:
    for sl_mult in [0.5, 1.0, 1.5, 2.0]:
        for tp_target in ['mid', 'opposite']:
            df_trades = run_mean_reversion(rsi_th, 100 - rsi_th, sl_mult, tp_target)
            
            if len(df_trades) >= 30:
                df_trades['real_pnl'] = df_trades['pnl'] - COST
                
                tp = df_trades[df_trades['reason'] == 'TP']
                sl = df_trades[df_trades['reason'] == 'SL']
                
                if len(tp) > 0 and len(sl) > 0:
                    win_rate = len(tp) / len(df_trades) * 100
                    avg_pnl = df_trades['pnl'].mean()
                    real_pnl = df_trades['real_pnl'].mean()
                    
                    # 기대값
                    tp_rate = len(tp) / len(df_trades)
                    sl_rate = len(sl) / len(df_trades)
                    expected = tp_rate * tp['real_pnl'].mean() + sl_rate * sl['real_pnl'].mean()
                    
                    print(f"{rsi_th:>8} {sl_mult:>8.1f} {tp_target:>10} {len(df_trades):>6} {win_rate:>8.1f}% {avg_pnl:>10.2f}% {real_pnl:>10.2f}% {expected:>10.3f}%")
                    
                    if expected > best_expected:
                        best_expected = expected
                        best_result = {
                            'rsi': rsi_th,
                            'sl_mult': sl_mult,
                            'tp_target': tp_target,
                            'trades': df_trades.copy()
                        }

# ============================================================
# 전략 2: BB + RSI + MACD 조합
# ============================================================
print("\n" + "=" * 80)
print("전략 2: BB + RSI + MACD 조합")
print("=" * 80)

def run_bb_rsi_macd(rsi_low=30, rsi_high=70, sl_mult=1.0, tp_target='opposite', use_macd=True):
    """BB + RSI + MACD 전략"""
    trades = []
    last_trade_idx = 0
    
    for i in range(50, len(close) - 200):
        if i < last_trade_idx + 4:
            continue
        
        # LONG: BB 하단 + RSI < 30 + MACD 상승 전환
        macd_bullish = macd_hist[i] > macd_hist[i-1] if use_macd else True
        macd_bearish = macd_hist[i] < macd_hist[i-1] if use_macd else True
        
        if close[i] <= bb_lower[i] and rsi[i] < rsi_low and macd_bullish:
            result = simulate_mean_reversion(i, 'LONG', sl_mult, tp_target)
            result['direction'] = 'LONG'
            result['rsi'] = rsi[i]
            result['macd_hist'] = macd_hist[i]
            trades.append(result)
            last_trade_idx = i
        
        # SHORT: BB 상단 + RSI > 70 + MACD 하락 전환
        elif close[i] >= bb_upper[i] and rsi[i] > rsi_high and macd_bearish:
            result = simulate_mean_reversion(i, 'SHORT', sl_mult, tp_target)
            result['direction'] = 'SHORT'
            result['rsi'] = rsi[i]
            result['macd_hist'] = macd_hist[i]
            trades.append(result)
            last_trade_idx = i
    
    return pd.DataFrame(trades)


print(f"\n{'MACD':>6} {'RSI':>6} {'SL':>6} {'TP':>10} {'건수':>6} {'승률':>8} {'실PnL':>10} {'기대값':>10}")
print("-" * 80)

for use_macd in [False, True]:
    for rsi_th in [25, 30]:
        for sl_mult in [1.0, 1.5]:
            for tp_target in ['mid', 'opposite']:
                df_trades = run_bb_rsi_macd(rsi_th, 100 - rsi_th, sl_mult, tp_target, use_macd)
                
                if len(df_trades) >= 20:
                    df_trades['real_pnl'] = df_trades['pnl'] - COST
                    
                    tp = df_trades[df_trades['reason'] == 'TP']
                    sl = df_trades[df_trades['reason'] == 'SL']
                    
                    if len(tp) > 0 and len(sl) > 0:
                        win_rate = len(tp) / len(df_trades) * 100
                        real_pnl = df_trades['real_pnl'].mean()
                        
                        tp_rate = len(tp) / len(df_trades)
                        sl_rate = len(sl) / len(df_trades)
                        expected = tp_rate * tp['real_pnl'].mean() + sl_rate * sl['real_pnl'].mean()
                        
                        macd_str = 'Y' if use_macd else 'N'
                        print(f"{macd_str:>6} {rsi_th:>6} {sl_mult:>6.1f} {tp_target:>10} {len(df_trades):>6} {win_rate:>8.1f}% {real_pnl:>10.2f}% {expected:>10.3f}%")
                        
                        if expected > best_expected:
                            best_expected = expected
                            best_result = {
                                'strategy': 'BB+RSI+MACD',
                                'rsi': rsi_th,
                                'sl_mult': sl_mult,
                                'tp_target': tp_target,
                                'use_macd': use_macd,
                                'trades': df_trades.copy()
                            }

# ============================================================
# 전략 3: Fade 전략 (밴드 터치 후 반전 확인)
# ============================================================
print("\n" + "=" * 80)
print("전략 3: Fade (밴드 터치 후 반전 캔들 확인)")
print("=" * 80)

def run_fade_strategy(rsi_low=30, rsi_high=70, sl_mult=1.5, tp_target='opposite'):
    """Fade 전략 - 반전 캔들 확인 후 진입"""
    trades = []
    last_trade_idx = 0
    
    for i in range(51, len(close) - 200):
        if i < last_trade_idx + 4:
            continue
        
        # LONG: 전봉 BB 하단 터치 + 현재봉 양봉 + RSI < 30
        prev_touched_lower = low[i-1] <= bb_lower[i-1]
        current_bullish = close[i] > open_price[i] if 'open' in df.columns else close[i] > close[i-1]
        
        if prev_touched_lower and current_bullish and rsi[i-1] < rsi_low:
            result = simulate_mean_reversion(i, 'LONG', sl_mult, tp_target)
            result['direction'] = 'LONG'
            result['rsi'] = rsi[i-1]
            trades.append(result)
            last_trade_idx = i
        
        # SHORT: 전봉 BB 상단 터치 + 현재봉 음봉 + RSI > 70
        prev_touched_upper = high[i-1] >= bb_upper[i-1]
        current_bearish = close[i] < open_price[i] if 'open' in df.columns else close[i] < close[i-1]
        
        if prev_touched_upper and current_bearish and rsi[i-1] > rsi_high:
            result = simulate_mean_reversion(i, 'SHORT', sl_mult, tp_target)
            result['direction'] = 'SHORT'
            result['rsi'] = rsi[i-1]
            trades.append(result)
            last_trade_idx = i
    
    return pd.DataFrame(trades)

# open 컬럼 확인 - open_price로 변경 (내장 함수와 충돌 방지)
if 'open' not in df.columns:
    df['open'] = df['close'].shift(1)
open_price = df['open'].values

print(f"\n{'RSI':>8} {'SL배수':>8} {'익절':>10} {'건수':>6} {'승률':>8} {'실제PnL':>10} {'기대값':>10}")
print("-" * 80)

for rsi_th in [25, 30, 35]:
    for sl_mult in [1.0, 1.5, 2.0]:
        for tp_target in ['mid', 'opposite']:
            df_trades = run_fade_strategy(rsi_th, 100 - rsi_th, sl_mult, tp_target)
            
            if len(df_trades) >= 20:
                df_trades['real_pnl'] = df_trades['pnl'] - COST
                
                tp = df_trades[df_trades['reason'] == 'TP']
                sl = df_trades[df_trades['reason'] == 'SL']
                
                if len(tp) > 0 and len(sl) > 0:
                    win_rate = len(tp) / len(df_trades) * 100
                    real_pnl = df_trades['real_pnl'].mean()
                    
                    tp_rate = len(tp) / len(df_trades)
                    sl_rate = len(sl) / len(df_trades)
                    expected = tp_rate * tp['real_pnl'].mean() + sl_rate * sl['real_pnl'].mean()
                    
                    print(f"{rsi_th:>8} {sl_mult:>8.1f} {tp_target:>10} {len(df_trades):>6} {win_rate:>8.1f}% {real_pnl:>10.2f}% {expected:>10.3f}%")
                    
                    if expected > best_expected:
                        best_expected = expected
                        best_result = {
                            'strategy': 'Fade',
                            'rsi': rsi_th,
                            'sl_mult': sl_mult,
                            'tp_target': tp_target,
                            'trades': df_trades.copy()
                        }

# ============================================================
# 최적 결과
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최적 전략 결과 ★★★")
print("=" * 80)

if best_result and best_expected > 0:
    trades = best_result['trades']
    trades['real_pnl'] = trades['pnl'] - COST
    
    tp = trades[trades['reason'] == 'TP']
    sl = trades[trades['reason'] == 'SL']
    
    print(f"""
■ 최적 파라미터:
  - RSI 임계값: {best_result['rsi']} / {100 - best_result['rsi']}
  - 손절: 밴드폭 × {best_result['sl_mult']}
  - 익절: {best_result['tp_target']}

■ 성과 (수수료 {COST}% 차감 후):
  - 총 거래: {len(trades)}건
  - 익절: {len(tp)}건 ({len(tp)/len(trades)*100:.1f}%)
  - 손절: {len(sl)}건 ({len(sl)/len(trades)*100:.1f}%)
  - 승률: {len(tp)/len(trades)*100:.1f}%
  - 평균 실제 PnL: {trades['real_pnl'].mean():.2f}%
  - 기대값: {best_expected:.3f}%

■ 손익 상세:
  - 익절 평균: +{tp['real_pnl'].mean():.2f}%
  - 손절 평균: {sl['real_pnl'].mean():.2f}%
  - 손익비: {abs(tp['real_pnl'].mean() / sl['real_pnl'].mean()):.2f}

■ 100회 거래 시 예상 수익:
  - 총 PnL: {best_expected * 100:.1f}%
""")
    
    # 저장
    trades.to_csv('bb_improved_results.csv', index=False)
    print(f"저장: bb_improved_results.csv")
    
else:
    print("\n수수료 차감 후 플러스 기대값 전략 없음")
    print("추가 개선 필요...")

# ============================================================
# 비교: 기존 vs 개선
# ============================================================
print("\n" + "=" * 80)
print("★ 기존 전략 vs 개선 전략 비교 ★")
print("=" * 80)

# 기존 전략 로드
try:
    df_old = pd.read_csv('bb_ema_all_signals.csv')
    df_old['real_pnl'] = df_old['pnl'] - COST
    
    old_tp = df_old[df_old['exit_reason'] == 'MID_LINE_TP']
    old_sl = df_old[df_old['exit_reason'] == 'STOP_LOSS']
    
    old_expected = (len(old_tp)/len(df_old)) * old_tp['real_pnl'].mean() + (len(old_sl)/len(df_old)) * old_sl['real_pnl'].mean()
    
    print(f"""
■ 기존 전략 (수축→돌파 + 중간선 익절):
  - 거래: {len(df_old)}건
  - 승률: {len(old_tp)/len(df_old)*100:.1f}%
  - 기대값: {old_expected:.3f}%

■ 개선 전략 (Mean Reversion + RSI):
  - 거래: {len(trades)}건
  - 승률: {len(tp)/len(trades)*100:.1f}%
  - 기대값: {best_expected:.3f}%

■ 개선 효과:
  - 기대값: {old_expected:.3f}% → {best_expected:.3f}% ({'+' if best_expected > old_expected else ''}{best_expected - old_expected:.3f}%p)
""")
except:
    print("기존 결과 파일 없음")
