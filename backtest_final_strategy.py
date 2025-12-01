#!/usr/bin/env python3
"""
최종 전략 백테스트 - 매매 횟수 및 성과 체크

전략:
1. 바닥 징후 신호 (RSI, MACD, Volume, BB, ATR)
2. 연속 LL 3~4번 포착
3. 다이버전스 확인
4. 첫 양봉 또는 RSI 회복 시 진입
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("최종 전략 백테스트 - 매매 횟수 체크")
print("=" * 70)

# ============================================================
# 데이터 로드 및 지표 계산
# ============================================================

def calculate_all_indicators(df):
    """모든 지표 계산"""
    df = df.copy()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd - signal
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Volume
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 캔들 방향
    df['is_green'] = df['close'] > df['open']
    
    return df

print("\n📊 데이터 로드 및 지표 계산 중...")
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = calculate_all_indicators(df)
df = df.dropna().reset_index(drop=True)

print(f"✅ 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"✅ 총 캔들 수: {len(df):,}개")
print(f"✅ 기간: {(df['datetime'].max() - df['datetime'].min()).days / 30:.1f}개월")

# ============================================================
# Swing Low 탐지
# ============================================================

print("\n🔍 Swing Low (L값) 탐지 중...")

order = 10
low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]

l_values = df.iloc[low_indices].copy()
l_values['l_price'] = l_values['low']
l_values = l_values.reset_index(drop=True)

print(f"✅ Swing Low: {len(l_values):,}개")

# ============================================================
# 연속 LL 추적 및 각 L의 연속 횟수 기록
# ============================================================

print("\n🔍 연속 LL 추적 중...")

consecutive_ll_count = 0
l_consecutive_map = {}  # L값 인덱스 → 연속 LL 횟수

for i in range(1, len(l_values)):
    prev_l = l_values.iloc[i - 1]
    curr_l = l_values.iloc[i]
    
    if curr_l['l_price'] < prev_l['l_price']:
        consecutive_ll_count += 1
        l_consecutive_map[i] = consecutive_ll_count
    else:
        l_consecutive_map[i] = 0
        consecutive_ll_count = 0

# ============================================================
# 전략 시그널 생성
# ============================================================

print("\n🔍 전략 시그널 생성 중...")

signals = []

for i in range(order + 1, len(df)):
    row = df.iloc[i]
    
    # 1. 바닥 징후 체크 (실시간 가능)
    bottom_signals = {
        'rsi_low': row['rsi'] < 30,
        'macd_deep': row['macd_hist'] < -50,
        'volume_high': row['volume_ratio'] > 3.0,
        'bb_bottom': row['bb_position'] < 0.1,
        'atr_high': row['atr_pct'] > 0.5,
    }
    
    bottom_signal_count = sum(bottom_signals.values())
    
    # 바닥 신호 4개 이상 필요
    if bottom_signal_count < 4:
        continue
    
    # 2. 최근 확정된 L값 확인 (현재 시점 이전의 L값)
    # 현재 캔들보다 최소 10개 이전에 확정된 L값만 사용
    recent_l_indices = [idx for idx in low_indices if idx < i - order]
    
    if len(recent_l_indices) < 3:
        continue
    
    # 최근 3개 L값
    last_3_l_indices = recent_l_indices[-3:]
    
    # L값 DataFrame에서 인덱스 찾기
    l_idx_in_df = None
    for idx, l_row in l_values.iterrows():
        if df[df['datetime'] == l_row['datetime']].index[0] in last_3_l_indices:
            l_idx_in_df = idx
    
    if l_idx_in_df is None or l_idx_in_df < 1:
        continue
    
    # 연속 LL 횟수 확인
    if l_idx_in_df not in l_consecutive_map:
        continue
    
    consecutive_ll = l_consecutive_map[l_idx_in_df]
    
    # 3~4번 연속 LL만 (또는 5번+)
    if consecutive_ll < 3:
        continue
    
    # 3. 다이버전스 체크 (옵션)
    # 최근 3개 L의 RSI/MACD 체크
    if l_idx_in_df >= 2:
        l1 = l_values.iloc[l_idx_in_df - 2]
        l2 = l_values.iloc[l_idx_in_df - 1]
        l3 = l_values.iloc[l_idx_in_df]
        
        price_declining = l1['l_price'] > l2['l_price'] > l3['l_price']
        rsi_rising = l1['rsi'] < l2['rsi'] < l3['rsi']
        rsi_divergence = price_declining and rsi_rising
    else:
        rsi_divergence = False
    
    # 4. 진입 트리거: 첫 양봉 또는 RSI 회복
    entry_trigger = row['is_green'] or row['rsi'] > 35
    
    if not entry_trigger:
        continue
    
    # 시그널 생성!
    signals.append({
        'signal_datetime': row['datetime'],
        'signal_idx': i,
        'entry_price': row['close'],
        'bottom_signal_count': bottom_signal_count,
        'consecutive_ll': consecutive_ll,
        'rsi_divergence': rsi_divergence,
        'rsi': row['rsi'],
        'macd_hist': row['macd_hist'],
        'volume_ratio': row['volume_ratio'],
        'atr_pct': row['atr_pct'],
    })

print(f"✅ 시그널 생성 완료: {len(signals)}개")

# ============================================================
# 백테스트 실행
# ============================================================

print("\n🔍 백테스트 실행 중...")

if len(signals) == 0:
    print("\n⚠️ 시그널이 없습니다. 조건을 완화하세요.")
    exit()

trades = []

for sig in signals:
    entry_idx = sig['signal_idx']
    entry_price = sig['entry_price']
    entry_time = sig['signal_datetime']
    
    # 청산 레벨
    tp1_price = entry_price * 1.015  # +1.5%
    tp2_price = entry_price * 1.035  # +3.5%
    sl_price = entry_price * 0.985   # -1.5%
    
    # 포지션 관리
    position = 1.0
    tp1_hit = False
    
    # 48개 봉 (12시간) 추적
    for offset in range(1, 49):
        if entry_idx + offset >= len(df):
            break
        
        bar = df.iloc[entry_idx + offset]
        
        # TP1 체크
        if not tp1_hit and bar['high'] >= tp1_price:
            position = 0.5
            tp1_hit = True
            sl_price = entry_price  # SL을 BE로
        
        # TP2 체크
        if tp1_hit and bar['high'] >= tp2_price:
            pnl_pct = 1.5 / 2 + 3.5 / 2  # (TP1 + TP2) / 2
            trades.append({
                'entry_time': entry_time,
                'exit_time': bar['datetime'],
                'entry_price': entry_price,
                'exit_price': tp2_price,
                'result': 'TP2',
                'pnl_pct': pnl_pct,
                'holding_bars': offset,
                'consecutive_ll': sig['consecutive_ll'],
                'rsi_divergence': sig['rsi_divergence'],
            })
            break
        
        # SL 체크
        if bar['low'] <= sl_price:
            if tp1_hit:
                pnl_pct = 1.5 / 2  # BE
                result = 'BE'
            else:
                pnl_pct = -1.5
                result = 'SL'
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': bar['datetime'],
                'entry_price': entry_price,
                'exit_price': sl_price,
                'result': result,
                'pnl_pct': pnl_pct,
                'holding_bars': offset,
                'consecutive_ll': sig['consecutive_ll'],
                'rsi_divergence': sig['rsi_divergence'],
            })
            break
    else:
        # Time Stop
        last_bar = df.iloc[min(entry_idx + 48, len(df) - 1)]
        exit_price = last_bar['close']
        
        if tp1_hit:
            pnl_pct = (1.5 / 2) + ((exit_price - entry_price) / entry_price * 100) * 0.5
        else:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': last_bar['datetime'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'result': 'TIME',
            'pnl_pct': pnl_pct,
            'holding_bars': min(48, len(df) - 1 - entry_idx),
            'consecutive_ll': sig['consecutive_ll'],
            'rsi_divergence': sig['rsi_divergence'],
        })

df_trades = pd.DataFrame(trades)

# ============================================================
# 결과 분석
# ============================================================

print("\n" + "=" * 70)
print("📊 백테스트 결과")
print("=" * 70)

total_trades = len(df_trades)

if total_trades == 0:
    print("\n⚠️ 거래가 없습니다.")
    exit()

# 기간 계산
start_date = df_trades['entry_time'].min()
end_date = df_trades['exit_time'].max()
months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
if months == 0:
    months = 1
years = months / 12

# 승률
win_trades = len(df_trades[df_trades['result'] != 'SL'])
loss_trades = len(df_trades[df_trades['result'] == 'SL'])
win_rate = win_trades / total_trades * 100

# 실전 승률 (PNL > 0)
practical_wins = len(df_trades[df_trades['pnl_pct'] > 0])
practical_win_rate = practical_wins / total_trades * 100

# 수익률
total_pnl = df_trades['pnl_pct'].sum()
avg_pnl = df_trades['pnl_pct'].mean()
monthly_avg = total_pnl / months

# MDD
cumulative_pnl = df_trades['pnl_pct'].cumsum()
running_max = cumulative_pnl.cummax()
drawdown = cumulative_pnl - running_max
mdd = drawdown.min()

print(f"\n기간: {start_date.date()} ~ {end_date.date()}")
print(f"총 기간: {years:.1f}년 ({months}개월)")
print(f"\n총 거래 수: {total_trades}건")
print(f"월평균 거래: {total_trades/months:.1f}건")
print(f"연평균 거래: {total_trades/years:.1f}건")

print(f"\n승률 (SL 회피): {win_rate:.1f}% ({win_trades}승 / {loss_trades}패)")
print(f"실전 승률 (PNL>0): {practical_win_rate:.1f}%")

print(f"\n총 수익: {total_pnl:.2f}%")
print(f"평균 수익: {avg_pnl:.2f}%")
print(f"월평균 수익: {monthly_avg:.2f}%")
print(f"연평균 수익: {monthly_avg * 12:.2f}%")
print(f"MDD: {mdd:.2f}%")

# 청산 유형별
print(f"\n청산 유형별 분석:")
for result_type in ['TP2', 'BE', 'TIME', 'SL']:
    subset = df_trades[df_trades['result'] == result_type]
    if len(subset) > 0:
        count = len(subset)
        pct = count / total_trades * 100
        avg_pnl = subset['pnl_pct'].mean()
        avg_bars = subset['holding_bars'].mean()
        print(f"   {result_type:4s}: {count:4d}건 ({pct:5.1f}%) | 평균: {avg_pnl:+6.2f}% | 홀딩: {avg_bars:.1f}봉")

# 연속 LL별
print(f"\n연속 LL 횟수별 성과:")
for ll_count in sorted(df_trades['consecutive_ll'].unique()):
    subset = df_trades[df_trades['consecutive_ll'] == ll_count]
    count = len(subset)
    pct = count / total_trades * 100
    avg_pnl = subset['pnl_pct'].mean()
    win_rate_ll = (subset['pnl_pct'] > 0).sum() / len(subset) * 100
    print(f"   {ll_count}번 LL: {count:4d}건 ({pct:5.1f}%) | 평균: {avg_pnl:+6.2f}% | 승률: {win_rate_ll:.1f}%")

# 다이버전스
print(f"\n다이버전스 유무별:")
for has_div in [True, False]:
    subset = df_trades[df_trades['rsi_divergence'] == has_div]
    if len(subset) > 0:
        count = len(subset)
        pct = count / total_trades * 100
        avg_pnl = subset['pnl_pct'].mean()
        win_rate_div = (subset['pnl_pct'] > 0).sum() / len(subset) * 100
        div_label = '다이버전스 O' if has_div else '다이버전스 X'
        print(f"   {div_label}: {count:4d}건 ({pct:5.1f}%) | 평균: {avg_pnl:+6.2f}% | 승률: {win_rate_div:.1f}%")

# ============================================================
# 저장
# ============================================================

df_trades.to_csv('final_strategy_backtest_results.csv', index=False)

print(f"\n💾 결과 저장: final_strategy_backtest_results.csv")

print("\n" + "=" * 70)
print("✅ 백테스트 완료!")
print("=" * 70)
