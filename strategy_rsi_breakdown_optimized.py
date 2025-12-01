#!/usr/bin/env python3
"""
RSI Breakdown Strategy (최적화 버전)

전략 규칙:
1. MACD < 0 (하락 추세)
2. RSI가 10 이하로 터진 후 다시 하락한 구간
3. RSI 10 이하 터진 구간의 저점(L라인)을 하향 돌파하면 숏 진입

진입: 15분봉 종가가 L라인 하향 돌파
청산:
- TP1: +1.5% (50% 청산, SL을 진입가로 이동)
- TP2: +3.5% (나머지 50% 청산)
- SL: -1.5%
- Time Stop: 12시간 (48봉)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드 및 지표 계산
# ============================================================

def calculate_indicators(df):
    """MACD, RSI 계산"""
    df = df.copy()
    
    # MACD 계산
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    df['macd'] = macd
    
    # RSI 계산
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI 10 이하 구간 마킹
    df['rsi_below_10'] = (df['rsi'] < 10).astype(int)
    
    # 각 RSI 10 이하 구간의 L라인(최저가) 계산
    df['zone_id'] = (df['rsi_below_10'].diff() == 1).cumsum() * df['rsi_below_10']
    
    return df

print("=" * 60)
print("RSI Breakdown Strategy Backtest (Optimized)")
print("=" * 60)

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_15m = calculate_indicators(df_15m)
df_15m = df_15m.dropna().reset_index(drop=True)

print(f"\n📊 데이터 기간: {df_15m['datetime'].min()} ~ {df_15m['datetime'].max()}")
print(f"📊 총 15분봉 개수: {len(df_15m):,}개")

# 각 존의 L라인 계산
print("\n🔍 RSI 10 이하 구간 및 L라인 계산 중...")
zone_l_lines = df_15m[df_15m['zone_id'] > 0].groupby('zone_id')['low'].min().to_dict()
print(f"✅ 총 {len(zone_l_lines)}개 RSI 과매도 구간 탐지됨")

# 각 행에 해당 존의 L라인 매핑
df_15m['current_l_line'] = df_15m['zone_id'].map(zone_l_lines)

# 존 종료 시점 이후에도 L라인 유지 (forward fill, 단 다음 존 시작 전까지만)
df_15m['l_line'] = np.nan
current_l = np.nan
for i in range(len(df_15m)):
    if df_15m.loc[i, 'zone_id'] > 0:
        current_l = df_15m.loc[i, 'current_l_line']
    df_15m.loc[i, 'l_line'] = current_l

# ============================================================
# 2. 신호 생성 (벡터화)
# ============================================================

print("\n📡 L라인 돌파 신호 생성 중...")

# 조건 계산
df_15m['macd_negative'] = df_15m['macd'] < 0
df_15m['rsi_declining'] = df_15m['rsi'] < df_15m['rsi'].shift(1)
df_15m['below_l_line'] = df_15m['close'] < df_15m['l_line']
df_15m['prev_above_l_line'] = df_15m['close'].shift(1) >= df_15m['l_line'].shift(1)
df_15m['breakdown'] = df_15m['below_l_line'] & df_15m['prev_above_l_line']

# 존 종료 후 구간만 (zone_id == 0이고 l_line이 있는 구간)
df_15m['in_post_zone'] = (df_15m['zone_id'] == 0) & (df_15m['l_line'].notna())

# 모든 조건 만족
df_15m['signal'] = (
    df_15m['macd_negative'] & 
    df_15m['rsi_declining'] & 
    df_15m['breakdown'] &
    df_15m['in_post_zone']
)

# 신호 추출
signals_df = df_15m[df_15m['signal']].copy()
print(f"✅ 총 {len(signals_df)}개 신호 생성됨")

if len(signals_df) == 0:
    print("\n⚠️ 생성된 신호가 없습니다. 전략 조건을 완화해주세요.")
    print("\n💡 디버그 정보:")
    print(f"   MACD < 0: {df_15m['macd_negative'].sum():,}개 봉")
    print(f"   RSI 하락 중: {df_15m['rsi_declining'].sum():,}개 봉")
    print(f"   L라인 존재: {df_15m['l_line'].notna().sum():,}개 봉")
    print(f"   L라인 하향돌파: {df_15m['breakdown'].sum():,}개 봉")
    exit()

# ============================================================
# 3. 백테스트 실행
# ============================================================

def backtest_strategy(df, signals_df, tp1_pct=1.5, tp2_pct=3.5, sl_pct=1.5, time_stop_bars=48):
    """백테스트 실행 (숏 포지션)"""
    trades = []
    
    for idx, sig in signals_df.iterrows():
        entry_idx = idx
        entry_price = sig['close']
        entry_time = sig['datetime']
        
        # 청산 레벨 계산 (숏이므로 반대)
        tp1_price = entry_price * (1 - tp1_pct / 100)
        tp2_price = entry_price * (1 - tp2_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
        
        # 포지션 관리
        position = 1.0  # 100% 포지션
        tp1_hit = False
        current_sl = sl_price
        
        for i in range(entry_idx + 1, min(entry_idx + 1 + time_stop_bars, len(df))):
            bar = df.iloc[i]
            
            # TP1 체크
            if not tp1_hit and bar['low'] <= tp1_price:
                position = 0.5
                tp1_hit = True
                current_sl = entry_price  # SL을 진입가로 이동
            
            # TP2 체크
            if tp1_hit and bar['low'] <= tp2_price:
                pnl = (tp1_pct / 2 + tp2_pct / 2)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': bar['datetime'],
                    'entry_price': entry_price,
                    'exit_price': tp2_price,
                    'result': 'TP2',
                    'pnl_pct': pnl,
                    'bars_held': i - entry_idx
                })
                break
            
            # SL 체크 (숏)
            if bar['high'] >= current_sl:
                if tp1_hit:
                    pnl = tp1_pct / 2  # TP1 도달 후 BE 청산
                    result = 'BE'
                else:
                    pnl = -sl_pct
                    result = 'SL'
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': bar['datetime'],
                    'entry_price': entry_price,
                    'exit_price': bar['high'],
                    'result': result,
                    'pnl_pct': pnl,
                    'bars_held': i - entry_idx
                })
                break
        else:
            # Time Stop
            last_bar = df.iloc[min(entry_idx + time_stop_bars, len(df) - 1)]
            exit_price = last_bar['close']
            
            if tp1_hit:
                pnl = (tp1_pct / 2) + ((entry_price - exit_price) / entry_price * 100) * 0.5
                result = 'TIME'
            else:
                pnl = (entry_price - exit_price) / entry_price * 100
                result = 'TIME'
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': last_bar['datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'result': result,
                'pnl_pct': pnl,
                'bars_held': min(time_stop_bars, len(df) - 1 - entry_idx)
            })
    
    return pd.DataFrame(trades)

print("\n⚙️ 백테스트 실행 중...")
print(f"   TP1: +1.5% (50% 청산, SL→BE)")
print(f"   TP2: +3.5% (50% 청산)")
print(f"   SL: -1.5%")
print(f"   Time Stop: 12시간 (48봉)")

df_trades = backtest_strategy(df_15m, signals_df)

# ============================================================
# 4. 결과 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 백테스트 결과")
print("=" * 60)

total_trades = len(df_trades)
if total_trades == 0:
    print("\n⚠️ 체결된 거래가 없습니다.")
    exit()

# 승률 계산 (SL 회피율)
win_trades = len(df_trades[df_trades['result'] != 'SL'])
loss_trades = len(df_trades[df_trades['result'] == 'SL'])
win_rate = win_trades / total_trades * 100

# 수익률 계산
total_pnl = df_trades['pnl_pct'].sum()
avg_pnl = df_trades['pnl_pct'].mean()

# 기간 계산
start_date = df_trades['entry_time'].min()
end_date = df_trades['exit_time'].max()
months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
if months == 0:
    months = 1
monthly_avg_pnl = total_pnl / months

# MDD 계산
cumulative_pnl = df_trades['pnl_pct'].cumsum()
running_max = cumulative_pnl.cummax()
drawdown = cumulative_pnl - running_max
mdd = drawdown.min()

print(f"\n📈 총 거래 수: {total_trades:,}건")
print(f"📈 승률 (SL 회피): {win_rate:.1f}% ({win_trades}승 / {loss_trades}패)")
print(f"📈 총 수익: {total_pnl:.1f}%")
print(f"📈 평균 수익: {avg_pnl:.2f}%")
print(f"📈 월평균 수익: {monthly_avg_pnl:.2f}%")
print(f"📈 MDD: {mdd:.1f}%")
print(f"📈 거래 기간: {start_date.date()} ~ {end_date.date()} ({months}개월)")

# 청산 유형별 분석
print(f"\n📊 청산 유형별 분석:")
for result_type in ['TP2', 'BE', 'TIME', 'SL']:
    subset = df_trades[df_trades['result'] == result_type]
    if len(subset) > 0:
        count = len(subset)
        pct = count / total_trades * 100
        avg_pnl_type = subset['pnl_pct'].mean()
        print(f"   {result_type:4s}: {count:4d}건 ({pct:5.1f}%) | 평균 수익: {avg_pnl_type:+6.2f}%")

# 실전 승률 계산 (PNL > 0)
practical_wins = len(df_trades[df_trades['pnl_pct'] > 0])
practical_win_rate = practical_wins / total_trades * 100
print(f"\n✅ 실전 승률 (PNL > 0): {practical_win_rate:.1f}%")

# 저장
df_trades.to_csv('strategy_rsi_breakdown_results.csv', index=False)
print(f"\n💾 결과 저장됨: strategy_rsi_breakdown_results.csv")

print("\n" + "=" * 60)
