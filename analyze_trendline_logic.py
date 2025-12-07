import pandas as pd
import numpy as np

# Load data
signals_df = pd.read_csv('valid_signals.csv')
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

TP = 5.0
SL = 2.0

def backtest_new_logic(signal):
    """
    새로운 로직:
    1. 진입
    2. 추세선 '터치'만으로는 청산 안 함
    3. 추세선 아래로 '마감'하면 손절
    4. TP 5% 도달하면 익절
    """
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    trendline_price = signal['trendline_price']
    h1_time = pd.to_datetime(signal['h1_time'])
    h2_time = pd.to_datetime(signal['h2_time'])
    h1_price = signal['h1_price']
    h2_price = signal['h2_price']
    
    # Get future candles (72시간 = 288 캔들)
    future = candles_df[candles_df['datetime'] > entry_time].head(288)
    
    if len(future) == 0:
        return None
    
    # 추세선 연장 계산
    h1_idx = candles_df[candles_df['datetime'] == h1_time].index[0]
    h2_idx = candles_df[candles_df['datetime'] == h2_time].index[0]
    slope = (h2_price - h1_price) / (h2_idx - h1_idx)
    
    exit_reason = None
    exit_time = None
    exit_price = None
    pnl = 0
    touched_trendline = False
    
    for i, (idx, candle) in enumerate(future.iterrows()):
        # 현재 추세선 가격 계산
        current_trendline = trendline_price + slope * i
        
        # Check TP first
        if candle['high'] >= entry_price * (1 + TP/100):
            exit_reason = 'TP'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 + TP/100)
            pnl = TP
            break
        
        # 추세선 터치 체크 (low가 닿음)
        if candle['low'] <= current_trendline:
            touched_trendline = True
        
        # 추세선 아래로 '마감' 체크 (close가 아래)
        if candle['close'] < current_trendline:
            exit_reason = 'TRENDLINE_BREAK_CLOSE'
            exit_time = candle['datetime']
            exit_price = candle['close']
            pnl = ((exit_price - entry_price) / entry_price) * 100
            break
        
        # SL 체크
        if candle['low'] <= entry_price * (1 - SL/100):
            exit_reason = 'SL'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 - SL/100)
            pnl = -SL
            break
    
    # Time limit (72시간)
    if exit_reason is None:
        last_candle = future.iloc[-1]
        exit_reason = 'TIME_72H'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
        pnl = ((exit_price - entry_price) / entry_price) * 100
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'touched_trendline': touched_trendline
    }

def backtest_old_logic(signal):
    """
    기존 로직: 추세선 터치만으로 즉시 청산
    """
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    trendline_price = signal['trendline_price']
    
    future = candles_df[candles_df['datetime'] > entry_time].head(96)  # 24시간
    
    if len(future) == 0:
        return None
    
    exit_reason = None
    exit_time = None
    exit_price = None
    pnl = 0
    
    for i, (idx, candle) in enumerate(future.iterrows()):
        # 추세선 터치 즉시 청산
        if candle['low'] <= trendline_price:
            exit_reason = 'RETEST_TOUCH'
            exit_time = candle['datetime']
            exit_price = trendline_price
            pnl = ((exit_price - entry_price) / entry_price) * 100
            break
        
        if candle['high'] >= entry_price * (1 + TP/100):
            exit_reason = 'TP'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 + TP/100)
            pnl = TP
            break
        
        if candle['low'] <= entry_price * (1 - SL/100):
            exit_reason = 'SL'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 - SL/100)
            pnl = -SL
            break
    
    if exit_reason is None:
        last_candle = future.iloc[-1]
        exit_reason = 'TIME_24H'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
        pnl = ((exit_price - entry_price) / entry_price) * 100
    
    return {
        'exit_reason': exit_reason,
        'pnl': pnl
    }

# 백테스트
print("=" * 80)
print("기존 로직 vs 새로운 로직 비교")
print("=" * 80)

old_trades = []
new_trades = []

for idx, signal in signals_df.iterrows():
    old_result = backtest_old_logic(signal)
    new_result = backtest_new_logic(signal)
    
    if old_result and new_result:
        old_trades.append({**signal.to_dict(), **old_result})
        new_trades.append({**signal.to_dict(), **new_result})

old_df = pd.DataFrame(old_trades)
new_df = pd.DataFrame(new_trades)

print("\n" + "=" * 80)
print("기존 로직: 추세선 '터치'만 해도 즉시 청산")
print("=" * 80)

print(f"\n총 거래: {len(old_df)}건")
print(f"평균 수익: {old_df['pnl'].mean():.2f}%")
print(f"승률: {(old_df['pnl'] > 0).mean()*100:.1f}%")

print("\n청산 이유별:")
for reason in old_df['exit_reason'].unique():
    subset = old_df[old_df['exit_reason'] == reason]
    print(f"  {reason}: {len(subset)}건 ({len(subset)/len(old_df)*100:.1f}%), 평균 {subset['pnl'].mean():.2f}%")

print("\n" + "=" * 80)
print("새 로직: 추세선 아래로 '마감'해야 손절 (터치는 OK)")
print("=" * 80)

print(f"\n총 거래: {len(new_df)}건")
print(f"평균 수익: {new_df['pnl'].mean():.2f}%")
print(f"승률: {(new_df['pnl'] > 0).mean()*100:.1f}%")
print(f"추세선 터치했지만 버틴 경우: {new_df['touched_trendline'].sum()}건")

print("\n청산 이유별:")
for reason in new_df['exit_reason'].unique():
    subset = new_df[new_df['exit_reason'] == reason]
    print(f"  {reason}: {len(subset)}건 ({len(subset)/len(new_df)*100:.1f}%), 평균 {subset['pnl'].mean():.2f}%")

print("\n" + "=" * 80)
print("🔥 개선 효과")
print("=" * 80)

improvement = new_df['pnl'].mean() - old_df['pnl'].mean()
print(f"\n평균 수익 개선: {old_df['pnl'].mean():.2f}% → {new_df['pnl'].mean():.2f}% (+{improvement:.2f}%)")

# TP 도달률 비교
old_tp_rate = (old_df['exit_reason'] == 'TP').sum() / len(old_df) * 100
new_tp_rate = (new_df['exit_reason'] == 'TP').sum() / len(new_df) * 100
print(f"TP 도달률: {old_tp_rate:.1f}% → {new_tp_rate:.1f}% (+{new_tp_rate-old_tp_rate:.1f}%p)")

# 조기 청산 방지
old_early_exit = (old_df['exit_reason'] == 'RETEST_TOUCH').sum()
new_early_exit = (new_df['exit_reason'] == 'TRENDLINE_BREAK_CLOSE').sum()
print(f"조기 청산: {old_early_exit}건 → {new_early_exit}건 (-{old_early_exit-new_early_exit}건)")

# 연 수익 추정
years = 5.7
old_annual = (1 + old_df['pnl'].mean()/100) ** (len(old_df)/years) - 1
new_annual = (1 + new_df['pnl'].mean()/100) ** (len(new_df)/years) - 1
print(f"\n연 수익률 (복리): {old_annual*100:.1f}% → {new_annual*100:.1f}%")

print("\n" + "=" * 80)
print("핵심 인사이트")
print("=" * 80)

# 터치했지만 다시 올라간 케이스
touched_but_recovered = new_df[
    (new_df['touched_trendline'] == True) & 
    (new_df['exit_reason'].isin(['TP', 'TIME_72H']))
]

if len(touched_but_recovered) > 0:
    print(f"\n✅ 추세선 터치했지만 다시 올라간 경우: {len(touched_but_recovered)}건")
    print(f"   평균 수익: {touched_but_recovered['pnl'].mean():.2f}%")
    print(f"   TP 도달: {(touched_but_recovered['exit_reason'] == 'TP').sum()}건")
    print(f"   → 기존 로직이었으면 본전(-0.2%) 청산했을 것들!")

# 마감으로 청산된 케이스
close_below = new_df[new_df['exit_reason'] == 'TRENDLINE_BREAK_CLOSE']
if len(close_below) > 0:
    print(f"\n🛑 추세선 아래로 마감해서 손절: {len(close_below)}건")
    print(f"   평균 손실: {close_below['pnl'].mean():.2f}%")
    print(f"   → 정확한 손절 타이밍!")

