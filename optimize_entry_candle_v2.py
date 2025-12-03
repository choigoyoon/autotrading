"""
진입 캔들 High/Low 기준 TP/SL 최적화 v2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
df_15m = pd.read_csv('analysis_15m.csv')

signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("진입 캔들 High/Low 기준 TP/SL 최적화 v2")
print("="*80)

# 각 시그널의 진입 캔들 정보 추출
for idx, signal in signals.iterrows():
    entry_time = signal['breakout_time']
    candle = df_15m[df_15m['datetime'] == entry_time]
    
    if len(candle) > 0:
        signals.loc[idx, 'candle_high'] = candle.iloc[0]['high']
        signals.loc[idx, 'candle_low'] = candle.iloc[0]['low']
        signals.loc[idx, 'candle_close'] = candle.iloc[0]['close']
    else:
        signals.loc[idx, 'candle_high'] = signal['breakout_price'] * 1.01
        signals.loc[idx, 'candle_low'] = signal['breakout_price'] * 0.99

print(f"\n시그널 수: {len(signals)}")

# 진입 캔들 통계
print("\n진입 캔들 통계:")
print(f"  High - Entry: 평균 {(signals['candle_high'] - signals['breakout_price']).mean()/signals['breakout_price'].mean()*100:.2f}%")
print(f"  Entry - Low: 평균 {(signals['breakout_price'] - signals['candle_low']).mean()/signals['breakout_price'].mean()*100:.2f}%")
print(f"  Entry - HL: 평균 {(signals['breakout_price'] - signals['hl_price']).mean()/signals['breakout_price'].mean()*100:.2f}%")

# 미래 데이터 캐시
def precompute_future_data(signals, df_15m, max_hours=72):
    all_future = {}
    for idx, signal in signals.iterrows():
        entry_time = signal['breakout_time']
        future = df_15m[df_15m['datetime'] > entry_time].head(max_hours * 4)
        
        if len(future) > 0:
            all_future[idx] = {
                'times': (future['datetime'] - entry_time).dt.total_seconds().values / 3600,
                'highs': future['high'].values,
                'lows': future['low'].values,
                'closes': future['close'].values
            }
    return all_future

print("\n미래 데이터 캐싱...")
future_cache = precompute_future_data(signals, df_15m)

def run_backtest(signals_df, future_cache, tp_multiplier, sl_type, sl_buffer_pct,
                 use_trailing, max_hold_hours, split_ratio=0.5):
    """
    tp_multiplier: 진입캔들 고점-진입가 거리의 배수 (1.0 = 고점, 2.0 = 2배)
    sl_type: 'hl' or 'candle_low'
    """
    results = []
    
    for idx, signal in signals_df.iterrows():
        if idx not in future_cache:
            continue
        
        entry_price = signal['breakout_price']
        candle_high = signal['candle_high']
        candle_low = signal['candle_low']
        hl_price = signal['hl_price']
        
        # TP 목표: 진입가 + (고점-진입가) * 배수
        tp_distance = candle_high - entry_price
        if tp_distance <= 0:
            tp_distance = entry_price * 0.01
        
        tp1_target = entry_price + tp_distance * tp_multiplier * 0.5
        tp2_target = entry_price + tp_distance * tp_multiplier
        
        # SL 결정
        if sl_type == 'hl':
            base_sl = hl_price
        else:
            base_sl = candle_low
        
        sl_price = base_sl * (1 - sl_buffer_pct / 100)
        
        if sl_price >= entry_price:
            sl_price = entry_price * 0.99
        
        # 캐시 데이터
        cache = future_cache[idx]
        times = cache['times']
        highs = cache['highs']
        lows = cache['lows']
        closes = cache['closes']
        
        # 시간 제한
        mask = times <= max_hold_hours
        times, highs, lows, closes = times[mask], highs[mask], lows[mask], closes[mask]
        
        if len(times) == 0:
            continue
        
        # 시뮬레이션
        position = 1.0
        total_pnl = 0
        tp1_done = False
        tp2_done = False
        sl_done = False
        current_sl = sl_price
        
        for i in range(len(times)):
            if lows[i] <= current_sl and position > 0:
                sl_pnl = (current_sl - entry_price) / entry_price * 100 * position
                total_pnl += sl_pnl
                sl_done = True
                position = 0
                break
            
            if not tp1_done and highs[i] >= tp1_target and position > 0:
                tp1_pnl = (tp1_target - entry_price) / entry_price * 100 * split_ratio
                total_pnl += tp1_pnl
                position -= split_ratio
                tp1_done = True
                
                if use_trailing:
                    current_sl = entry_price
            
            if not tp2_done and highs[i] >= tp2_target and position > 0:
                tp2_pnl = (tp2_target - entry_price) / entry_price * 100 * position
                total_pnl += tp2_pnl
                position = 0
                tp2_done = True
                break
        
        if position > 0 and len(closes) > 0:
            pnl = (closes[-1] - entry_price) / entry_price * 100 * position
            total_pnl += pnl
        
        risk = abs(entry_price - sl_price) / entry_price * 100
        reward = tp_distance * tp_multiplier / entry_price * 100
        
        results.append({
            'total_pnl': total_pnl,
            'tp1_done': tp1_done,
            'tp2_done': tp2_done,
            'sl_done': sl_done,
            'risk': risk,
            'reward': reward
        })
    
    return pd.DataFrame(results)

# 최적화
print("\n" + "="*80)
print("파라미터 최적화 시작")
print("="*80)

tp_multipliers = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
sl_types = ['hl', 'candle_low']
sl_buffers = [0.0, 0.3, 0.5, 1.0]
trailing_options = [False, True]
max_hold_options = [24, 48, 72]
split_options = [0.3, 0.5, 0.7]

results = []
count = 0

for tp_mult in tp_multipliers:
    for sl_type in sl_types:
        for sl_buf in sl_buffers:
            for trailing in trailing_options:
                for max_hold in max_hold_options:
                    for split in split_options:
                        count += 1
                        
                        df = run_backtest(
                            signals, future_cache,
                            tp_mult, sl_type, sl_buf, trailing, max_hold, split
                        )
                        
                        if len(df) == 0:
                            continue
                        
                        results.append({
                            'tp_multiplier': tp_mult,
                            'sl_type': sl_type,
                            'sl_buffer': sl_buf,
                            'trailing': trailing,
                            'max_hold': max_hold,
                            'split': split,
                            'signals': len(df),
                            'tp1_rate': df['tp1_done'].mean() * 100,
                            'tp2_rate': df['tp2_done'].mean() * 100,
                            'sl_rate': df['sl_done'].mean() * 100,
                            'win_rate': (df['total_pnl'] > 0).mean() * 100,
                            'avg_pnl': df['total_pnl'].mean(),
                            'total_pnl': df['total_pnl'].sum(),
                            'avg_risk': df['risk'].mean(),
                            'avg_reward': df['reward'].mean()
                        })

print(f"총 {count}개 조합 테스트")

opt_df = pd.DataFrame(results)
opt_df = opt_df.sort_values('avg_pnl', ascending=False)

print("\n" + "="*80)
print("🏆 TOP 15 (평균 수익 기준)")
print("="*80)

for i, row in opt_df.head(15).iterrows():
    print(f"\nTP배수: {row['tp_multiplier']:.1f}x | SL: {row['sl_type']} -{row['sl_buffer']:.1f}% | Split: {row['split']*100:.0f}%")
    print(f"  Trailing: {row['trailing']} | 보유: {row['max_hold']}h")
    print(f"  TP1: {row['tp1_rate']:.1f}% | TP2: {row['tp2_rate']:.1f}% | SL: {row['sl_rate']:.1f}%")
    print(f"  승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}% | 총: {row['total_pnl']:+.1f}%")

print("\n" + "="*80)
print("🎯 TOP 10 (승률 기준)")
print("="*80)

for i, row in opt_df.sort_values('win_rate', ascending=False).head(10).iterrows():
    print(f"TP: {row['tp_multiplier']:.1f}x | SL: {row['sl_type']} | 승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}%")

# 최종 추천
print("\n" + "="*80)
print("🏆 최종 추천")
print("="*80)

filtered = opt_df[(opt_df['win_rate'] >= 55) & (opt_df['avg_pnl'] >= 0.5)]
if len(filtered) > 0:
    best = filtered.sort_values('avg_pnl', ascending=False).iloc[0]
    print("\n✅ 균형 조건 (승률 55%+, 평균 0.5%+)")
else:
    filtered = opt_df[(opt_df['win_rate'] >= 50) & (opt_df['avg_pnl'] >= 0.3)]
    if len(filtered) > 0:
        best = filtered.sort_values('avg_pnl', ascending=False).iloc[0]
        print("\n✅ 균형 조건 (승률 50%+, 평균 0.3%+)")
    else:
        best = opt_df.iloc[0]
        print("\n✅ 평균 수익 최고")

print(f"\n   === 전략 파라미터 ===")
print(f"   TP 배수: {best['tp_multiplier']:.1f}x (진입캔들 고점-진입가 거리)")
print(f"   SL 기준: {best['sl_type']} -{best['sl_buffer']:.1f}%")
print(f"   분할 비율: TP1에서 {best['split']*100:.0f}% 매도")
print(f"   Trailing SL: {best['trailing']}")
print(f"   최대 보유: {best['max_hold']}시간")
print(f"\n   === 성과 ===")
print(f"   TP1 달성률: {best['tp1_rate']:.1f}%")
print(f"   TP2 달성률: {best['tp2_rate']:.1f}%")
print(f"   손절률: {best['sl_rate']:.1f}%")
print(f"   승률: {best['win_rate']:.1f}%")
print(f"   평균 수익: {best['avg_pnl']:+.2f}%")
print(f"   총 수익: {best['total_pnl']:+.1f}%")

opt_df.to_csv('entry_candle_optimization.csv', index=False)
print(f"\n저장: entry_candle_optimization.csv")

