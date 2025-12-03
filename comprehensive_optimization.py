"""
종합 TP/SL 최적화
여러 방식 비교:
1. 고정 % TP/SL
2. HL 기준 TP/SL
3. 진입캔들 기준 TP/SL
4. ATR 기준 TP/SL
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
print("종합 TP/SL 최적화")
print("="*80)

# 각 시그널에 추가 정보 계산
for idx, signal in signals.iterrows():
    entry_time = signal['breakout_time']
    candle = df_15m[df_15m['datetime'] == entry_time]
    
    if len(candle) > 0:
        signals.loc[idx, 'candle_high'] = candle.iloc[0]['high']
        signals.loc[idx, 'candle_low'] = candle.iloc[0]['low']
        
        # 최근 20봉 ATR 계산
        recent_20 = df_15m[df_15m['datetime'] <= entry_time].tail(20)
        if len(recent_20) > 0:
            atr = (recent_20['high'] - recent_20['low']).mean()
            signals.loc[idx, 'atr'] = atr
        else:
            signals.loc[idx, 'atr'] = signal['breakout_price'] * 0.01
    else:
        signals.loc[idx, 'candle_high'] = signal['breakout_price'] * 1.01
        signals.loc[idx, 'candle_low'] = signal['breakout_price'] * 0.99
        signals.loc[idx, 'atr'] = signal['breakout_price'] * 0.01

print(f"시그널 수: {len(signals)}")

# ATR 통계
atr_pct = (signals['atr'] / signals['breakout_price'] * 100).mean()
print(f"평균 ATR: {atr_pct:.2f}%")

# Entry - HL 거리
entry_hl_dist = ((signals['breakout_price'] - signals['hl_price']) / signals['breakout_price'] * 100).mean()
print(f"평균 Entry-HL 거리: {entry_hl_dist:.2f}%")

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

def run_backtest(signals_df, future_cache, tp_type, tp_value, sl_type, sl_value,
                 use_trailing, max_hold_hours, split_ratio=0.5):
    """
    범용 백테스트 함수
    
    tp_type: 'fixed' (고정 %), 'hl_mult' (HL 거리 배수), 'atr_mult' (ATR 배수)
    tp_value: tp_type에 따른 값
    sl_type: 'hl', 'candle_low', 'fixed', 'atr_mult'
    sl_value: sl_type에 따른 값
    """
    results = []
    
    for idx, signal in signals_df.iterrows():
        if idx not in future_cache:
            continue
        
        entry_price = signal['breakout_price']
        hl_price = signal['hl_price']
        candle_low = signal['candle_low']
        atr = signal['atr']
        
        # TP 계산
        entry_hl_dist = entry_price - hl_price
        if entry_hl_dist <= 0:
            entry_hl_dist = entry_price * 0.01
        
        if tp_type == 'fixed':
            tp1_target = entry_price * (1 + tp_value / 100 * 0.5)
            tp2_target = entry_price * (1 + tp_value / 100)
        elif tp_type == 'hl_mult':
            tp1_target = entry_price + entry_hl_dist * tp_value * 0.5
            tp2_target = entry_price + entry_hl_dist * tp_value
        elif tp_type == 'atr_mult':
            tp1_target = entry_price + atr * tp_value * 0.5
            tp2_target = entry_price + atr * tp_value
        
        # SL 계산
        if sl_type == 'hl':
            sl_price = hl_price * (1 - sl_value / 100)
        elif sl_type == 'candle_low':
            sl_price = candle_low * (1 - sl_value / 100)
        elif sl_type == 'fixed':
            sl_price = entry_price * (1 - sl_value / 100)
        elif sl_type == 'atr_mult':
            sl_price = entry_price - atr * sl_value
        
        if sl_price >= entry_price:
            sl_price = entry_price * 0.99
        
        # 캐시 데이터
        cache = future_cache[idx]
        times = cache['times']
        highs = cache['highs']
        lows = cache['lows']
        closes = cache['closes']
        
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
        reward = (tp2_target - entry_price) / entry_price * 100
        
        results.append({
            'total_pnl': total_pnl,
            'tp1_done': tp1_done,
            'tp2_done': tp2_done,
            'sl_done': sl_done,
            'risk': risk,
            'reward': reward
        })
    
    return pd.DataFrame(results)

# 최적화 실행
print("\n" + "="*80)
print("각 방식별 최적화")
print("="*80)

all_results = []

# 1. 고정 % TP/SL
print("\n[1] 고정 % TP/SL 테스트...")
for tp_pct in [2, 3, 4, 5, 7, 10]:
    for sl_pct in [1, 1.5, 2, 3]:
        for trailing in [False, True]:
            for max_hold in [24, 48, 72]:
                for split in [0.3, 0.5, 0.7]:
                    df = run_backtest(signals, future_cache, 
                                     'fixed', tp_pct, 'fixed', sl_pct,
                                     trailing, max_hold, split)
                    if len(df) > 0:
                        all_results.append({
                            'method': f'Fixed TP{tp_pct}%/SL{sl_pct}%',
                            'tp_type': 'fixed', 'tp_value': tp_pct,
                            'sl_type': 'fixed', 'sl_value': sl_pct,
                            'trailing': trailing, 'max_hold': max_hold, 'split': split,
                            'tp1_rate': df['tp1_done'].mean() * 100,
                            'tp2_rate': df['tp2_done'].mean() * 100,
                            'sl_rate': df['sl_done'].mean() * 100,
                            'win_rate': (df['total_pnl'] > 0).mean() * 100,
                            'avg_pnl': df['total_pnl'].mean(),
                            'total_pnl': df['total_pnl'].sum()
                        })

# 2. HL 기준 TP / HL SL
print("[2] HL 기준 TP/SL 테스트...")
for tp_mult in [1, 2, 3, 5, 7, 10]:
    for sl_buf in [0, 0.3, 0.5, 1.0]:
        for trailing in [False, True]:
            for max_hold in [24, 48, 72]:
                for split in [0.3, 0.5, 0.7]:
                    df = run_backtest(signals, future_cache,
                                     'hl_mult', tp_mult, 'hl', sl_buf,
                                     trailing, max_hold, split)
                    if len(df) > 0:
                        all_results.append({
                            'method': f'HL TP{tp_mult}x/SL HL-{sl_buf}%',
                            'tp_type': 'hl_mult', 'tp_value': tp_mult,
                            'sl_type': 'hl', 'sl_value': sl_buf,
                            'trailing': trailing, 'max_hold': max_hold, 'split': split,
                            'tp1_rate': df['tp1_done'].mean() * 100,
                            'tp2_rate': df['tp2_done'].mean() * 100,
                            'sl_rate': df['sl_done'].mean() * 100,
                            'win_rate': (df['total_pnl'] > 0).mean() * 100,
                            'avg_pnl': df['total_pnl'].mean(),
                            'total_pnl': df['total_pnl'].sum()
                        })

# 3. ATR 기준 TP/SL
print("[3] ATR 기준 TP/SL 테스트...")
for tp_mult in [1, 2, 3, 4, 5]:
    for sl_mult in [0.5, 1, 1.5, 2]:
        for trailing in [False, True]:
            for max_hold in [24, 48, 72]:
                for split in [0.3, 0.5, 0.7]:
                    df = run_backtest(signals, future_cache,
                                     'atr_mult', tp_mult, 'atr_mult', sl_mult,
                                     trailing, max_hold, split)
                    if len(df) > 0:
                        all_results.append({
                            'method': f'ATR TP{tp_mult}x/SL{sl_mult}x',
                            'tp_type': 'atr_mult', 'tp_value': tp_mult,
                            'sl_type': 'atr_mult', 'sl_value': sl_mult,
                            'trailing': trailing, 'max_hold': max_hold, 'split': split,
                            'tp1_rate': df['tp1_done'].mean() * 100,
                            'tp2_rate': df['tp2_done'].mean() * 100,
                            'sl_rate': df['sl_done'].mean() * 100,
                            'win_rate': (df['total_pnl'] > 0).mean() * 100,
                            'avg_pnl': df['total_pnl'].mean(),
                            'total_pnl': df['total_pnl'].sum()
                        })

opt_df = pd.DataFrame(all_results)
opt_df = opt_df.sort_values('avg_pnl', ascending=False)

print(f"\n총 {len(opt_df)}개 조합 테스트 완료")

# 방식별 최고 성과
print("\n" + "="*80)
print("방식별 최고 성과")
print("="*80)

for tp_type in ['fixed', 'hl_mult', 'atr_mult']:
    subset = opt_df[opt_df['tp_type'] == tp_type]
    if len(subset) > 0:
        best = subset.iloc[0]
        print(f"\n[{tp_type.upper()}] 최고: {best['method']}")
        print(f"  Trailing: {best['trailing']} | 보유: {best['max_hold']}h | Split: {best['split']*100:.0f}%")
        print(f"  TP1: {best['tp1_rate']:.1f}% | TP2: {best['tp2_rate']:.1f}% | SL: {best['sl_rate']:.1f}%")
        print(f"  승률: {best['win_rate']:.1f}% | 평균: {best['avg_pnl']:+.2f}% | 총: {best['total_pnl']:+.1f}%")

# 전체 TOP 20
print("\n" + "="*80)
print("🏆 전체 TOP 20 (평균 수익 기준)")
print("="*80)

for i, row in opt_df.head(20).iterrows():
    print(f"\n{row['method']} | Trailing: {row['trailing']} | {row['max_hold']}h | {row['split']*100:.0f}%")
    print(f"  승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}% | TP1: {row['tp1_rate']:.1f}%")

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

print(f"\n   방식: {best['method']}")
print(f"   Trailing: {best['trailing']}")
print(f"   최대 보유: {best['max_hold']}시간")
print(f"   분할: {best['split']*100:.0f}%")
print(f"\n   TP1 달성: {best['tp1_rate']:.1f}%")
print(f"   TP2 달성: {best['tp2_rate']:.1f}%")
print(f"   손절: {best['sl_rate']:.1f}%")
print(f"   승률: {best['win_rate']:.1f}%")
print(f"   평균 수익: {best['avg_pnl']:+.2f}%")
print(f"   총 수익: {best['total_pnl']:+.1f}%")

opt_df.to_csv('comprehensive_optimization.csv', index=False)
print(f"\n저장: comprehensive_optimization.csv")

# 기존 대비 비교
print("\n" + "="*80)
print("📊 기존 전략 대비")
print("="*80)
print(f"기존: TP달성 17.3%, 승률 26.4%, 평균 +0.60%")
print(f"최적화: TP1달성 {best['tp1_rate']:.1f}%, 승률 {best['win_rate']:.1f}%, 평균 {best['avg_pnl']:+.2f}%")

