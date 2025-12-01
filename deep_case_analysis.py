import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("🔬 실제 캔들 움직임 상세 분석 - 각 케이스별")
print("=" * 80)

# Load data
df = pd.read_csv('backtest_confirmation_space_results.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])

ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
ohlcv = ohlcv.sort_values('datetime').reset_index(drop=True)

def analyze_candle_movement(entry_time, entry_price, h1, h2, h3, sl, candle_count=20):
    """진입 후 실제 캔들 움직임 분석"""
    entry_idx = ohlcv[ohlcv['datetime'] == entry_time].index
    if len(entry_idx) == 0:
        return None
    
    entry_idx = entry_idx[0]
    candles = ohlcv.iloc[entry_idx:entry_idx+candle_count].copy()
    
    if len(candles) < 2:
        return None
    
    movements = []
    for i in range(1, len(candles)):
        c = candles.iloc[i]
        movements.append({
            'candle': i,
            'time': c['datetime'],
            'open': c['open'],
            'high': c['high'],
            'low': c['low'],
            'close': c['close'],
            'high_from_entry': ((c['high'] - entry_price) / entry_price) * 100,
            'low_from_entry': ((c['low'] - entry_price) / entry_price) * 100,
            'close_from_entry': ((c['close'] - entry_price) / entry_price) * 100,
            'touched_sl': c['low'] <= sl,
            'touched_h3': c['low'] <= h3,
            'reached_h2': c['high'] >= h2,
            'reached_h1': c['high'] >= h1,
        })
    
    return movements

# 1. SL 케이스 - 진입 직후 하락 타입 (가장 많음 46개)
print("\n" + "=" * 80)
print("1️⃣ SL 실패 케이스: '진입 직후 하락' 유형 - 대표 사례 3개")
print("=" * 80)

sl_cases = df[df['exit_reason'] == 'SL'].head(3)

for idx, trade in sl_cases.iterrows():
    print(f"\n{'='*60}")
    print(f"진입 시간: {trade['entry_time']}")
    print(f"진입 가격: ${trade['entry_price']:.2f}")
    print(f"H1: ${trade['h1_price']:.2f} | H2: ${trade['h2_price']:.2f} | H3: ${trade['h3_price']:.2f}")
    print(f"SL: ${trade['sl_price']:.2f} (-0.5%)")
    print(f"Power Score: {trade['power_score']}")
    print(f"{'='*60}")
    
    movements = analyze_candle_movement(
        trade['entry_time'], 
        trade['entry_price'],
        trade['h1_price'],
        trade['h2_price'],
        trade['h3_price'],
        trade['sl_price']
    )
    
    if movements:
        print("\n캔들 | 시간          | High%  | Low%   | Close% | SL타격 | H3터치 | H2도달")
        print("-" * 80)
        for m in movements[:10]:  # First 10 candles
            print(f"{m['candle']:2d}   | {m['time'].strftime('%m-%d %H:%M')} | "
                  f"{m['high_from_entry']:+6.2f} | {m['low_from_entry']:+6.2f} | "
                  f"{m['close_from_entry']:+6.2f} | {'✓' if m['touched_sl'] else ' '} | "
                  f"{'✓' if m['touched_h3'] else ' '} | {'✓' if m['reached_h2'] else ' '}")

# 2. TP1 Breakeven 케이스 - TP2 직전 되돌림 (43개)
print("\n\n" + "=" * 80)
print("2️⃣ TP1 Breakeven 실패: 'TP2 직전 되돌림' 유형 - 대표 사례 3개")
print("=" * 80)

tp1_be_cases = df[df['exit_reason'] == 'TP1_Breakeven'].head(3)

for idx, trade in tp1_be_cases.iterrows():
    print(f"\n{'='*60}")
    print(f"진입 시간: {trade['entry_time']}")
    print(f"진입 가격: ${trade['entry_price']:.2f}")
    print(f"H1 (TP2): ${trade['h1_price']:.2f} | H2 (TP1): ${trade['h2_price']:.2f}")
    print(f"Power Score: {trade['power_score']}")
    print(f"{'='*60}")
    
    movements = analyze_candle_movement(
        trade['entry_time'], 
        trade['entry_price'],
        trade['h1_price'],
        trade['h2_price'],
        trade['h3_price'],
        trade['sl_price'],
        candle_count=30
    )
    
    if movements:
        print("\n캔들 | 시간          | High%  | Low%   | Close% | H2도달 | H1도달")
        print("-" * 70)
        for m in movements[:15]:  # First 15 candles
            print(f"{m['candle']:2d}   | {m['time'].strftime('%m-%d %H:%M')} | "
                  f"{m['high_from_entry']:+6.2f} | {m['low_from_entry']:+6.2f} | "
                  f"{m['close_from_entry']:+6.2f} | {'✓' if m['reached_h2'] else ' '} | "
                  f"{'✓' if m['reached_h1'] else ' '}")

# 3. TP2 Full 성공 케이스 비교
print("\n\n" + "=" * 80)
print("3️⃣ TP2 Full 성공 케이스 - 대표 사례 3개")
print("=" * 80)

tp2_cases = df[df['exit_reason'] == 'TP2_Full'].head(3)

for idx, trade in tp2_cases.iterrows():
    print(f"\n{'='*60}")
    print(f"진입 시간: {trade['entry_time']}")
    print(f"진입 가격: ${trade['entry_price']:.2f}")
    print(f"H1 (TP2): ${trade['h1_price']:.2f} | H2 (TP1): ${trade['h2_price']:.2f}")
    print(f"Power Score: {trade['power_score']}")
    print(f"{'='*60}")
    
    movements = analyze_candle_movement(
        trade['entry_time'], 
        trade['entry_price'],
        trade['h1_price'],
        trade['h2_price'],
        trade['h3_price'],
        trade['sl_price'],
        candle_count=20
    )
    
    if movements:
        print("\n캔들 | 시간          | High%  | Low%   | Close% | H2도달 | H1도달")
        print("-" * 70)
        for m in movements[:10]:
            print(f"{m['candle']:2d}   | {m['time'].strftime('%m-%d %H:%M')} | "
                  f"{m['high_from_entry']:+6.2f} | {m['low_from_entry']:+6.2f} | "
                  f"{m['close_from_entry']:+6.2f} | {'✓' if m['reached_h2'] else ' '} | "
                  f"{'✓' if m['reached_h1'] else ' '}")

# 4. 핵심 차이점 요약
print("\n\n" + "=" * 80)
print("🎯 핵심 차이점 요약")
print("=" * 80)

sl_movements_all = []
tp1_movements_all = []
tp2_movements_all = []

for idx, trade in df[df['exit_reason'] == 'SL'].iterrows():
    mvmt = analyze_candle_movement(trade['entry_time'], trade['entry_price'],
                                   trade['h1_price'], trade['h2_price'], 
                                   trade['h3_price'], trade['sl_price'])
    if mvmt:
        sl_movements_all.extend(mvmt[:5])  # First 5 candles

for idx, trade in df[df['exit_reason'] == 'TP1_Breakeven'].iterrows():
    mvmt = analyze_candle_movement(trade['entry_time'], trade['entry_price'],
                                   trade['h1_price'], trade['h2_price'], 
                                   trade['h3_price'], trade['sl_price'])
    if mvmt:
        tp1_movements_all.extend(mvmt[:5])

for idx, trade in df[df['exit_reason'] == 'TP2_Full'].iterrows():
    mvmt = analyze_candle_movement(trade['entry_time'], trade['entry_price'],
                                   trade['h1_price'], trade['h2_price'], 
                                   trade['h3_price'], trade['sl_price'])
    if mvmt:
        tp2_movements_all.extend(mvmt[:5])

if sl_movements_all:
    sl_df = pd.DataFrame(sl_movements_all)
    print(f"\n❌ SL 케이스 (진입 후 5캔들 평균):")
    print(f"  - 평균 High: {sl_df['high_from_entry'].mean():+.2f}%")
    print(f"  - 평균 Low: {sl_df['low_from_entry'].mean():+.2f}%")
    print(f"  - 평균 Close: {sl_df['close_from_entry'].mean():+.2f}%")
    print(f"  - H3 터치율: {(sl_df['touched_h3'].sum() / len(sl_df) * 100):.1f}%")

if tp2_movements_all:
    tp2_df = pd.DataFrame(tp2_movements_all)
    print(f"\n✅ TP2 Full 케이스 (진입 후 5캔들 평균):")
    print(f"  - 평균 High: {tp2_df['high_from_entry'].mean():+.2f}%")
    print(f"  - 평균 Low: {tp2_df['low_from_entry'].mean():+.2f}%")
    print(f"  - 평균 Close: {tp2_df['close_from_entry'].mean():+.2f}%")
    print(f"  - H3 터치율: {(tp2_df['touched_h3'].sum() / len(tp2_df) * 100):.1f}%")

print("\n" + "=" * 80)
print("✅ 분석 완료")
print("=" * 80)
