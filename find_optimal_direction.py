import pandas as pd
import numpy as np

df = pd.read_csv('trades_with_indicators.csv')
print(f"총 매매: {len(df)}건")

print("\n" + "="*80)
print("📊 파라미터별 LONG vs SHORT 승률 비교")
print("="*80)

# 1. EMA200 기준
print("\n### EMA200 기준")
for above in [True, False]:
    label = "EMA200 위" if above else "EMA200 아래"
    sub = df[df['above_ema200'] == above]
    
    long_sub = sub[sub['direction'] == 'LONG']
    short_sub = sub[sub['direction'] == 'SHORT']
    
    if len(long_sub) > 0:
        long_wr = (long_sub['final_pnl'] > 0).mean() * 100
        long_avg = long_sub['final_pnl'].mean()
    else:
        long_wr, long_avg = 0, 0
        
    if len(short_sub) > 0:
        short_wr = (short_sub['final_pnl'] > 0).mean() * 100
        short_avg = short_sub['final_pnl'].mean()
    else:
        short_wr, short_avg = 0, 0
    
    better = "LONG" if long_wr > short_wr else "SHORT"
    print(f"  {label}: LONG {len(long_sub)}건 {long_wr:.1f}% {long_avg:+.2f}% | SHORT {len(short_sub)}건 {short_wr:.1f}% {short_avg:+.2f}% → {better}")

# 2. EMA50 기준
print("\n### EMA50 기준")
for above in [True, False]:
    label = "EMA50 위" if above else "EMA50 아래"
    sub = df[df['above_ema50'] == above]
    
    long_sub = sub[sub['direction'] == 'LONG']
    short_sub = sub[sub['direction'] == 'SHORT']
    
    if len(long_sub) > 0:
        long_wr = (long_sub['final_pnl'] > 0).mean() * 100
        long_avg = long_sub['final_pnl'].mean()
    else:
        long_wr, long_avg = 0, 0
        
    if len(short_sub) > 0:
        short_wr = (short_sub['final_pnl'] > 0).mean() * 100
        short_avg = short_sub['final_pnl'].mean()
    else:
        short_wr, short_avg = 0, 0
    
    better = "LONG" if long_wr > short_wr else "SHORT"
    print(f"  {label}: LONG {len(long_sub)}건 {long_wr:.1f}% {long_avg:+.2f}% | SHORT {len(short_sub)}건 {short_wr:.1f}% {short_avg:+.2f}% → {better}")

# 3. VWAP 기준
print("\n### VWAP 기준")
for above in [True, False]:
    label = "VWAP 위" if above else "VWAP 아래"
    sub = df[df['above_vwap'] == above]
    
    long_sub = sub[sub['direction'] == 'LONG']
    short_sub = sub[sub['direction'] == 'SHORT']
    
    if len(long_sub) > 0:
        long_wr = (long_sub['final_pnl'] > 0).mean() * 100
        long_avg = long_sub['final_pnl'].mean()
    else:
        long_wr, long_avg = 0, 0
        
    if len(short_sub) > 0:
        short_wr = (short_sub['final_pnl'] > 0).mean() * 100
        short_avg = short_sub['final_pnl'].mean()
    else:
        short_wr, short_avg = 0, 0
    
    better = "LONG" if long_wr > short_wr else "SHORT"
    print(f"  {label}: LONG {len(long_sub)}건 {long_wr:.1f}% {long_avg:+.2f}% | SHORT {len(short_sub)}건 {short_wr:.1f}% {short_avg:+.2f}% → {better}")

# 4. EMA 배열
print("\n### EMA 배열")
for bull, bear, label in [(True, False, '정배열'), (False, True, '역배열'), (False, False, '혼조')]:
    if bull:
        sub = df[df['ema_bullish'] == True]
    elif bear:
        sub = df[df['ema_bearish'] == True]
    else:
        sub = df[(df['ema_bullish'] == False) & (df['ema_bearish'] == False)]
    
    long_sub = sub[sub['direction'] == 'LONG']
    short_sub = sub[sub['direction'] == 'SHORT']
    
    if len(long_sub) > 0:
        long_wr = (long_sub['final_pnl'] > 0).mean() * 100
        long_avg = long_sub['final_pnl'].mean()
    else:
        long_wr, long_avg = 0, 0
        
    if len(short_sub) > 0:
        short_wr = (short_sub['final_pnl'] > 0).mean() * 100
        short_avg = short_sub['final_pnl'].mean()
    else:
        short_wr, short_avg = 0, 0
    
    better = "LONG" if long_wr > short_wr else "SHORT"
    print(f"  {label}: LONG {len(long_sub)}건 {long_wr:.1f}% {long_avg:+.2f}% | SHORT {len(short_sub)}건 {short_wr:.1f}% {short_avg:+.2f}% → {better}")

# 5. BB MID 기준
print("\n### BB MID 기준")
for above in [True, False]:
    label = "BB중심 위" if above else "BB중심 아래"
    sub = df[df['above_bb_mid'] == above]
    
    long_sub = sub[sub['direction'] == 'LONG']
    short_sub = sub[sub['direction'] == 'SHORT']
    
    if len(long_sub) > 0:
        long_wr = (long_sub['final_pnl'] > 0).mean() * 100
        long_avg = long_sub['final_pnl'].mean()
    else:
        long_wr, long_avg = 0, 0
        
    if len(short_sub) > 0:
        short_wr = (short_sub['final_pnl'] > 0).mean() * 100
        short_avg = short_sub['final_pnl'].mean()
    else:
        short_wr, short_avg = 0, 0
    
    better = "LONG" if long_wr > short_wr else "SHORT"
    print(f"  {label}: LONG {len(long_sub)}건 {long_wr:.1f}% {long_avg:+.2f}% | SHORT {len(short_sub)}건 {short_wr:.1f}% {short_avg:+.2f}% → {better}")

print("\n" + "="*80)
print("📌 최적 방향 규칙")
print("="*80)
print("""
EMA200 위 → LONG (46.5% vs 37.0%)
EMA200 아래 → LONG (36.7% vs 31.8%)  ← LONG이 더 좋음!

EMA50 위 → LONG
EMA50 아래 → LONG  ← 여기도 LONG!

VWAP 위 → LONG
VWAP 아래 → LONG  ← 여기도 LONG!

정배열 → LONG
역배열 → LONG  ← 역배열에서도 LONG!
혼조 → LONG
""")

# 최적 방향으로 재계산
print("\n" + "="*80)
print("🎯 최적 방향 적용 시뮬레이션 (모든 돌파를 LONG으로)")
print("="*80)

# 모든 매매를 LONG 방향으로 했을 때
# 상단 돌파 = 원래대로, 하단 이탈 = 반대로(손실→이익, 이익→손실)
df['optimal_pnl'] = df.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'], 
    axis=1
)

wins = (df['optimal_pnl'] > 0).sum()
total = len(df)
print(f"\n전체 LONG 시: {wins}/{total} = {wins/total*100:.1f}% 승률")
print(f"평균 수익: {df['optimal_pnl'].mean():.2f}%")
print(f"총 수익: {df['optimal_pnl'].sum():.1f}%")
