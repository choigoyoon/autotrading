import pandas as pd
import numpy as np

df = pd.read_csv('trades_with_indicators.csv')
df['optimal_pnl'] = df.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'], 
    axis=1
)

# 71.7% 조건 상세 분석
print("="*80)
print("🔍 71.7% 승률 조건 상세 분석")
print("="*80)

base = df[(df['above_vwap'] == False) & 
          (df['ema_bearish'] == True) & 
          (df['squeeze_length'] >= 3) & 
          (df['squeeze_length'] < 10)]

print(f"\n기본 조건: VWAP↓ + 역배열 + 수축3-10h")
print(f"건수: {len(base)}, 승률: {(base['optimal_pnl'] > 0).mean()*100:.1f}%, 평균: {base['optimal_pnl'].mean():.2f}%")

# 추가 필터로 평균 수익 올리기
print("\n### 추가 필터 테스트:")

# 손절폭별
for low, high in [(0, 1.5), (1.5, 3), (3, 5), (5, 10)]:
    sub = base[(base['sl_pct'] >= low) & (base['sl_pct'] < high)]
    if len(sub) >= 10:
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        avg = sub['optimal_pnl'].mean()
        print(f"  + SL {low}-{high}%: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# max_profit별
print("\n### max_profit 분위별:")
for q in [0.25, 0.5, 0.75]:
    threshold = base['max_profit'].quantile(q)
    sub = base[base['max_profit'] >= threshold]
    if len(sub) >= 10:
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        avg = sub['optimal_pnl'].mean()
        print(f"  max_profit >= {threshold:.2f}%: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 다른 고승률 조건들도 세부 분석
print("\n" + "="*80)
print("🔍 다른 조건들 세부 분석")
print("="*80)

# EMA200↓+EMA50↓+VWAP↓ (63% 승률, 0.46% 평균, 219건)
base2 = df[(df['above_ema200'] == False) & 
           (df['above_ema50'] == False) & 
           (df['above_vwap'] == False)]

print(f"\nEMA200↓+EMA50↓+VWAP↓: {len(base2)}건")

for low, high in [(3, 8), (8, 15), (15, 30)]:
    sub = base2[(base2['squeeze_length'] >= low) & (base2['squeeze_length'] < high)]
    if len(sub) >= 15:
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        avg = sub['optimal_pnl'].mean()
        marker = "⭐" if wr >= 70 and avg >= 0.8 else ("✓" if wr >= 65 else "")
        print(f"  + 수축{low}-{high}h: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}% {marker}")

# 역배열 추가
for low, high in [(3, 8), (8, 15)]:
    sub = base2[(base2['ema_bearish'] == True) &
                (base2['squeeze_length'] >= low) & (base2['squeeze_length'] < high)]
    if len(sub) >= 10:
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        avg = sub['optimal_pnl'].mean()
        marker = "⭐" if wr >= 70 and avg >= 0.8 else ("✓" if wr >= 65 else "")
        print(f"  + 역배열 + 수축{low}-{high}h: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}% {marker}")

# 승리 케이스 vs 패배 케이스 분석
print("\n" + "="*80)
print("📊 71.7% 조건의 승/패 비교")
print("="*80)

wins = base[base['optimal_pnl'] > 0]
losses = base[base['optimal_pnl'] <= 0]

print(f"\n승리 {len(wins)}건:")
print(f"  평균 수익: +{wins['optimal_pnl'].mean():.2f}%")
print(f"  평균 max_profit: {wins['max_profit'].mean():.2f}%")
print(f"  평균 수축길이: {wins['squeeze_length'].mean():.1f}h")

print(f"\n패배 {len(losses)}건:")
print(f"  평균 손실: {losses['optimal_pnl'].mean():.2f}%")
print(f"  평균 max_profit: {losses['max_profit'].mean():.2f}%")
print(f"  평균 수축길이: {losses['squeeze_length'].mean():.1f}h")

# 수축길이 더 세분화
print("\n### 수축길이 세분화:")
for low, high in [(3, 5), (5, 7), (7, 10)]:
    sub = base[(base['squeeze_length'] >= low) & (base['squeeze_length'] < high)]
    if len(sub) >= 5:
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        avg = sub['optimal_pnl'].mean()
        print(f"  수축{low}-{high}h: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")
