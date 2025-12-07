import pandas as pd
import numpy as np

df = pd.read_csv('trades_with_indicators.csv')
df['optimal_pnl'] = df.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'], 
    axis=1
)
print(f"총 매매: {len(df)}건")
print(f"현재: 승률 {(df['optimal_pnl'] > 0).mean()*100:.1f}%, 평균 {df['optimal_pnl'].mean():.2f}%")

print("\n" + "="*80)
print("🎯 최소 200건 이상에서 승률/수익 찾기")
print("="*80)

# 현재 데이터로는 841건에서 53.9% 승률, 0.37% 평균
# 이걸로는 70% 승률 불가능

# 문제: 손절에 걸리면 손실이 큼
# 해결: 손절 조건 자체를 바꿔야 함

print("\n### 청산 사유별 현황:")
for reason in df['exit_reason'].unique():
    sub = df[df['exit_reason'] == reason]
    wr = (sub['optimal_pnl'] > 0).mean() * 100
    avg = sub['optimal_pnl'].mean()
    print(f"  {reason}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 손절 안 맞은 것만
print("\n### NEXT_SQUEEZE 청산만 (손절 제외):")
ns = df[df['exit_reason'] == 'NEXT_SQUEEZE']
print(f"  전체: {len(ns)}건, 승률 {(ns['optimal_pnl'] > 0).mean()*100:.1f}%, 평균 {ns['optimal_pnl'].mean():.2f}%")

# 이것도 53% 승률밖에 안됨
# 문제는 방향 판단 자체가 틀림

print("\n" + "="*80)
print("📊 원본 데이터 다시 분석 - 실제 돌파 방향별")
print("="*80)

# 상단 돌파 vs 하단 돌파
long_trades = df[df['direction'] == 'LONG']
short_trades = df[df['direction'] == 'SHORT']

print(f"\n상단 돌파 (원래 LONG): {len(long_trades)}건")
print(f"  LONG 유지 승률: {(long_trades['final_pnl'] > 0).mean()*100:.1f}%")
print(f"  SHORT 전환 승률: {(long_trades['final_pnl'] < 0).mean()*100:.1f}%")

print(f"\n하단 이탈 (원래 SHORT): {len(short_trades)}건")
print(f"  SHORT 유지 승률: {(short_trades['final_pnl'] > 0).mean()*100:.1f}%")
print(f"  LONG 전환 승률: {(short_trades['final_pnl'] < 0).mean()*100:.1f}%")

# 파라미터별로 어느 방향이 더 좋은지
print("\n" + "="*80)
print("🔍 파라미터별 최적 방향 (상세)")
print("="*80)

print("\n### EMA200 + 돌파방향:")
for ema200 in [True, False]:
    for orig_dir in ['LONG', 'SHORT']:
        sub = df[(df['above_ema200'] == ema200) & (df['direction'] == orig_dir)]
        if len(sub) >= 50:
            # 원래 방향 유지
            orig_wr = (sub['final_pnl'] > 0).mean() * 100
            # 반대 방향
            rev_wr = (sub['final_pnl'] < 0).mean() * 100
            
            label = f"EMA200{'↑' if ema200 else '↓'} + {orig_dir}"
            better = orig_dir if orig_wr > rev_wr else ('LONG' if orig_dir == 'SHORT' else 'SHORT')
            print(f"  {label}: {len(sub)}건, {orig_dir}:{orig_wr:.1f}% vs 반대:{rev_wr:.1f}% → {better}")

print("\n### VWAP + 돌파방향:")
for vwap in [True, False]:
    for orig_dir in ['LONG', 'SHORT']:
        sub = df[(df['above_vwap'] == vwap) & (df['direction'] == orig_dir)]
        if len(sub) >= 50:
            orig_wr = (sub['final_pnl'] > 0).mean() * 100
            rev_wr = (sub['final_pnl'] < 0).mean() * 100
            
            label = f"VWAP{'↑' if vwap else '↓'} + {orig_dir}"
            better = orig_dir if orig_wr > rev_wr else ('LONG' if orig_dir == 'SHORT' else 'SHORT')
            print(f"  {label}: {len(sub)}건, {orig_dir}:{orig_wr:.1f}% vs 반대:{rev_wr:.1f}% → {better}")

print("\n### EMA배열 + 돌파방향:")
for bull in [True, False]:
    for orig_dir in ['LONG', 'SHORT']:
        if bull:
            sub = df[(df['ema_bullish'] == True) & (df['direction'] == orig_dir)]
            label = f"정배열 + {orig_dir}"
        else:
            sub = df[(df['ema_bearish'] == True) & (df['direction'] == orig_dir)]
            label = f"역배열 + {orig_dir}"
        
        if len(sub) >= 50:
            orig_wr = (sub['final_pnl'] > 0).mean() * 100
            rev_wr = (sub['final_pnl'] < 0).mean() * 100
            better = orig_dir if orig_wr > rev_wr else ('LONG' if orig_dir == 'SHORT' else 'SHORT')
            print(f"  {label}: {len(sub)}건, {orig_dir}:{orig_wr:.1f}% vs 반대:{rev_wr:.1f}% → {better}")
