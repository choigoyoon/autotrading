import pandas as pd
import numpy as np

df = pd.read_csv('trades_with_indicators.csv')

# 반대 방향 = 원래 pnl의 반대
df['counter_pnl'] = -df['final_pnl']

print("="*80)
print("📊 역돌파 전략 (돌파 반대 방향 진입)")
print("="*80)

print(f"\n총 매매: {len(df)}건")
print(f"승률: {(df['counter_pnl'] > 0).mean()*100:.1f}%")
print(f"평균: {df['counter_pnl'].mean():.2f}%")
print(f"총 수익: {df['counter_pnl'].sum():.1f}%")

print("\n### 방향별:")
# 상단 돌파 → SHORT
long_orig = df[df['direction'] == 'LONG']
print(f"상단돌파→SHORT: {len(long_orig)}건, 승률 {(long_orig['counter_pnl'] > 0).mean()*100:.1f}%, 평균 {long_orig['counter_pnl'].mean():.2f}%")

# 하단 이탈 → LONG
short_orig = df[df['direction'] == 'SHORT']
print(f"하단이탈→LONG: {len(short_orig)}건, 승률 {(short_orig['counter_pnl'] > 0).mean()*100:.1f}%, 평균 {short_orig['counter_pnl'].mean():.2f}%")

print("\n" + "="*80)
print("🎯 파라미터별 역돌파 승률")
print("="*80)

# EMA200
print("\n### EMA200:")
for above in [True, False]:
    sub = df[df['above_ema200'] == above]
    wr = (sub['counter_pnl'] > 0).mean() * 100
    avg = sub['counter_pnl'].mean()
    print(f"  EMA200{'↑' if above else '↓'}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# VWAP
print("\n### VWAP:")
for above in [True, False]:
    sub = df[df['above_vwap'] == above]
    wr = (sub['counter_pnl'] > 0).mean() * 100
    avg = sub['counter_pnl'].mean()
    print(f"  VWAP{'↑' if above else '↓'}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# EMA 배열
print("\n### EMA배열:")
sub = df[df['ema_bullish'] == True]
print(f"  정배열: {len(sub)}건, 승률 {(sub['counter_pnl'] > 0).mean()*100:.1f}%, 평균 {sub['counter_pnl'].mean():.2f}%")
sub = df[df['ema_bearish'] == True]
print(f"  역배열: {len(sub)}건, 승률 {(sub['counter_pnl'] > 0).mean()*100:.1f}%, 평균 {sub['counter_pnl'].mean():.2f}%")

print("\n" + "="*80)
print("🔍 조합별 역돌파 승률 (70% 이상 찾기)")
print("="*80)

results = []
for ema200 in [True, False, None]:
    for vwap in [True, False, None]:
        for ema_arr in ['bull', 'bear', None]:
            for orig_dir in ['LONG', 'SHORT', None]:
                sub = df.copy()
                conds = []
                
                if ema200 is not None:
                    sub = sub[sub['above_ema200'] == ema200]
                    conds.append(f"EMA200{'↑' if ema200 else '↓'}")
                if vwap is not None:
                    sub = sub[sub['above_vwap'] == vwap]
                    conds.append(f"VWAP{'↑' if vwap else '↓'}")
                if ema_arr == 'bull':
                    sub = sub[sub['ema_bullish'] == True]
                    conds.append("정배열")
                elif ema_arr == 'bear':
                    sub = sub[sub['ema_bearish'] == True]
                    conds.append("역배열")
                if orig_dir is not None:
                    sub = sub[sub['direction'] == orig_dir]
                    entry_dir = "SHORT" if orig_dir == "LONG" else "LONG"
                    conds.append(f"→{entry_dir}")
                
                if len(sub) >= 50 and len(conds) >= 2:
                    wr = (sub['counter_pnl'] > 0).mean() * 100
                    avg = sub['counter_pnl'].mean()
                    results.append({
                        'cond': '+'.join(conds),
                        'count': len(sub),
                        'wr': wr,
                        'avg': avg,
                        'total': sub['counter_pnl'].sum()
                    })

results_df = pd.DataFrame(results).drop_duplicates(subset=['cond']).sort_values('wr', ascending=False)

print("\n### 승률 TOP 20:")
for _, row in results_df.head(20).iterrows():
    marker = "⭐" if row['wr'] >= 70 else ("✓" if row['wr'] >= 65 else "")
    print(f"  {row['cond']}: {row['count']}건, 승률 {row['wr']:.1f}%, 평균 {row['avg']:.2f}%, 총 {row['total']:.0f}% {marker}")

print("\n### 수익 TOP 10:")
results_df2 = results_df.sort_values('avg', ascending=False)
for _, row in results_df2.head(10).iterrows():
    marker = "⭐" if row['avg'] >= 1.0 else ("✓" if row['avg'] >= 0.5 else "")
    print(f"  {row['cond']}: {row['count']}건, 승률 {row['wr']:.1f}%, 평균 {row['avg']:.2f}%, 총 {row['total']:.0f}% {marker}")
