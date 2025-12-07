import pandas as pd
import numpy as np

df = pd.read_csv('trades_with_indicators.csv')

# 반대 방향으로 수익 계산
df['reverse_pnl'] = -df['final_pnl']

print(f"총 매매: {len(df)}건")
print(f"\n기존 (돌파 방향): 승률 {(df['final_pnl'] > 0).mean()*100:.1f}%, 평균 {df['final_pnl'].mean():.2f}%")
print(f"반전 (돌파 반대): 승률 {(df['reverse_pnl'] > 0).mean()*100:.1f}%, 평균 {df['reverse_pnl'].mean():.2f}%")

print("\n" + "="*80)
print("🔄 반전 전략 (돌파 반대 방향) 상세")
print("="*80)

# 상단 돌파 → SHORT
long_trades = df[df['direction'] == 'LONG']
print(f"\n상단 돌파 → SHORT: {len(long_trades)}건")
print(f"  승률: {(long_trades['reverse_pnl'] > 0).mean()*100:.1f}%")
print(f"  평균: {long_trades['reverse_pnl'].mean():.2f}%")

# 하단 이탈 → LONG  
short_trades = df[df['direction'] == 'SHORT']
print(f"\n하단 이탈 → LONG: {len(short_trades)}건")
print(f"  승률: {(short_trades['reverse_pnl'] > 0).mean()*100:.1f}%")
print(f"  평균: {short_trades['reverse_pnl'].mean():.2f}%")

print("\n" + "="*80)
print("🎯 하단 이탈 → LONG 조건 최적화 (65.8% 베이스)")
print("="*80)

base = short_trades.copy()
base['opt_pnl'] = base['reverse_pnl']

print(f"\n베이스: {len(base)}건, 승률 {(base['opt_pnl'] > 0).mean()*100:.1f}%, 평균 {base['opt_pnl'].mean():.2f}%")

# 파라미터별
print("\n### EMA200:")
for above in [True, False]:
    sub = base[base['above_ema200'] == above]
    if len(sub) >= 50:
        wr = (sub['opt_pnl'] > 0).mean() * 100
        avg = sub['opt_pnl'].mean()
        print(f"  EMA200{'↑' if above else '↓'}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

print("\n### VWAP:")
for above in [True, False]:
    sub = base[base['above_vwap'] == above]
    if len(sub) >= 50:
        wr = (sub['opt_pnl'] > 0).mean() * 100
        avg = sub['opt_pnl'].mean()
        print(f"  VWAP{'↑' if above else '↓'}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

print("\n### EMA배열:")
for bull in [True, False]:
    if bull:
        sub = base[base['ema_bullish'] == True]
        label = "정배열"
    else:
        sub = base[base['ema_bearish'] == True]
        label = "역배열"
    if len(sub) >= 50:
        wr = (sub['opt_pnl'] > 0).mean() * 100
        avg = sub['opt_pnl'].mean()
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 조합
print("\n### 조합 (70% 이상 찾기):")
results = []
for ema200 in [True, False, None]:
    for vwap in [True, False, None]:
        for bull in [True, False, None]:
            sub = base.copy()
            conds = []
            
            if ema200 is not None:
                sub = sub[sub['above_ema200'] == ema200]
                conds.append(f"EMA200{'↑' if ema200 else '↓'}")
            if vwap is not None:
                sub = sub[sub['above_vwap'] == vwap]
                conds.append(f"VWAP{'↑' if vwap else '↓'}")
            if bull is True:
                sub = sub[sub['ema_bullish'] == True]
                conds.append("정배열")
            elif bull is False:
                sub = sub[sub['ema_bearish'] == True]
                conds.append("역배열")
            
            if len(sub) >= 30 and len(conds) >= 1:
                wr = (sub['opt_pnl'] > 0).mean() * 100
                avg = sub['opt_pnl'].mean()
                results.append({
                    'cond': '+'.join(conds),
                    'count': len(sub),
                    'wr': wr,
                    'avg': avg
                })

results_df = pd.DataFrame(results).sort_values('wr', ascending=False)
for _, row in results_df.head(15).iterrows():
    marker = "⭐" if row['wr'] >= 70 else ""
    print(f"  {row['cond']}: {row['count']}건, 승률 {row['wr']:.1f}%, 평균 {row['avg']:.2f}% {marker}")

print("\n" + "="*80)
print("📊 전체 반전 전략 결과")  
print("="*80)
print(f"\n전체: {len(df)}건, 승률 {(df['reverse_pnl'] > 0).mean()*100:.1f}%, 평균 {df['reverse_pnl'].mean():.2f}%")
print(f"총 수익: {df['reverse_pnl'].sum():.1f}%")
