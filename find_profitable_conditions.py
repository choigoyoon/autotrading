import pandas as pd
import numpy as np

df = pd.read_csv('trades_with_indicators.csv')
print(f"총 매매: {len(df)}건")

# 기존 결과에 최적 방향(LONG) 적용
df['optimal_pnl'] = df.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'], 
    axis=1
)

print("\n" + "="*80)
print("🔍 평균 수익 1% 이상 조건 찾기")
print("="*80)

# 단일 조건
print("\n### 단일 조건 (최소 30건)")
results = []

# EMA200
for above in [True, False]:
    sub = df[df['above_ema200'] == above]
    if len(sub) >= 30:
        avg = sub['optimal_pnl'].mean()
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        results.append({'조건': f"EMA200 {'위' if above else '아래'}", 'count': len(sub), 'wr': wr, 'avg': avg})

# EMA50
for above in [True, False]:
    sub = df[df['above_ema50'] == above]
    if len(sub) >= 30:
        avg = sub['optimal_pnl'].mean()
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        results.append({'조건': f"EMA50 {'위' if above else '아래'}", 'count': len(sub), 'wr': wr, 'avg': avg})

# VWAP
for above in [True, False]:
    sub = df[df['above_vwap'] == above]
    if len(sub) >= 30:
        avg = sub['optimal_pnl'].mean()
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        results.append({'조건': f"VWAP {'위' if above else '아래'}", 'count': len(sub), 'wr': wr, 'avg': avg})

# EMA 배열
for bull, bear, label in [(True, False, '정배열'), (False, True, '역배열')]:
    if bull:
        sub = df[df['ema_bullish'] == True]
    else:
        sub = df[df['ema_bearish'] == True]
    if len(sub) >= 30:
        avg = sub['optimal_pnl'].mean()
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        results.append({'조건': label, 'count': len(sub), 'wr': wr, 'avg': avg})

# 수축길이
for low, high in [(3, 10), (10, 20), (20, 40), (40, 100)]:
    sub = df[(df['squeeze_length'] >= low) & (df['squeeze_length'] < high)]
    if len(sub) >= 30:
        avg = sub['optimal_pnl'].mean()
        wr = (sub['optimal_pnl'] > 0).mean() * 100
        results.append({'조건': f"수축{low}-{high}h", 'count': len(sub), 'wr': wr, 'avg': avg})

results_df = pd.DataFrame(results).sort_values('avg', ascending=False)
for _, row in results_df.iterrows():
    marker = "✓" if row['avg'] >= 1.0 else ""
    print(f"  {row['조건']}: {row['count']}건, 승률 {row['wr']:.1f}%, 평균 {row['avg']:.2f}% {marker}")

# 2개 조합
print("\n### 2개 조합 (최소 30건, 평균 0.5% 이상만)")
results2 = []

for ema200 in [True, False]:
    for vwap in [True, False]:
        sub = df[(df['above_ema200'] == ema200) & (df['above_vwap'] == vwap)]
        if len(sub) >= 30:
            avg = sub['optimal_pnl'].mean()
            wr = (sub['optimal_pnl'] > 0).mean() * 100
            if avg >= 0.5:
                results2.append({
                    '조건': f"EMA200{'↑' if ema200 else '↓'}+VWAP{'↑' if vwap else '↓'}",
                    'count': len(sub), 'wr': wr, 'avg': avg
                })

for ema200 in [True, False]:
    for bull in [True, False]:
        if bull:
            sub = df[(df['above_ema200'] == ema200) & (df['ema_bullish'] == True)]
        else:
            sub = df[(df['above_ema200'] == ema200) & (df['ema_bearish'] == True)]
        if len(sub) >= 30:
            avg = sub['optimal_pnl'].mean()
            wr = (sub['optimal_pnl'] > 0).mean() * 100
            label = '정배열' if bull else '역배열'
            if avg >= 0.5:
                results2.append({
                    '조건': f"EMA200{'↑' if ema200 else '↓'}+{label}",
                    'count': len(sub), 'wr': wr, 'avg': avg
                })

for ema200 in [True, False]:
    for low, high in [(3, 15), (15, 50)]:
        sub = df[(df['above_ema200'] == ema200) & 
                 (df['squeeze_length'] >= low) & (df['squeeze_length'] < high)]
        if len(sub) >= 30:
            avg = sub['optimal_pnl'].mean()
            wr = (sub['optimal_pnl'] > 0).mean() * 100
            if avg >= 0.5:
                results2.append({
                    '조건': f"EMA200{'↑' if ema200 else '↓'}+수축{low}-{high}h",
                    'count': len(sub), 'wr': wr, 'avg': avg
                })

results2_df = pd.DataFrame(results2).sort_values('avg', ascending=False)
for _, row in results2_df.iterrows():
    marker = "✓" if row['avg'] >= 1.0 else ""
    print(f"  {row['조건']}: {row['count']}건, 승률 {row['wr']:.1f}%, 평균 {row['avg']:.2f}% {marker}")

# 3개 조합
print("\n### 3개 조합 (최소 20건)")
results3 = []

for ema200 in [True, False]:
    for vwap in [True, False]:
        for bull in [True, False, None]:
            if bull is True:
                sub = df[(df['above_ema200'] == ema200) & 
                         (df['above_vwap'] == vwap) & 
                         (df['ema_bullish'] == True)]
            elif bull is False:
                sub = df[(df['above_ema200'] == ema200) & 
                         (df['above_vwap'] == vwap) & 
                         (df['ema_bearish'] == True)]
            else:
                continue
            
            if len(sub) >= 20:
                avg = sub['optimal_pnl'].mean()
                wr = (sub['optimal_pnl'] > 0).mean() * 100
                label = '정배열' if bull else '역배열'
                results3.append({
                    '조건': f"EMA200{'↑' if ema200 else '↓'}+VWAP{'↑' if vwap else '↓'}+{label}",
                    'count': len(sub), 'wr': wr, 'avg': avg
                })

for ema200 in [True, False]:
    for vwap in [True, False]:
        for low, high in [(3, 15), (15, 50)]:
            sub = df[(df['above_ema200'] == ema200) & 
                     (df['above_vwap'] == vwap) &
                     (df['squeeze_length'] >= low) & (df['squeeze_length'] < high)]
            if len(sub) >= 20:
                avg = sub['optimal_pnl'].mean()
                wr = (sub['optimal_pnl'] > 0).mean() * 100
                results3.append({
                    '조건': f"EMA200{'↑' if ema200 else '↓'}+VWAP{'↑' if vwap else '↓'}+수축{low}-{high}h",
                    'count': len(sub), 'wr': wr, 'avg': avg
                })

results3_df = pd.DataFrame(results3).sort_values('avg', ascending=False)
print("\n평균 수익 TOP 15:")
for _, row in results3_df.head(15).iterrows():
    marker = "✓✓" if row['avg'] >= 1.0 else ("✓" if row['avg'] >= 0.7 else "")
    print(f"  {row['조건']}: {row['count']}건, 승률 {row['wr']:.1f}%, 평균 {row['avg']:.2f}% {marker}")

# 청산 사유별로도 확인
print("\n" + "="*80)
print("📊 청산 사유별 분석")
print("="*80)

for reason in ['NEXT_SQUEEZE', 'SL']:
    sub = df[df['exit_reason'] == reason]
    avg = sub['optimal_pnl'].mean()
    wr = (sub['optimal_pnl'] > 0).mean() * 100
    print(f"  {reason}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# NEXT_SQUEEZE만 보면
print("\n### NEXT_SQUEEZE 청산만 (손절 제외)")
ns_df = df[df['exit_reason'] == 'NEXT_SQUEEZE']
print(f"전체: {len(ns_df)}건, 승률 {(ns_df['optimal_pnl'] > 0).mean()*100:.1f}%, 평균 {ns_df['optimal_pnl'].mean():.2f}%")

# NEXT_SQUEEZE + EMA200 위
sub = ns_df[ns_df['above_ema200'] == True]
print(f"EMA200↑: {len(sub)}건, 승률 {(sub['optimal_pnl'] > 0).mean()*100:.1f}%, 평균 {sub['optimal_pnl'].mean():.2f}%")

# NEXT_SQUEEZE + 정배열
sub = ns_df[ns_df['ema_bullish'] == True]
print(f"정배열: {len(sub)}건, 승률 {(sub['optimal_pnl'] > 0).mean()*100:.1f}%, 평균 {sub['optimal_pnl'].mean():.2f}%")
