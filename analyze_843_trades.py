import pandas as pd
import numpy as np

# 843건 데이터 로드
df = pd.read_csv('bb30_strategy_v4_results.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"총 매매: {len(df)}건")

wins = df[df['final_pnl'] > 0]
losses = df[df['final_pnl'] <= 0]

print(f"\n승: {len(wins)}건 ({len(wins)/len(df)*100:.1f}%)")
print(f"패: {len(losses)}건 ({len(losses)/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("📊 승/패 파라미터 비교")
print("="*80)

# 1. 방향별
print("\n### 1. 방향 (direction)")
for d in ['LONG', 'SHORT']:
    w = len(wins[wins['direction'] == d])
    l = len(losses[losses['direction'] == d])
    total = w + l
    if total > 0:
        print(f"  {d}: 승 {w}건, 패 {l}건, 승률 {w/total*100:.1f}%")

# 2. 마지막 캔들 방향
print("\n### 2. 마지막 캔들 방향 (last_direction)")
for d in ['UP', 'DOWN']:
    w = len(wins[wins['last_direction'] == d])
    l = len(losses[losses['last_direction'] == d])
    total = w + l
    if total > 0:
        print(f"  {d}: 승 {w}건, 패 {l}건, 승률 {w/total*100:.1f}%")

# 3. 수축 길이
print("\n### 3. 수축 길이 (squeeze_length)")
print(f"  승리 평균: {wins['squeeze_length'].mean():.1f}시간")
print(f"  패배 평균: {losses['squeeze_length'].mean():.1f}시간")
for low, high in [(3, 6), (6, 10), (10, 15), (15, 25), (25, 50), (50, 200)]:
    sub = df[(df['squeeze_length'] >= low) & (df['squeeze_length'] < high)]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"  {low}-{high}h: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 4. 손절폭
print("\n### 4. 손절폭 (sl_pct)")
print(f"  승리 평균: {wins['sl_pct'].mean():.2f}%")
print(f"  패배 평균: {losses['sl_pct'].mean():.2f}%")
for low, high in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10), (10, 20)]:
    sub = df[(df['sl_pct'] >= low) & (df['sl_pct'] < high)]
    if len(sub) >= 15:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"  {low}-{high}%: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 5. 수축 범위
print("\n### 5. 수축 범위 (squeeze_range_pct)")
print(f"  승리 평균: {wins['squeeze_range_pct'].mean():.2f}%")
print(f"  패배 평균: {losses['squeeze_range_pct'].mean():.2f}%")
for low, high in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 6), (6, 10)]:
    sub = df[(df['squeeze_range_pct'] >= low) & (df['squeeze_range_pct'] < high)]
    if len(sub) >= 15:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"  {low}-{high}%: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 6. 청산 사유
print("\n### 6. 청산 사유 (exit_reason)")
for reason in df['exit_reason'].unique():
    sub = df[df['exit_reason'] == reason]
    wr = (sub['final_pnl'] > 0).mean() * 100
    avg = sub['final_pnl'].mean()
    print(f"  {reason}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 7. 방향 + 마지막캔들 조합
print("\n### 7. 방향 + 마지막캔들 조합")
for d in ['LONG', 'SHORT']:
    for last in ['UP', 'DOWN']:
        sub = df[(df['direction'] == d) & (df['last_direction'] == last)]
        if len(sub) >= 20:
            wr = (sub['final_pnl'] > 0).mean() * 100
            avg = sub['final_pnl'].mean()
            print(f"  {d} + {last}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 8. max_profit 분석
print("\n### 8. 최대 수익 (max_profit)")
print(f"  승리 평균 max_profit: {wins['max_profit'].mean():.2f}%")
print(f"  패배 평균 max_profit: {losses['max_profit'].mean():.2f}%")

# 9. max_loss 분석
print("\n### 9. 최대 손실 (max_loss)")
print(f"  승리 평균 max_loss: {wins['max_loss'].mean():.2f}%")
print(f"  패배 평균 max_loss: {losses['max_loss'].mean():.2f}%")

# 10. 보유시간
print("\n### 10. 보유시간 (hold_hours)")
print(f"  승리 평균: {wins['hold_hours'].mean():.1f}시간")
print(f"  패배 평균: {losses['hold_hours'].mean():.1f}시간")

print("\n" + "="*80)
print("🔍 승률 높은 조합 찾기")
print("="*80)

# 조합 탐색
results = []
for d in ['LONG', 'SHORT']:
    for last in ['UP', 'DOWN']:
        for sl_low, sl_high in [(0, 2), (2, 4), (4, 7)]:
            for sq_low, sq_high in [(0, 1.5), (1.5, 3), (3, 6)]:
                sub = df[(df['direction'] == d) & 
                         (df['last_direction'] == last) &
                         (df['sl_pct'] >= sl_low) & (df['sl_pct'] < sl_high) &
                         (df['squeeze_range_pct'] >= sq_low) & (df['squeeze_range_pct'] < sq_high)]
                if len(sub) >= 15:
                    wr = (sub['final_pnl'] > 0).mean() * 100
                    avg = sub['final_pnl'].mean()
                    results.append({
                        'combo': f"{d}+{last}+SL{sl_low}-{sl_high}+범위{sq_low}-{sq_high}",
                        'count': len(sub),
                        'win_rate': wr,
                        'avg_pnl': avg
                    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('win_rate', ascending=False)
print("\n### 승률 TOP 10 조합")
for i, row in results_df.head(10).iterrows():
    print(f"  {row['combo']}: {row['count']}건, 승률 {row['win_rate']:.1f}%, 평균 {row['avg_pnl']:.2f}%")

results_df = results_df.sort_values('avg_pnl', ascending=False)
print("\n### 평균수익 TOP 10 조합")
for i, row in results_df.head(10).iterrows():
    print(f"  {row['combo']}: {row['count']}건, 승률 {row['win_rate']:.1f}%, 평균 {row['avg_pnl']:.2f}%")
