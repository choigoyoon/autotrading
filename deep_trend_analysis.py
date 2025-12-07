import pandas as pd
import numpy as np

trades = pd.read_csv('trades_with_staircase_trend.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

print("=" * 60)
print("문제 분석: 왜 DOWNTREND + SHORT가 안 되는가?")
print("=" * 60)

# DOWNTREND + SHORT 케이스 상세 분석
down_short = trades[(trades['staircase_trend'] == 'DOWNTREND') & 
                    (trades['direction'] == 'SHORT')]

print(f"\n총 {len(down_short)}건")
print(f"승률: {(down_short['final_pnl'] > 0).mean()*100:.1f}%")
print(f"평균손익: {down_short['final_pnl'].mean():.2f}%")
print()

# exit_reason별 분석
print("=== Exit Reason 분석 ===")
for reason in down_short['exit_reason'].unique():
    subset = down_short[down_short['exit_reason'] == reason]
    wr = (subset['final_pnl'] > 0).mean() * 100
    avg = subset['final_pnl'].mean()
    print(f"{reason}: {len(subset)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

print()

# 문제: 계단식 하락인데도 반등이 많음
# → 계단식 하락 = LL + LH 구간
# → 근데 그 안에서도 반등 파동이 있음

print("=" * 60)
print("핵심 문제: '계단식 추세' 라벨의 한계")
print("=" * 60)
print("""
계단식 하락 구간이라도:
- 하락 파동 중간에 반등 파동이 존재
- 수축/발산 시점이 반등 시작점일 수 있음
- 72시간 윈도우로 감지한 추세는 '큰 그림'
- 수축/발산은 몇 시간~며칠 단위의 '작은 파동'

→ 큰 그림과 작은 파동의 타이밍이 안 맞음
""")

print("=" * 60)
print("해결책: 연속 수축/발산 분석")
print("=" * 60)
print("""
단일 수축/발산만 보지 말고:
1. 이전 N개 수축/발산의 방향 연속성 확인
2. 연속 상승 발산 → 계단식 상승 중
3. 연속 하락 발산 → 계단식 하락 중
4. 추세와 같은 방향으로만 진입
""")

# 연속 발산 분석
trades_sorted = trades.sort_values('datetime').reset_index(drop=True)

# 이전 N개 발산 방향 계산
def get_previous_directions(df, n=3):
    df = df.copy()
    for i in range(1, n+1):
        df[f'prev_{i}_direction'] = df['direction'].shift(i)
        df[f'prev_{i}_pnl'] = df['final_pnl'].shift(i)
    return df

trades_with_prev = get_previous_directions(trades_sorted, n=5)

# 연속 방향 패턴 분석
def count_consecutive_direction(row, n=3):
    """이전 n개 발산이 같은 방향인지 확인"""
    long_count = 0
    short_count = 0
    for i in range(1, n+1):
        prev_dir = row.get(f'prev_{i}_direction')
        if prev_dir == 'LONG':
            long_count += 1
        elif prev_dir == 'SHORT':
            short_count += 1
    return long_count, short_count

trades_with_prev['prev_long_count'] = trades_with_prev.apply(
    lambda x: count_consecutive_direction(x, 3)[0], axis=1
)
trades_with_prev['prev_short_count'] = trades_with_prev.apply(
    lambda x: count_consecutive_direction(x, 3)[1], axis=1
)

print("\n=== 이전 3개 발산 연속성 기반 분석 ===")

# 이전 3개 중 LONG이 많으면 상승 추세
# → 현재 LONG으로 진입
mostly_long = trades_with_prev[trades_with_prev['prev_long_count'] >= 2].copy()
mostly_long['adjusted_pnl'] = mostly_long.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'],
    axis=1
)
print(f"\n[이전 3개 중 LONG 2개 이상 → LONG 진입] ({len(mostly_long)}건)")
print(f"  승률: {(mostly_long['adjusted_pnl'] > 0).mean()*100:.1f}%")
print(f"  평균: {mostly_long['adjusted_pnl'].mean():.2f}%")

# 이전 3개 중 SHORT이 많으면 하락 추세
# → 현재 SHORT으로 진입
mostly_short = trades_with_prev[trades_with_prev['prev_short_count'] >= 2].copy()
mostly_short['adjusted_pnl'] = mostly_short.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'SHORT' else -x['final_pnl'],
    axis=1
)
print(f"\n[이전 3개 중 SHORT 2개 이상 → SHORT 진입] ({len(mostly_short)}건)")
print(f"  승률: {(mostly_short['adjusted_pnl'] > 0).mean()*100:.1f}%")
print(f"  평균: {mostly_short['adjusted_pnl'].mean():.2f}%")

# 이전 발산이 수익이었던 방향으로 진입
print("\n=== 이전 발산 수익 기반 분석 ===")

# 이전 발산이 LONG이고 수익이었으면 → LONG 계속
# 이전 발산이 SHORT이고 수익이었으면 → SHORT 계속
trades_with_prev['follow_winner'] = None

for idx, row in trades_with_prev.iterrows():
    prev_dir = row['prev_1_direction']
    prev_pnl = row['prev_1_pnl']
    
    if pd.isna(prev_dir) or pd.isna(prev_pnl):
        continue
    
    # 이전 발산이 수익이었으면 같은 방향 유지
    if prev_pnl > 0:
        trades_with_prev.loc[idx, 'follow_winner'] = prev_dir
    else:
        # 이전이 손실이었으면 반대 방향
        trades_with_prev.loc[idx, 'follow_winner'] = 'SHORT' if prev_dir == 'LONG' else 'LONG'

# follow_winner 전략 성과
valid_follow = trades_with_prev[trades_with_prev['follow_winner'].notna()].copy()
valid_follow['follow_pnl'] = valid_follow.apply(
    lambda x: x['final_pnl'] if x['direction'] == x['follow_winner'] else -x['final_pnl'],
    axis=1
)

print(f"\n[이전 수익 방향 추종 전략] ({len(valid_follow)}건)")
print(f"  승률: {(valid_follow['follow_pnl'] > 0).mean()*100:.1f}%")
print(f"  평균: {valid_follow['follow_pnl'].mean():.2f}%")
print(f"  총수익: {valid_follow['follow_pnl'].sum():.1f}%")

# 저장
trades_with_prev.to_csv('trades_with_sequence.csv', index=False)
print("\n✅ 저장: trades_with_sequence.csv")

