import pandas as pd
import numpy as np

# 데이터 로드
df_trend = pd.read_csv('btc_with_staircase_trend.csv')
df_trend['datetime'] = pd.to_datetime(df_trend['datetime'])

trades = pd.read_csv('trades_with_indicators.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

print("=" * 60)
print("수축/발산을 계단식 추세 안에서 분석")
print("=" * 60)
print(f"총 거래: {len(trades)}건")
print()

# 각 거래에 계단식 추세 라벨 매핑
def get_trend_at_time(dt, trend_df):
    """특정 시점의 계단식 추세 반환"""
    idx = trend_df[trend_df['datetime'] <= dt].index
    if len(idx) == 0:
        return 'UNKNOWN'
    return trend_df.loc[idx[-1], 'staircase_trend']

trades['staircase_trend'] = trades['datetime'].apply(
    lambda x: get_trend_at_time(x, df_trend)
)

print("=== 거래별 계단식 추세 분포 ===")
print(trades['staircase_trend'].value_counts())
print()

# 기존 로직: 원래 방향대로 진입 (direction 컬럼)
# 승/패 계산
trades['is_win'] = trades['final_pnl'] > 0

print("=" * 60)
print("1. 원래 방향 진입 결과 (현재 로직)")
print("=" * 60)

# 계단식 추세별 성과
for trend in ['UPTREND', 'DOWNTREND', 'TRANSITION']:
    subset = trades[trades['staircase_trend'] == trend]
    if len(subset) == 0:
        continue
    
    win_rate = subset['is_win'].mean() * 100
    avg_pnl = subset['final_pnl'].mean()
    total_pnl = subset['final_pnl'].sum()
    
    print(f"\n[{trend}] ({len(subset)}건)")
    print(f"  승률: {win_rate:.1f}%, 평균수익: {avg_pnl:.2f}%, 총수익: {total_pnl:.1f}%")
    
    # 방향별 세부
    for direction in ['LONG', 'SHORT']:
        sub = subset[subset['direction'] == direction]
        if len(sub) == 0:
            continue
        wr = sub['is_win'].mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"    {direction}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

print("\n")
print("=" * 60)
print("2. 추세 방향으로만 진입 (추세추종)")
print("=" * 60)

# 계단식 상승 중 → LONG만
# 계단식 하락 중 → SHORT만
def calculate_trend_following():
    results = []
    
    # UPTREND에서 LONG만
    uptrend_long = trades[(trades['staircase_trend'] == 'UPTREND') & 
                          (trades['direction'] == 'LONG')]
    if len(uptrend_long) > 0:
        results.append({
            'condition': 'UPTREND → LONG',
            'count': len(uptrend_long),
            'win_rate': uptrend_long['is_win'].mean() * 100,
            'avg_pnl': uptrend_long['final_pnl'].mean(),
            'total_pnl': uptrend_long['final_pnl'].sum()
        })
    
    # DOWNTREND에서 SHORT만
    downtrend_short = trades[(trades['staircase_trend'] == 'DOWNTREND') & 
                             (trades['direction'] == 'SHORT')]
    if len(downtrend_short) > 0:
        results.append({
            'condition': 'DOWNTREND → SHORT',
            'count': len(downtrend_short),
            'win_rate': downtrend_short['is_win'].mean() * 100,
            'avg_pnl': downtrend_short['final_pnl'].mean(),
            'total_pnl': downtrend_short['final_pnl'].sum()
        })
    
    # UPTREND에서 SHORT (역추세) - 비교용
    uptrend_short = trades[(trades['staircase_trend'] == 'UPTREND') & 
                           (trades['direction'] == 'SHORT')]
    if len(uptrend_short) > 0:
        results.append({
            'condition': 'UPTREND → SHORT (역추세)',
            'count': len(uptrend_short),
            'win_rate': uptrend_short['is_win'].mean() * 100,
            'avg_pnl': uptrend_short['final_pnl'].mean(),
            'total_pnl': uptrend_short['final_pnl'].sum()
        })
    
    # DOWNTREND에서 LONG (역추세) - 비교용
    downtrend_long = trades[(trades['staircase_trend'] == 'DOWNTREND') & 
                            (trades['direction'] == 'LONG')]
    if len(downtrend_long) > 0:
        results.append({
            'condition': 'DOWNTREND → LONG (역추세)',
            'count': len(downtrend_long),
            'win_rate': downtrend_long['is_win'].mean() * 100,
            'avg_pnl': downtrend_long['final_pnl'].mean(),
            'total_pnl': downtrend_long['final_pnl'].sum()
        })
    
    return results

trend_results = calculate_trend_following()
for r in trend_results:
    marker = "⭐" if r['win_rate'] >= 50 and r['avg_pnl'] >= 0.3 else ""
    print(f"{r['condition']}: {r['count']}건")
    print(f"  승률: {r['win_rate']:.1f}%, 평균: {r['avg_pnl']:.2f}%, 총: {r['total_pnl']:.1f}% {marker}")
    print()

print("=" * 60)
print("3. 역발상: 추세 반대로 진입")
print("=" * 60)

# 계단식 상승 중 하방발산 → 되돌림이니까 LONG
# 계단식 하락 중 상방발산 → 반등이니까 SHORT
def calculate_counter_breakout():
    results = []
    
    # UPTREND에서 하방 발산(SHORT) 했지만 → 실제로는 LONG 진입
    # 즉, 원래 SHORT 신호인데 LONG으로 뒤집기
    uptrend_counter = trades[(trades['staircase_trend'] == 'UPTREND') & 
                             (trades['direction'] == 'SHORT')]
    if len(uptrend_counter) > 0:
        # 원래 SHORT으로 계산된 수익을 뒤집으면
        # 실제로는 반대 포지션이므로 대략 반대 수익
        # 여기서는 원래 수익이 음수면 반대 진입 시 양수가 될 가능성
        counter_pnl = -uptrend_counter['final_pnl']  # 단순 반전
        results.append({
            'condition': 'UPTREND + 하방발산 → LONG (역발상)',
            'count': len(uptrend_counter),
            'original_win_rate': uptrend_counter['is_win'].mean() * 100,
            'original_avg_pnl': uptrend_counter['final_pnl'].mean(),
            'counter_win_rate': (counter_pnl > 0).mean() * 100,
            'counter_avg_pnl': counter_pnl.mean()
        })
    
    # DOWNTREND에서 상방 발산(LONG) 했지만 → 실제로는 SHORT 진입
    downtrend_counter = trades[(trades['staircase_trend'] == 'DOWNTREND') & 
                               (trades['direction'] == 'LONG')]
    if len(downtrend_counter) > 0:
        counter_pnl = -downtrend_counter['final_pnl']
        results.append({
            'condition': 'DOWNTREND + 상방발산 → SHORT (역발상)',
            'count': len(downtrend_counter),
            'original_win_rate': downtrend_counter['is_win'].mean() * 100,
            'original_avg_pnl': downtrend_counter['final_pnl'].mean(),
            'counter_win_rate': (counter_pnl > 0).mean() * 100,
            'counter_avg_pnl': counter_pnl.mean()
        })
    
    return results

counter_results = calculate_counter_breakout()
for r in counter_results:
    print(f"\n{r['condition']}: {r['count']}건")
    print(f"  원래 방향: 승률 {r['original_win_rate']:.1f}%, 평균 {r['original_avg_pnl']:.2f}%")
    print(f"  역발상:    승률 {r['counter_win_rate']:.1f}%, 평균 {r['counter_avg_pnl']:.2f}%")

print("\n")
print("=" * 60)
print("4. 핵심 인사이트: 계단식 추세 + 홀딩 전략")
print("=" * 60)

# 계단식 상승 중이면 → 어떤 발산이든 LONG 홀딩
# 계단식 하락 중이면 → 어떤 발산이든 SHORT 홀딩

# ALL LONG in UPTREND
uptrend_all = trades[trades['staircase_trend'] == 'UPTREND'].copy()
# LONG은 그대로, SHORT은 뒤집기
uptrend_all['adjusted_pnl'] = uptrend_all.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'], 
    axis=1
)
uptrend_all['adjusted_win'] = uptrend_all['adjusted_pnl'] > 0

print(f"\n[UPTREND 구간에서 무조건 LONG] ({len(uptrend_all)}건)")
print(f"  승률: {uptrend_all['adjusted_win'].mean()*100:.1f}%")
print(f"  평균: {uptrend_all['adjusted_pnl'].mean():.2f}%")
print(f"  총수익: {uptrend_all['adjusted_pnl'].sum():.1f}%")

# ALL SHORT in DOWNTREND
downtrend_all = trades[trades['staircase_trend'] == 'DOWNTREND'].copy()
downtrend_all['adjusted_pnl'] = downtrend_all.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'SHORT' else -x['final_pnl'], 
    axis=1
)
downtrend_all['adjusted_win'] = downtrend_all['adjusted_pnl'] > 0

print(f"\n[DOWNTREND 구간에서 무조건 SHORT] ({len(downtrend_all)}건)")
print(f"  승률: {downtrend_all['adjusted_win'].mean()*100:.1f}%")
print(f"  평균: {downtrend_all['adjusted_pnl'].mean():.2f}%")
print(f"  총수익: {downtrend_all['adjusted_pnl'].sum():.1f}%")

# 합산
combined = pd.concat([uptrend_all, downtrend_all])
print(f"\n[추세추종 전략 합산] ({len(combined)}건)")
print(f"  승률: {combined['adjusted_win'].mean()*100:.1f}%")
print(f"  평균: {combined['adjusted_pnl'].mean():.2f}%")
print(f"  총수익: {combined['adjusted_pnl'].sum():.1f}%")

# 결과 저장
trades.to_csv('trades_with_staircase_trend.csv', index=False)
print("\n✅ 저장: trades_with_staircase_trend.csv")

