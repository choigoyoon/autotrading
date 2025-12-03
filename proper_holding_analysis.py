import pandas as pd
import numpy as np

print("=" * 70)
print("올바른 분석: 추세 안에서 홀딩하면 어떻게 되는가?")
print("=" * 70)

# 원본 1시간봉 데이터와 추세 데이터 로드
df = pd.read_csv('btc_with_staircase_trend.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

trades = pd.read_csv('trades_with_staircase_trend.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

print(f"총 캔들: {len(df)}")
print(f"총 거래 신호: {len(trades)}")
print()

# 핵심 변경:
# 기존: 다음 수축까지만 홀딩 (~27시간)
# 개선: 추세가 바뀔 때까지 홀딩

# 추세 변경 시점 찾기
df['trend_change'] = df['staircase_trend'] != df['staircase_trend'].shift(1)
trend_changes = df[df['trend_change']].copy()

print(f"추세 변경 시점: {len(trend_changes)}개")
print()

# 각 거래에 대해 "추세가 바뀔 때까지 홀딩" 수익 계산
def calculate_extended_holding(trade_row, df, trend_changes):
    """추세가 바뀔 때까지 홀딩했을 때의 수익 계산"""
    entry_time = trade_row['datetime']
    entry_trend = trade_row['staircase_trend']
    direction = trade_row['direction']
    
    # 진입 시점의 가격
    entry_idx = df[df['datetime'] >= entry_time].index
    if len(entry_idx) == 0:
        return None, None, None
    entry_idx = entry_idx[0]
    entry_price = df.loc[entry_idx, 'close']
    
    # 추세가 바뀌는 시점 찾기
    next_trend_change = trend_changes[trend_changes['datetime'] > entry_time]
    
    if len(next_trend_change) == 0:
        # 추세 변경이 없으면 데이터 끝까지
        exit_idx = len(df) - 1
    else:
        exit_time = next_trend_change.iloc[0]['datetime']
        exit_idx = df[df['datetime'] >= exit_time].index[0]
    
    exit_price = df.loc[exit_idx, 'close']
    holding_hours = exit_idx - entry_idx
    
    # 수익 계산
    if direction == 'LONG':
        pnl = (exit_price - entry_price) / entry_price * 100
    else:  # SHORT
        pnl = (entry_price - exit_price) / entry_price * 100
    
    # 홀딩 기간 중 최대 수익/손실
    holding_data = df.loc[entry_idx:exit_idx]
    if direction == 'LONG':
        max_profit = (holding_data['high'].max() - entry_price) / entry_price * 100
        max_loss = (holding_data['low'].min() - entry_price) / entry_price * 100
    else:
        max_profit = (entry_price - holding_data['low'].min()) / entry_price * 100
        max_loss = (entry_price - holding_data['high'].max()) / entry_price * 100
    
    return pnl, holding_hours, max_profit, max_loss

# 샘플로 계산 (전체는 시간 오래 걸림)
print("=== 추세 종료까지 홀딩 시 결과 ===\n")

results = []
for idx, row in trades.iterrows():
    result = calculate_extended_holding(row, df, trend_changes)
    if result[0] is not None:
        results.append({
            'datetime': row['datetime'],
            'direction': row['direction'],
            'trend': row['staircase_trend'],
            'original_pnl': row['final_pnl'],
            'extended_pnl': result[0],
            'holding_hours': result[1],
            'max_profit': result[2],
            'max_loss': result[3]
        })

results_df = pd.DataFrame(results)

print(f"분석 대상: {len(results_df)}건")
print()

# 기존 vs 확장 홀딩 비교
print("=== 기존 전략 (다음 수축까지) ===")
print(f"승률: {(results_df['original_pnl'] > 0).mean()*100:.1f}%")
print(f"평균수익: {results_df['original_pnl'].mean():.2f}%")
print(f"총수익: {results_df['original_pnl'].sum():.1f}%")
print()

print("=== 확장 홀딩 (추세 종료까지) ===")
print(f"승률: {(results_df['extended_pnl'] > 0).mean()*100:.1f}%")
print(f"평균수익: {results_df['extended_pnl'].mean():.2f}%")
print(f"총수익: {results_df['extended_pnl'].sum():.1f}%")
print(f"평균 홀딩: {results_df['holding_hours'].mean():.0f}시간")
print()

# 추세별 분석
print("=== 추세별 확장 홀딩 결과 ===")
for trend in ['UPTREND', 'DOWNTREND', 'TRANSITION']:
    subset = results_df[results_df['trend'] == trend]
    if len(subset) == 0:
        continue
    
    print(f"\n[{trend}] ({len(subset)}건)")
    
    # 방향별
    for direction in ['LONG', 'SHORT']:
        sub = subset[subset['direction'] == direction]
        if len(sub) == 0:
            continue
        
        print(f"  {direction}:")
        print(f"    기존 - 승률 {(sub['original_pnl']>0).mean()*100:.1f}%, 평균 {sub['original_pnl'].mean():.2f}%")
        print(f"    확장 - 승률 {(sub['extended_pnl']>0).mean()*100:.1f}%, 평균 {sub['extended_pnl'].mean():.2f}%")
        print(f"    홀딩 {sub['holding_hours'].mean():.0f}h, max수익 {sub['max_profit'].mean():.1f}%, max손실 {sub['max_loss'].mean():.1f}%")

# 핵심 전략: 추세 방향으로만 진입 + 추세 종료까지 홀딩
print("\n" + "=" * 70)
print("최종 전략: 추세 방향 진입 + 추세 종료 홀딩")
print("=" * 70)

# UPTREND에서 LONG만
uptrend_long = results_df[(results_df['trend'] == 'UPTREND') & 
                          (results_df['direction'] == 'LONG')]
# DOWNTREND에서 SHORT만
downtrend_short = results_df[(results_df['trend'] == 'DOWNTREND') & 
                             (results_df['direction'] == 'SHORT')]

combined_trend = pd.concat([uptrend_long, downtrend_short])

print(f"\n추세추종 전략 ({len(combined_trend)}건)")
print(f"  기존 - 승률 {(combined_trend['original_pnl']>0).mean()*100:.1f}%, 평균 {combined_trend['original_pnl'].mean():.2f}%")
print(f"  확장 - 승률 {(combined_trend['extended_pnl']>0).mean()*100:.1f}%, 평균 {combined_trend['extended_pnl'].mean():.2f}%")
print(f"  총수익: {combined_trend['extended_pnl'].sum():.1f}%")

# 역발상: UPTREND에서 SHORT 발생 → LONG으로 뒤집기
print("\n" + "=" * 70)
print("역발상: 발산 방향 무시 + 추세 방향 진입")
print("=" * 70)

# UPTREND에서 모든 신호 → LONG
uptrend_all = results_df[results_df['trend'] == 'UPTREND'].copy()
uptrend_all['trend_pnl'] = uptrend_all.apply(
    lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'],
    axis=1
)

# DOWNTREND에서 모든 신호 → SHORT
downtrend_all = results_df[results_df['trend'] == 'DOWNTREND'].copy()
downtrend_all['trend_pnl'] = downtrend_all.apply(
    lambda x: x['extended_pnl'] if x['direction'] == 'SHORT' else -x['extended_pnl'],
    axis=1
)

print(f"\n[UPTREND 무조건 LONG] ({len(uptrend_all)}건)")
print(f"  승률: {(uptrend_all['trend_pnl']>0).mean()*100:.1f}%")
print(f"  평균: {uptrend_all['trend_pnl'].mean():.2f}%")
print(f"  총수익: {uptrend_all['trend_pnl'].sum():.1f}%")

print(f"\n[DOWNTREND 무조건 SHORT] ({len(downtrend_all)}건)")
print(f"  승률: {(downtrend_all['trend_pnl']>0).mean()*100:.1f}%")
print(f"  평균: {downtrend_all['trend_pnl'].mean():.2f}%")
print(f"  총수익: {downtrend_all['trend_pnl'].sum():.1f}%")

total_trend = pd.concat([uptrend_all[['trend_pnl']], downtrend_all[['trend_pnl']]])
print(f"\n[합산] ({len(total_trend)}건)")
print(f"  승률: {(total_trend['trend_pnl']>0).mean()*100:.1f}%")
print(f"  평균: {total_trend['trend_pnl'].mean():.2f}%")
print(f"  총수익: {total_trend['trend_pnl'].sum():.1f}%")

# 저장
results_df.to_csv('extended_holding_results.csv', index=False)
print("\n✅ 저장: extended_holding_results.csv")

