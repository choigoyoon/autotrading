"""
L값 근처 하이킨아시 도지 전략
가설: L값 근처에서 도지 → 방향성 가속화
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("L값 근처 하이킨아시 도지 전략")
print("=" * 80)
print()

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv('btcusdt_1h_raw.csv', parse_dates=['datetime'])
l_values = pd.read_csv('all_L_values.csv', parse_dates=['datetime'])
print(f"  1시간봉: {len(df):,}개")
print(f"  L값: {len(l_values):,}개")
print()

# 하이킨아시 변환
print("하이킨아시 변환...")
df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['ha_open'] = 0.0
df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

for i in range(1, len(df)):
    df.loc[i, 'ha_open'] = (df.loc[i-1, 'ha_open'] + df.loc[i-1, 'ha_close']) / 2

df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

# 도지 정의
df['ha_range'] = df['ha_high'] - df['ha_low']
df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
df['ha_body_pct'] = df['ha_body'] / df['ha_range']

# 도지 = body가 range의 10% 미만
df['is_doji'] = (df['ha_body_pct'] < 0.1) & (df['ha_range'] > 0)
print(f"  도지: {df['is_doji'].sum():,}개")
print()

# 각 캔들에서 가장 가까운 L값까지의 거리 계산
print("L값 근접도 계산 중...")
df['nearest_L_dist_pct'] = np.nan
df['nearest_L_datetime'] = pd.NaT
df['nearest_L_value'] = np.nan

# L값을 datetime으로 인덱싱
l_dict = dict(zip(l_values['datetime'], l_values['L_value']))

for i in range(len(df)):
    curr_dt = df.loc[i, 'datetime']
    curr_price = df.loc[i, 'close']

    # 현재 시점 이전의 L값들만 고려 (미래 데이터 방지)
    past_l_values = l_values[l_values['datetime'] <= curr_dt]

    if len(past_l_values) == 0:
        continue

    # 가장 가까운 L값 찾기
    past_l_values['price_dist'] = abs(past_l_values['L_value'] - curr_price) / curr_price * 100
    nearest = past_l_values.nsmallest(1, 'price_dist').iloc[0]

    df.loc[i, 'nearest_L_dist_pct'] = nearest['price_dist']
    df.loc[i, 'nearest_L_datetime'] = nearest['datetime']
    df.loc[i, 'nearest_L_value'] = nearest['L_value']

print("  완료!")
print()

# 도지 분류
# 1. L값 근처 도지 (±1% 이내)
# 2. L값 근처 도지 (±2% 이내)
# 3. L값 근처 도지 (±3% 이내)
# 4. L값에서 먼 도지 (3% 초과)

df['doji_near_L_1pct'] = df['is_doji'] & (df['nearest_L_dist_pct'] <= 1.0)
df['doji_near_L_2pct'] = df['is_doji'] & (df['nearest_L_dist_pct'] <= 2.0)
df['doji_near_L_3pct'] = df['is_doji'] & (df['nearest_L_dist_pct'] <= 3.0)
df['doji_far_from_L'] = df['is_doji'] & (df['nearest_L_dist_pct'] > 3.0)

print("도지 분류:")
print(f"  L값 ±1% 이내 도지: {df['doji_near_L_1pct'].sum():,}개")
print(f"  L값 ±2% 이내 도지: {df['doji_near_L_2pct'].sum():,}개")
print(f"  L값 ±3% 이내 도지: {df['doji_near_L_3pct'].sum():,}개")
print(f"  L값에서 먼 도지: {df['doji_far_from_L'].sum():,}개")
print()

# 백테스트 함수
def backtest(df, signal_col, name):
    """
    간단한 백테스트: 도지 다음봉에서 진입
    TP: 2%, SL: -1.5%, MAX_HOLD: 50봉
    """
    TP = 0.02
    SL = -0.015
    MAX_HOLD = 50
    COST = 0.0021  # 0.21%

    trades = []

    signal_indices = df[df[signal_col]].index.tolist()

    for i in signal_indices:
        # 다음봉 체크
        if i + 1 >= len(df):
            continue

        next_bar = df.iloc[i+1]

        # 다음봉이 상승인지 확인
        if next_bar['ha_close'] <= next_bar['ha_open']:
            continue

        # 진입
        if i + 2 >= len(df):
            continue

        entry_idx = i + 2
        entry_price = df.iloc[entry_idx]['open']
        entry_time = df.iloc[entry_idx]['datetime']

        tp_level = entry_price * (1 + TP)
        sl_level = entry_price * (1 + SL)

        # 홀딩
        exit_reason = 'TIMEOUT'
        exit_idx = min(entry_idx + MAX_HOLD, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']

        for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, len(df))):
            bar = df.iloc[j]

            if bar['high'] >= tp_level:
                exit_reason = 'TP'
                exit_idx = j
                exit_price = tp_level
                break
            elif bar['low'] <= sl_level:
                exit_reason = 'SL'
                exit_idx = j
                exit_price = sl_level
                break

        gross_pnl = (exit_price - entry_price) / entry_price
        net_pnl = gross_pnl - COST

        trades.append({
            'doji_time': df.iloc[i]['datetime'],
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': df.iloc[exit_idx]['datetime'],
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'gross_pnl': gross_pnl * 100,
            'net_pnl': net_pnl * 100,
            'nearest_L_dist': df.iloc[i]['nearest_L_dist_pct'],
            'nearest_L_value': df.iloc[i]['nearest_L_value']
        })

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)

    win_rate = (trades_df['net_pnl'] > 0).sum() / len(trades_df) * 100
    avg_pnl = trades_df['net_pnl'].mean()
    total_pnl = trades_df['net_pnl'].sum()

    return {
        'name': name,
        'trades': len(trades_df),
        'trades_per_year': len(trades_df) / 5,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'df': trades_df
    }

# 백테스트 실행
print("=" * 80)
print("백테스트 실행")
print("=" * 80)
print()
print("조건: 도지 다음봉 상승 확인, TP 2%, SL -1.5%, 비용 0.21%")
print()

strategies = [
    ('doji_near_L_1pct', 'L값 ±1% 도지'),
    ('doji_near_L_2pct', 'L값 ±2% 도지'),
    ('doji_near_L_3pct', 'L값 ±3% 도지'),
    ('doji_far_from_L', 'L값 먼 도지 (>3%)'),
    ('is_doji', '전체 도지 (비교군)'),
]

results = []

for signal_col, name in strategies:
    result = backtest(df, signal_col, name)
    if result:
        results.append(result)
        print(f"【{name}】")
        print(f"  거래: {result['trades']}개 (연 {result['trades_per_year']:.1f}개)")
        print(f"  승률: {result['win_rate']:.2f}%")
        print(f"  평균: {result['avg_pnl']:+.3f}%")
        print(f"  5년: {result['total_pnl']:+.2f}%")

        if result['win_rate'] >= 51.6:
            print(f"  ⭐ 우수 (승률 51.6%+ 달성)")
        elif result['win_rate'] >= 50:
            print(f"  ✅ 양호")
        else:
            print(f"  ❌ 개선 필요")
        print()

# 추가 분석: 다음봉 상승폭별 성과
print("=" * 80)
print("추가 분석: L값 근접도와 다음봉 상승폭")
print("=" * 80)
print()

# L값 ±2% 도지만 필터링
l_near_doji = df[df['doji_near_L_2pct']].copy()

if len(l_near_doji) > 0:
    # 다음봉 상승폭 계산
    next_gains = []

    for i in l_near_doji.index:
        if i + 1 >= len(df):
            continue

        curr = df.iloc[i]
        next_bar = df.iloc[i+1]

        gain = (next_bar['ha_close'] - curr['ha_close']) / curr['ha_close'] * 100
        next_gains.append({
            'datetime': curr['datetime'],
            'L_dist': curr['nearest_L_dist_pct'],
            'next_gain': gain,
            'is_bullish': next_bar['ha_close'] > next_bar['ha_open']
        })

    next_df = pd.DataFrame(next_gains)

    print(f"L값 ±2% 도지: {len(next_df)}개")
    print()
    print("다음봉 상승폭 분포:")
    print(f"  평균: {next_df['next_gain'].mean():+.3f}%")
    print(f"  중앙값: {next_df['next_gain'].median():+.3f}%")
    print(f"  상승봉 비율: {next_df['is_bullish'].sum() / len(next_df) * 100:.1f}%")
    print()

    # 다음봉 +0.5%+ 상승 필터
    strong_next = next_df[next_df['next_gain'] > 0.5]
    print(f"다음봉 +0.5%+ 상승: {len(strong_next)}개")
    if len(strong_next) > 0:
        print(f"  L값 평균 거리: {strong_next['L_dist'].mean():.2f}%")
    print()

# 최종 추천
print("=" * 80)
print("결론")
print("=" * 80)
print()

# 최고 성과 찾기
best = max(results, key=lambda x: x['win_rate'])

print(f"🏆 최고 승률: {best['name']}")
print(f"   승률: {best['win_rate']:.2f}%")
print(f"   거래: {best['trades']}개 (연 {best['trades_per_year']:.1f}개)")
print(f"   5년 수익: {best['total_pnl']:+.2f}%")
print()

# L값 근처 vs 먼 곳 비교
l_near = [r for r in results if 'L값' in r['name'] and '먼' not in r['name']]
l_far = [r for r in results if '먼' in r['name']]

if l_near and l_far:
    best_near = max(l_near, key=lambda x: x['win_rate'])
    far = l_far[0]

    print("📊 L값 근접 효과:")
    print(f"  L값 근처: {best_near['win_rate']:.2f}% 승률")
    print(f"  L값 멀리: {far['win_rate']:.2f}% 승률")
    print(f"  차이: {best_near['win_rate'] - far['win_rate']:+.2f}%p")
    print()

    if best_near['win_rate'] > far['win_rate']:
        print("✅ 가설 검증: L값 근처 도지가 더 좋은 성과!")
    else:
        print("❌ 가설 기각: L값 근처 도지가 더 나쁜 성과")

print()
print("=" * 80)
