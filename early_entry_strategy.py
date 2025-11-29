"""
조기 진입 전략: L값 반등 + 추세선 근접
- L값 확정 및 반등 확인
- 추세선 돌파 직전 진입
- 목표: 추세선 돌파 전략보다 빠르고, L값 반등보다 안전
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("조기 진입 전략 분석")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# 추세선 돌파 데이터
df_breakouts = pd.read_csv('backtest_filtered_10bars.csv')
df_breakouts['datetime'] = pd.to_datetime(df_breakouts['datetime'])

# 최근 2년
cutoff = df['datetime'].max() - timedelta(days=730)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

# L값 수집
l_vals = []
for i in range(1, len(df)):
    if df.iloc[i-1]['macd_hist'] < 0 and df.iloc[i]['macd_hist'] >= 0:
        l_vals.append({
            'idx': i,
            'datetime': df.iloc[i]['datetime'],
            'price': df.iloc[i]['low']
        })

print(f"L값: {len(l_vals)}개")
print(f"추세선 돌파: {len(df_breakouts)}개")
print()

# 전략: L값 반등 확인 → 추세선 근처까지 상승 → 진입
print("전략 로직:")
print("  1. L값 확정 및 반등 확인 (1% 이상)")
print("  2. 반등 후 상승 추세 확인")
print("  3. 추세선 돌파 5봉 이내 진입")
print()

trades = []

for l in l_vals[::3]:
    l_idx = l['idx']
    l_time = l['datetime']
    l_price = l['price']

    if l_idx + 50 >= len(df):
        continue

    # 1단계: 강한 반등 확인
    bounce_idx = None
    for i in range(1, 6):
        c = df.iloc[l_idx + i]
        bounce_str = (c['close'] - l_price) / l_price * 100
        if c['close'] > c['open'] and bounce_str >= 1.0:
            bounce_idx = l_idx + i
            break

    if bounce_idx is None:
        continue

    # 2단계: 추세선 돌파 찾기
    breakout_idx = None
    for i in range(bounce_idx, min(bounce_idx + 30, len(df))):
        check_time = df.iloc[i]['datetime']
        matching = df_breakouts[
            (df_breakouts['datetime'] >= check_time - timedelta(minutes=15)) &
            (df_breakouts['datetime'] <= check_time + timedelta(minutes=15))
        ]
        if len(matching) > 0:
            breakout_idx = i
            break

    if breakout_idx is None:
        continue

    # 3단계: 돌파 직전 진입 (5봉 전)
    entry_idx = max(bounce_idx + 1, breakout_idx - 5)

    if entry_idx >= len(df):
        continue

    # 백테스트
    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price * 1.02
    sl_price = entry_price * 0.98

    pnl = None
    exit_type = None

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        c = df.iloc[j]

        if c['low'] <= sl_price:
            pnl = -2.0
            exit_type = 'SL'
            break
        elif c['high'] >= tp_price:
            pnl = 2.0
            exit_type = 'TP'
            break

    if pnl is None:
        pnl = (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100
        exit_type = 'TIMEOUT'

    trades.append({
        'pnl': pnl,
        'exit_type': exit_type,
        'bars_before_breakout': breakout_idx - entry_idx
    })

# 결과
print("=" * 70)
print("백테스트 결과")
print("=" * 70)
print()

if len(trades) > 0:
    df_trades = pd.DataFrame(trades)

    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100
    avg_pnl = df_trades['pnl'].mean()
    avg_bars = df_trades['bars_before_breakout'].mean()

    print(f"총 거래: {len(df_trades)}개")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 PnL: {avg_pnl:+.2f}%")
    print(f"평균 진입 시점: 돌파 {avg_bars:.1f}봉 전")
    print()

    print("비교:")
    print("-" * 70)
    print(f"{'전략':<30} {'승률':<15} {'평균 PnL':<15}")
    print("-" * 70)
    print(f"{'L값 즉시 진입':<30} {'50.8%':<15} {'+0.06%':<15}")
    print(f"{'L값 반등 진입':<30} {'52.1%':<15} {'+0.11%':<15}")
    print(f"{'강한 반등 진입':<30} {'65.3%':<15} {'?':<15}")
    print(f"{'조기 진입 (새로운)':<30} {f'{win_rate:.1f}%':<15} {f'{avg_pnl:+.2f}%':<15}")
    print(f"{'추세선 돌파 (기존)':<30} {'79.8%':<15} {'+0.96%':<15}")
    print()

    if win_rate > 65:
        print("✅ 조기 진입 전략 유효!")
        print(f"   - 추세선 돌파보다 {avg_bars:.1f}봉 빠름")
        print(f"   - 승률: {win_rate:.1f}% (목표: 70%+)")
    else:
        print("⚠️ 추가 개선 필요")
        print(f"   - 현재 승률: {win_rate:.1f}%")
        print(f"   - 목표 승률: 70%+")
else:
    print("거래 없음")

print()
print("✅ 분석 완료")
