"""
역방향 접근: 추세선 돌파 신호를 지표로 필터링
======================================================================
기존 문제: 지표를 PRIMARY 신호로 사용 → 50% 승률

새 접근:
1. 추세선 돌파 신호 시작 (983개, 79.8% 승률)
2. 지표로 나쁜 돌파 제거
3. 최종: 고품질 신호만 남김

필터 테스트:
- MACD 상태 (기울기, 수렴)
- 과매도 확인 (RSI, Stoch, CCI)
- 캔들 확인
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("추세선 돌파 필터링 전략")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

df_breakouts = pd.read_csv('backtest_filtered_10bars.csv')
df_breakouts['datetime'] = pd.to_datetime(df_breakouts['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)
df_breakouts = df_breakouts[df_breakouts['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"추세선 돌파: {len(df_breakouts):,}개\n")

# ═══════════════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# MACD 기울기
df['macd_slope'] = df['macd'].diff()

# MACD 수렴
df['macd_convergence'] = abs(df['macd'] - df['macd_signal'])

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Stochastic
low_14 = df['low'].rolling(window=14).min()
high_14 = df['high'].rolling(window=14).max()
df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)

# CCI
tp = (df['high'] + df['low'] + df['close']) / 3
df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

# 캔들
df['is_bullish'] = df['close'] > df['open']

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 백테스트 함수
# ═══════════════════════════════════════════════════════════════════

def backtest_entry(entry_idx):
    if entry_idx >= len(df):
        return None

    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price * 1.02
    sl_price = entry_price * 0.98

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        c = df.iloc[j]
        if c['low'] <= sl_price:
            return -2.0
        elif c['high'] >= tp_price:
            return 2.0

    return (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100

# ═══════════════════════════════════════════════════════════════════
# 필터 테스트
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("필터별 성과")
print("=" * 70)
print()

results = []

# 기준: 모든 돌파
print("기준: 모든 추세선 돌파")
baseline_trades = []
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        idx = matching.index[0]
        pnl = backtest_entry(idx + 1)
        if pnl is not None:
            baseline_trades.append(pnl)

baseline_win_rate = len([p for p in baseline_trades if p > 0]) / len(baseline_trades) * 100
baseline_avg = np.mean(baseline_trades)

results.append({
    'filter': '필터 없음 (기준)',
    'trades': len(baseline_trades),
    'win_rate': baseline_win_rate,
    'avg_pnl': baseline_avg
})

print(f"  거래: {len(baseline_trades)}개")
print(f"  승률: {baseline_win_rate:.1f}%")
print(f"  평균 PnL: {baseline_avg:+.2f}%\n")

# 필터 1: MACD 상승 중
print("필터 1: MACD 상승 중 (기울기 > 0)")
filter1_trades = []
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        idx = matching.index[0]
        row = df.iloc[idx]

        if pd.notna(row['macd_slope']) and row['macd_slope'] > 0:
            pnl = backtest_entry(idx + 1)
            if pnl is not None:
                filter1_trades.append(pnl)

if len(filter1_trades) > 0:
    win_rate = len([p for p in filter1_trades if p > 0]) / len(filter1_trades) * 100
    avg_pnl = np.mean(filter1_trades)
    results.append({
        'filter': '+ MACD 상승',
        'trades': len(filter1_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(filter1_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# 필터 2: RSI 과매도 (< 40)
print("필터 2: RSI < 40 (과매도)")
filter2_trades = []
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        idx = matching.index[0]
        row = df.iloc[idx]

        if pd.notna(row['rsi']) and row['rsi'] < 40:
            pnl = backtest_entry(idx + 1)
            if pnl is not None:
                filter2_trades.append(pnl)

if len(filter2_trades) > 0:
    win_rate = len([p for p in filter2_trades if p > 0]) / len(filter2_trades) * 100
    avg_pnl = np.mean(filter2_trades)
    results.append({
        'filter': '+ RSI < 40',
        'trades': len(filter2_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(filter2_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# 필터 3: MACD 수렴 중
print("필터 3: MACD 수렴 중 (< 200)")
filter3_trades = []
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        idx = matching.index[0]
        row = df.iloc[idx]

        if pd.notna(row['macd_convergence']) and row['macd_convergence'] < 200:
            pnl = backtest_entry(idx + 1)
            if pnl is not None:
                filter3_trades.append(pnl)

if len(filter3_trades) > 0:
    win_rate = len([p for p in filter3_trades if p > 0]) / len(filter3_trades) * 100
    avg_pnl = np.mean(filter3_trades)
    results.append({
        'filter': '+ MACD 수렴',
        'trades': len(filter3_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(filter3_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# 필터 4: 복합 (MACD 상승 + RSI < 40)
print("필터 4: 복합 (MACD 상승 + RSI < 40)")
filter4_trades = []
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        idx = matching.index[0]
        row = df.iloc[idx]

        if (pd.notna(row['macd_slope']) and row['macd_slope'] > 0 and
            pd.notna(row['rsi']) and row['rsi'] < 40):
            pnl = backtest_entry(idx + 1)
            if pnl is not None:
                filter4_trades.append(pnl)

if len(filter4_trades) > 0:
    win_rate = len([p for p in filter4_trades if p > 0]) / len(filter4_trades) * 100
    avg_pnl = np.mean(filter4_trades)
    results.append({
        'filter': '복합 (MACD+RSI)',
        'trades': len(filter4_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(filter4_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# 필터 5: 복합 강화 (MACD 상승 + RSI < 40 + Stoch < 40)
print("필터 5: 복합 강화 (MACD + RSI + Stoch)")
filter5_trades = []
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        idx = matching.index[0]
        row = df.iloc[idx]

        if (pd.notna(row['macd_slope']) and row['macd_slope'] > 0 and
            pd.notna(row['rsi']) and row['rsi'] < 40 and
            pd.notna(row['stoch_k']) and row['stoch_k'] < 40):
            pnl = backtest_entry(idx + 1)
            if pnl is not None:
                filter5_trades.append(pnl)

if len(filter5_trades) > 0:
    win_rate = len([p for p in filter5_trades if p > 0]) / len(filter5_trades) * 100
    avg_pnl = np.mean(filter5_trades)
    results.append({
        'filter': '복합 강화 (3지표)',
        'trades': len(filter5_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(filter5_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 결과 요약
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("필터 효과 요약")
print("=" * 70)
print()

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('win_rate', ascending=False)

print(f"{'필터':<25} {'거래수':<10} {'승률':<12} {'평균 PnL':<12} {'개선':<10}")
print("-" * 70)

for _, row in df_results.iterrows():
    improvement = row['win_rate'] - baseline_win_rate
    print(f"{row['filter']:<25} {row['trades']:<10} {row['win_rate']:>6.1f}%      {row['avg_pnl']:>+6.2f}%      {improvement:>+5.1f}%p")

print()

# 최고 성과
best = df_results.iloc[0]
improvement = best['win_rate'] - baseline_win_rate

print("=" * 70)
print("결론")
print("=" * 70)
print()

if improvement > 5:
    print(f"✅ 유효한 필터 발견: {best['filter']}")
    print(f"   승률: {baseline_win_rate:.1f}% → {best['win_rate']:.1f}% ({improvement:+.1f}%p)")
    print(f"   거래 수: {best['trades']}개")
    print(f"   평균 PnL: {best['avg_pnl']:+.2f}%")
elif improvement > 2:
    print(f"✓ 미세 개선: {best['filter']}")
    print(f"   승률: {baseline_win_rate:.1f}% → {best['win_rate']:.1f}% ({improvement:+.1f}%p)")
else:
    print("⚠️  필터 효과 미미")
    print(f"   최대 개선: {improvement:+.1f}%p")
    print("   → 추세선 돌파 그대로 사용이 최선")

print()
print("✅ 분석 완료")
