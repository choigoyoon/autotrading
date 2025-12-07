"""
나머지 캔들에서 거래 기회 찾기
======================================================================
추세선 돌파: 983개 (0.56%)
나머지: 174,217개 (99.44%) ← 여기서 뭘 할 수 있나?

전략 포트폴리오:
1. 추세 전환 (Trend Reversal) - 추세선 돌파
2. 레인지 바운스 (Range Bounce) - 지지/저항 반등
3. 평균 회귀 (Mean Reversion) - L값 반등
4. 변동성 돌파 (Volatility Breakout) - 큰 움직임 후
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("나머지 캔들 기회 분석")
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
print(f"총 캔들: {len(df):,}개")
print(f"추세선 돌파: {len(df_breakouts):,}개 ({len(df_breakouts)/len(df)*100:.2f}%)")
print(f"나머지: {len(df) - len(df_breakouts):,}개 ({(1-len(df_breakouts)/len(df))*100:.2f}%)\n")

# ═══════════════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# 가격 변동성
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_pct'] = df['atr'] / df['close'] * 100

# 볼륨
df['vol_ma'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma']

# 레인지 식별 (20봉 기준)
df['high_20'] = df['high'].rolling(20).max()
df['low_20'] = df['low'].rolling(20).min()
df['range_size'] = (df['high_20'] - df['low_20']) / df['low_20'] * 100

# 지지/저항 근접도
df['near_high'] = (df['high_20'] - df['close']) / df['high_20'] * 100  # 저항 근처
df['near_low'] = (df['close'] - df['low_20']) / df['low_20'] * 100   # 지지 근처

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 돌파 제외한 캔들들
# ═══════════════════════════════════════════════════════════════════

# 돌파 인덱스 찾기
breakout_indices = set()
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        breakout_indices.add(matching.index[0])

# 나머지 캔들
non_breakout_df = df[~df.index.isin(breakout_indices)].copy()

print(f"돌파 제외 캔들: {len(non_breakout_df):,}개\n")

# ═══════════════════════════════════════════════════════════════════
# L값 수집
# ═══════════════════════════════════════════════════════════════════

l_values = []
for i in range(1, len(df)):
    if df.iloc[i-1]['macd_hist'] < 0 and df.iloc[i]['macd_hist'] >= 0:
        l_values.append({
            'idx': i,
            'datetime': df.iloc[i]['datetime'],
            'price': df.iloc[i]['low']
        })

print(f"L값: {len(l_values):,}개\n")

# ═══════════════════════════════════════════════════════════════════
# 백테스트 함수
# ═══════════════════════════════════════════════════════════════════

def backtest_entry(entry_idx):
    if entry_idx >= len(df):
        return None, None

    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price * 1.02
    sl_price = entry_price * 0.98

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        c = df.iloc[j]
        if c['low'] <= sl_price:
            return -2.0, 'SL'
        elif c['high'] >= tp_price:
            return 2.0, 'TP'

    pnl = (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100
    return pnl, 'TIMEOUT'

# ═══════════════════════════════════════════════════════════════════
# 추가 전략 테스트
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("나머지 캔들 전략 테스트")
print("=" * 70)
print()

strategies = []

# ═══════════════════════════════════════════════════════════════════
# 전략 1: L값 강한 반등 (돌파 없어도)
# ═══════════════════════════════════════════════════════════════════

print("전략 1: L값 강한 반등 (1%+ 반등)")
l_bounce_trades = []

for l in l_values:
    l_idx = l['idx']
    l_price = l['price']

    # 돌파 신호에 포함된 경우 스킵
    if l_idx in breakout_indices:
        continue

    if l_idx + 50 >= len(df):
        continue

    # 1%+ 강한 반등 확인
    for i in range(1, 6):
        c = df.iloc[l_idx + i]
        bounce = (c['close'] - l_price) / l_price * 100

        if c['close'] > c['open'] and bounce >= 1.0:
            pnl, exit_type = backtest_entry(l_idx + i + 1)
            if pnl is not None:
                l_bounce_trades.append({
                    'pnl': pnl,
                    'exit': exit_type
                })
            break

if len(l_bounce_trades) > 0:
    win_rate = len([t for t in l_bounce_trades if t['pnl'] > 0]) / len(l_bounce_trades) * 100
    avg_pnl = np.mean([t['pnl'] for t in l_bounce_trades])
    strategies.append({
        'strategy': 'L값 강한 반등',
        'trades': len(l_bounce_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(l_bounce_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 전략 2: 레인지 하단 바운스
# ═══════════════════════════════════════════════════════════════════

print("전략 2: 레인지 하단 바운스")
print("  조건: 20봉 최저가 근처 (<2%) + MACD < 0")

range_trades = []

for idx in non_breakout_df.index:
    if idx < 50 or idx + 50 >= len(df):
        continue

    row = df.iloc[idx]

    # 레인지 하단 근처 + MACD 음수
    if (pd.notna(row['near_low']) and
        row['near_low'] < 2.0 and  # 최저가 2% 이내
        row['macd_hist'] < 0 and
        pd.notna(row['range_size']) and
        row['range_size'] < 10):  # 레인지가 10% 이내 (박스권)

        pnl, exit_type = backtest_entry(idx + 1)
        if pnl is not None:
            range_trades.append({
                'pnl': pnl,
                'exit': exit_type
            })

        if len(range_trades) >= 1000:  # 제한
            break

if len(range_trades) > 0:
    win_rate = len([t for t in range_trades if t['pnl'] > 0]) / len(range_trades) * 100
    avg_pnl = np.mean([t['pnl'] for t in range_trades])
    strategies.append({
        'strategy': '레인지 하단 바운스',
        'trades': len(range_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(range_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 전략 3: 변동성 돌파 (큰 하락 후 반등)
# ═══════════════════════════════════════════════════════════════════

print("전략 3: 변동성 돌파 (3봉 -2% 하락 후)")

vol_trades = []

for idx in non_breakout_df.index:
    if idx < 50 or idx + 50 >= len(df):
        continue

    # 최근 3봉 -2% 이상 하락
    price_3ago = df.iloc[idx - 3]['close']
    current_price = df.iloc[idx]['close']
    drop = (current_price - price_3ago) / price_3ago * 100

    if drop <= -2.0 and df.iloc[idx]['macd_hist'] < 0:
        # 반등 캔들 (양봉)
        if df.iloc[idx]['close'] > df.iloc[idx]['open']:
            pnl, exit_type = backtest_entry(idx + 1)
            if pnl is not None:
                vol_trades.append({
                    'pnl': pnl,
                    'exit': exit_type
                })

            if len(vol_trades) >= 1000:
                break

if len(vol_trades) > 0:
    win_rate = len([t for t in vol_trades if t['pnl'] > 0]) / len(vol_trades) * 100
    avg_pnl = np.mean([t['pnl'] for t in vol_trades])
    strategies.append({
        'strategy': '변동성 돌파',
        'trades': len(vol_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(vol_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 전략 4: 볼륨 급증 반전
# ═══════════════════════════════════════════════════════════════════

print("전략 4: 볼륨 급증 반전 (2배+ 볼륨)")

volume_trades = []

for idx in non_breakout_df.index:
    if idx < 50 or idx + 50 >= len(df):
        continue

    row = df.iloc[idx]

    # 볼륨 2배 이상 + 하락 중 + 양봉
    if (pd.notna(row['vol_ratio']) and
        row['vol_ratio'] >= 2.0 and
        row['macd_hist'] < 0 and
        row['close'] > row['open']):

        pnl, exit_type = backtest_entry(idx + 1)
        if pnl is not None:
            volume_trades.append({
                'pnl': pnl,
                'exit': exit_type
            })

        if len(volume_trades) >= 1000:
            break

if len(volume_trades) > 0:
    win_rate = len([t for t in volume_trades if t['pnl'] > 0]) / len(volume_trades) * 100
    avg_pnl = np.mean([t['pnl'] for t in volume_trades])
    strategies.append({
        'strategy': '볼륨 급증 반전',
        'trades': len(volume_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(volume_trades)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 결과 요약
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("전략 포트폴리오 요약")
print("=" * 70)
print()

# 추세선 돌파 추가
strategies.insert(0, {
    'strategy': '추세선 돌파',
    'trades': 983,
    'win_rate': 86.7,
    'avg_pnl': 1.26
})

df_strategies = pd.DataFrame(strategies)
df_strategies = df_strategies.sort_values('win_rate', ascending=False)

print(f"{'전략':<20} {'거래수':<10} {'승률':<12} {'평균 PnL':<12}")
print("-" * 70)
for _, row in df_strategies.iterrows():
    print(f"{row['strategy']:<20} {row['trades']:<10} {row['win_rate']:>6.1f}%      {row['avg_pnl']:>+6.2f}%")

print()

# 총합
total_trades = df_strategies['trades'].sum()
coverage = total_trades / len(df) * 100

print("=" * 70)
print("포트폴리오 효과")
print("=" * 70)
print()
print(f"총 전략: {len(df_strategies)}개")
print(f"총 거래: {total_trades:,}개")
print(f"커버리지: {coverage:.2f}% (vs 기존 0.56%)")
print(f"개선: {coverage / 0.56:.1f}배 증가")
print()

# 우수 전략 필터
good_strategies = df_strategies[
    (df_strategies['win_rate'] >= 70) &
    (df_strategies['avg_pnl'] >= 0.5)
]

if len(good_strategies) > 0:
    print("✅ 우수 전략:")
    for _, row in good_strategies.iterrows():
        print(f"  • {row['strategy']}: {row['trades']}개, {row['win_rate']:.1f}% 승률, {row['avg_pnl']:+.2f}% PnL")

    good_total = good_strategies['trades'].sum()
    print(f"\n  우수 전략 총 거래: {good_total:,}개")
    print(f"  커버리지: {good_total/len(df)*100:.2f}%")

print()
print("✅ 분석 완료")
