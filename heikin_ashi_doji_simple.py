"""
하이킨아시 도지 + 다음 1봉 확인 (단순 버전)
======================================================================
전략:
1. 하이킨아시 도지 발견
2. 다음 1봉만 확인
3. 다음 봉이 상승 → 진입
4. TP 2% / SL -1.5%

훨씬 단순하고 빠른 반응
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("하이킨아시 도지 + 다음 1봉 (단순)")
print("=" * 80)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"15분 데이터: {len(df):,}개\n")

# 하이킨아시 변환
print("하이킨아시 변환 중...")

df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['ha_open'] = np.nan

df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

for i in range(1, len(df)):
    df.loc[i, 'ha_open'] = (df.loc[i-1, 'ha_open'] + df.loc[i-1, 'ha_close']) / 2

df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
df['ha_range'] = df['ha_high'] - df['ha_low']
df['ha_body_pct'] = df['ha_body'] / df['ha_range']

print("  완료!\n")

# RSI
print("RSI 계산 중...")
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))
print("  완료!\n")

# 전략 설정
SLIPPAGE = 0.05
FEE = 0.055
TOTAL_COST = (SLIPPAGE + FEE) * 2
TP_PCT = 2.0
SL_PCT = -1.5
MAX_HOLD = 50

print("=" * 80)
print("백테스트")
print("=" * 80)
print()
print(f"TP: {TP_PCT}% (순: {TP_PCT - TOTAL_COST:.2f}%)")
print(f"SL: {SL_PCT}% (순: {SL_PCT - TOTAL_COST:.2f}%)\n")

trades = []
doji_count = 0

for i in range(50, len(df) - MAX_HOLD - 2):
    curr = df.iloc[i]

    # 도지 확인
    is_doji = curr['ha_body_pct'] < 0.1 and curr['ha_range'] > 0
    if not is_doji:
        continue

    doji_count += 1

    # 다음 1봉만 확인
    next_bar = df.iloc[i+1]
    is_bullish = next_bar['ha_close'] > next_bar['ha_open']

    if not is_bullish:
        continue

    # 필터
    filter_none = True
    filter_rsi30 = curr['rsi'] < 30
    filter_rsi40 = curr['rsi'] < 40
    filter_rsi50 = curr['rsi'] < 50

    # 다음 봉 상승폭
    next_gain = (next_bar['ha_close'] - curr['ha_close']) / curr['ha_close'] * 100
    filter_gain05 = next_gain > 0.5
    filter_gain10 = next_gain > 1.0

    # 진입 (도지 후 2번째 봉 시가)
    entry_idx = i + 2
    entry_price = df.iloc[entry_idx]['open']
    entry_time = df.iloc[entry_idx]['datetime']

    tp_level = entry_price * (1 + TP_PCT / 100)
    sl_level = entry_price * (1 + SL_PCT / 100)

    # 청산
    exit_bar = None
    exit_price = None
    exit_reason = None

    for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, len(df))):
        bar = df.iloc[j]
        hit_tp = bar['high'] >= tp_level
        hit_sl = bar['low'] <= sl_level

        if hit_tp and hit_sl:
            if bar['open'] <= sl_level:
                exit_bar = j
                exit_price = sl_level
                exit_reason = 'SL'
                break
            elif bar['open'] >= tp_level:
                exit_bar = j
                exit_price = tp_level
                exit_reason = 'TP'
                break
            else:
                exit_bar = j
                exit_price = sl_level
                exit_reason = 'SL'
                break
        elif hit_sl:
            exit_bar = j
            exit_price = sl_level
            exit_reason = 'SL'
            break
        elif hit_tp:
            exit_bar = j
            exit_price = tp_level
            exit_reason = 'TP'
            break

    if exit_bar is None:
        exit_bar = min(entry_idx + MAX_HOLD, len(df) - 1)
        exit_price = df.iloc[exit_bar]['close']
        exit_reason = 'TIMEOUT'

    gross_pnl = (exit_price - entry_price) / entry_price * 100
    net_pnl = gross_pnl - TOTAL_COST

    trades.append({
        'doji_time': curr['datetime'],
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': df.iloc[exit_bar]['datetime'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'next_gain': next_gain,
        'doji_rsi': curr['rsi'],
        'f_none': filter_none,
        'f_rsi30': filter_rsi30,
        'f_rsi40': filter_rsi40,
        'f_rsi50': filter_rsi50,
        'f_gain05': filter_gain05,
        'f_gain10': filter_gain10,
    })

df_trades = pd.DataFrame(trades)

print(f"도지 발견: {doji_count}개")
print(f"상승 확인: {len(df_trades)}개\n")

# 분석
def analyze(df_trades, filter_col, name):
    filtered = df_trades[df_trades[filter_col]]
    if len(filtered) == 0:
        print(f"{name}: 거래 없음\n")
        return None

    wins = (filtered['net_pnl'] > 0).sum()
    win_rate = wins / len(filtered) * 100
    avg_pnl = filtered['net_pnl'].mean()
    total_pnl = filtered['net_pnl'].sum()

    tp_count = (filtered['exit_reason'] == 'TP').sum()
    sl_count = (filtered['exit_reason'] == 'SL').sum()

    date_range = 5.0
    per_year = len(filtered) / date_range

    # MDD
    filtered_sorted = filtered.sort_values('entry_time').reset_index(drop=True)
    filtered_sorted['cumulative'] = filtered_sorted['net_pnl'].cumsum()
    filtered_sorted['cummax'] = filtered_sorted['cumulative'].cummax()
    filtered_sorted['dd'] = filtered_sorted['cumulative'] - filtered_sorted['cummax']
    mdd = filtered_sorted['dd'].min()

    print(f"【{name}】")
    print(f"  거래: {len(filtered)}개 (연 {per_year:.0f}개)")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  평균: {avg_pnl:+.3f}%")
    print(f"  5년 누적: {total_pnl:+.2f}%")
    print(f"  MDD: {mdd:+.2f}%")
    print(f"  TP/SL: {tp_count}개 / {sl_count}개")

    if avg_pnl > 0 and win_rate > 48:
        print(f"  ✅ 수익")
    else:
        print(f"  ❌ 손실")
    print()

    return {'name': name, 'win_rate': win_rate, 'avg_pnl': avg_pnl, 'total': total_pnl}

print("=" * 80)
print("성과")
print("=" * 80)
print()

results = []
results.append(analyze(df_trades, 'f_none', '1. 필터 없음'))
results.append(analyze(df_trades, 'f_rsi30', '2. RSI<30'))
results.append(analyze(df_trades, 'f_rsi40', '3. RSI<40'))
results.append(analyze(df_trades, 'f_rsi50', '4. RSI<50'))
results.append(analyze(df_trades, 'f_gain05', '5. 다음봉 +0.5%+'))
results.append(analyze(df_trades, 'f_gain10', '6. 다음봉 +1.0%+'))

results = [r for r in results if r is not None]
if results:
    best = max(results, key=lambda x: x['avg_pnl'] if x['win_rate'] > 48 else -999)
    if best['avg_pnl'] > 0:
        print(f"🏆 최고: {best['name']} (승률 {best['win_rate']:.1f}%, 평균 {best['avg_pnl']:+.3f}%)")

print()
df_trades.to_csv('ha_doji_next1_trades.csv', index=False)
print("💾 저장: ha_doji_next1_trades.csv")
print()
print("=" * 80)
print("✅ 완료")
print("=" * 80)
