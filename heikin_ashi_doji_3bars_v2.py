"""
하이킨아시 도지 + 방향성 확인 (3봉)
======================================================================
전략:
1. 도지 캔들 발견
2. 다음 3봉의 방향성 확인
   - 다음봉 (1번)
   - 다다음봉 (2번)
   - 다다다음봉 (3번)
3. 3봉의 방향이 일치하면 → 진입
4. TP 2% / SL -1.5%
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("하이킨아시 도지 + 3봉 방향성")
print("=" * 80)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"데이터: {len(df):,}개\n")

# 하이킨아시
print("하이킨아시 변환...")
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
print("RSI...")
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))
print("  완료!\n")

# 설정
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
print(f"TP: {TP_PCT}% | SL: {SL_PCT}% | 비용: {TOTAL_COST:.2f}%\n")

trades = []
doji_count = 0

for i in range(50, len(df) - MAX_HOLD - 4):
    curr = df.iloc[i]

    # 도지
    is_doji = curr['ha_body_pct'] < 0.1 and curr['ha_range'] > 0
    if not is_doji:
        continue

    doji_count += 1

    # 다음 3봉
    next1 = df.iloc[i+1]  # 다음봉
    next2 = df.iloc[i+2]  # 다다음봉
    next3 = df.iloc[i+3]  # 다다다음봉

    # 방향 확인
    dir1 = 1 if next1['ha_close'] > next1['ha_open'] else -1
    dir2 = 1 if next2['ha_close'] > next2['ha_open'] else -1
    dir3 = 1 if next3['ha_close'] > next3['ha_open'] else -1

    # 필터
    filter_all3_up = (dir1 == 1) and (dir2 == 1) and (dir3 == 1)  # 3개 모두 상승
    filter_2of3_up = (dir1 + dir2 + dir3) >= 2  # 3개 중 2개 이상 상승
    filter_majority_up = (dir1 + dir2 + dir3) > 0  # 과반 상승

    # 3봉 평균 상승률
    avg_close = (next1['ha_close'] + next2['ha_close'] + next3['ha_close']) / 3
    gain_3bars = (avg_close - curr['ha_close']) / curr['ha_close'] * 100

    filter_all3_gain05 = filter_all3_up and gain_3bars > 0.5
    filter_all3_gain10 = filter_all3_up and gain_3bars > 1.0

    # RSI 필터
    filter_all3_rsi40 = filter_all3_up and curr['rsi'] < 40
    filter_all3_rsi50 = filter_all3_up and curr['rsi'] < 50

    # 연속성 (1→2→3 순서대로 높아지는지)
    sequential_up = (next1['ha_close'] < next2['ha_close']) and (next2['ha_close'] < next3['ha_close'])
    filter_sequential = filter_all3_up and sequential_up

    if not filter_majority_up:  # 최소한 과반은 상승이어야 거래 기록
        continue

    # 진입 (4번째 봉)
    entry_idx = i + 4
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
                exit_bar, exit_price, exit_reason = j, sl_level, 'SL'
                break
            elif bar['open'] >= tp_level:
                exit_bar, exit_price, exit_reason = j, tp_level, 'TP'
                break
            else:
                exit_bar, exit_price, exit_reason = j, sl_level, 'SL'
                break
        elif hit_sl:
            exit_bar, exit_price, exit_reason = j, sl_level, 'SL'
            break
        elif hit_tp:
            exit_bar, exit_price, exit_reason = j, tp_level, 'TP'
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
        'dir1': dir1,
        'dir2': dir2,
        'dir3': dir3,
        'gain_3bars': gain_3bars,
        'rsi': curr['rsi'],
        'f_all3_up': filter_all3_up,
        'f_2of3_up': filter_2of3_up,
        'f_majority_up': filter_majority_up,
        'f_all3_gain05': filter_all3_gain05,
        'f_all3_gain10': filter_all3_gain10,
        'f_all3_rsi40': filter_all3_rsi40,
        'f_all3_rsi50': filter_all3_rsi50,
        'f_sequential': filter_sequential,
    })

df_trades = pd.DataFrame(trades)

print(f"도지: {doji_count}개")
print(f"과반 상승: {len(df_trades)}개\n")

# 분석
def analyze(df_trades, filter_col, name):
    filtered = df_trades[df_trades[filter_col]]
    if len(filtered) == 0:
        print(f"{name}: 없음\n")
        return None

    wins = (filtered['net_pnl'] > 0).sum()
    win_rate = wins / len(filtered) * 100
    avg_pnl = filtered['net_pnl'].mean()
    total_pnl = filtered['net_pnl'].sum()

    per_year = len(filtered) / 5.0

    # MDD
    sorted_df = filtered.sort_values('entry_time').reset_index(drop=True)
    sorted_df['cum'] = sorted_df['net_pnl'].cumsum()
    sorted_df['cummax'] = sorted_df['cum'].cummax()
    sorted_df['dd'] = sorted_df['cum'] - sorted_df['cummax']
    mdd = sorted_df['dd'].min()

    print(f"【{name}】")
    print(f"  거래: {len(filtered)}개 (연 {per_year:.0f}개)")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  평균: {avg_pnl:+.3f}%")
    print(f"  5년: {total_pnl:+.2f}%")
    print(f"  MDD: {mdd:+.2f}%")

    if avg_pnl > 0 and win_rate > 48:
        print(f"  ✅ 수익")
    else:
        print(f"  ❌ 손실")
    print()

    return {'name': name, 'wr': win_rate, 'avg': avg_pnl, 'total': total_pnl, 'trades': len(filtered)}

print("=" * 80)
print("성과")
print("=" * 80)
print()

r = []
r.append(analyze(df_trades, 'f_majority_up', '1. 과반 상승 (2개 이상)'))
r.append(analyze(df_trades, 'f_2of3_up', '2. 2개 이상 상승'))
r.append(analyze(df_trades, 'f_all3_up', '3. 3개 모두 상승'))
r.append(analyze(df_trades, 'f_all3_gain05', '4. 3개 상승 + 평균 0.5%+'))
r.append(analyze(df_trades, 'f_all3_gain10', '5. 3개 상승 + 평균 1.0%+'))
r.append(analyze(df_trades, 'f_all3_rsi40', '6. 3개 상승 + RSI<40'))
r.append(analyze(df_trades, 'f_all3_rsi50', '7. 3개 상승 + RSI<50'))
r.append(analyze(df_trades, 'f_sequential', '8. 3개 상승 + 순차 증가'))

r = [x for x in r if x is not None]
if r:
    best = max(r, key=lambda x: x['avg'] if x['wr'] > 48 else -999)
    if best['avg'] > 0:
        print(f"🏆 최고: {best['name']}")
        print(f"   승률 {best['wr']:.1f}% | 평균 {best['avg']:+.3f}% | 5년 {best['total']:+.2f}%")

print()
df_trades.to_csv('ha_doji_3bars_direction.csv', index=False)
print("💾 저장: ha_doji_3bars_direction.csv")
print()
print("=" * 80)
print("✅ 완료")
print("=" * 80)
