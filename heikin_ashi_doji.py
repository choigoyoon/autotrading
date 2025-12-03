"""
하이킨아시 도지 패턴 전략
======================================================================
전략:
1. 하이킨아시 차트 변환
2. 도지 캔들 찾기 (몸통 작음, open ≈ close)
3. 도지 후 다음 3봉 관찰
4. 3봉이 상승 방향 → 진입
5. TP 2% / SL -1.5%

하이킨아시 계산:
- HA_Close = (Open + High + Low + Close) / 4
- HA_Open = (HA_Open[prev] + HA_Close[prev]) / 2
- HA_High = Max(High, HA_Open, HA_Close)
- HA_Low = Min(Low, HA_Open, HA_Close)

도지 정의:
- |HA_Close - HA_Open| < (HA_High - HA_Low) * 0.1  # 몸통이 전체 범위의 10% 미만
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("하이킨아시 도지 패턴 전략")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════

df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"15분 데이터: {len(df):,}개 (5년)\n")

# ═══════════════════════════════════════════════════════════════════
# 하이킨아시 변환
# ═══════════════════════════════════════════════════════════════════

print("하이킨아시 변환 중...")

df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['ha_open'] = np.nan
df['ha_high'] = np.nan
df['ha_low'] = np.nan

# 첫 번째 HA_Open
df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

for i in range(1, len(df)):
    # HA_Open = (이전 HA_Open + 이전 HA_Close) / 2
    df.loc[i, 'ha_open'] = (df.loc[i-1, 'ha_open'] + df.loc[i-1, 'ha_close']) / 2

    # HA_High = Max(High, HA_Open, HA_Close)
    df.loc[i, 'ha_high'] = max(df.loc[i, 'high'], df.loc[i, 'ha_open'], df.loc[i, 'ha_close'])

    # HA_Low = Min(Low, HA_Open, HA_Close)
    df.loc[i, 'ha_low'] = min(df.loc[i, 'low'], df.loc[i, 'ha_open'], df.loc[i, 'ha_close'])

# 첫 번째 HA_High, HA_Low
df.loc[0, 'ha_high'] = df.loc[0, 'high']
df.loc[0, 'ha_low'] = df.loc[0, 'low']

# 하이킨아시 몸통 크기
df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
df['ha_range'] = df['ha_high'] - df['ha_low']
df['ha_body_pct'] = df['ha_body'] / df['ha_range']

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# RSI 계산 (필터용)
# ═══════════════════════════════════════════════════════════════════

print("RSI 계산 중...")

delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 전략 백테스트
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("도지 패턴 백테스트")
print("=" * 80)
print()

SLIPPAGE = 0.05
FEE = 0.055
TOTAL_COST = (SLIPPAGE + FEE) * 2

TP_PCT = 2.0
SL_PCT = -1.5
MAX_HOLD = 50

print(f"TP: {TP_PCT}% (순: {TP_PCT - TOTAL_COST:.2f}%)")
print(f"SL: {SL_PCT}% (순: {SL_PCT - TOTAL_COST:.2f}%)")
print(f"비용: {TOTAL_COST:.2f}%\n")

trades = []
doji_count = 0

for i in range(50, len(df) - MAX_HOLD - 4):
    curr = df.iloc[i]

    # ─────────────────────────────────────────────────────────────
    # 도지 캔들 확인
    # ─────────────────────────────────────────────────────────────

    # 도지 정의: 몸통이 전체 범위의 10% 미만
    is_doji = curr['ha_body_pct'] < 0.1 and curr['ha_range'] > 0

    if not is_doji:
        continue

    doji_count += 1

    # ─────────────────────────────────────────────────────────────
    # 다음 3봉 관찰 (방향 예측)
    # ─────────────────────────────────────────────────────────────

    next_3 = df.iloc[i+1:i+4]

    if len(next_3) < 3:
        continue

    # 3봉의 방향 확인
    bullish_count = 0
    bearish_count = 0

    for _, bar in next_3.iterrows():
        if bar['ha_close'] > bar['ha_open']:
            bullish_count += 1
        else:
            bearish_count += 1

    # ─────────────────────────────────────────────────────────────
    # 진입 필터
    # ─────────────────────────────────────────────────────────────

    # 필터 1: 3봉 모두 상승
    filter_all_bullish = bullish_count == 3

    # 필터 2: 3봉 중 2개 이상 상승
    filter_2of3_bullish = bullish_count >= 2

    # 필터 3: 3봉 모두 상승 + RSI < 40
    filter_bullish_rsi40 = filter_all_bullish and curr['rsi'] < 40

    # 필터 4: 2개 이상 상승 + RSI < 40
    filter_2of3_rsi40 = filter_2of3_bullish and curr['rsi'] < 40

    # 필터 5: 3봉 모두 상승 + RSI < 50
    filter_bullish_rsi50 = filter_all_bullish and curr['rsi'] < 50

    # 필터 6: 3봉의 HA_Close 평균이 도지보다 0.5% 이상 상승
    avg_close_3 = next_3['ha_close'].mean()
    price_gain = (avg_close_3 - curr['ha_close']) / curr['ha_close'] * 100
    filter_gain05 = filter_all_bullish and price_gain > 0.5

    # ─────────────────────────────────────────────────────────────
    # 진입 (4번째 봉 시가 = 도지 후 3봉 확인 후)
    # ─────────────────────────────────────────────────────────────

    entry_idx = i + 4  # 도지(i) → 3봉 관찰(i+1,2,3) → 진입(i+4)
    entry_price = df.iloc[entry_idx]['open']
    entry_time = df.iloc[entry_idx]['datetime']

    tp_level = entry_price * (1 + TP_PCT / 100)
    sl_level = entry_price * (1 + SL_PCT / 100)

    # ─────────────────────────────────────────────────────────────
    # 청산 시뮬레이션
    # ─────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────
    # 손익 계산
    # ─────────────────────────────────────────────────────────────

    gross_pnl = (exit_price - entry_price) / entry_price * 100
    net_pnl = gross_pnl - TOTAL_COST
    hold_bars = exit_bar - entry_idx

    trades.append({
        'doji_time': curr['datetime'],
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': df.iloc[exit_bar]['datetime'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_bars': hold_bars,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'bullish_count': bullish_count,
        'price_gain': price_gain,
        'doji_rsi': curr['rsi'],
        # 필터
        'f_all_bullish': filter_all_bullish,
        'f_2of3_bullish': filter_2of3_bullish,
        'f_bullish_rsi40': filter_bullish_rsi40,
        'f_2of3_rsi40': filter_2of3_rsi40,
        'f_bullish_rsi50': filter_bullish_rsi50,
        'f_gain05': filter_gain05,
    })

df_trades = pd.DataFrame(trades)

print(f"도지 캔들 발견: {doji_count}개")
print(f"거래 기회: {len(df_trades)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 성과 분석
# ═══════════════════════════════════════════════════════════════════

def analyze(df_trades, filter_col, name):
    filtered = df_trades[df_trades[filter_col]]

    if len(filtered) == 0:
        print(f"{name}: 거래 없음\n")
        return None

    wins = (filtered['net_pnl'] > 0).sum()
    losses = (filtered['net_pnl'] <= 0).sum()
    win_rate = wins / len(filtered) * 100

    avg_pnl = filtered['net_pnl'].mean()
    total_pnl = filtered['net_pnl'].sum()

    avg_win = filtered[filtered['net_pnl'] > 0]['net_pnl'].mean() if wins > 0 else 0
    avg_loss = filtered[filtered['net_pnl'] <= 0]['net_pnl'].mean() if losses > 0 else 0

    tp_count = (filtered['exit_reason'] == 'TP').sum()
    sl_count = (filtered['exit_reason'] == 'SL').sum()

    date_range = (df['datetime'].max() - df['datetime'].min()).days / 365.25
    per_year = len(filtered) / date_range

    # MDD
    filtered_sorted = filtered.sort_values('entry_time').reset_index(drop=True)
    filtered_sorted['cumulative'] = filtered_sorted['net_pnl'].cumsum()
    filtered_sorted['cummax'] = filtered_sorted['cumulative'].cummax()
    filtered_sorted['dd'] = filtered_sorted['cumulative'] - filtered_sorted['cummax']
    mdd = filtered_sorted['dd'].min()

    print(f"【{name}】")
    print(f"  거래: {len(filtered)}개 (연 {per_year:.0f}개)")
    print(f"  승률: {win_rate:.2f}% (승 {wins} / 패 {losses})")
    print(f"  평균 손익: {avg_pnl:+.3f}%")
    print(f"  평균 승: {avg_win:+.3f}% | 평균 패: {avg_loss:+.3f}%")
    print(f"  누적 손익: {total_pnl:+.2f}%")
    print(f"  MDD: {mdd:+.2f}%")
    print(f"  청산: TP {tp_count}개 ({tp_count/len(filtered)*100:.0f}%) | SL {sl_count}개 ({sl_count/len(filtered)*100:.0f}%)")

    if avg_pnl > 0 and win_rate > 48:
        print(f"  ✅ 수익 가능")
    else:
        print(f"  ❌ 손실 또는 승률 부족")
    print()

    return {'name': name, 'trades': len(filtered), 'win_rate': win_rate,
            'avg_pnl': avg_pnl, 'total_pnl': total_pnl, 'mdd': mdd}

print("=" * 80)
print("전략별 성과")
print("=" * 80)
print()

results = []
results.append(analyze(df_trades, 'f_all_bullish', '전략 1: 3봉 모두 상승'))
results.append(analyze(df_trades, 'f_2of3_bullish', '전략 2: 3봉 중 2개 이상 상승'))
results.append(analyze(df_trades, 'f_bullish_rsi40', '전략 3: 3봉 상승 + RSI<40'))
results.append(analyze(df_trades, 'f_2of3_rsi40', '전략 4: 2개 이상 상승 + RSI<40'))
results.append(analyze(df_trades, 'f_bullish_rsi50', '전략 5: 3봉 상승 + RSI<50'))
results.append(analyze(df_trades, 'f_gain05', '전략 6: 3봉 상승 + 0.5%+ 상승'))

# ═══════════════════════════════════════════════════════════════════
# 결론
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("결론")
print("=" * 80)
print()

print("✅ 하이킨아시 도지 패턴:")
print("  • 도지 캔들 = 추세 전환 가능성")
print("  • 다음 3봉으로 방향 확인")
print("  • 상승 확인 후 진입")
print()

results = [r for r in results if r is not None]
if results:
    best = max(results, key=lambda x: x['avg_pnl'] if x['win_rate'] > 48 else -999)
    if best['avg_pnl'] > 0:
        print(f"🏆 최고 전략: {best['name']}")
        print(f"   승률: {best['win_rate']:.1f}% | 평균: {best['avg_pnl']:+.3f}%")
        print(f"   5년 수익: {best['total_pnl']:+.2f}% | MDD: {best['mdd']:+.2f}%")

print()
df_trades.to_csv('heikin_ashi_doji_trades.csv', index=False)
print("💾 거래 내역 저장: heikin_ashi_doji_trades.csv")
print()
print("=" * 80)
print("✅ 하이킨아시 도지 백테스트 완료")
print("=" * 80)
