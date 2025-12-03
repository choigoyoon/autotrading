"""
나우캐스트 준수 전략 (Golden Cross 확정 후 진입)
======================================================================
올바른 로직:
1. Golden Cross 발생 (확정)
2. 이전 구간(Dead~Golden) 최저점 = L값 (사후 확정)
3. Golden Cross 다음 봉에서 진입
4. L값 대비 상승률 체크 (이미 너무 올랐으면 스킵)

미래데이터 없음:
- Golden Cross 발생 시점 = 현재
- L값은 과거 확정 데이터
- 진입은 현재 이후
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("나우캐스트 준수 전략 - Golden Cross 확정 후 진입")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════

df_15m = pd.read_csv('output_phase1_labeled.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

df_1h = pd.read_csv('btcusdt_1h_raw.csv')
df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])

# 5년 데이터
cutoff = df_15m['datetime'].max() - timedelta(days=1825)
df_15m = df_15m[df_15m['datetime'] >= cutoff].reset_index(drop=True)
df_1h = df_1h[df_1h['datetime'] >= cutoff].reset_index(drop=True)

print(f"15분 데이터: {len(df_15m):,}개")
print(f"1시간 데이터: {len(df_1h):,}개\n")

# ═══════════════════════════════════════════════════════════════════
# 1H MACD 계산
# ═══════════════════════════════════════════════════════════════════

print("1H MACD 계산 중...")

ema12 = df_1h['close'].ewm(span=12, adjust=False).mean()
ema26 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = ema12 - ema26
df_1h['macd_signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['macd_hist'] = df_1h['macd'] - df_1h['macd_signal']

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 15분 RSI 계산
# ═══════════════════════════════════════════════════════════════════

print("15분 RSI 계산 중...")

delta = df_15m['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df_15m['rsi'] = 100 - (100 / (1 + rs))

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# Dead Cross / Golden Cross 찾기 (나우캐스트 준수)
# ═══════════════════════════════════════════════════════════════════

print("1H MACD Cross 찾기...")

dead_crosses = []
golden_crosses = []

for i in range(1, len(df_1h)):
    prev = df_1h.iloc[i-1]
    curr = df_1h.iloc[i]

    # Dead Cross
    if prev['macd'] >= prev['macd_signal'] and curr['macd'] < curr['macd_signal']:
        dead_crosses.append({
            'idx': i,
            'datetime': curr['datetime'],
            'price': curr['close']
        })

    # Golden Cross
    if prev['macd'] <= prev['macd_signal'] and curr['macd'] > curr['macd_signal']:
        golden_crosses.append({
            'idx': i,
            'datetime': curr['datetime'],
            'price': curr['close']
        })

print(f"  Dead Cross: {len(dead_crosses)}개")
print(f"  Golden Cross: {len(golden_crosses)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 나우캐스트 준수 전략 백테스트
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("나우캐스트 백테스트 (Golden Cross 후 진입)")
print("=" * 80)
print()

SLIPPAGE = 0.05
FEE = 0.055
TOTAL_COST = (SLIPPAGE + FEE) * 2

TP_PCT = 2.0
SL_PCT = -1.5
MAX_HOLD = 50

NET_TP = TP_PCT - TOTAL_COST
NET_SL = SL_PCT - TOTAL_COST

print(f"TP: {TP_PCT}% (순: {NET_TP:.2f}%)")
print(f"SL: {SL_PCT}% (순: {NET_SL:.2f}%)")
print(f"비용: {TOTAL_COST:.2f}%\n")

trades = []

for gc_idx, golden in enumerate(golden_crosses):
    gc_time = golden['datetime']
    gc_1h_idx = golden['idx']

    # ─────────────────────────────────────────────────────────────
    # 이전 Dead Cross 찾기 (과거 확정 데이터)
    # ─────────────────────────────────────────────────────────────

    prev_dead = None
    for dead in reversed(dead_crosses):
        if dead['datetime'] < gc_time:
            prev_dead = dead
            break

    if prev_dead is None:
        continue

    dead_time = prev_dead['datetime']

    # ─────────────────────────────────────────────────────────────
    # L값 찾기: Dead ~ Golden 사이 최저점 (과거 확정 데이터)
    # ─────────────────────────────────────────────────────────────

    period_1h = df_1h[
        (df_1h['datetime'] >= dead_time) &
        (df_1h['datetime'] <= gc_time)
    ]

    if len(period_1h) == 0:
        continue

    l_row_1h = period_1h.loc[period_1h['low'].idxmin()]
    l_time = l_row_1h['datetime']
    l_price = l_row_1h['low']

    # 15분 데이터에서 정확한 L값
    period_15m = df_15m[
        (df_15m['datetime'] >= l_time) &
        (df_15m['datetime'] < l_time + timedelta(hours=1))
    ]

    if len(period_15m) == 0:
        continue

    l_row_15m = period_15m.loc[period_15m['low'].idxmin()]
    l_idx_15m = l_row_15m.name
    l_price_exact = l_row_15m['low']

    # ─────────────────────────────────────────────────────────────
    # 진입 시점: Golden Cross 다음 봉 (나우캐스트 준수)
    # ─────────────────────────────────────────────────────────────

    # Golden Cross 시점의 15분 인덱스 찾기
    gc_15m_idx = df_15m[df_15m['datetime'] >= gc_time].index[0] if len(df_15m[df_15m['datetime'] >= gc_time]) > 0 else None

    if gc_15m_idx is None or gc_15m_idx + MAX_HOLD >= len(df_15m):
        continue

    # 진입: Golden Cross 다음 봉 시가
    entry_idx = gc_15m_idx + 1
    entry_price = df_15m.iloc[entry_idx]['open']
    entry_time = df_15m.iloc[entry_idx]['datetime']

    # ─────────────────────────────────────────────────────────────
    # 진입 필터 (선택적)
    # ─────────────────────────────────────────────────────────────

    # L값 대비 이미 너무 올랐는지 체크
    gain_from_l = (entry_price - l_price_exact) / l_price_exact * 100

    # 필터 1: 필터 없음
    filter_none = True

    # 필터 2: L값 대비 3% 미만 상승
    filter_gain3 = gain_from_l < 3.0

    # 필터 3: L값 대비 5% 미만 상승
    filter_gain5 = gain_from_l < 5.0

    # 필터 4: 진입 시점 RSI < 50
    entry_candle = df_15m.iloc[entry_idx]
    filter_rsi50 = entry_candle['rsi'] < 50

    # 필터 5: L값 대비 3% 미만 + RSI < 50
    filter_gain3_rsi50 = filter_gain3 and filter_rsi50

    # ─────────────────────────────────────────────────────────────
    # 청산 시뮬레이션
    # ─────────────────────────────────────────────────────────────

    tp_level = entry_price * (1 + TP_PCT / 100)
    sl_level = entry_price * (1 + SL_PCT / 100)

    exit_bar = None
    exit_price = None
    exit_reason = None

    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, len(df_15m))):
        bar = df_15m.iloc[i]

        hit_tp = bar['high'] >= tp_level
        hit_sl = bar['low'] <= sl_level

        if hit_tp and hit_sl:
            if bar['open'] <= sl_level:
                exit_bar = i
                exit_price = sl_level
                exit_reason = 'SL'
                break
            elif bar['open'] >= tp_level:
                exit_bar = i
                exit_price = tp_level
                exit_reason = 'TP'
                break
            else:
                exit_bar = i
                exit_price = sl_level
                exit_reason = 'SL'
                break
        elif hit_sl:
            exit_bar = i
            exit_price = sl_level
            exit_reason = 'SL'
            break
        elif hit_tp:
            exit_bar = i
            exit_price = tp_level
            exit_reason = 'TP'
            break

    if exit_bar is None:
        exit_bar = min(entry_idx + MAX_HOLD, len(df_15m) - 1)
        exit_price = df_15m.iloc[exit_bar]['close']
        exit_reason = 'TIMEOUT'

    # ─────────────────────────────────────────────────────────────
    # 손익 계산
    # ─────────────────────────────────────────────────────────────

    gross_pnl = (exit_price - entry_price) / entry_price * 100
    net_pnl = gross_pnl - TOTAL_COST
    hold_bars = exit_bar - entry_idx

    trades.append({
        'dead_cross_time': dead_time,
        'l_time': l_row_15m['datetime'],
        'l_price': l_price_exact,
        'golden_cross_time': gc_time,
        'entry_time': entry_time,
        'entry_price': entry_price,
        'gain_from_l': gain_from_l,
        'exit_time': df_15m.iloc[exit_bar]['datetime'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_bars': hold_bars,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        # 필터
        'f_none': filter_none,
        'f_gain3': filter_gain3,
        'f_gain5': filter_gain5,
        'f_rsi50': filter_rsi50,
        'f_gain3_rsi50': filter_gain3_rsi50,
    })

df_trades = pd.DataFrame(trades)

print(f"총 거래 기회: {len(df_trades)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 성과 분석
# ═══════════════════════════════════════════════════════════════════

def analyze_strategy(df_trades, filter_col, name):
    filtered = df_trades[df_trades[filter_col]]

    if len(filtered) == 0:
        print(f"{name}: 거래 없음\n")
        return

    wins = (filtered['net_pnl'] > 0).sum()
    losses = (filtered['net_pnl'] <= 0).sum()
    win_rate = wins / len(filtered) * 100

    avg_pnl = filtered['net_pnl'].mean()
    total_pnl = filtered['net_pnl'].sum()

    avg_gain_from_l = filtered['gain_from_l'].mean()

    tp_count = (filtered['exit_reason'] == 'TP').sum()
    sl_count = (filtered['exit_reason'] == 'SL').sum()

    date_range = (df_15m['datetime'].max() - df_15m['datetime'].min()).days / 365.25
    trades_per_year = len(filtered) / date_range

    print(f"【{name}】")
    print(f"  거래: {len(filtered)}개 (연 {trades_per_year:.0f}개)")
    print(f"  승률: {win_rate:.2f}% (승 {wins} / 패 {losses})")
    print(f"  평균 손익: {avg_pnl:+.3f}%")
    print(f"  누적 손익: {total_pnl:+.2f}%")
    print(f"  L값 대비 평균 상승: {avg_gain_from_l:.2f}%")
    print(f"  청산: TP {tp_count}개 ({tp_count/len(filtered)*100:.0f}%) | SL {sl_count}개 ({sl_count/len(filtered)*100:.0f}%)")

    if avg_pnl > 0:
        print(f"  ✅ 수익 가능")
    else:
        print(f"  ❌ 손실")
    print()

print("=" * 80)
print("전략별 성과")
print("=" * 80)
print()

analyze_strategy(df_trades, 'f_none', '전략 1: 필터 없음')
analyze_strategy(df_trades, 'f_gain3', '전략 2: L값 대비 +3% 미만')
analyze_strategy(df_trades, 'f_gain5', '전략 3: L값 대비 +5% 미만')
analyze_strategy(df_trades, 'f_rsi50', '전략 4: 진입 RSI < 50')
analyze_strategy(df_trades, 'f_gain3_rsi50', '전략 5: L+3% 미만 & RSI<50')

# ═══════════════════════════════════════════════════════════════════
# 결론
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("결론")
print("=" * 80)
print()

print("✅ 나우캐스트 준수:")
print("  • Golden Cross 발생 (확정)")
print("  • L값은 과거 데이터 (Dead~Golden 사이 최저점)")
print("  • 진입은 Golden Cross 다음 봉")
print("  • 미래 정보 없음")
print()

if len(df_trades) > 0:
    best_filter = None
    best_pnl = -999

    for col in ['f_none', 'f_gain3', 'f_gain5', 'f_rsi50', 'f_gain3_rsi50']:
        filtered = df_trades[df_trades[col]]
        if len(filtered) > 10:
            avg = filtered['net_pnl'].mean()
            if avg > best_pnl:
                best_pnl = avg
                best_filter = col

    if best_pnl > 0:
        print(f"🏆 최고 전략: {best_filter} (평균 {best_pnl:+.3f}%)")
    else:
        print(f"⚠️  모든 전략 손실 또는 거래 부족")

print()
df_trades.to_csv('golden_cross_nowcast_trades.csv', index=False)
print("💾 거래 내역 저장: golden_cross_nowcast_trades.csv")
print()
print("=" * 80)
print("✅ 나우캐스트 백테스트 완료")
print("=" * 80)
