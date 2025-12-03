"""
실매매 가능 전략 - 15분 MACD 0-Cross
======================================================================
진입 신호:
- 15분 MACD Hist: 음수 → 양수 전환 (현재 봉에서 확인 가능)
- RSI < 30 (과매도 확인)
- 음수 구간 지속 > 10봉 (충분한 하락 확인)

청산:
- TP: +2.0%
- SL: -1.5%
- 최대: 50봉

미래데이터: 없음
실시간 가능: 예
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("실매매 가능 전략 - 15분 MACD 0-Cross")
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
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# MACD는 이미 있음 (macd_hist)

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 실시간 시뮬레이션 (나우캐스트 준수)
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("실매매 시뮬레이션")
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
last_entry_idx = -100  # 최소 거래 간격

for i in range(50, len(df) - MAX_HOLD):
    # ─────────────────────────────────────────────────────────────
    # 현재 시점 데이터 (실시간 가능)
    # ─────────────────────────────────────────────────────────────

    curr = df.iloc[i]
    prev = df.iloc[i-1]

    # 거래 간격
    if i - last_entry_idx < 10:
        continue

    # ─────────────────────────────────────────────────────────────
    # 진입 신호: MACD Hist 0-Cross (음 → 양)
    # ─────────────────────────────────────────────────────────────

    macd_cross = prev['macd_hist'] < 0 and curr['macd_hist'] >= 0

    if not macd_cross:
        continue

    # ─────────────────────────────────────────────────────────────
    # 추가 필터 (현재 시점 확인 가능)
    # ─────────────────────────────────────────────────────────────

    # 필터 1: 없음
    filter_none = True

    # 필터 2: RSI < 30
    filter_rsi30 = curr['rsi'] < 30

    # 필터 3: RSI < 35
    filter_rsi35 = curr['rsi'] < 35

    # 필터 4: 음수 구간 지속 (충분한 하락)
    neg_duration = 0
    for j in range(i-1, max(i-50, 0), -1):
        if df.iloc[j]['macd_hist'] < 0:
            neg_duration += 1
        else:
            break
    filter_duration10 = neg_duration >= 10
    filter_duration20 = neg_duration >= 20

    # 필터 5: RSI < 30 + 지속 10봉+
    filter_rsi30_dur10 = filter_rsi30 and filter_duration10

    # 필터 6: RSI < 35 + 지속 20봉+
    filter_rsi35_dur20 = filter_rsi35 and filter_duration20

    # 필터 7: MACD 깊이
    recent_macd = df.iloc[max(i-20, 0):i]['macd_hist']
    macd_min = recent_macd.min()
    filter_macd_deep = macd_min < -30

    # ─────────────────────────────────────────────────────────────
    # 진입 (다음 봉 시가)
    # ─────────────────────────────────────────────────────────────

    entry_idx = i + 1
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
        'signal_time': curr['datetime'],
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': df.iloc[exit_bar]['datetime'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_bars': hold_bars,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'rsi': curr['rsi'],
        'macd_hist': curr['macd_hist'],
        'neg_duration': neg_duration,
        'macd_min': macd_min,
        # 필터
        'f_none': filter_none,
        'f_rsi30': filter_rsi30,
        'f_rsi35': filter_rsi35,
        'f_duration10': filter_duration10,
        'f_duration20': filter_duration20,
        'f_rsi30_dur10': filter_rsi30_dur10,
        'f_rsi35_dur20': filter_rsi35_dur20,
        'f_macd_deep': filter_macd_deep,
    })

    last_entry_idx = i

df_trades = pd.DataFrame(trades)

print(f"총 거래: {len(df_trades)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 성과 분석
# ═══════════════════════════════════════════════════════════════════

def analyze(df_trades, filter_col, name):
    filtered = df_trades[df_trades[filter_col]]

    if len(filtered) == 0:
        print(f"{name}: 거래 없음\n")
        return

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

    # MDD 계산
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

print("=" * 80)
print("전략별 성과")
print("=" * 80)
print()

analyze(df_trades, 'f_none', '전략 1: 필터 없음 (모든 MACD Cross)')
analyze(df_trades, 'f_rsi30', '전략 2: RSI < 30')
analyze(df_trades, 'f_rsi35', '전략 3: RSI < 35')
analyze(df_trades, 'f_duration10', '전략 4: 음수 지속 10봉+')
analyze(df_trades, 'f_duration20', '전략 5: 음수 지속 20봉+')
analyze(df_trades, 'f_rsi30_dur10', '전략 6: RSI<30 + 지속10봉+')
analyze(df_trades, 'f_rsi35_dur20', '전략 7: RSI<35 + 지속20봉+')
analyze(df_trades, 'f_macd_deep', '전략 8: MACD 최소 < -30')

# ═══════════════════════════════════════════════════════════════════
# 결론
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("결론")
print("=" * 80)
print()

print("✅ 실매매 가능:")
print("  • 15분 MACD Hist 0-Cross 감지 (실시간)")
print("  • RSI, 지속 기간 등 현재 데이터로 필터")
print("  • 다음 봉 시가 진입")
print("  • 미래 정보 불필요")
print()

if len(df_trades) > 0:
    best = None
    best_score = -999

    for col in ['f_none', 'f_rsi30', 'f_rsi35', 'f_duration10', 'f_duration20',
                'f_rsi30_dur10', 'f_rsi35_dur20', 'f_macd_deep']:
        filtered = df_trades[df_trades[col]]
        if len(filtered) > 20:
            avg = filtered['net_pnl'].mean()
            wr = (filtered['net_pnl'] > 0).sum() / len(filtered) * 100
            score = avg * (wr / 50)  # 평균 손익 * 승률 가중
            if score > best_score:
                best_score = score
                best = col

    if best:
        best_data = df_trades[df_trades[best]]
        best_avg = best_data['net_pnl'].mean()
        best_wr = (best_data['net_pnl'] > 0).sum() / len(best_data) * 100
        print(f"🏆 최고 전략: {best}")
        print(f"   평균: {best_avg:+.3f}% | 승률: {best_wr:.1f}%")

print()
df_trades.to_csv('macd_cross_live_ready_trades.csv', index=False)
print("💾 거래 내역 저장: macd_cross_live_ready_trades.csv")
print()
print("=" * 80)
print("✅ 실매매 가능 전략 백테스트 완료")
print("=" * 80)
