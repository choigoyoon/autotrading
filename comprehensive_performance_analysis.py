"""
필터 적용 후 종합 성과 분석

1. 기본 성과 (매매횟수, 승률, 수익률)
2. MDD (Maximum Drawdown)
3. FVG 유무별 성과
4. 동적 TP 적용 시뮬레이션
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("필터 적용 후 종합 성과 분석")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

breakouts = pd.read_csv('output_phase4_breakouts.csv')
breakouts['datetime'] = breakouts['break_idx'].apply(lambda x: df.iloc[x]['datetime'] if x < len(df) else None)
breakouts = breakouts.dropna(subset=['datetime'])

# 필터링된 돌파 (10봉 간격)
filtered_breakouts = []
last_break_idx = -999

for idx, row in breakouts.iterrows():
    break_idx = row['break_idx']
    if break_idx - last_break_idx >= 10:
        filtered_breakouts.append(row)
        last_break_idx = break_idx

filtered_df = pd.DataFrame(filtered_breakouts)

# Trendline up만 (롱 전략)
trendline_up = filtered_df[filtered_df['type'] == 'trendline_up'].copy()

print(f"\n필터링 후 Trendline Up: {len(trendline_up):,}개")

# ═══════════════════════════════════════════════════════════
# 1. 기본 성과 (TP 2%, SL 2%)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1. 기본 성과 (TP 2%, SL 2%)")
print("=" * 70)

def backtest_strategy(signals_df, df, tp_pct=2.0, sl_pct=2.0, position_size=0.3, leverage=3):
    """백테스트 실행"""
    trades = []

    for idx, signal in signals_df.iterrows():
        break_idx = signal['break_idx']
        entry_price = signal['break_price']

        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

        # 향후 50봉 스캔
        max_idx = min(break_idx + 50, len(df) - 1)
        future = df.iloc[break_idx:max_idx+1]

        hit_tp = False
        hit_sl = False
        exit_idx = None
        exit_price = None

        for i in range(1, len(future)):
            candle = future.iloc[i]

            if candle['high'] >= tp_price:
                hit_tp = True
                exit_idx = break_idx + i
                exit_price = tp_price
                break

            if candle['low'] <= sl_price:
                hit_sl = True
                exit_idx = break_idx + i
                exit_price = sl_price
                break

        if exit_idx is None:
            exit_idx = max_idx
            exit_price = future.iloc[-1]['close']

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_idx': break_idx,
            'entry_price': entry_price,
            'exit_idx': exit_idx,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'datetime': signal['datetime'],
        })

    return pd.DataFrame(trades)

# 백테스트 실행
trades_df = backtest_strategy(trendline_up, df, tp_pct=2.0, sl_pct=2.0)

# 통계
win_trades = trades_df[trades_df['pnl_pct'] > 0]
loss_trades = trades_df[trades_df['pnl_pct'] < 0]

win_rate = len(win_trades) / len(trades_df) * 100
avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
avg_loss = loss_trades['pnl_pct'].mean() if len(loss_trades) > 0 else 0
avg_pnl = trades_df['pnl_pct'].mean()

# 수수료 적용
fee_pct = 0.11
avg_pnl_after_fee = avg_pnl - fee_pct

print(f"\n총 거래: {len(trades_df):,}개")
print(f"승리: {len(win_trades):,}개 | 손실: {len(loss_trades):,}개")
print(f"\n승률: {win_rate:.2f}%")
print(f"평균 승리: +{avg_win:.2f}%")
print(f"평균 손실: {avg_loss:.2f}%")
print(f"평균 PnL: {avg_pnl:.3f}%")
print(f"수수료 후: {avg_pnl_after_fee:.3f}%")

# 누적 수익률 (복리)
position_size = 0.3
leverage = 3

print(f"\n레버리지 설정: {leverage}배, 포지션: {position_size*100:.0f}%")

cumulative_return = 1.0
for pnl in trades_df['pnl_pct']:
    # 실제 수익 = (PnL - 수수료) × 레버리지 × 포지션
    actual_pnl = (pnl - fee_pct) * leverage * position_size / 100
    cumulative_return *= (1 + actual_pnl)

total_return_pct = (cumulative_return - 1) * 100

print(f"총 수익률 (복리): {total_return_pct:.1f}%")
print(f"연간 수익률: {total_return_pct / 5:.1f}% (5년 기준)")

# 월간 통계
monthly_trades = len(trades_df) / 60  # 5년 = 60개월
monthly_return = total_return_pct / 60

print(f"\n월간 거래: {monthly_trades:.1f}회")
print(f"월간 수익: {monthly_return:.2f}%")

# ═══════════════════════════════════════════════════════════
# 2. MDD (Maximum Drawdown)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2. MDD (Maximum Drawdown)")
print("=" * 70)

# 복리 잔고 계산
balance = 1000  # $1,000 시작
balances = [balance]

for pnl in trades_df['pnl_pct']:
    actual_pnl = (pnl - fee_pct) * leverage * position_size / 100
    balance *= (1 + actual_pnl)
    balances.append(balance)

balance_series = pd.Series(balances)

# MDD 계산
peak = balance_series.expanding(min_periods=1).max()
drawdown = (balance_series - peak) / peak * 100
max_drawdown = drawdown.min()

print(f"\nMDD: {max_drawdown:.2f}%")
print(f"최종 잔고: ${balance:.2f}")
print(f"초기 대비: {(balance/1000 - 1) * 100:.1f}%")

# 최대 연속 손실
consecutive_losses = 0
max_consecutive_losses = 0

for pnl in trades_df['pnl_pct']:
    if pnl < 0:
        consecutive_losses += 1
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    else:
        consecutive_losses = 0

print(f"\n최대 연속 손실: {max_consecutive_losses}회")

# 회복 시간
dd_periods = []
in_drawdown = False
dd_start = 0

for i in range(len(balance_series)):
    if balance_series[i] < peak[i] and not in_drawdown:
        in_drawdown = True
        dd_start = i
    elif balance_series[i] >= peak[i] and in_drawdown:
        in_drawdown = False
        dd_periods.append(i - dd_start)

if len(dd_periods) > 0:
    avg_recovery = np.mean(dd_periods)
    print(f"평균 회복 시간: {avg_recovery:.0f}회 거래")

# ═══════════════════════════════════════════════════════════
# 3. FVG 유무별 성과
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3. FVG 유무별 성과")
print("=" * 70)

def detect_fvg(df):
    """FVG 감지"""
    fvg_flags = []

    for i in range(len(df)):
        has_fvg = False

        if i >= 2:
            candle_1 = df.iloc[i-2]
            candle_3 = df.iloc[i]

            # Bullish FVG
            if candle_1['high'] < candle_3['low']:
                has_fvg = True

        fvg_flags.append(has_fvg)

    return fvg_flags

# FVG 감지
fvg_flags = detect_fvg(df)
fvg_df = pd.DataFrame({'has_fvg': fvg_flags})

# Trendline up에 FVG 정보 추가
trendline_up['has_fvg'] = trendline_up['break_idx'].apply(
    lambda x: fvg_df.iloc[x]['has_fvg'] if x < len(fvg_df) else False
)

# FVG 있는 경우 vs 없는 경우
with_fvg = trendline_up[trendline_up['has_fvg'] == True]
without_fvg = trendline_up[trendline_up['has_fvg'] == False]

print(f"\nFVG 있음: {len(with_fvg):,}개 ({len(with_fvg)/len(trendline_up)*100:.1f}%)")
print(f"FVG 없음: {len(without_fvg):,}개 ({len(without_fvg)/len(trendline_up)*100:.1f}%)")

# 각각 백테스트
if len(with_fvg) > 0:
    trades_with_fvg = backtest_strategy(with_fvg, df, tp_pct=2.0, sl_pct=2.0)

    win_rate_fvg = (trades_with_fvg['pnl_pct'] > 0).sum() / len(trades_with_fvg) * 100
    avg_pnl_fvg = trades_with_fvg['pnl_pct'].mean() - fee_pct

    print(f"\n[FVG 있음]")
    print(f"  승률: {win_rate_fvg:.1f}%")
    print(f"  평균 PnL: {avg_pnl_fvg:.3f}%")

if len(without_fvg) > 0:
    trades_without_fvg = backtest_strategy(without_fvg, df, tp_pct=2.0, sl_pct=2.0)

    win_rate_no_fvg = (trades_without_fvg['pnl_pct'] > 0).sum() / len(trades_without_fvg) * 100
    avg_pnl_no_fvg = trades_without_fvg['pnl_pct'].mean() - fee_pct

    print(f"\n[FVG 없음]")
    print(f"  승률: {win_rate_no_fvg:.1f}%")
    print(f"  평균 PnL: {avg_pnl_no_fvg:.3f}%")

# ═══════════════════════════════════════════════════════════
# 4. 동적 TP - FVG 기반
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4. 동적 TP 전략 (FVG 기반)")
print("=" * 70)

def backtest_dynamic_tp(signals_df, df, has_fvg_flags):
    """FVG 유무에 따라 다른 TP 적용"""
    trades = []

    for idx, signal in signals_df.iterrows():
        break_idx = signal['break_idx']
        entry_price = signal['break_price']
        has_fvg = signal.get('has_fvg', False)

        # FVG 있으면 공격적, 없으면 보수적
        if has_fvg:
            tp_pct = 2.5  # 공격적
            sl_pct = 2.0
        else:
            tp_pct = 1.5  # 보수적
            sl_pct = 2.0

        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

        # 향후 50봉 스캔
        max_idx = min(break_idx + 50, len(df) - 1)
        future = df.iloc[break_idx:max_idx+1]

        hit_tp = False
        hit_sl = False
        exit_idx = None
        exit_price = None

        for i in range(1, len(future)):
            candle = future.iloc[i]

            if candle['high'] >= tp_price:
                hit_tp = True
                exit_idx = break_idx + i
                exit_price = tp_price
                break

            if candle['low'] <= sl_price:
                hit_sl = True
                exit_idx = break_idx + i
                exit_price = sl_price
                break

        if exit_idx is None:
            exit_idx = max_idx
            exit_price = future.iloc[-1]['close']

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_idx': break_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'has_fvg': has_fvg,
            'tp_pct': tp_pct,
        })

    return pd.DataFrame(trades)

# 동적 TP 백테스트
trades_dynamic = backtest_dynamic_tp(trendline_up, df, trendline_up['has_fvg'])

win_rate_dynamic = (trades_dynamic['pnl_pct'] > 0).sum() / len(trades_dynamic) * 100
avg_pnl_dynamic = trades_dynamic['pnl_pct'].mean() - fee_pct

print(f"\n동적 TP 적용:")
print(f"  - FVG 있음: TP 2.5%, SL 2.0%")
print(f"  - FVG 없음: TP 1.5%, SL 2.0%")

print(f"\n총 거래: {len(trades_dynamic):,}개")
print(f"승률: {win_rate_dynamic:.1f}%")
print(f"평균 PnL: {avg_pnl_dynamic:.3f}%")

# 복리 수익률
cumulative_dynamic = 1.0
for pnl in trades_dynamic['pnl_pct']:
    actual_pnl = (pnl - fee_pct) * leverage * position_size / 100
    cumulative_dynamic *= (1 + actual_pnl)

total_return_dynamic = (cumulative_dynamic - 1) * 100

print(f"\n총 수익률 (복리): {total_return_dynamic:.1f}%")
print(f"월간 수익: {total_return_dynamic / 60:.2f}%")

# 기본 전략 대비
improvement = total_return_dynamic - total_return_pct
print(f"\n기본 대비 개선: {improvement:+.1f}%p")

# ═══════════════════════════════════════════════════════════
# 5. 종합 비교표
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5. 종합 비교")
print("=" * 70)

comparison = pd.DataFrame({
    '전략': ['기본 (TP 2%)', '동적 TP (FVG 기반)'],
    '거래수': [f"{len(trades_df):,}", f"{len(trades_dynamic):,}"],
    '승률': [f"{win_rate:.1f}%", f"{win_rate_dynamic:.1f}%"],
    '평균PnL': [f"{avg_pnl_after_fee:.3f}%", f"{avg_pnl_dynamic:.3f}%"],
    '총수익': [f"{total_return_pct:.1f}%", f"{total_return_dynamic:.1f}%"],
    'MDD': [f"{max_drawdown:.2f}%", "계산중..."],
})

print("\n" + comparison.to_string(index=False))

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
