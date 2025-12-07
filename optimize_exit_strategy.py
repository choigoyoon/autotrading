"""
진입 후 청산 최적화
- 승률 향상 (빠른 손절)
- 수익 극대화 (익절 최적화)
- 안정성 향상 (변동성 감소)
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("진입 후 청산 최적화")
print("=" * 60)

# 데이터 로드
df = pd.read_csv("output_phase1_labeled.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
breakouts_df = pd.read_csv("output_phase4_breakouts.csv")

trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]
labeled_df = df[df['label'].notna()].set_index(df[df['label'].notna()].index)

print(f"\n데이터: {len(df):,}봉")
print(f"추세선 돌파: {len(trendline_breakouts):,}개")

# 진입 파라미터 (검증된 최적값)
PB_MIN = -1.0
PB_MAX = -0.2
SUPPORT_BARS = 2

def backtest_exit_strategy(strategy_name, exit_func):
    """청산 전략 백테스트"""

    trades = []
    balance = 10000
    balance_compound = 10000
    running_max = 10000

    for idx, breakout in trendline_breakouts.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        end_idx = min(break_idx + 60, len(df) - 1)
        if end_idx <= break_idx + 1:
            continue

        window = df.iloc[break_idx:end_idx+1].copy()
        break_price = window.iloc[0]['close']

        # 진입 조건 (기존 검증된 로직)
        if direction == 'long':
            pullback = (window['close'] - break_price) / break_price * 100
            rising = window['close'].diff() > 0
        else:
            pullback = (break_price - window['close']) / break_price * 100
            rising = window['close'].diff() < 0

        entry_mask = (pullback >= PB_MIN) & (pullback <= PB_MAX)
        rising_sum = rising.rolling(SUPPORT_BARS).sum()
        support_confirmed = rising_sum >= SUPPORT_BARS
        entry_candidates = entry_mask & support_confirmed

        if not entry_candidates.any():
            continue

        entry_idx_local = entry_candidates.idxmax()
        entry_price = window.loc[entry_idx_local, 'close']
        entry_time = window.loc[entry_idx_local, 'datetime']

        after_entry = window.loc[entry_idx_local:]

        if len(after_entry) < 2:
            continue

        # 청산 전략 실행
        exit_result = exit_func(after_entry, entry_price, direction, labeled_df)

        if exit_result is None:
            continue

        exit_pl, bars_held, exit_reason = exit_result

        # 슬리피지
        exit_pl_net = exit_pl - 0.2

        # 복리
        balance_compound *= (1 + exit_pl_net / 100)

        if balance_compound > running_max:
            running_max = balance_compound

        trades.append({
            'pl_gross': exit_pl,
            'pl_net': exit_pl_net,
            'bars_held': bars_held,
            'exit_reason': exit_reason,
            'balance_compound': balance_compound
        })

    return pd.DataFrame(trades)


# ============================================================
# 청산 전략들
# ============================================================

def strategy_1_current(after_entry, entry_price, direction, labeled_df):
    """전략 1: 현재 방식 (기준선)"""

    # 손익 계산
    if direction == 'long':
        pl = (after_entry['close'] - entry_price) / entry_price * 100
        max_pl = (after_entry['high'] - entry_price) / entry_price * 100
        min_pl = (after_entry['low'] - entry_price) / entry_price * 100
    else:
        pl = (entry_price - after_entry['close']) / entry_price * 100
        max_pl = (entry_price - after_entry['low']) / entry_price * 100
        min_pl = (entry_price - after_entry['high']) / entry_price * 100

    # MACD, 모멘텀
    price_change = after_entry['close'].diff()
    if direction == 'long':
        macd_reversal = (after_entry['macd_hist'].diff() < 0) & \
                        (after_entry['macd_hist'].shift(1).diff() < 0)
        momentum_slow = (price_change < 0).rolling(3).sum() >= 2
    else:
        macd_reversal = (after_entry['macd_hist'].diff() > 0) & \
                        (after_entry['macd_hist'].shift(1).diff() > 0)
        momentum_slow = (price_change > 0).rolling(3).sum() >= 2

    # 청산 신호
    for i, idx in enumerate(after_entry.index):
        current_pl = pl.iloc[i]
        current_min = min_pl.iloc[i]

        # SL
        if current_min <= -0.8:
            return (current_min, i, 'SL')

        # 수익 구간별
        if current_pl >= 0.5 and current_pl < 1.0:
            if momentum_slow.iloc[i]:
                return (current_pl, i, 'momentum')
        elif current_pl >= 1.0 and current_pl < 2.0:
            if momentum_slow.iloc[i] and macd_reversal.iloc[i]:
                return (current_pl, i, 'momentum+macd')
        elif current_pl >= 2.0:
            if momentum_slow.iloc[i] or macd_reversal.iloc[i]:
                return (current_pl, i, 'profit_protect')

    # 타임아웃
    return (pl.iloc[-1], len(after_entry)-1, 'timeout')


def strategy_2_quick_cut(after_entry, entry_price, direction, labeled_df):
    """전략 2: 빠른 손절 + 익절 유지 (승률↑)"""

    if direction == 'long':
        pl = (after_entry['close'] - entry_price) / entry_price * 100
        min_pl = (after_entry['low'] - entry_price) / entry_price * 100
    else:
        pl = (entry_price - after_entry['close']) / entry_price * 100
        min_pl = (entry_price - after_entry['high']) / entry_price * 100

    price_change = after_entry['close'].diff()
    momentum_slow = (price_change < 0).rolling(2).sum() >= 2 if direction == 'long' else \
                    (price_change > 0).rolling(2).sum() >= 2

    for i, idx in enumerate(after_entry.index):
        current_pl = pl.iloc[i]
        current_min = min_pl.iloc[i]

        # 타이트 SL
        if current_min <= -0.5:
            return (current_min, i, 'quick_SL')

        # 소액 손실에서도 빠르게 청산
        if current_pl < 0 and momentum_slow.iloc[i]:
            return (current_pl, i, 'quick_cut')

        # 수익은 여유롭게
        if current_pl >= 1.5:
            if momentum_slow.iloc[i]:
                return (current_pl, i, 'profit_exit')

    return (pl.iloc[-1], len(after_entry)-1, 'timeout')


def strategy_3_let_winners_run(after_entry, entry_price, direction, labeled_df):
    """전략 3: 승자 달리게 + Trailing (수익↑)"""

    if direction == 'long':
        pl = (after_entry['close'] - entry_price) / entry_price * 100
        max_pl = (after_entry['high'] - entry_price) / entry_price * 100
        min_pl = (after_entry['low'] - entry_price) / entry_price * 100
    else:
        pl = (entry_price - after_entry['close']) / entry_price * 100
        max_pl = (entry_price - after_entry['low']) / entry_price * 100
        min_pl = (entry_price - after_entry['high']) / entry_price * 100

    peak_pl = 0

    for i, idx in enumerate(after_entry.index):
        current_pl = pl.iloc[i]
        current_max = max_pl.iloc[i]
        current_min = min_pl.iloc[i]

        # 피크 추적
        if current_max > peak_pl:
            peak_pl = current_max

        # 기본 SL
        if current_min <= -0.8:
            return (current_min, i, 'SL')

        # Trailing stop (수익 나면 트레일링)
        if peak_pl >= 1.0:
            # 피크에서 50% 되돌리면 청산
            if current_pl <= peak_pl * 0.5:
                return (current_pl, i, 'trailing_50pct')

        if peak_pl >= 2.0:
            # 피크에서 30% 되돌리면 청산
            if current_pl <= peak_pl * 0.7:
                return (current_pl, i, 'trailing_30pct')

    return (pl.iloc[-1], len(after_entry)-1, 'timeout')


def strategy_4_partial_exit(after_entry, entry_price, direction, labeled_df):
    """전략 4: 부분 익절 (안정성↑)"""

    if direction == 'long':
        pl = (after_entry['close'] - entry_price) / entry_price * 100
        min_pl = (after_entry['low'] - entry_price) / entry_price * 100
    else:
        pl = (entry_price - after_entry['close']) / entry_price * 100
        min_pl = (entry_price - after_entry['high']) / entry_price * 100

    price_change = after_entry['close'].diff()
    momentum_slow = (price_change < 0).rolling(3).sum() >= 2 if direction == 'long' else \
                    (price_change > 0).rolling(3).sum() >= 2

    position_remaining = 1.0
    total_pl = 0

    for i, idx in enumerate(after_entry.index):
        current_pl = pl.iloc[i]
        current_min = min_pl.iloc[i]

        # SL
        if current_min <= -0.8:
            total_pl += current_min * position_remaining
            return (total_pl, i, 'SL')

        # 부분 익절 1: +0.7% 도달 → 50% 청산
        if position_remaining == 1.0 and current_pl >= 0.7:
            total_pl += current_pl * 0.5
            position_remaining = 0.5

        # 부분 익절 2: +1.5% 도달 → 나머지 50% 청산
        if position_remaining == 0.5 and current_pl >= 1.5:
            total_pl += current_pl * 0.5
            return (total_pl, i, 'partial_complete')

        # 나머지 포지션 청산 조건
        if position_remaining > 0 and momentum_slow.iloc[i]:
            total_pl += current_pl * position_remaining
            return (total_pl, i, 'remainder_exit')

    # 타임아웃
    total_pl += pl.iloc[-1] * position_remaining
    return (total_pl, len(after_entry)-1, 'timeout')


def strategy_5_hybrid(after_entry, entry_price, direction, labeled_df):
    """전략 5: 하이브리드 (빠른손절 + 부분익절 + 트레일링)"""

    if direction == 'long':
        pl = (after_entry['close'] - entry_price) / entry_price * 100
        max_pl = (after_entry['high'] - entry_price) / entry_price * 100
        min_pl = (after_entry['low'] - entry_price) / entry_price * 100
    else:
        pl = (entry_price - after_entry['close']) / entry_price * 100
        max_pl = (entry_price - after_entry['low']) / entry_price * 100
        min_pl = (entry_price - after_entry['high']) / entry_price * 100

    price_change = after_entry['close'].diff()
    momentum_slow = (price_change < 0).rolling(2).sum() >= 2 if direction == 'long' else \
                    (price_change > 0).rolling(2).sum() >= 2

    position_remaining = 1.0
    total_pl = 0
    peak_pl = 0

    for i, idx in enumerate(after_entry.index):
        current_pl = pl.iloc[i]
        current_max = max_pl.iloc[i]
        current_min = min_pl.iloc[i]

        if current_max > peak_pl:
            peak_pl = current_max

        # 빠른 손절
        if current_min <= -0.5:
            total_pl += current_min * position_remaining
            return (total_pl, i, 'quick_SL')

        # 소액 손실 빠른 컷
        if current_pl < -0.2 and momentum_slow.iloc[i]:
            total_pl += current_pl * position_remaining
            return (total_pl, i, 'quick_cut')

        # 부분 익절: +0.8% → 50%
        if position_remaining == 1.0 and current_pl >= 0.8:
            total_pl += current_pl * 0.5
            position_remaining = 0.5

        # 트레일링 (나머지 50%)
        if position_remaining == 0.5:
            if peak_pl >= 1.5:
                # 피크에서 40% 되돌리면 청산
                if current_pl <= peak_pl * 0.6:
                    total_pl += current_pl * 0.5
                    return (total_pl, i, 'trailing')

    # 타임아웃
    total_pl += pl.iloc[-1] * position_remaining
    return (total_pl, len(after_entry)-1, 'timeout')


# ============================================================
# 전략 테스트
# ============================================================

print("\n" + "=" * 60)
print("전략 백테스트 중...")
print("=" * 60)

strategies = [
    ("현재 방식 (기준)", strategy_1_current),
    ("빠른 손절", strategy_2_quick_cut),
    ("승자 달리게", strategy_3_let_winners_run),
    ("부분 익절", strategy_4_partial_exit),
    ("하이브리드", strategy_5_hybrid),
]

results = []

for name, strategy_func in strategies:
    print(f"\n테스트: {name}")

    trades_df = backtest_exit_strategy(name, strategy_func)

    if len(trades_df) == 0:
        print("  거래 없음")
        continue

    total_trades = len(trades_df)
    wins = (trades_df['pl_net'] > 0).sum()
    losses = (trades_df['pl_net'] <= 0).sum()
    win_rate = wins / total_trades * 100

    avg_win = trades_df[trades_df['pl_net'] > 0]['pl_net'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pl_net'] <= 0]['pl_net'].mean() if losses > 0 else 0

    avg_pl = trades_df['pl_net'].mean()
    total_pl = trades_df['pl_net'].sum()

    # MDD
    equity = trades_df['balance_compound'].values
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_dd = drawdown.min()

    final_balance = trades_df.iloc[-1]['balance_compound']
    compound_return = (final_balance - 10000) / 10000 * 100

    avg_bars = trades_df['bars_held'].mean()

    print(f"  거래: {total_trades}회")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 수익: {avg_pl:.3f}%")
    print(f"  평균 승/패: {avg_win:.3f}% / {avg_loss:.3f}%")
    print(f"  복리 수익: {compound_return:.2f}%")
    print(f"  MDD: {max_dd:.2f}%")
    print(f"  평균 보유: {avg_bars:.1f}봉")

    results.append({
        'strategy': name,
        'trades': total_trades,
        'win_rate': win_rate,
        'avg_pl': avg_pl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'compound': compound_return,
        'mdd': max_dd,
        'avg_bars': avg_bars,
    })

# 최종 비교
print("\n" + "=" * 60)
print("📊 전략 비교")
print("=" * 60)

comparison = pd.DataFrame(results)
print("\n" + comparison.to_string(index=False))

# 최적 전략
best_winrate = comparison.loc[comparison['win_rate'].idxmax()]
best_profit = comparison.loc[comparison['compound'].idxmax()]
best_mdd = comparison.loc[comparison['mdd'].idxmax()]

print("\n" + "=" * 60)
print("🏆 최적 전략")
print("=" * 60)

print(f"\n승률 최고: {best_winrate['strategy']}")
print(f"  승률: {best_winrate['win_rate']:.1f}%")
print(f"  수익: {best_winrate['compound']:.1f}%")
print(f"  MDD: {best_mdd['mdd']:.1f}%")

print(f"\n수익 최고: {best_profit['strategy']}")
print(f"  승률: {best_profit['win_rate']:.1f}%")
print(f"  수익: {best_profit['compound']:.1f}%")
print(f"  MDD: {best_profit['mdd']:.1f}%")

print(f"\nMDD 최소: {best_mdd['strategy']}")
print(f"  승률: {best_mdd['win_rate']:.1f}%")
print(f"  수익: {best_mdd['compound']:.1f}%")
print(f"  MDD: {best_mdd['mdd']:.1f}%")

print("\n" + "=" * 60)
print("완료")
print("=" * 60)
