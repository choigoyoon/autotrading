"""
청산 체크박스 백테스트
- 진입: L + 추세선 + 되돌림 (검증된 조건)
- 청산: 체크박스 조건 모니터링
- 5년 전체 백테스트
- 손익비, 승율, MDD, 복리, 단리 계산
"""

import pandas as pd
import numpy as np
from itertools import product

print("=" * 60)
print("청산 체크박스 백테스트")
print("=" * 60)

# 데이터 로드
print("\n데이터 로드 중...")
df = pd.read_csv("output_phase1_labeled.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
breakouts_df = pd.read_csv("output_phase4_breakouts.csv")

print(f"데이터: {len(df):,}봉 (5년)")
print(f"돌파: {len(breakouts_df):,}개")

# 추세선 돌파만
trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]
print(f"추세선 돌파: {len(trendline_breakouts):,}개\n")


def calculate_exit_checkboxes(window, entry_price, entry_idx_local, direction, labeled_df):
    """진입 후 각 봉마다 청산 체크박스 계산"""

    after_entry = window.loc[entry_idx_local:].copy()

    if len(after_entry) < 2:
        return pd.DataFrame()

    # 현재 손익
    if direction == 'long':
        after_entry['pl'] = (after_entry['close'] - entry_price) / entry_price * 100
        after_entry['max_pl'] = (after_entry['high'] - entry_price) / entry_price * 100
        after_entry['min_pl'] = (after_entry['low'] - entry_price) / entry_price * 100
    else:
        after_entry['pl'] = (entry_price - after_entry['close']) / entry_price * 100
        after_entry['max_pl'] = (entry_price - after_entry['low']) / entry_price * 100
        after_entry['min_pl'] = (entry_price - after_entry['high']) / entry_price * 100

    # 체크박스 1: MACD 히스토그램 반전
    after_entry['macd_reversal'] = False
    if direction == 'long':
        # 상승 중 하락 전환
        after_entry['macd_reversal'] = (after_entry['macd_hist'].diff() < 0) & \
                                        (after_entry['macd_hist'].shift(1).diff() < 0)
    else:
        # 하락 중 상승 전환
        after_entry['macd_reversal'] = (after_entry['macd_hist'].diff() > 0) & \
                                        (after_entry['macd_hist'].shift(1).diff() > 0)

    # 체크박스 2: 모멘텀 둔화 (3봉 연속)
    price_change = after_entry['close'].diff()
    if direction == 'long':
        after_entry['momentum_slow'] = (price_change < 0).rolling(3).sum() >= 2
    else:
        after_entry['momentum_slow'] = (price_change > 0).rolling(3).sum() >= 2

    # 체크박스 3: 새로운 H/L 형성 (1봉 딜레이 - 나우캐스트)
    after_entry['new_hl'] = False
    for idx in after_entry.index:
        if direction == 'long':
            # 이전봉이 H로 확정되었는지 체크 (현재봉에서 확인 가능)
            if idx-1 in labeled_df.index and labeled_df.loc[idx-1, 'label'] == 'H':
                after_entry.loc[idx, 'new_hl'] = True
        else:
            # 이전봉이 L로 확정되었는지 체크 (현재봉에서 확인 가능)
            if idx-1 in labeled_df.index and labeled_df.loc[idx-1, 'label'] == 'L':
                after_entry.loc[idx, 'new_hl'] = True

    # 체크박스 4: 수익 구간별 조건 (개선 버전)
    # 승률↑: 빠른 손절 (-0.8% → -0.6%)
    # 수익↑: 트레일링 스톱
    # 안정성↑: 부분 익절 개념
    after_entry['exit_signal'] = False
    after_entry['peak_pl'] = after_entry['max_pl'].cummax()  # 누적 최고

    for idx in after_entry.index:
        pl = after_entry.loc[idx, 'pl']
        peak = after_entry.loc[idx, 'peak_pl']

        # 승률↑: 타이트한 손절 -0.6% (기존 -0.8%)
        if after_entry.loc[idx, 'min_pl'] <= -0.6:
            after_entry.loc[idx, 'exit_signal'] = True
            continue

        # 소액 손실 빠른 컷 (승률↑)
        if pl < -0.2 and after_entry.loc[idx, 'momentum_slow']:
            after_entry.loc[idx, 'exit_signal'] = True
            continue

        # 수익 구간별
        if pl < 0.5:
            # 손실 구간: 빠른 청산
            if after_entry.loc[idx, 'momentum_slow']:
                after_entry.loc[idx, 'exit_signal'] = True
        elif pl < 1.0:
            # 0.5-1%: 모멘텀 둔화
            if after_entry.loc[idx, 'momentum_slow']:
                after_entry.loc[idx, 'exit_signal'] = True
        elif pl < 1.5:
            # 1-1.5%: 모멘텀 + MACD
            if after_entry.loc[idx, 'momentum_slow'] and after_entry.loc[idx, 'macd_reversal']:
                after_entry.loc[idx, 'exit_signal'] = True
        else:
            # 1.5%+: 트레일링 스톱 (수익↑)
            # 피크에서 40% 되돌리면 청산
            if peak >= 1.5 and pl <= peak * 0.6:
                after_entry.loc[idx, 'exit_signal'] = True
            # 또는 모멘텀 + MACD 둘 다
            elif after_entry.loc[idx, 'momentum_slow'] and after_entry.loc[idx, 'macd_reversal']:
                after_entry.loc[idx, 'exit_signal'] = True

    return after_entry


def backtest_with_exit_checkboxes(df, trendline_breakouts, labeled_df,
                                   pb_min, pb_max, support_bars):
    """청산 체크박스 백테스트"""

    trades = []
    equity_curve = []
    balance = 10000  # 초기 자본
    balance_compound = 10000

    for idx, breakout in trendline_breakouts.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        end_idx = min(break_idx + 60, len(df) - 1)
        if end_idx <= break_idx + 1:
            continue

        window = df.iloc[break_idx:end_idx+1].copy()
        break_price = window.iloc[0]['close']

        # 진입: 되돌림 + 지지
        if direction == 'long':
            pullback = (window['close'] - break_price) / break_price * 100
        else:
            pullback = (break_price - window['close']) / break_price * 100

        entry_mask = (pullback >= pb_min) & (pullback <= pb_max)

        if not entry_mask.any():
            continue

        # 지지 확인
        if direction == 'long':
            rising = window['close'].diff() > 0
            rising_sum = rising.rolling(support_bars).sum()
            support_confirmed = rising_sum >= support_bars
        else:
            falling = window['close'].diff() < 0
            falling_sum = falling.rolling(support_bars).sum()
            support_confirmed = falling_sum >= support_bars

        entry_candidates = entry_mask & support_confirmed

        if not entry_candidates.any():
            continue

        entry_idx_local = entry_candidates.idxmax()
        entry_price = window.loc[entry_idx_local, 'close']
        entry_time = window.loc[entry_idx_local, 'datetime']

        # 청산 체크박스 계산
        exit_data = calculate_exit_checkboxes(window, entry_price, entry_idx_local,
                                               direction, labeled_df)

        if len(exit_data) == 0:
            continue

        # 청산 타이밍 찾기
        exit_signals = exit_data[exit_data['exit_signal'] == True]

        if len(exit_signals) > 0:
            # 첫 청산 신호
            exit_idx_local = exit_signals.index[0]
            exit_pl = exit_data.loc[exit_idx_local, 'pl']
            exit_time = exit_data.loc[exit_idx_local, 'datetime']
            bars_held = exit_signals.index.get_loc(exit_idx_local)
            exit_reason = 'checkbox'
        else:
            # 청산 신호 없으면 마지막 봉
            exit_pl = exit_data.iloc[-1]['pl']
            exit_time = exit_data.iloc[-1]['datetime']
            bars_held = len(exit_data) - 1
            exit_reason = 'timeout'

        # 슬리피지
        exit_pl_net = exit_pl - 0.2

        # 단리 (고정 금액)
        balance += (10000 * exit_pl_net / 100)

        # 복리 (잔고 비율)
        balance_compound *= (1 + exit_pl_net / 100)

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'direction': direction,
            'pl_gross': exit_pl,
            'pl_net': exit_pl_net,
            'bars_held': bars_held,
            'exit_reason': exit_reason,
            'balance': balance,
            'balance_compound': balance_compound
        })

    return pd.DataFrame(trades)


# 파라미터 그리드 (검증된 범위)
param_configs = [
    {'name': '얕은 되돌림', 'pb_min': -0.3, 'pb_max': 0.0, 'support': 2},
    {'name': '중간 되돌림', 'pb_min': -0.5, 'pb_max': -0.3, 'support': 2},
    {'name': '깊은 되돌림', 'pb_min': -0.8, 'pb_max': -0.5, 'support': 2},
    {'name': '최적 범위', 'pb_min': -1.0, 'pb_max': -0.2, 'support': 2},
]

print("=" * 60)
print("백테스트 실행")
print("=" * 60)

labeled_df = df[df['label'].notna()].set_index(df[df['label'].notna()].index)

all_results = []

for config in param_configs:
    print(f"\n테스트 중: {config['name']} (pb {config['pb_min']}~{config['pb_max']}%)")

    trades_df = backtest_with_exit_checkboxes(
        df, trendline_breakouts, labeled_df,
        config['pb_min'], config['pb_max'], config['support']
    )

    if len(trades_df) == 0:
        print("  거래 없음")
        continue

    # 통계 계산
    total_trades = len(trades_df)
    wins = (trades_df['pl_net'] > 0).sum()
    losses = (trades_df['pl_net'] <= 0).sum()
    win_rate = wins / total_trades * 100

    avg_win = trades_df[trades_df['pl_net'] > 0]['pl_net'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['pl_net'] <= 0]['pl_net'].mean() if losses > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    total_pl = trades_df['pl_net'].sum()
    avg_pl = trades_df['pl_net'].mean()

    # MDD 계산 (복리)
    equity = trades_df['balance_compound'].values
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_dd = drawdown.min()

    # 최종 수익
    final_balance_simple = trades_df.iloc[-1]['balance']
    final_balance_compound = trades_df.iloc[-1]['balance_compound']

    simple_return = (final_balance_simple - 10000) / 10000 * 100
    compound_return = (final_balance_compound - 10000) / 10000 * 100

    # 평균 보유 기간
    avg_bars = trades_df['bars_held'].mean()

    # 청산 사유
    checkbox_exits = (trades_df['exit_reason'] == 'checkbox').sum()
    timeout_exits = (trades_df['exit_reason'] == 'timeout').sum()

    print(f"\n📊 {config['name']} 결과:")
    print(f"  총 거래: {total_trades}회")
    print(f"  승: {wins}회, 패: {losses}회")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 수익: {avg_pl:.3f}%")
    print(f"  평균 승: {avg_win:.3f}%, 평균 패: {avg_loss:.3f}%")
    print(f"  손익비: {profit_factor:.2f}")
    print(f"  MDD: {max_dd:.2f}%")
    print(f"  단리 수익: {simple_return:.2f}% (₩{final_balance_simple:,.0f})")
    print(f"  복리 수익: {compound_return:.2f}% (₩{final_balance_compound:,.0f})")
    print(f"  평균 보유: {avg_bars:.1f}봉")
    print(f"  청산: 체크박스 {checkbox_exits}회, 타임아웃 {timeout_exits}회")

    all_results.append({
        'config': config['name'],
        'trades': total_trades,
        'win_rate': win_rate,
        'avg_pl': avg_pl,
        'profit_factor': profit_factor,
        'mdd': max_dd,
        'simple_return': simple_return,
        'compound_return': compound_return,
        'avg_bars': avg_bars,
        'checkbox_exits': checkbox_exits,
        'trades_df': trades_df
    })


# 최종 비교
print("\n" + "=" * 60)
print("📊 전체 비교")
print("=" * 60)

comparison = pd.DataFrame([{
    '설정': r['config'],
    '거래': r['trades'],
    '승률': f"{r['win_rate']:.1f}%",
    '평균수익': f"{r['avg_pl']:.3f}%",
    '손익비': f"{r['profit_factor']:.2f}",
    'MDD': f"{r['mdd']:.1f}%",
    '복리수익': f"{r['compound_return']:.1f}%",
    '체크박스청산': f"{r['checkbox_exits']}회"
} for r in all_results])

print("\n" + comparison.to_string(index=False))

# 최고 성과
best = max(all_results, key=lambda x: x['compound_return'])

print("\n" + "=" * 60)
print("🏆 최고 성과")
print("=" * 60)
print(f"\n설정: {best['config']}")
print(f"총 거래: {best['trades']}회")
print(f"승률: {best['win_rate']:.1f}%")
print(f"평균 수익: {best['avg_pl']:.3f}%")
print(f"손익비: {best['profit_factor']:.2f}")
print(f"MDD: {best['mdd']:.2f}%")
print(f"단리 수익: {best['simple_return']:.2f}%")
print(f"복리 수익: {best['compound_return']:.2f}%")
print(f"평균 보유: {best['avg_bars']:.1f}봉 (15분 = {best['avg_bars'] * 15 / 60:.1f}시간)")
print(f"체크박스 청산: {best['checkbox_exits']}회 ({best['checkbox_exits']/best['trades']*100:.1f}%)")

# 거래 내역 저장
best['trades_df'].to_csv('backtest_exit_checkboxes_trades.csv', index=False)
print(f"\n거래 내역 저장: backtest_exit_checkboxes_trades.csv")

print("\n" + "=" * 60)
print("백테스트 완료")
print("=" * 60)
