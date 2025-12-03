"""
5년 데이터 종합 분석 - MDD, 수익률, 승률 등
======================================================================
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("5년 데이터 종합 분석 보고서")
print("=" * 80)
print()

# 거래 내역 로드
df_trades = pd.read_csv('l_value_nowcast_trades.csv')
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])

# 전체 데이터 기간
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

start_date = df['datetime'].min()
end_date = df['datetime'].max()
total_days = (end_date - start_date).days

print(f"분석 기간: {start_date.date()} ~ {end_date.date()}")
print(f"총 기간: {total_days}일 ({total_days/365.25:.1f}년)")
print()

# ═══════════════════════════════════════════════════════════════════
# 전략별 종합 분석
# ═══════════════════════════════════════════════════════════════════

def comprehensive_analysis(df_trades, filter_col, strategy_name):
    """전략별 종합 성과 분석"""

    filtered = df_trades[df_trades[filter_col]].copy()

    if len(filtered) == 0:
        print(f"\n{strategy_name}: 거래 없음")
        return None

    # 날짜순 정렬
    filtered = filtered.sort_values('entry_time').reset_index(drop=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 기본 통계
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    total_trades = len(filtered)
    wins = (filtered['net_pnl'] > 0).sum()
    losses = (filtered['net_pnl'] <= 0).sum()
    win_rate = wins / total_trades * 100

    avg_pnl = filtered['net_pnl'].mean()
    median_pnl = filtered['net_pnl'].median()
    std_pnl = filtered['net_pnl'].std()

    avg_win = filtered[filtered['net_pnl'] > 0]['net_pnl'].mean() if wins > 0 else 0
    avg_loss = filtered[filtered['net_pnl'] <= 0]['net_pnl'].mean() if losses > 0 else 0

    max_win = filtered['net_pnl'].max()
    max_loss = filtered['net_pnl'].min()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 누적 손익 및 MDD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    filtered['cumulative_pnl'] = filtered['net_pnl'].cumsum()
    filtered['cumulative_max'] = filtered['cumulative_pnl'].cummax()
    filtered['drawdown'] = filtered['cumulative_pnl'] - filtered['cumulative_max']

    total_return = filtered['cumulative_pnl'].iloc[-1]
    mdd = filtered['drawdown'].min()
    mdd_pct = (mdd / filtered['cumulative_max'].max() * 100) if filtered['cumulative_max'].max() > 0 else 0

    # MDD 발생 시점
    mdd_idx = filtered['drawdown'].idxmin()
    mdd_date = filtered.loc[mdd_idx, 'entry_time']

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. 연속 승/패
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    filtered['is_win'] = filtered['net_pnl'] > 0

    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for is_win in filtered['is_win']:
        if is_win:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 시간 분석
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    trade_start = filtered['entry_time'].min()
    trade_end = filtered['entry_time'].max()
    trading_days = (trade_end - trade_start).days
    trading_years = trading_days / 365.25

    trades_per_year = total_trades / trading_years if trading_years > 0 else 0
    trades_per_month = total_trades / (trading_days / 30) if trading_days > 0 else 0

    avg_hold_time = filtered['hold_bars'].mean() * 15 / 60  # 15분봉 → 시간

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Risk/Reward 지표
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    profit_factor = abs(filtered[filtered['net_pnl'] > 0]['net_pnl'].sum() /
                        filtered[filtered['net_pnl'] <= 0]['net_pnl'].sum()) if losses > 0 else 999

    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 999

    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Sharpe Ratio (일별 수익률 기준)
    if len(filtered) > 1:
        filtered['days_since_start'] = (filtered['entry_time'] - filtered['entry_time'].iloc[0]).dt.days
        daily_returns = filtered.groupby('days_since_start')['net_pnl'].sum()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. 청산 분석
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    tp_count = (filtered['exit_reason'] == 'TP').sum()
    sl_count = (filtered['exit_reason'] == 'SL').sum()
    timeout_count = (filtered['exit_reason'] == 'TIMEOUT').sum()

    tp_rate = tp_count / total_trades * 100
    sl_rate = sl_count / total_trades * 100
    timeout_rate = timeout_count / total_trades * 100

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. 연도별 분석
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    filtered['year'] = filtered['entry_time'].dt.year
    yearly_stats = []

    for year in sorted(filtered['year'].unique()):
        year_data = filtered[filtered['year'] == year]
        yearly_stats.append({
            'year': year,
            'trades': len(year_data),
            'win_rate': (year_data['net_pnl'] > 0).sum() / len(year_data) * 100,
            'total_pnl': year_data['net_pnl'].sum(),
            'avg_pnl': year_data['net_pnl'].mean()
        })

    df_yearly = pd.DataFrame(yearly_stats)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 출력
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print()
    print("=" * 80)
    print(f"【{strategy_name}】")
    print("=" * 80)
    print()

    print("📊 기본 통계")
    print("-" * 80)
    print(f"  총 거래 수: {total_trades}개")
    print(f"  승/패: {wins}승 {losses}패")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  평균 손익: {avg_pnl:+.3f}% (중앙값: {median_pnl:+.3f}%)")
    print(f"  표준편차: {std_pnl:.3f}%")
    print(f"  평균 수익: {avg_win:+.3f}% | 평균 손실: {avg_loss:+.3f}%")
    print(f"  최대 수익: {max_win:+.3f}% | 최대 손실: {max_loss:+.3f}%")
    print()

    print("💰 누적 손익 및 리스크")
    print("-" * 80)
    print(f"  총 수익률: {total_return:+.2f}%")
    print(f"  MDD (최대 낙폭): {mdd:+.2f}% ({mdd_pct:+.1f}% 대비 고점)")
    print(f"  MDD 발생일: {mdd_date.date()}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Risk/Reward: {risk_reward:.2f}")
    print(f"  기댓값: {expectancy:+.3f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print()

    print("🔄 연속 승/패")
    print("-" * 80)
    print(f"  최대 연속 승: {max_consecutive_wins}회")
    print(f"  최대 연속 패: {max_consecutive_losses}회")
    print()

    print("⏱️  거래 빈도")
    print("-" * 80)
    print(f"  거래 기간: {trade_start.date()} ~ {trade_end.date()} ({trading_years:.1f}년)")
    print(f"  연간 거래: {trades_per_year:.0f}회")
    print(f"  월간 거래: {trades_per_month:.1f}회")
    print(f"  평균 보유시간: {avg_hold_time:.1f}시간")
    print()

    print("🎯 청산 분석")
    print("-" * 80)
    print(f"  TP 도달: {tp_count}회 ({tp_rate:.1f}%)")
    print(f"  SL 도달: {sl_count}회 ({sl_rate:.1f}%)")
    print(f"  타임아웃: {timeout_count}회 ({timeout_rate:.1f}%)")
    print()

    print("📅 연도별 성과")
    print("-" * 80)
    for _, row in df_yearly.iterrows():
        print(f"  {int(row['year'])}년: {int(row['trades'])}회 | 승률 {row['win_rate']:.1f}% | "
              f"수익 {row['total_pnl']:+.2f}% | 평균 {row['avg_pnl']:+.3f}%")
    print()

    return {
        'strategy': strategy_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_pnl': avg_pnl,
        'mdd': mdd,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'trades_per_year': trades_per_year
    }

# ═══════════════════════════════════════════════════════════════════
# 주요 전략 분석
# ═══════════════════════════════════════════════════════════════════

strategies = [
    ('f_none', '전략 1: 필터 없음 (모든 L값)'),
    ('f_rsi30_and_macd', '전략 2: RSI<30 AND MACD<0 (최고 승률)'),
    ('f_rsi35', '전략 3: RSI<35 (균형)'),
    ('f_macd', '전략 4: MACD<0 (많은 기회)'),
]

results = []
for col, name in strategies:
    result = comprehensive_analysis(df_trades, col, name)
    if result:
        results.append(result)

# ═══════════════════════════════════════════════════════════════════
# 비교 요약
# ═══════════════════════════════════════════════════════════════════

if results:
    print()
    print("=" * 80)
    print("📊 전략 비교 요약")
    print("=" * 80)
    print()

    df_summary = pd.DataFrame(results)

    print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print()

    # 최고 전략
    best_sharpe = df_summary.loc[df_summary['sharpe'].idxmax()]
    best_return = df_summary.loc[df_summary['total_return'].idxmax()]
    best_winrate = df_summary.loc[df_summary['win_rate'].idxmax()]
    lowest_mdd = df_summary.loc[df_summary['mdd'].idxmax()]  # MDD는 음수, 가장 작은 낙폭

    print("🏆 최고 성과")
    print("-" * 80)
    print(f"  최고 Sharpe: {best_sharpe['strategy']}")
    print(f"  최고 수익률: {best_return['strategy']} ({best_return['total_return']:+.2f}%)")
    print(f"  최고 승률: {best_winrate['strategy']} ({best_winrate['win_rate']:.2f}%)")
    print(f"  최소 MDD: {lowest_mdd['strategy']} ({lowest_mdd['mdd']:+.2f}%)")

print()
print("=" * 80)
print("✅ 종합 분석 완료")
print("=" * 80)
