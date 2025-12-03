#!/usr/bin/env python3
"""
백테스트 문제점 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("백테스트 문제점 시각화")
print("=" * 80)

# 데이터 로드
results = pd.read_csv('backtest_HL_strategy_results.csv')
results['entry_time'] = pd.to_datetime(results['entry_time'])
results['exit_time'] = pd.to_datetime(results['exit_time'])
results['year'] = results['entry_time'].dt.year

# 그림 생성
fig = plt.figure(figsize=(20, 24))

# 1. 연도별 누적 수익률
ax1 = plt.subplot(6, 2, 1)
yearly_cumsum = results.groupby('year')['pnl_pct'].sum()
ax1.bar(yearly_cumsum.index, yearly_cumsum.values, color=['green' if x > 0 else 'red' for x in yearly_cumsum.values])
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.set_title('1. Yearly Total PnL (%)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total PnL (%)')
ax1.grid(True, alpha=0.3)
for i, (year, val) in enumerate(yearly_cumsum.items()):
    ax1.text(year, val, f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

# 2. 전체 기간 누적 수익률
ax2 = plt.subplot(6, 2, 2)
results['cumulative_pnl'] = results['pnl_pct'].cumsum()
ax2.plot(results['entry_time'], results['cumulative_pnl'], linewidth=2, color='blue')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_title('2. Cumulative PnL Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative PnL (%)')
ax2.grid(True, alpha=0.3)
ax2.fill_between(results['entry_time'], 0, results['cumulative_pnl'], 
                  where=(results['cumulative_pnl'] >= 0), alpha=0.3, color='green')
ax2.fill_between(results['entry_time'], 0, results['cumulative_pnl'], 
                  where=(results['cumulative_pnl'] < 0), alpha=0.3, color='red')

# 3. 청산 사유별 비율
ax3 = plt.subplot(6, 2, 3)
exit_counts = results['exit_reason'].value_counts()
colors_exit = {'SL': 'red', 'TP2_Full': 'green', 'TP1_Breakeven': 'orange'}
ax3.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
        colors=[colors_exit.get(x, 'gray') for x in exit_counts.index],
        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax3.set_title('3. Exit Reason Distribution', fontsize=14, fontweight='bold')

# 4. 청산 사유별 PnL
ax4 = plt.subplot(6, 2, 4)
exit_pnl = results.groupby('exit_reason')['pnl_pct'].agg(['sum', 'mean', 'count'])
x_pos = np.arange(len(exit_pnl))
bars = ax4.bar(x_pos, exit_pnl['sum'].values, color=['red', 'orange', 'green'])
ax4.set_xticks(x_pos)
ax4.set_xticklabels(exit_pnl.index, rotation=45, ha='right')
ax4.set_title('4. Total PnL by Exit Reason', fontsize=14, fontweight='bold')
ax4.set_ylabel('Total PnL (%)')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
for i, (idx, row) in enumerate(exit_pnl.iterrows()):
    ax4.text(i, row['sum'], f"{row['sum']:.1f}%\n({int(row['count'])} trades)", 
             ha='center', va='bottom' if row['sum'] > 0 else 'top', fontweight='bold', fontsize=9)

# 5. HL 강도별 성과
ax5 = plt.subplot(6, 2, 5)
results['strength_group'] = pd.cut(results['HL_strength'], 
                                    bins=[0, 0.5, 1, 2, 5, 100],
                                    labels=['Weak\n(0-0.5%)', 'Medium\n(0.5-1%)', 'Strong\n(1-2%)', 
                                           'Very Strong\n(2-5%)', 'Extreme\n(5%+)'])
strength_pnl = results.groupby('strength_group', observed=True)['pnl_pct'].agg(['sum', 'mean', 'count'])
x_pos = np.arange(len(strength_pnl))
bars = ax5.bar(x_pos, strength_pnl['mean'].values, 
               color=['red' if x < 0 else 'green' for x in strength_pnl['mean'].values])
ax5.set_xticks(x_pos)
ax5.set_xticklabels(strength_pnl.index, fontsize=9)
ax5.set_title('5. Average PnL by HL Strength', fontsize=14, fontweight='bold')
ax5.set_ylabel('Average PnL (%)')
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
for i, (idx, row) in enumerate(strength_pnl.iterrows()):
    ax5.text(i, row['mean'], f"{row['mean']:.3f}%\n({int(row['count'])})", 
             ha='center', va='bottom' if row['mean'] > 0 else 'top', fontweight='bold', fontsize=8)

# 6. 보유 시간 분포
ax6 = plt.subplot(6, 2, 6)
ax6.hist(results['hours_held'], bins=50, alpha=0.7, color='blue', edgecolor='black')
ax6.axvline(x=results['hours_held'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {results["hours_held"].mean():.1f}h')
ax6.axvline(x=results['hours_held'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Median: {results["hours_held"].median():.1f}h')
ax6.set_title('6. Holding Time Distribution', fontsize=14, fontweight='bold')
ax6.set_xlabel('Hours Held')
ax6.set_ylabel('Frequency')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# 7. PnL 분포
ax7 = plt.subplot(6, 2, 7)
ax7.hist(results['pnl_pct'], bins=100, alpha=0.7, color='purple', edgecolor='black')
ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax7.axvline(x=results['pnl_pct'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {results["pnl_pct"].mean():.3f}%')
ax7.axvline(x=results['pnl_pct'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Median: {results["pnl_pct"].median():.3f}%')
ax7.set_title('7. PnL Distribution', fontsize=14, fontweight='bold')
ax7.set_xlabel('PnL (%)')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# 8. 승률 및 손익비
ax8 = plt.subplot(6, 2, 8)
wins = results[results['pnl_pct'] > 0]
losses = results[results['pnl_pct'] < 0]
win_rate = len(wins) / len(results) * 100
avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
profit_factor = abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()) if len(losses) > 0 else 0

stats_data = {
    'Win Rate': [win_rate, 100-win_rate],
    'Avg Win': [avg_win, 0],
    'Avg Loss': [0, abs(avg_loss)]
}
x = np.arange(2)
width = 0.25
colors = ['green', 'red']

ax8.bar(x - width, [win_rate, 100-win_rate], width, label='Win/Loss Rate', color=colors)
ax8.bar(x, [avg_win, 0], width, label='Avg Win', color='lightgreen')
ax8.bar(x + width, [0, abs(avg_loss)], width, label='Avg Loss', color='lightcoral')

ax8.set_xticks(x)
ax8.set_xticklabels(['Win', 'Loss'])
ax8.set_title(f'8. Win Rate & Profit Factor\nWin Rate: {win_rate:.1f}% | PF: {profit_factor:.2f}', 
              fontsize=14, fontweight='bold')
ax8.set_ylabel('Value')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. 월별 거래 수
ax9 = plt.subplot(6, 2, 9)
results['month'] = results['entry_time'].dt.to_period('M')
monthly_trades = results.groupby('month').size()
monthly_trades.plot(kind='bar', ax=ax9, color='steelblue', alpha=0.7)
ax9.set_title('9. Monthly Trade Count', fontsize=14, fontweight='bold')
ax9.set_xlabel('Month')
ax9.set_ylabel('Number of Trades')
ax9.grid(True, alpha=0.3, axis='y')
ax9.tick_params(axis='x', rotation=45, labelsize=7)

# 10. 월별 수익률
ax10 = plt.subplot(6, 2, 10)
monthly_pnl = results.groupby('month')['pnl_pct'].sum()
colors_monthly = ['green' if x > 0 else 'red' for x in monthly_pnl.values]
monthly_pnl.plot(kind='bar', ax=ax10, color=colors_monthly, alpha=0.7)
ax10.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax10.set_title('10. Monthly Total PnL', fontsize=14, fontweight='bold')
ax10.set_xlabel('Month')
ax10.set_ylabel('Total PnL (%)')
ax10.grid(True, alpha=0.3, axis='y')
ax10.tick_params(axis='x', rotation=45, labelsize=7)

# 11. 연속 손실/이익 분석
ax11 = plt.subplot(6, 2, 11)
results['is_win'] = results['pnl_pct'] > 0
streak = []
current_streak = 0
streak_type = None

for win in results['is_win']:
    if streak_type is None:
        streak_type = win
        current_streak = 1
    elif win == streak_type:
        current_streak += 1
    else:
        streak.append(current_streak if streak_type else -current_streak)
        streak_type = win
        current_streak = 1

win_streaks = [s for s in streak if s > 0]
loss_streaks = [abs(s) for s in streak if s < 0]

ax11.hist([win_streaks, loss_streaks], bins=range(1, 11), label=['Win Streaks', 'Loss Streaks'],
          color=['green', 'red'], alpha=0.7, edgecolor='black')
ax11.set_title(f'11. Win/Loss Streaks\nMax Win: {max(win_streaks) if win_streaks else 0} | Max Loss: {max(loss_streaks) if loss_streaks else 0}',
               fontsize=14, fontweight='bold')
ax11.set_xlabel('Streak Length')
ax11.set_ylabel('Frequency')
ax11.legend()
ax11.grid(True, alpha=0.3, axis='y')

# 12. 문제점 요약
ax12 = plt.subplot(6, 2, 12)
ax12.axis('off')

# 주요 문제점 텍스트
problems = f"""
MAJOR ISSUES IDENTIFIED:

1. HIGH SL RATE: {(results['exit_reason'] == 'SL').sum() / len(results) * 100:.1f}%
   - {(results['exit_reason'] == 'SL').sum()} trades out of {len(results)}
   - Total loss: {results[results['exit_reason'] == 'SL']['pnl_pct'].sum():.1f}%
   
2. POOR RECENT PERFORMANCE:
   - 2020-2021: +{results[results['year'].isin([2020, 2021])]['pnl_pct'].sum():.1f}%
   - 2022-2025: {results[results['year'] >= 2022]['pnl_pct'].sum():.1f}%
   
3. LOW WIN RATE: {win_rate:.1f}%
   - Worse than random (50%)
   - Need better entry filters
   
4. SMALL PROFIT FACTOR: {profit_factor:.2f}
   - Avg Win: +{avg_win:.3f}%
   - Avg Loss: {avg_loss:.3f}%
   - Winners barely cover losers
   
5. STRATEGY DETERIORATION:
   - Works well in bull market (2021)
   - Fails in sideways/bear (2022-2025)
   - Need market regime filter

CONCLUSION:
Strategy needs major improvements!
"""

ax12.text(0.05, 0.95, problems, transform=ax12.transAxes, fontsize=11,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('backtest_issues_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ 그래프 저장: backtest_issues_analysis.png")

# 추가: 간단한 요약 차트
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# 상단 좌측: 주요 지표
ax = axes[0, 0]
ax.axis('off')
summary_text = f"""
BACKTEST SUMMARY
{'='*50}

Period: {results['entry_time'].min().strftime('%Y-%m-%d')} to {results['exit_time'].max().strftime('%Y-%m-%d')}
Duration: 5.66 years

Total Trades: {len(results)}
Total PnL: {results['pnl_pct'].sum():.2f}%
Annual Return: {results['pnl_pct'].sum() / 5.66:.2f}%

Win Rate: {win_rate:.2f}%
Profit Factor: {profit_factor:.2f}

Exit Reasons:
  - SL: {(results['exit_reason'] == 'SL').sum()} ({(results['exit_reason'] == 'SL').sum()/len(results)*100:.1f}%)
  - TP2 Full: {(results['exit_reason'] == 'TP2_Full').sum()} ({(results['exit_reason'] == 'TP2_Full').sum()/len(results)*100:.1f}%)
  - TP1 BE: {(results['exit_reason'] == 'TP1_Breakeven').sum()} ({(results['exit_reason'] == 'TP1_Breakeven').sum()/len(results)*100:.1f}%)

Avg Holding Time: {results['hours_held'].mean():.1f} hours
"""
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=13,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.set_title('STRATEGY OVERVIEW', fontsize=16, fontweight='bold', pad=20)

# 상단 우측: 연도별 성과
ax = axes[0, 1]
yearly_data = results.groupby('year').agg({
    'pnl_pct': 'sum',
    'entry_time': 'count'
}).rename(columns={'entry_time': 'trades'})

x = np.arange(len(yearly_data))
width = 0.35

ax.bar(x - width/2, yearly_data['pnl_pct'], width, label='Total PnL (%)',
       color=['green' if p > 0 else 'red' for p in yearly_data['pnl_pct']], alpha=0.8)
ax2 = ax.twinx()
ax2.plot(x, yearly_data['trades'], 'b-o', linewidth=2, markersize=8, label='# Trades')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total PnL (%)', fontsize=12, color='black')
ax2.set_ylabel('Number of Trades', fontsize=12, color='blue')
ax.set_xticks(x)
ax.set_xticklabels(yearly_data.index)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.set_title('YEARLY PERFORMANCE', fontsize=14, fontweight='bold')

# 하단 좌측: 누적 수익 곡선
ax = axes[1, 0]
ax.plot(results['entry_time'], results['cumulative_pnl'], linewidth=2.5, color='darkblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.fill_between(results['entry_time'], 0, results['cumulative_pnl'],
                 where=(results['cumulative_pnl'] >= 0), alpha=0.3, color='green', label='Profit')
ax.fill_between(results['entry_time'], 0, results['cumulative_pnl'],
                 where=(results['cumulative_pnl'] < 0), alpha=0.3, color='red', label='Loss')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative PnL (%)', fontsize=12)
ax.set_title('CUMULATIVE PERFORMANCE', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# 하단 우측: 주요 문제점
ax = axes[1, 1]
ax.axis('off')

problem_text = f"""
KEY PROBLEMS:
{'='*50}

1. EXCESSIVE STOP LOSSES (65.4%)
   → Need better entry timing
   → SL placement too tight
   
2. STRATEGY DECAY OVER TIME
   → 2021: +47.4% (best year)
   → 2025: -17.9% (worst year)
   → Market changed, strategy didn't adapt
   
3. LOW WIN RATE (35.8%)
   → Poor entry signals
   → Need additional filters
   
4. SMALL AVERAGE WIN
   → Avg Win: +{avg_win:.2f}%
   → Avg Loss: {avg_loss:.2f}%
   → Risk/Reward imbalance

RECOMMENDATIONS:
• Tighten entry filters (reduce SL rate)
• Add market regime detection
• Adjust TP/SL ratios
• Consider position sizing
• Add volatility filters
"""

ax.text(0.05, 0.95, problem_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('ISSUES & RECOMMENDATIONS', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('backtest_summary.png', dpi=150, bbox_inches='tight')
print("✅ 요약 차트 저장: backtest_summary.png")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
