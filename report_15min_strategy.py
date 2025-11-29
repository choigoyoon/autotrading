"""
BTC 15분봉 전략 최종 성과 보고서
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("BTC 15분봉 추세선 돌파 전략 - 최종 성과 보고서")
print("=" * 70)

# 데이터 로드
df_trades = pd.read_csv('backtest_filtered_10bars.csv')
df_trades['datetime'] = pd.to_datetime(df_trades['datetime'])

# 기본 정보
print(f"\n📊 기본 정보")
print(f"{'='*70}")
print(f"기간: {df_trades['datetime'].min()} ~ {df_trades['datetime'].max()}")

total_days = (df_trades['datetime'].max() - df_trades['datetime'].min()).days
total_years = total_days / 365
total_months = total_years * 12

print(f"총 일수: {total_days:,}일")
print(f"총 기간: {total_years:.2f}년 ({total_months:.1f}개월)")

# 거래 통계
print(f"\n📈 거래 통계")
print(f"{'='*70}")

total_trades = len(df_trades)
wins = (df_trades['pnl'] > 0).sum()
losses = (df_trades['pnl'] <= 0).sum()
win_rate = wins / total_trades * 100

print(f"총 거래: {total_trades:,}개")
print(f"승리: {wins:,}개")
print(f"손실: {losses:,}개")
print(f"승률: {win_rate:.1f}%")

# 월별/일별 거래 빈도
trades_per_month = total_trades / total_months
trades_per_day = total_trades / total_days

print(f"\n월 평균 거래: {trades_per_month:.1f}회")
print(f"일 평균 거래: {trades_per_day:.2f}회")

# 수익률 분석
print(f"\n💰 수익률 분석")
print(f"{'='*70}")

avg_pnl = df_trades['pnl'].mean()
avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean()

print(f"평균 PnL: {avg_pnl:.3f}%")
print(f"평균 승리: {avg_win:.3f}%")
print(f"평균 손실: {avg_loss:.3f}%")

max_win = df_trades['pnl'].max()
max_loss = df_trades['pnl'].min()

print(f"\n최대 승리: {max_win:.3f}%")
print(f"최대 손실: {max_loss:.3f}%")

# 단리 수익
print(f"\n💵 단리 수익 (레버리지 없음)")
print(f"{'='*70}")

total_pnl_simple = df_trades['pnl'].sum()
monthly_pnl_simple = total_pnl_simple / total_months
annual_pnl_simple = monthly_pnl_simple * 12

print(f"총 수익: {total_pnl_simple:.2f}%")
print(f"월 평균 수익: {monthly_pnl_simple:.2f}%")
print(f"연 평균 수익: {annual_pnl_simple:.2f}%")

# 복리 수익
print(f"\n💎 복리 수익 (레버리지 없음)")
print(f"{'='*70}")

balance = 10000  # 초기 $10,000
for pnl in df_trades['pnl']:
    balance *= (1 + pnl / 100)

total_return = (balance - 10000) / 10000 * 100
annual_return_compound = ((balance / 10000) ** (1 / total_years) - 1) * 100
monthly_return_compound = ((balance / 10000) ** (1 / total_months) - 1) * 100

print(f"초기 자본: $10,000")
print(f"최종 잔고: ${balance:,.2f}")
print(f"총 수익: {total_return:,.1f}%")
print(f"연 복리 수익: {annual_return_compound:.2f}%")
print(f"월 복리 수익: {monthly_return_compound:.2f}%")

# 레버리지 3배 (30% 포지션)
print(f"\n🚀 복리 수익 (레버리지 3배, 30% 포지션)")
print(f"{'='*70}")

balance_lev = 10000
for pnl in df_trades['pnl']:
    # 30% 포지션에 3배 레버리지
    leveraged_pnl = pnl * 3 * 0.3
    balance_lev *= (1 + leveraged_pnl / 100)

total_return_lev = (balance_lev - 10000) / 10000 * 100
annual_return_lev = ((balance_lev / 10000) ** (1 / total_years) - 1) * 100
monthly_return_lev = ((balance_lev / 10000) ** (1 / total_months) - 1) * 100

print(f"초기 자본: $10,000")
print(f"최종 잔고: ${balance_lev:,.2f}")
print(f"총 수익: {total_return_lev:,.1f}%")
print(f"연 복리 수익: {annual_return_lev:.2f}%")
print(f"월 복리 수익: {monthly_return_lev:.2f}%")

# MDD 계산
print(f"\n📉 리스크 분석")
print(f"{'='*70}")

cumulative = (1 + df_trades['pnl'] / 100).cumprod()
peak = cumulative.expanding(min_periods=1).max()
drawdown = (cumulative - peak) / peak * 100
mdd = drawdown.min()

# 최대 연속 손실
consecutive_losses = 0
max_consecutive_losses = 0
for pnl in df_trades['pnl']:
    if pnl <= 0:
        consecutive_losses += 1
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    else:
        consecutive_losses = 0

print(f"MDD (Maximum Drawdown): {mdd:.2f}%")
print(f"최대 연속 손실: {max_consecutive_losses}회")

# 연도별 성과
print(f"\n📅 연도별 성과")
print(f"{'='*70}")

df_trades['year'] = df_trades['datetime'].dt.year

for year in sorted(df_trades['year'].unique()):
    year_trades = df_trades[df_trades['year'] == year]
    year_count = len(year_trades)
    year_wins = (year_trades['pnl'] > 0).sum()
    year_win_rate = year_wins / year_count * 100
    year_avg_pnl = year_trades['pnl'].mean()
    year_total_pnl = year_trades['pnl'].sum()

    print(f"\n{year}년:")
    print(f"  거래: {year_count}개")
    print(f"  승률: {year_win_rate:.1f}%")
    print(f"  평균 PnL: {year_avg_pnl:+.3f}%")
    print(f"  총 수익: {year_total_pnl:+.2f}%")

# 요약 테이블
print(f"\n" + "=" * 70)
print("📋 핵심 지표 요약")
print(f"=" * 70)

print(f"""
기간: {total_years:.1f}년 ({total_months:.0f}개월)
총 거래: {total_trades:,}개
월 평균: {trades_per_month:.1f}회

승률: {win_rate:.1f}%
평균 PnL: {avg_pnl:.3f}%
MDD: {mdd:.2f}%

단리 (레버 없음):
  월 수익: {monthly_pnl_simple:.2f}%
  연 수익: {annual_pnl_simple:.2f}%

복리 (레버 없음):
  월 수익: {monthly_return_compound:.2f}%
  연 수익: {annual_return_compound:.2f}%
  $10,000 → ${balance:,.0f} ({total_return:,.0f}%)

복리 (레버 3배, 30% 포지션):
  월 수익: {monthly_return_lev:.2f}%
  연 수익: {annual_return_lev:.2f}%
  $10,000 → ${balance_lev:,.0f} ({total_return_lev:,.0f}%)
""")

print("=" * 70)
print("보고서 완료!")
print("=" * 70)
