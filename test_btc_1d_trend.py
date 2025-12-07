"""
BTC 1일봉 추세 추종 vs 스윙 비교
"""

import pandas as pd
import sys

# 기존 BTC 1D 데이터 로드
try:
    df = pd.read_csv('btc_1d_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
except:
    print("⚠️ btc_1d_data.csv 없음")
    sys.exit(1)

# doge_1d_strategy 함수들 임포트
exec(open('doge_1d_strategy.py').read().split('if __name__')[0])

print("=" * 70)
print("BTC 1일봉: 추세 추종 vs 스윙 트레이딩 비교")
print("=" * 70)

# MACD 계산
df = calculate_macd(df)
df = label_hl(df)

print(f"\n데이터: {len(df)}개 캔들")
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

# 추세선 생성
trendlines = generate_trendlines(df, min_touches=2)
print(f"추세선: {len(trendlines)}개")

# 돌파 감지
breakouts = detect_breakouts(df, trendlines)
print(f"돌파: {len(breakouts)}개")

# 10일 필터
filtered = filter_consecutive_signals(breakouts, min_interval=10)
print(f"필터 후: {len(filtered)}개")

# ═══════════════════════════════════════════════════════════
# 모드 1: 추세 추종
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("모드 1: 추세 추종 (트레일링 스탑 + MACD 반전)")
print("=" * 70)

trades_trend = backtest(df, filtered, mode='TREND')

if len(trades_trend) > 0:
    win_rate = (trades_trend['pnl_after_fee'] > 0).sum() / len(trades_trend) * 100
    avg_pnl = trades_trend['pnl_after_fee'].mean()

    # 복리
    balance = 1000
    for pnl in trades_trend['pnl_after_fee']:
        balance *= (1 + pnl / 100)

    total_return = (balance - 1000) / 1000 * 100

    print(f"\n총 거래: {len(trades_trend)}개")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 PnL: {avg_pnl:.2f}% (수수료 후)")
    print(f"\n출구 타입:")
    for exit_type in trades_trend['exit_type'].unique():
        count = (trades_trend['exit_type'] == exit_type).sum()
        print(f"  {exit_type}: {count}개")

    print(f"\n최종 잔고: ${balance:,.2f}")
    print(f"총 수익: {total_return:.1f}%")

    # MDD
    cumulative = (1 + trades_trend['pnl_after_fee'] / 100).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak * 100
    mdd = drawdown.min()
    print(f"MDD: {mdd:.2f}%")

    # 최대 수익 거래
    max_trade = trades_trend.loc[trades_trend['pnl_after_fee'].idxmax()]
    print(f"\n최대 수익 거래:")
    print(f"  진입: {max_trade['entry_date']}, ${max_trade['entry_price']:,.2f}")
    print(f"  청산: {max_trade['exit_date']}, ${max_trade['exit_price']:,.2f}")
    print(f"  수익: {max_trade['pnl_after_fee']:.2f}%")
    print(f"  출구: {max_trade['exit_type']}")

# ═══════════════════════════════════════════════════════════
# 모드 2: 고정 TP/SL (50%/10%)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("모드 2: 고정 TP/SL (50%/10%)")
print("=" * 70)

trades_fixed = backtest(df, filtered, mode='FIXED')

if len(trades_fixed) > 0:
    win_rate = (trades_fixed['pnl_after_fee'] > 0).sum() / len(trades_fixed) * 100
    avg_pnl = trades_fixed['pnl_after_fee'].mean()

    # 복리
    balance = 1000
    for pnl in trades_fixed['pnl_after_fee']:
        balance *= (1 + pnl / 100)

    total_return = (balance - 1000) / 1000 * 100

    print(f"\n총 거래: {len(trades_fixed)}개")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 PnL: {avg_pnl:.2f}% (수수료 후)")
    print(f"\n출구 타입:")
    for exit_type in trades_fixed['exit_type'].unique():
        count = (trades_fixed['exit_type'] == exit_type).sum()
        print(f"  {exit_type}: {count}개")

    print(f"\n최종 잔고: ${balance:,.2f}")
    print(f"총 수익: {total_return:.1f}%")

    # MDD
    cumulative = (1 + trades_fixed['pnl_after_fee'] / 100).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak * 100
    mdd = drawdown.min()
    print(f"MDD: {mdd:.2f}%")

# ═══════════════════════════════════════════════════════════
# 비교
# ═══════════════════════════════════════════════════════════

if len(trades_trend) > 0 and len(trades_fixed) > 0:
    print("\n" + "=" * 70)
    print("비교 결과")
    print("=" * 70)

    trend_balance = 1000
    for pnl in trades_trend['pnl_after_fee']:
        trend_balance *= (1 + pnl / 100)

    fixed_balance = 1000
    for pnl in trades_fixed['pnl_after_fee']:
        fixed_balance *= (1 + pnl / 100)

    trend_return = (trend_balance - 1000) / 1000 * 100
    fixed_return = (fixed_balance - 1000) / 1000 * 100

    print(f"\n추세 추종:")
    print(f"  승률: {(trades_trend['pnl_after_fee'] > 0).sum() / len(trades_trend) * 100:.1f}%")
    print(f"  평균: {trades_trend['pnl_after_fee'].mean():.2f}%")
    print(f"  총 수익: {trend_return:.1f}%")

    print(f"\n고정 TP/SL:")
    print(f"  승률: {(trades_fixed['pnl_after_fee'] > 0).sum() / len(trades_fixed) * 100:.1f}%")
    print(f"  평균: {trades_fixed['pnl_after_fee'].mean():.2f}%")
    print(f"  총 수익: {fixed_return:.1f}%")

    diff = trend_return - fixed_return
    print(f"\n차이: {diff:+.1f}%p")

    if trend_return > fixed_return:
        print(f"\n✅ 추세 추종이 {abs(diff):.1f}%p 더 좋음!")
    else:
        print(f"\n✅ 고정 TP/SL이 {abs(diff):.1f}%p 더 좋음!")

print("\n" + "=" * 70)
