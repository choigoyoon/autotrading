import pandas as pd
import numpy as np

# Load the original backtest results
print("Loading original backtest results (SL -0.5%)...")
df_original = pd.read_csv('backtest_complete_results.csv')

print(f"Original results: {len(df_original)} trades")
print()

# Calculate what would happen with different SL levels
# We need to check for each trade: would it survive with -1.0% or -1.5% SL?

print("=" * 80)
print("📊 SL 변경 시뮬레이션 (기존 백테스트 데이터 기반)")
print("=" * 80)
print()

# Assumptions:
# - If a trade hit SL at -0.5%, we need to check if it would survive at -1.0% or -1.5%
# - For simplicity, we'll use a conservative estimate:
#   * Trades that hit SL will be split into:
#     - 50% would still hit SL at -1.0% (those that dropped more)
#     - 50% would survive and potentially hit TP1/TP2 (those that only dropped -0.5% to -1.0%)

# Original stats
sl_count = len(df_original[df_original['exit_reason'] == 'SL'])
tp1_count = len(df_original[df_original['exit_reason'] == 'TP1_Breakeven'])
tp2_count = len(df_original[df_original['exit_reason'] == 'TP2_Full'])
total = len(df_original)

original_total_pnl = df_original['pnl_pct'].sum()
original_avg_pnl = df_original['pnl_pct'].mean()

print("Original (SL -0.5%):")
print(f"  총 거래: {total}개")
print(f"  SL: {sl_count}개 ({sl_count/total*100:.1f}%)")
print(f"  TP1_Breakeven: {tp1_count}개 ({tp1_count/total*100:.1f}%)")
print(f"  TP2_Full: {tp2_count}개 ({tp2_count/total*100:.1f}%)")
print(f"  총 PnL: {original_total_pnl:.2f}%")
print(f"  평균 PnL: {original_avg_pnl:.2f}%")
print()

# Estimate for SL -1.0%
# Conservative: 40% of SL trades survive and reach TP1 (25% reach TP2)
sl_survivors_10 = int(sl_count * 0.4)
new_sl_count_10 = sl_count - sl_survivors_10
new_tp1_count_10 = tp1_count + int(sl_survivors_10 * 0.6)
new_tp2_count_10 = tp2_count + int(sl_survivors_10 * 0.4)

# PnL calculation
# SL trades that now reach TP1: change from -0.5% to TP1 average
# SL trades that now reach TP2: change from -0.5% to TP2 average
tp1_avg_pnl = df_original[df_original['exit_reason'] == 'TP1_Breakeven']['pnl_pct'].mean()
tp2_avg_pnl = df_original[df_original['exit_reason'] == 'TP2_Full']['pnl_pct'].mean()

# Original SL loss
original_sl_loss = -0.5 * sl_count

# New SL -1.0%
new_sl_loss_10 = -1.0 * new_sl_count_10
new_tp1_gain_10 = tp1_avg_pnl * int(sl_survivors_10 * 0.6)
new_tp2_gain_10 = tp2_avg_pnl * int(sl_survivors_10 * 0.4)

# Keep original TP1 and TP2 gains
original_tp1_gain = df_original[df_original['exit_reason'] == 'TP1_Breakeven']['pnl_pct'].sum()
original_tp2_gain = df_original[df_original['exit_reason'] == 'TP2_Full']['pnl_pct'].sum()

new_total_pnl_10 = new_sl_loss_10 + original_tp1_gain + original_tp2_gain + new_tp1_gain_10 + new_tp2_gain_10
new_avg_pnl_10 = new_total_pnl_10 / total
new_win_rate_10 = (new_tp1_count_10 + new_tp2_count_10) / total * 100

print("Estimated (SL -1.0%):")
print(f"  총 거래: {total}개")
print(f"  SL: {new_sl_count_10}개 ({new_sl_count_10/total*100:.1f}%) - 개선: {sl_count - new_sl_count_10}개 적음")
print(f"  TP1_Breakeven: {new_tp1_count_10}개 ({new_tp1_count_10/total*100:.1f}%)")
print(f"  TP2_Full: {new_tp2_count_10}개 ({new_tp2_count_10/total*100:.1f}%)")
print(f"  총 PnL: {new_total_pnl_10:.2f}% (개선: {new_total_pnl_10 - original_total_pnl:+.2f}%p)")
print(f"  평균 PnL: {new_avg_pnl_10:.2f}% (개선: {new_avg_pnl_10 - original_avg_pnl:+.2f}%p)")
print(f"  승률: {new_win_rate_10:.1f}%")
print()

# Estimate for SL -1.5%
# More aggressive: 60% of SL trades survive
sl_survivors_15 = int(sl_count * 0.6)
new_sl_count_15 = sl_count - sl_survivors_15
new_tp1_count_15 = tp1_count + int(sl_survivors_15 * 0.6)
new_tp2_count_15 = tp2_count + int(sl_survivors_15 * 0.4)

new_sl_loss_15 = -1.5 * new_sl_count_15
new_tp1_gain_15 = tp1_avg_pnl * int(sl_survivors_15 * 0.6)
new_tp2_gain_15 = tp2_avg_pnl * int(sl_survivors_15 * 0.4)

new_total_pnl_15 = new_sl_loss_15 + original_tp1_gain + original_tp2_gain + new_tp1_gain_15 + new_tp2_gain_15
new_avg_pnl_15 = new_total_pnl_15 / total
new_win_rate_15 = (new_tp1_count_15 + new_tp2_count_15) / total * 100

print("Estimated (SL -1.5%):")
print(f"  총 거래: {total}개")
print(f"  SL: {new_sl_count_15}개 ({new_sl_count_15/total*100:.1f}%) - 개선: {sl_count - new_sl_count_15}개 적음")
print(f"  TP1_Breakeven: {new_tp1_count_15}개 ({new_tp1_count_15/total*100:.1f}%)")
print(f"  TP2_Full: {new_tp2_count_15}개 ({new_tp2_count_15/total*100:.1f}%)")
print(f"  총 PnL: {new_total_pnl_15:.2f}% (개선: {new_total_pnl_15 - original_total_pnl:+.2f}%p)")
print(f"  평균 PnL: {new_avg_pnl_15:.2f}% (개선: {new_avg_pnl_15 - original_avg_pnl:+.2f}%p)")
print(f"  승률: {new_win_rate_15:.1f}%")
print()

# Comparison table
print("=" * 80)
print("📈 비교표")
print("=" * 80)
print()

comparison = pd.DataFrame([
    {
        'SL 설정': 'H3 - 0.5%',
        '총 거래': total,
        '총 PnL': f"{original_total_pnl:.2f}%",
        '평균 PnL': f"{original_avg_pnl:.2f}%",
        '승률': f"{(tp1_count + tp2_count)/total*100:.1f}%",
        'SL 비율': f"{sl_count/total*100:.1f}%",
        'TP2 비율': f"{tp2_count/total*100:.1f}%"
    },
    {
        'SL 설정': 'H3 - 1.0%',
        '총 거래': total,
        '총 PnL': f"{new_total_pnl_10:.2f}%",
        '평균 PnL': f"{new_avg_pnl_10:.2f}%",
        '승률': f"{new_win_rate_10:.1f}%",
        'SL 비율': f"{new_sl_count_10/total*100:.1f}%",
        'TP2 비율': f"{new_tp2_count_10/total*100:.1f}%"
    },
    {
        'SL 설정': 'H3 - 1.5%',
        '총 거래': total,
        '총 PnL': f"{new_total_pnl_15:.2f}%",
        '평균 PnL': f"{new_avg_pnl_15:.2f}%",
        '승률': f"{new_win_rate_15:.1f}%",
        'SL 비율': f"{new_sl_count_15/total*100:.1f}%",
        'TP2 비율': f"{new_tp2_count_15/total*100:.1f}%"
    }
])

print(comparison.to_string(index=False))
print()

# Final recommendation
print("=" * 80)
print("🎯 최종 권장사항")
print("=" * 80)
print()

if new_total_pnl_10 > 0 and new_total_pnl_10 > new_total_pnl_15:
    print("✅ 최적 SL 설정: **H3 - 1.0%**")
    print(f"   - 총 PnL: {new_total_pnl_10:.2f}% (개선: {new_total_pnl_10 - original_total_pnl:+.2f}%p)")
    print(f"   - SL 비율: {new_sl_count_10/total*100:.1f}% (개선: {(sl_count - new_sl_count_10)/total*100:.1f}%p)")
    print(f"   - 승률: {new_win_rate_10:.1f}%")
elif new_total_pnl_15 > 0:
    print("✅ 최적 SL 설정: **H3 - 1.5%**")
    print(f"   - 총 PnL: {new_total_pnl_15:.2f}% (개선: {new_total_pnl_15 - original_total_pnl:+.2f}%p)")
    print(f"   - SL 비율: {new_sl_count_15/total*100:.1f}% (개선: {(sl_count - new_sl_count_15)/total*100:.1f}%p)")
    print(f"   - 승률: {new_win_rate_15:.1f}%")
else:
    print("⚠️ 두 옵션 모두 수익 개선 실패")
    print("   다른 전략 개선 필요 (TP 타이밍, 진입 필터 등)")

print()
print("=" * 80)
print("💡 주의사항")
print("=" * 80)
print()
print("이 시뮬레이션은 보수적 추정치입니다.")
print("실제 결과는 다음 요인에 따라 달라질 수 있습니다:")
print("  - SL 터진 거래들의 실제 최대 낙폭")
print("  - 시장 변동성 및 노이즈")
print("  - TP 도달 타이밍 변화")
print()
print("실전 적용 전 정밀 백테스트 권장!")

