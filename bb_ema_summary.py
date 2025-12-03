#!/usr/bin/env python3
"""
BB + EMA 전략 최종 요약
"""

import pandas as pd

print("=" * 80)
print("BB 중간선 익절 + EMA 필터 최종 요약")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('bb_ema_all_signals.csv')
print(f"\n전체 신호: {len(df)}건")

# ============================================================
# 핵심 비교: LONG vs SHORT
# ============================================================
print("\n" + "=" * 80)
print("★ 방향별 성과 비교 ★")
print("=" * 80)

print(f"\n{'조건':>50} {'건수':>6} {'PnL':>8} {'승률':>8} {'손절률':>8} {'익절률':>8}")
print("-" * 100)

# 전체
print(f"{'[전체]':>50} {len(df):>6} {df['pnl'].mean():>8.2f}% {(df['pnl'] > 0).mean() * 100:>8.1f}% {(df['exit_reason'] == 'STOP_LOSS').mean() * 100:>8.1f}% {(df['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}%")

# LONG 전체
long_all = df[df['direction'] == 'LONG']
print(f"{'LONG 전체':>50} {len(long_all):>6} {long_all['pnl'].mean():>8.2f}% {(long_all['pnl'] > 0).mean() * 100:>8.1f}% {(long_all['exit_reason'] == 'STOP_LOSS').mean() * 100:>8.1f}% {(long_all['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}%")

# SHORT 전체
short_all = df[df['direction'] == 'SHORT']
print(f"{'SHORT 전체':>50} {len(short_all):>6} {short_all['pnl'].mean():>8.2f}% {(short_all['pnl'] > 0).mean() * 100:>8.1f}% {(short_all['exit_reason'] == 'STOP_LOSS').mean() * 100:>8.1f}% {(short_all['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}%")

print()

# EMA200 트렌드 필터
# LONG + EMA200 위
long_trend = df[(df['direction'] == 'LONG') & (df['close'] > df['ema_200'])]
print(f"{'LONG + EMA200 위 (상승트렌드)':>50} {len(long_trend):>6} {long_trend['pnl'].mean():>8.2f}% {(long_trend['pnl'] > 0).mean() * 100:>8.1f}% {(long_trend['exit_reason'] == 'STOP_LOSS').mean() * 100:>8.1f}% {(long_trend['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}%")

# SHORT + EMA200 아래
short_trend = df[(df['direction'] == 'SHORT') & (df['close'] < df['ema_200'])]
print(f"{'SHORT + EMA200 아래 (하락트렌드)':>50} {len(short_trend):>6} {short_trend['pnl'].mean():>8.2f}% {(short_trend['pnl'] > 0).mean() * 100:>8.1f}% {(short_trend['exit_reason'] == 'STOP_LOSS').mean() * 100:>8.1f}% {(short_trend['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}%")

# ============================================================
# 수축 기간 50봉 이상 + EMA200 트렌드
# ============================================================
print("\n" + "=" * 80)
print("★ 최적 조건: 수축 50봉 이상 + EMA200 트렌드 ★")
print("=" * 80)

# 수축 50봉 이상
squeeze_50 = df[df['squeeze_duration'] >= 50]
print(f"\n수축 50봉 이상 전체: {len(squeeze_50)}건")
print(f"  평균 PnL: {squeeze_50['pnl'].mean():.2f}%")
print(f"  승률: {(squeeze_50['pnl'] > 0).mean() * 100:.1f}%")
print(f"  손절률: {(squeeze_50['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%")
print(f"  익절률: {(squeeze_50['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}%")

# 수축 50봉 이상 + EMA200 트렌드
squeeze_50_trend = squeeze_50[
    ((squeeze_50['direction'] == 'LONG') & (squeeze_50['close'] > squeeze_50['ema_200'])) |
    ((squeeze_50['direction'] == 'SHORT') & (squeeze_50['close'] < squeeze_50['ema_200']))
]
print(f"\n수축 50봉 이상 + EMA200 트렌드: {len(squeeze_50_trend)}건")
print(f"  평균 PnL: {squeeze_50_trend['pnl'].mean():.2f}%")
print(f"  승률: {(squeeze_50_trend['pnl'] > 0).mean() * 100:.1f}%")
print(f"  손절률: {(squeeze_50_trend['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%")
print(f"  익절률: {(squeeze_50_trend['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}%")

# 익절만
tp_only = squeeze_50_trend[squeeze_50_trend['exit_reason'] == 'MID_LINE_TP']
print(f"\n  익절 케이스: {len(tp_only)}건")
print(f"    평균 PnL: {tp_only['pnl'].mean():.2f}%")
print(f"    승률: {(tp_only['pnl'] > 0).mean() * 100:.1f}%")
print(f"    평균 보유: {tp_only['bars'].mean() * 15 / 60:.1f}시간")

# ============================================================
# 다양한 EMA 조합 테스트
# ============================================================
print("\n" + "=" * 80)
print("★ EMA 조합 테스트 (수축 50봉 이상) ★")
print("=" * 80)

print(f"\n{'EMA 조합':>40} {'건수':>6} {'PnL':>8} {'승률':>8} {'손절률':>8} {'익절률':>8}")
print("-" * 90)

for ema in ['ema_20', 'ema_50', 'ema_100', 'ema_200']:
    # 트렌드
    trend = squeeze_50[
        ((squeeze_50['direction'] == 'LONG') & (squeeze_50['close'] > squeeze_50[ema])) |
        ((squeeze_50['direction'] == 'SHORT') & (squeeze_50['close'] < squeeze_50[ema]))
    ]
    if len(trend) >= 10:
        print(f"{f'{ema.upper()} 트렌드':>40} {len(trend):>6} {trend['pnl'].mean():>8.2f}% {(trend['pnl'] > 0).mean() * 100:>8.1f}% {(trend['exit_reason'] == 'STOP_LOSS').mean() * 100:>8.1f}% {(trend['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}%")

# ============================================================
# LONG만 분석 (상승트렌드에서 LONG)
# ============================================================
print("\n" + "=" * 80)
print("★ LONG 전용 전략 (수축 50봉 이상 + EMA 트렌드) ★")
print("=" * 80)

print(f"\n{'조건':>50} {'건수':>6} {'PnL':>8} {'승률':>8} {'익절률':>8} {'익절PnL':>10}")
print("-" * 110)

for ema in ['ema_20', 'ema_50', 'ema_100', 'ema_200']:
    long_trend_sq50 = squeeze_50[(squeeze_50['direction'] == 'LONG') & (squeeze_50['close'] > squeeze_50[ema])]
    if len(long_trend_sq50) >= 5:
        tp = long_trend_sq50[long_trend_sq50['exit_reason'] == 'MID_LINE_TP']
        print(f"{f'LONG + {ema.upper()} 위 + 수축50+':>50} {len(long_trend_sq50):>6} {long_trend_sq50['pnl'].mean():>8.2f}% {(long_trend_sq50['pnl'] > 0).mean() * 100:>8.1f}% {(long_trend_sq50['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}% {tp['pnl'].mean() if len(tp) > 0 else 0:>10.2f}%")

# ============================================================
# SHORT만 분석 (하락트렌드에서 SHORT)
# ============================================================
print("\n" + "=" * 80)
print("★ SHORT 전용 전략 (수축 50봉 이상 + EMA 트렌드) ★")
print("=" * 80)

print(f"\n{'조건':>50} {'건수':>6} {'PnL':>8} {'승률':>8} {'익절률':>8} {'익절PnL':>10}")
print("-" * 110)

for ema in ['ema_20', 'ema_50', 'ema_100', 'ema_200']:
    short_trend_sq50 = squeeze_50[(squeeze_50['direction'] == 'SHORT') & (squeeze_50['close'] < squeeze_50[ema])]
    if len(short_trend_sq50) >= 5:
        tp = short_trend_sq50[short_trend_sq50['exit_reason'] == 'MID_LINE_TP']
        print(f"{f'SHORT + {ema.upper()} 아래 + 수축50+':>50} {len(short_trend_sq50):>6} {short_trend_sq50['pnl'].mean():>8.2f}% {(short_trend_sq50['pnl'] > 0).mean() * 100:>8.1f}% {(short_trend_sq50['exit_reason'] == 'MID_LINE_TP').mean() * 100:>8.1f}% {tp['pnl'].mean() if len(tp) > 0 else 0:>10.2f}%")

# ============================================================
# 최종 권장 전략
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최종 권장 전략 ★★★")
print("=" * 80)

# 최적: 수축 50봉 이상 + EMA200 트렌드
best = squeeze_50_trend.copy()
best_tp = best[best['exit_reason'] == 'MID_LINE_TP']

# 기준선
baseline = df.copy()

print(f"""
■ 기준선 (필터 없음):
  - 건수: {len(baseline)}건
  - 평균 PnL: {baseline['pnl'].mean():.2f}%
  - 승률: {(baseline['pnl'] > 0).mean() * 100:.1f}%
  - 손절률: {(baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%
  - 익절률: {(baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}%

■ 권장 전략 (수축 50봉+ & EMA200 트렌드):
  - 건수: {len(best)}건
  - 평균 PnL: {best['pnl'].mean():.2f}%
  - 승률: {(best['pnl'] > 0).mean() * 100:.1f}%
  - 손절률: {(best['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%
  - 익절률: {(best['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}%
  
■ 익절 케이스만:
  - 건수: {len(best_tp)}건 (전체의 {len(best_tp)/len(best)*100:.1f}%)
  - 평균 PnL: {best_tp['pnl'].mean():.2f}%
  - 승률: {(best_tp['pnl'] > 0).mean() * 100:.1f}%
  - 평균 보유: {best_tp['bars'].mean() * 15 / 60:.1f}시간

■ 개선 효과:
  - 손절률: {(baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}% → {(best['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}% ({(best['exit_reason'] == 'STOP_LOSS').mean() * 100 - (baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:+.1f}%p)
  - 익절률: {(baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}% → {(best['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}% ({(best['exit_reason'] == 'MID_LINE_TP').mean() * 100 - (baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100:+.1f}%p)

■ 전략 규칙:
  1. BB 수축 50봉 이상 후 확장 시점 진입
  2. LONG: 가격 > EMA200 & BB 상단 돌파
  3. SHORT: 가격 < EMA200 & BB 하단 돌파
  4. 익절: 중간선 도달 시
  5. 손절: 밴드폭의 50%
""")
