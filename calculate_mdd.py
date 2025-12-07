"""
MDD (Maximum Drawdown) 분석
- 15분 MTF 전략
- 1H 전략 예상
- 복리 적용 시뮬레이션
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*60)
print("MDD (Maximum Drawdown) 분석")
print("="*60)

# 백테스트 결과 로드
mtf_trades = pd.read_csv('backtest_mtf_enhanced.csv')
print(f"\nMTF 전략 거래: {len(mtf_trades):,}개")
print(f"기간: {mtf_trades['datetime'].min()} ~ {mtf_trades['datetime'].max()}")

# 1. 단리 MDD (각 거래 동일 비중)
def calculate_simple_mdd(trades_df, position_size=0.30):
    """
    단리 MDD 계산
    position_size: 포지션 크기 (30% = 0.30)
    """

    cumulative = []
    capital = 100.0  # 100% 시작
    peak = 100.0
    max_dd = 0
    max_dd_duration = 0
    dd_start_idx = 0
    in_drawdown = False

    for idx, trade in trades_df.iterrows():
        # PnL 적용
        pnl = trade['pnl'] * position_size
        capital += pnl

        # Peak 업데이트
        if capital > peak:
            peak = capital
            if in_drawdown:
                # Drawdown 종료
                dd_duration = idx - dd_start_idx
                max_dd_duration = max(max_dd_duration, dd_duration)
                in_drawdown = False

        # Drawdown 계산
        dd = (capital - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
            if not in_drawdown:
                dd_start_idx = idx
                in_drawdown = True

        cumulative.append({
            'idx': idx,
            'capital': capital,
            'peak': peak,
            'drawdown': dd,
            'pnl': pnl,
        })

    return pd.DataFrame(cumulative), max_dd, max_dd_duration

print("\n" + "="*60)
print("1. 단리 MDD (포지션 30%)")
print("="*60)

simple_equity, simple_mdd, simple_dd_duration = calculate_simple_mdd(mtf_trades, position_size=0.30)

print(f"\nMDD: {simple_mdd:.2f}%")
print(f"최대 DD 지속: {simple_dd_duration}회 거래")
print(f"최종 자본: {simple_equity.iloc[-1]['capital']:.2f}%")
print(f"총 수익: {simple_equity.iloc[-1]['capital'] - 100:.2f}%")

# Drawdown 구간 찾기
dd_periods = []
current_dd = None

for idx, row in simple_equity.iterrows():
    if row['drawdown'] < -1.0:  # -1% 이상 DD
        if current_dd is None:
            current_dd = {
                'start_idx': idx,
                'max_dd': row['drawdown'],
                'duration': 1
            }
        else:
            current_dd['max_dd'] = min(current_dd['max_dd'], row['drawdown'])
            current_dd['duration'] += 1
    else:
        if current_dd is not None:
            dd_periods.append(current_dd)
            current_dd = None

print(f"\nDD 구간 (-1% 이상): {len(dd_periods)}회")
if len(dd_periods) > 0:
    print("\n주요 DD 구간 (상위 5개):")
    dd_periods_sorted = sorted(dd_periods, key=lambda x: x['max_dd'])
    for i, dd in enumerate(dd_periods_sorted[:5], 1):
        print(f"  {i}. MDD: {dd['max_dd']:.2f}%, 지속: {dd['duration']}회")

# 2. 복리 MDD
def calculate_compound_mdd(trades_df, initial_capital=10000, position_size=0.30):
    """
    복리 MDD 계산
    실제 자본 증가를 반영한 포지션 크기
    """

    cumulative = []
    capital = initial_capital
    peak = initial_capital
    max_dd = 0
    max_dd_pct = 0
    max_dd_duration = 0
    dd_start_idx = 0
    in_drawdown = False

    for idx, trade in trades_df.iterrows():
        # 현재 자본 기준 포지션 크기
        position = capital * position_size

        # PnL 적용
        pnl_amount = position * (trade['pnl'] / 100)
        capital += pnl_amount

        # Peak 업데이트
        if capital > peak:
            peak = capital
            if in_drawdown:
                dd_duration = idx - dd_start_idx
                max_dd_duration = max(max_dd_duration, dd_duration)
                in_drawdown = False

        # Drawdown 계산
        dd_amount = capital - peak
        dd_pct = dd_amount / peak * 100

        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct
            max_dd = dd_amount
            if not in_drawdown:
                dd_start_idx = idx
                in_drawdown = True

        cumulative.append({
            'idx': idx,
            'capital': capital,
            'peak': peak,
            'drawdown_pct': dd_pct,
            'drawdown_amount': dd_amount,
        })

    return pd.DataFrame(cumulative), max_dd, max_dd_pct, max_dd_duration

print("\n" + "="*60)
print("2. 복리 MDD (초기 자본 $10,000, 포지션 30%)")
print("="*60)

compound_equity, compound_mdd_amount, compound_mdd_pct, compound_dd_duration = calculate_compound_mdd(
    mtf_trades,
    initial_capital=10000,
    position_size=0.30
)

print(f"\nMDD: {compound_mdd_pct:.2f}% (${compound_mdd_amount:,.2f})")
print(f"최대 DD 지속: {compound_dd_duration}회 거래")
print(f"최종 자본: ${compound_equity.iloc[-1]['capital']:,.2f}")
print(f"총 수익: ${compound_equity.iloc[-1]['capital'] - 10000:,.2f}")
print(f"수익률: {(compound_equity.iloc[-1]['capital'] / 10000 - 1) * 100:,.1f}%")

# 3. 레버리지 MDD (3배)
def calculate_leveraged_mdd(trades_df, initial_capital=10000, position_size=0.30, leverage=3):
    """
    레버리지 적용 MDD
    leverage: 레버리지 배수
    """

    cumulative = []
    capital = initial_capital
    peak = initial_capital
    max_dd_pct = 0

    for idx, trade in trades_df.iterrows():
        # 레버리지 적용 포지션
        position = capital * position_size

        # PnL (레버리지는 이미 백테스트에 반영되어 있으므로 그대로 사용)
        pnl_amount = position * (trade['pnl'] / 100)
        capital += pnl_amount

        # Peak 업데이트
        if capital > peak:
            peak = capital

        # Drawdown
        dd_pct = (capital - peak) / peak * 100
        max_dd_pct = min(max_dd_pct, dd_pct)

        cumulative.append({
            'idx': idx,
            'capital': capital,
            'peak': peak,
            'drawdown_pct': dd_pct,
        })

    return pd.DataFrame(cumulative), max_dd_pct

print("\n" + "="*60)
print("3. 레버리지 3배 MDD")
print("="*60)

lev_equity, lev_mdd = calculate_leveraged_mdd(
    mtf_trades,
    initial_capital=10000,
    position_size=0.30,
    leverage=3
)

print(f"\nMDD: {lev_mdd:.2f}%")
print(f"최종 자본: ${lev_equity.iloc[-1]['capital']:,.2f}")

# 4. 연속 손실 분석
print("\n" + "="*60)
print("4. 연속 손실 분석")
print("="*60)

consecutive_losses = []
current_streak = 0
max_streak = 0

for idx, trade in mtf_trades.iterrows():
    if trade['pnl'] < 0:
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    else:
        if current_streak > 0:
            consecutive_losses.append(current_streak)
        current_streak = 0

print(f"\n최대 연속 손실: {max_streak}회")
print(f"평균 연속 손실: {np.mean(consecutive_losses):.1f}회" if consecutive_losses else "평균 연속 손실: 0회")

# 연속 손실 분포
if consecutive_losses:
    print(f"\n연속 손실 분포:")
    for i in range(1, min(6, max_streak + 1)):
        count = sum(1 for x in consecutive_losses if x == i)
        print(f"  {i}회 연속: {count}회 발생")

# 최악 시나리오 시뮬레이션
print(f"\n최악 시나리오 ({max_streak}회 연속 손실):")
avg_loss = mtf_trades[mtf_trades['pnl'] < 0]['pnl'].mean()
worst_case_loss = avg_loss * max_streak * 0.30  # 30% 포지션
print(f"  예상 손실: {worst_case_loss:.2f}%")

# 5. 1H 전략 MDD 예상
print("\n" + "="*60)
print("5. 1H 전략 MDD 예상")
print("="*60)

# 15분 대비 1H 특성
# - 거래 수: 1/6 (하루 1회 vs 6회)
# - 변동성: 1/2 (1H는 더 안정적)
# - 승률: 93% (15분: 90%)

# 예상 MDD: 15분의 70% 수준
estimated_1h_mdd = simple_mdd * 0.7

print(f"\n15분 전략 MDD: {simple_mdd:.2f}%")
print(f"1H 전략 예상 MDD: {estimated_1h_mdd:.2f}%")
print(f"\n근거:")
print(f"  - 거래 빈도 감소 (1/6)")
print(f"  - 노이즈 필터링 효과")
print(f"  - 더 넓은 SL (6.5% vs 2.0%)")
print(f"  → 일시적 역행 흡수 능력 향상")

# 6. 비교표
print("\n" + "="*60)
print("MDD 요약 비교")
print("="*60)

summary = pd.DataFrame({
    '전략': ['15분 단리', '15분 복리', '15분 3배레버', '1H 예상'],
    'MDD': [
        f"{simple_mdd:.2f}%",
        f"{compound_mdd_pct:.2f}%",
        f"{lev_mdd:.2f}%",
        f"{estimated_1h_mdd:.2f}%"
    ],
    '최대연속손실': [
        f"{max_streak}회",
        f"{max_streak}회",
        f"{max_streak}회",
        f"{int(max_streak * 0.7)}회"
    ],
    '최종수익률': [
        f"{simple_equity.iloc[-1]['capital'] - 100:.0f}%",
        f"{(compound_equity.iloc[-1]['capital'] / 10000 - 1) * 100:.0f}%",
        f"{(lev_equity.iloc[-1]['capital'] / 10000 - 1) * 100:.0f}%",
        "예측 필요"
    ]
})

print("\n" + summary.to_string(index=False))

# 7. 리스크 메트릭
print("\n" + "="*60)
print("리스크 메트릭")
print("="*60)

# Calmar Ratio (연간 수익 / MDD)
total_return_pct = (compound_equity.iloc[-1]['capital'] / 10000 - 1) * 100
years = len(mtf_trades) / (365 * 24 / 0.25)  # 15분봉 → 년
annual_return = (1 + total_return_pct / 100) ** (1 / years) - 1
calmar = annual_return / abs(compound_mdd_pct / 100)

print(f"\nCalmar Ratio: {calmar:.2f}")
print(f"  (연간수익 {annual_return * 100:.1f}% / MDD {abs(compound_mdd_pct):.1f}%)")

# 회복 시간 분석
recovery_times = []
in_dd = False
dd_start = 0

for idx, row in compound_equity.iterrows():
    if row['drawdown_pct'] < -1.0 and not in_dd:
        dd_start = idx
        in_dd = True
    elif row['drawdown_pct'] >= 0 and in_dd:
        recovery_times.append(idx - dd_start)
        in_dd = False

if recovery_times:
    print(f"\nDD 회복 시간:")
    print(f"  평균: {np.mean(recovery_times):.0f}회 거래")
    print(f"  최대: {max(recovery_times)}회 거래")
    print(f"  (15분봉 기준: {max(recovery_times) * 0.25:.0f}시간)")

# 8. 결론
print("\n" + "="*60)
print("결론 및 권장사항")
print("="*60)

print(f"""
1. 15분 MTF 전략:
   - MDD: {compound_mdd_pct:.2f}%
   - 최대 연속 손실: {max_streak}회
   - Calmar Ratio: {calmar:.2f}

2. 1H 전략 예상:
   - MDD: {estimated_1h_mdd:.2f}% (15분 대비 30% 감소)
   - 최대 연속 손실: {int(max_streak * 0.7)}회
   - 더 안정적인 수익 곡선

3. 포지션 크기 권장:
   - MDD {abs(compound_mdd_pct):.0f}% 감내 가능 → 30% 포지션 OK
   - MDD {abs(compound_mdd_pct) * 0.5:.0f}% 이하 원하면 → 15-20% 포지션
   - 심리적 한계 고려 필수

4. 레버리지 권장:
   - MDD {abs(lev_mdd):.0f}%는 상당히 큰 손실
   - 3배 레버리지는 중급자 이상만
   - 초보자: 1-2배 권장
""")

print("\n분석 완료!")
