"""
MDD 감소 전략 분석
- 드로다운 시기 분석
- 연속 손실 패턴
- 변동성 기반 필터
- 체크박스로 MDD 줄이기
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("MDD 감소 체크박스 분석")
print("=" * 60)

# 기존 거래 내역 로드
trades_df = pd.read_csv('backtest_exit_checkboxes_trades.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print(f"\n총 거래: {len(trades_df)}회")
print(f"현재 MDD: -33.83%")

# 1. 드로다운 기간 분석
print("\n" + "=" * 60)
print("1. 드로다운 기간 분석")
print("=" * 60)

equity = trades_df['balance_compound'].values
running_max = np.maximum.accumulate(equity)
drawdown = (equity - running_max) / running_max * 100

trades_df['equity'] = equity
trades_df['running_max'] = running_max
trades_df['drawdown'] = drawdown

# 최악의 드로다운 시기
max_dd_idx = drawdown.argmin()
max_dd = drawdown.min()

print(f"\n최대 드로다운: {max_dd:.2f}%")
print(f"발생 시점: {trades_df.iloc[max_dd_idx]['exit_time']}")
print(f"거래 번호: {max_dd_idx} / {len(trades_df)}")

# 드로다운 >10% 구간
high_dd_mask = drawdown < -10
high_dd_periods = trades_df[high_dd_mask]

print(f"\n드로다운 >10% 거래: {len(high_dd_periods)}회 ({len(high_dd_periods)/len(trades_df)*100:.1f}%)")

# 2. 연속 손실 패턴 분석
print("\n" + "=" * 60)
print("2. 연속 손실 패턴")
print("=" * 60)

trades_df['is_win'] = trades_df['pl_net'] > 0
trades_df['is_loss'] = trades_df['pl_net'] <= 0

# 연속 손실 계산
consecutive_losses = []
current_streak = 0

for i, row in trades_df.iterrows():
    if row['is_loss']:
        current_streak += 1
    else:
        if current_streak > 0:
            consecutive_losses.append(current_streak)
        current_streak = 0

if current_streak > 0:
    consecutive_losses.append(current_streak)

print(f"\n최대 연속 손실: {max(consecutive_losses)}회")
print(f"평균 연속 손실: {np.mean(consecutive_losses):.1f}회")
print(f"\n연속 손실 분포:")
print(f"  2회 이상: {sum(1 for x in consecutive_losses if x >= 2)}회")
print(f"  3회 이상: {sum(1 for x in consecutive_losses if x >= 3)}회")
print(f"  5회 이상: {sum(1 for x in consecutive_losses if x >= 5)}회")

# 3. 드로다운 중 거래 특성
print("\n" + "=" * 60)
print("3. 드로다운 중 거래 특성")
print("=" * 60)

# 드로다운 >5% vs 정상
dd_trades = trades_df[trades_df['drawdown'] < -5]
normal_trades = trades_df[trades_df['drawdown'] >= -5]

print(f"\n드로다운 중 (DD >5%):")
print(f"  거래: {len(dd_trades)}회")
print(f"  평균 수익: {dd_trades['pl_net'].mean():.3f}%")
print(f"  승률: {(dd_trades['pl_net'] > 0).sum() / len(dd_trades) * 100:.1f}%")

print(f"\n정상 기간:")
print(f"  거래: {len(normal_trades)}회")
print(f"  평균 수익: {normal_trades['pl_net'].mean():.3f}%")
print(f"  승률: {(normal_trades['pl_net'] > 0).sum() / len(normal_trades) * 100:.1f}%")

# 4. MDD 감소 체크박스 전략
print("\n" + "=" * 60)
print("4. MDD 감소 체크박스 테스트")
print("=" * 60)

strategies = [
    {
        'name': '기본 (현재)',
        'skip_on_dd': False,
        'skip_on_streak': 999,
        'reduce_size_on_dd': False,
    },
    {
        'name': '연속 손실 3회 → 스킵',
        'skip_on_dd': False,
        'skip_on_streak': 3,
        'reduce_size_on_dd': False,
    },
    {
        'name': '연속 손실 5회 → 스킵',
        'skip_on_dd': False,
        'skip_on_streak': 5,
        'reduce_size_on_dd': False,
    },
    {
        'name': 'DD >10% → 스킵',
        'skip_on_dd': 10,
        'skip_on_streak': 999,
        'reduce_size_on_dd': False,
    },
    {
        'name': 'DD >10% → 50% 포지션',
        'skip_on_dd': False,
        'skip_on_streak': 999,
        'reduce_size_on_dd': 10,
        'reduce_pct': 0.5,
    },
    {
        'name': '연속 3회 손실 → 스킵 + DD >10% → 50%',
        'skip_on_dd': False,
        'skip_on_streak': 3,
        'reduce_size_on_dd': 10,
        'reduce_pct': 0.5,
    },
]

results = []

for strategy in strategies:
    balance = 10000
    running_max_equity = 10000
    max_dd = 0
    trades_taken = 0
    trades_skipped = 0

    consecutive_loss_count = 0
    equity_curve = []

    for i, row in trades_df.iterrows():
        current_dd = (balance - running_max_equity) / running_max_equity * 100

        # 체크박스 1: 연속 손실 체크
        skip_streak = consecutive_loss_count >= strategy['skip_on_streak']

        # 체크박스 2: 드로다운 스킵
        skip_dd = strategy.get('skip_on_dd', False) and current_dd < -strategy['skip_on_dd']

        # 체크박스 3: 드로다운 시 포지션 축소
        reduce_dd = strategy.get('reduce_size_on_dd', False) and current_dd < -strategy['reduce_size_on_dd']

        if skip_streak or skip_dd:
            trades_skipped += 1
            # 연속 손실 카운트는 리셋하지 않음 (스킵하는 동안 유지)
            equity_curve.append(balance)
            continue

        # 거래 실행
        trades_taken += 1

        # 포지션 크기 조정
        if reduce_dd:
            position_size = 10000 * strategy['reduce_pct']
        else:
            position_size = 10000

        # 손익 적용 (복리)
        pnl = position_size * row['pl_net'] / 100
        balance += pnl

        # 연속 손실 카운트
        if row['pl_net'] <= 0:
            consecutive_loss_count += 1
        else:
            consecutive_loss_count = 0

        # MDD 계산
        if balance > running_max_equity:
            running_max_equity = balance

        current_dd = (balance - running_max_equity) / running_max_equity * 100
        if current_dd < max_dd:
            max_dd = current_dd

        equity_curve.append(balance)

    final_return = (balance - 10000) / 10000 * 100

    results.append({
        'strategy': strategy['name'],
        'trades': trades_taken,
        'skipped': trades_skipped,
        'final_balance': balance,
        'return': final_return,
        'mdd': max_dd,
    })

# 결과 출력
results_df = pd.DataFrame(results)

print("\n전략별 비교:")
print(results_df[['strategy', 'trades', 'skipped', 'return', 'mdd']].to_string(index=False))

# 최적 전략 (MDD 기준)
best_mdd = results_df.loc[results_df['mdd'].idxmax()]
best_return = results_df.loc[results_df['return'].idxmax()]

print("\n" + "=" * 60)
print("📊 최적 전략")
print("=" * 60)

print(f"\nMDD 최소화:")
print(f"  전략: {best_mdd['strategy']}")
print(f"  거래: {best_mdd['trades']:.0f}회 (스킵: {best_mdd['skipped']:.0f}회)")
print(f"  수익: {best_mdd['return']:.2f}%")
print(f"  MDD: {best_mdd['mdd']:.2f}%")
print(f"  개선: {best_mdd['mdd'] - results_df.iloc[0]['mdd']:.2f}%")

print(f"\n수익 최대화:")
print(f"  전략: {best_return['strategy']}")
print(f"  거래: {best_return['trades']:.0f}회 (스킵: {best_return['skipped']:.0f}회)")
print(f"  수익: {best_return['return']:.2f}%")
print(f"  MDD: {best_return['mdd']:.2f}%")

# 5. 추가 체크박스 제안
print("\n" + "=" * 60)
print("5. 추가 체크박스 제안")
print("=" * 60)

print("\n📌 제안 1: 이동 평균 승률")
print("  최근 20거래 승률 < 50% → 스킵 또는 포지션 축소")

print("\n📌 제안 2: 변동성 필터")
print("  ATR이 평균의 2배 이상 → 스킵 (과도한 변동성)")

print("\n📌 제안 3: 시간대 필터")
print("  특정 시간대 승률 낮으면 → 스킵")

print("\n📌 제안 4: 연속 승리 후 보수적")
print("  연속 5회 승리 → 다음 거래 포지션 축소 (과신 방지)")

print("\n" + "=" * 60)
print("분석 완료")
print("=" * 60)
