"""
단계별 필터링 퍼널 전략
======================================================================
1단계: MACD 히스토 < 0 (40% 남음)
2단계: MACD 선 상태 분석 (20% 남음)
  - 기울기 (상승 전환 중?)
  - 0선 거리 (얼마나 과매도?)
  - 수렴 (Signal 선과 가까워지는 중?)
3단계: 다중 지표 미세 조정 (10% 남음)
  - RSI 범위 내 위치
  - BB %B 위치
  - CCI 위치
  - Stochastic 위치
4단계: 캔들/FVG 트리거 (진입)
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("단계별 필터링 퍼널 전략")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"총 데이터: {len(df):,}개 캔들\n")

# ═══════════════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# MACD는 이미 있음 (macd, macd_signal, macd_hist)
# MACD 기울기
df['macd_slope'] = df['macd'].diff()
df['macd_signal_slope'] = df['macd_signal'].diff()

# MACD 0선 거리
df['macd_distance'] = abs(df['macd'])

# MACD 수렴 (MACD와 Signal 간 거리)
df['macd_convergence'] = abs(df['macd'] - df['macd_signal'])

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# BB
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

# BB %B (0~1, 0=하단, 1=상단)
df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# CCI
tp = (df['high'] + df['low'] + df['close']) / 3
df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

# Stochastic
low_14 = df['low'].rolling(window=14).min()
high_14 = df['high'].rolling(window=14).max()
df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)

# 캔들 패턴
df['body'] = abs(df['close'] - df['open'])
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)

# 해머 패턴 (긴 아래꼬리)
df['is_hammer'] = (
    (df['body'] > 0) &
    (df['lower_wick'] >= 2 * df['body']) &
    (df['upper_wick'] < 0.5 * df['body'])
)

# 강세 캔들
df['is_bullish'] = df['close'] > df['open']

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 퍼널 단계별 필터링
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("퍼널 단계별 필터링")
print("=" * 70)
print()

total_candles = len(df)

# 1단계: MACD 히스토 < 0
print("1단계: MACD 히스토 < 0")
stage1 = df[df['macd_hist'] < 0].copy()
stage1_pct = len(stage1) / total_candles * 100
print(f"  → {len(stage1):,}개 ({stage1_pct:.1f}%) 남음\n")

# 2단계: MACD 선 상태
print("2단계: MACD 선 상태 분석")
print("  조건:")
print("  - MACD 기울기 > 0 (상승 전환 중)")
print("  - MACD 0선 거리 < 500 (과도하게 하락하지 않음)")
print("  - MACD 수렴 중 (Signal 선과 거리 < 200)")

stage2 = stage1[
    (stage1['macd_slope'] > 0) &  # 상승 전환
    (stage1['macd_distance'] < 500) &  # 적절한 거리
    (stage1['macd_convergence'] < 200)  # 수렴 중
].copy()

stage2_pct = len(stage2) / total_candles * 100
print(f"  → {len(stage2):,}개 ({stage2_pct:.1f}%) 남음\n")

# 3단계: 다중 지표 미세 조정
print("3단계: 다중 지표 미세 조정")
print("  조건:")
print("  - RSI 20~40 범위 (과매도 영역)")
print("  - BB %B < 0.3 (하단 근처)")
print("  - CCI < -50 (과매도)")
print("  - Stoch < 30 (과매도)")

stage3 = stage2[
    stage2['rsi'].between(20, 40) &
    (stage2['bb_percent_b'] < 0.3) &
    (stage2['cci'] < -50) &
    (stage2['stoch_k'] < 30)
].copy()

stage3_pct = len(stage3) / total_candles * 100
print(f"  → {len(stage3):,}개 ({stage3_pct:.1f}%) 남음\n")

# 4단계: 캔들 트리거
print("4단계: 캔들 트리거")
print("  조건:")
print("  - 해머 캔들 OR")
print("  - 강세 캔들 (양봉)")

stage4 = stage3[
    stage3['is_hammer'] | stage3['is_bullish']
].copy()

stage4_pct = len(stage4) / total_candles * 100
print(f"  → {len(stage4):,}개 ({stage4_pct:.1f}%) 남음\n")

# ═══════════════════════════════════════════════════════════════════
# 백테스트
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("백테스트 실행")
print("=" * 70)
print()

entry_signals = stage4.index.tolist()

trades = []

for idx in entry_signals:
    if idx + 50 >= len(df):
        continue

    entry_idx = idx + 1
    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price * 1.02
    sl_price = entry_price * 0.98

    # TP/SL 체크
    pnl = None
    exit_type = None

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        c = df.iloc[j]
        if c['low'] <= sl_price:
            pnl = -2.0
            exit_type = 'SL'
            break
        elif c['high'] >= tp_price:
            pnl = 2.0
            exit_type = 'TP'
            break

    if pnl is None:
        pnl = (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100
        exit_type = 'TIMEOUT'

    trades.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'pnl': pnl,
        'exit': exit_type
    })

print(f"총 거래: {len(trades)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 성과 분석
# ═══════════════════════════════════════════════════════════════════

if len(trades) > 0:
    df_trades = pd.DataFrame(trades)

    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100
    avg_pnl = df_trades['pnl'].mean()

    # 복리 계산
    initial_capital = 10000
    capital = initial_capital

    for pnl in df_trades['pnl']:
        capital = capital * (1 + pnl / 100)

    total_return = (capital - initial_capital) / initial_capital * 100

    days = (df['datetime'].max() - df['datetime'].min()).days
    years = days / 365.25
    annual_return = ((capital / initial_capital) ** (1 / years) - 1) * 100

    # MDD
    cumulative = [initial_capital]
    for pnl in df_trades['pnl']:
        cumulative.append(cumulative[-1] * (1 + pnl / 100))

    cumulative = np.array(cumulative)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    mdd = drawdown.min()

    annual_trades = len(df_trades) / years
    monthly_trades = annual_trades / 12

    print("=" * 70)
    print("성과 분석")
    print("=" * 70)
    print()

    print(f"총 거래: {len(df_trades)}개")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 PnL: {avg_pnl:+.2f}%")
    print()
    print(f"초기 자본: ${initial_capital:,.2f}")
    print(f"최종 자본: ${capital:,.2f}")
    print(f"총 수익률: {total_return:,.2f}%")
    print(f"연복리: {annual_return:.2f}%")
    print(f"MDD: {mdd:.2f}%")
    print()
    print(f"연간 거래: {annual_trades:.1f}회")
    print(f"월간 거래: {monthly_trades:.1f}회")
    print()

    # 비교
    print("=" * 70)
    print("vs 기준 전략")
    print("=" * 70)
    print()
    print(f"{'전략':<30} {'거래':<12} {'승률':<12} {'평균 PnL':<12}")
    print("-" * 70)
    print(f"{'추세선 돌파 (기준)':<30} {'983개':<12} {'79.8%':<12} {'+0.96%':<12}")
    print(f"{'필터 없는 RSI<30':<30} {'2000개':<12} {'57.8%':<12} {'+0.20%':<12}")
    print(f"{'퍼널 전략 (신규)':<30} {f'{len(df_trades)}개':<12} {f'{win_rate:.1f}%':<12} {f'{avg_pnl:+.2f}%':<12}")
    print()

    if win_rate >= 70:
        print("✅ 우수한 성과!")
        print(f"   승률: {win_rate:.1f}%")
        print(f"   기준 대비: {win_rate - 79.8:+.1f}%p")
    elif win_rate >= 65:
        print("✓ 양호한 성과")
        print(f"   승률: {win_rate:.1f}%")
    else:
        print("⚠️  추가 개선 필요")
        print(f"   승률: {win_rate:.1f}% (목표: 70%+)")

else:
    print("⚠️  진입 신호 없음 - 필터 조건 완화 필요")

print()
print("=" * 70)
print("퍼널 효율성")
print("=" * 70)
print()
print(f"1단계 (MACD<0):      {stage1_pct:>5.1f}% 남음")
print(f"2단계 (MACD 상태):   {stage2_pct:>5.1f}% 남음")
print(f"3단계 (다중 지표):   {stage3_pct:>5.1f}% 남음")
print(f"4단계 (캔들):        {stage4_pct:>5.1f}% 남음")

print()
print("✅ 분석 완료")
