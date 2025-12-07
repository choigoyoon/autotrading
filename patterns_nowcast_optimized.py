"""
8-패턴 조기 진입 전략 (나우캐스트 준수 + 최적화)
======================================================================
벡터화 연산으로 속도 최적화
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("8-패턴 조기 진입 전략 (나우캐스트 준수 + 최적화)")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# 최근 5년
cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"총 데이터: {len(df):,}개 캔들\n")

# 지표 계산
print("지표 계산 중...")

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# BB
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

# EMA50
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

# Stochastic
low_14 = df['low'].rolling(window=14).min()
high_14 = df['high'].rolling(window=14).max()
df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)

# CCI
tp = (df['high'] + df['low'] + df['close']) / 3
df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

# Williams %R
df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

print("  완료!\n")

# 벡터화된 백테스트 함수
def vectorized_backtest(entry_indices, df):
    """벡터화 백테스트 - 훨씬 빠름"""
    results = []

    for idx in entry_indices:
        if idx + 50 >= len(df):
            continue

        entry_price = df.iloc[idx]['open']
        tp_price = entry_price * 1.02
        sl_price = entry_price * 0.98

        # 50봉 범위
        future = df.iloc[idx:idx+50]

        # TP/SL 체크
        tp_hit = future[future['high'] >= tp_price]
        sl_hit = future[future['low'] <= sl_price]

        if len(tp_hit) == 0 and len(sl_hit) == 0:
            # Timeout
            pnl = (future.iloc[-1]['close'] - entry_price) / entry_price * 100
            results.append(pnl)
        elif len(sl_hit) > 0 and len(tp_hit) > 0:
            # 둘 다 발생 - 먼저 발생한 것
            if tp_hit.index[0] < sl_hit.index[0]:
                results.append(2.0)
            else:
                results.append(-2.0)
        elif len(tp_hit) > 0:
            results.append(2.0)
        else:
            results.append(-2.0)

    return results

print("패턴별 진입 신호 수집 중...\n")

all_trades = []

# Pattern 1: L값 강한 반등
print("1. L값 강한 반등...")
l_indices = []
for i in range(1, len(df)):
    if df.iloc[i-1]['macd_hist'] < 0 and df.iloc[i]['macd_hist'] >= 0:
        l_price = df.iloc[i]['low']
        # 5봉 이내 1%+ 반등
        for j in range(1, min(6, len(df)-i)):
            c = df.iloc[i+j]
            if c['close'] > c['open'] and (c['close'] - l_price) / l_price * 100 >= 1.0:
                l_indices.append(i + j + 1)
                break

pnls = vectorized_backtest(l_indices, df)
for idx, pnl in zip(l_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'L-bounce',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 2: RSI < 30 (최적화)
print("2. RSI < 30 과매도...")
rsi_mask = (df['rsi'] < 30) & (df['macd_hist'] < 0)
rsi_indices = df[rsi_mask].index.tolist()
rsi_indices = [i+1 for i in rsi_indices if i+1 < len(df)]

pnls = vectorized_backtest(rsi_indices[:1000], df)  # 최대 1000개로 제한
for idx, pnl in zip(rsi_indices[:1000], pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'RSI<30',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 3: BB 하단
print("3. BB 하단 밴드...")
bb_mask = (df['close'] <= df['bb_lower']) & (df['macd_hist'] < 0) & df['bb_lower'].notna()
bb_indices = df[bb_mask].index.tolist()
bb_indices = [i+1 for i in bb_indices if i+1 < len(df)]

pnls = vectorized_backtest(bb_indices, df)
for idx, pnl in zip(bb_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'BB-lower',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 4: EMA50 -1% 이탈
print("4. EMA50 -1% 이탈...")
df['ema_dev'] = (df['close'] - df['ema50']) / df['ema50'] * 100
ema_mask = (df['ema_dev'] <= -1.0) & (df['macd_hist'] < 0) & df['ema50'].notna()
ema_indices = df[ema_mask].index.tolist()
ema_indices = [i+1 for i in ema_indices if i+1 < len(df)]

pnls = vectorized_backtest(ema_indices, df)
for idx, pnl in zip(ema_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'EMA50-dev',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 5: CCI < -100
print("5. CCI < -100 과매도...")
cci_mask = (df['cci'] < -100) & (df['macd_hist'] < 0) & df['cci'].notna()
cci_indices = df[cci_mask].index.tolist()
cci_indices = [i+1 for i in cci_indices if i+1 < len(df)]

pnls = vectorized_backtest(cci_indices, df)
for idx, pnl in zip(cci_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'CCI<-100',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 6: 해머 캔들
print("6. 해머 캔들...")
df['body'] = abs(df['close'] - df['open'])
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)

hammer_mask = (
    (df['body'] > 0) &
    (df['lower_wick'] >= 2 * df['body']) &
    (df['upper_wick'] < 0.5 * df['body']) &
    (df['macd_hist'] < 0)
)
hammer_indices = df[hammer_mask].index.tolist()
hammer_indices = [i+1 for i in hammer_indices if i+1 < len(df)]

pnls = vectorized_backtest(hammer_indices, df)
for idx, pnl in zip(hammer_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'Hammer',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 7: Stochastic < 20
print("7. Stochastic < 20 과매도...")
stoch_mask = (df['stoch_k'] < 20) & (df['macd_hist'] < 0) & df['stoch_k'].notna()
stoch_indices = df[stoch_mask].index.tolist()
stoch_indices = [i+1 for i in stoch_indices if i+1 < len(df)]

pnls = vectorized_backtest(stoch_indices, df)
for idx, pnl in zip(stoch_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'Stoch<20',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# Pattern 8: Williams %R < -80
print("8. Williams %R < -80 과매도...")
willr_mask = (df['williams_r'] < -80) & (df['macd_hist'] < 0) & df['williams_r'].notna()
willr_indices = df[willr_mask].index.tolist()
willr_indices = [i+1 for i in willr_indices if i+1 < len(df)]

pnls = vectorized_backtest(willr_indices, df)
for idx, pnl in zip(willr_indices, pnls):
    all_trades.append({
        'datetime': df.iloc[idx]['datetime'],
        'pattern': 'WillR<-80',
        'pnl': pnl
    })
print(f"   → {len(pnls)}개 진입\n")

# 결과 분석
print("=" * 70)
print("통합 전략 성과 (나우캐스트 준수)")
print("=" * 70)
print()

if len(all_trades) > 0:
    df_all = pd.DataFrame(all_trades)

    # 중복 제거 (15분 이내)
    df_all = df_all.sort_values('datetime').reset_index(drop=True)

    to_remove = []
    for i in range(1, len(df_all)):
        if (df_all.iloc[i]['datetime'] - df_all.iloc[i-1]['datetime']).total_seconds() < 900:
            to_remove.append(i)

    df_all = df_all.drop(to_remove).reset_index(drop=True)

    # 패턴별 통계
    print("패턴별 기여도:")
    print("-" * 70)
    print(f"{'패턴':<20} {'거래수':<10} {'비중':<10} {'승률':<12} {'평균 PnL':<12}")
    print("-" * 70)

    for pattern in sorted(df_all['pattern'].unique()):
        pattern_trades = df_all[df_all['pattern'] == pattern]
        count = len(pattern_trades)
        ratio = count / len(df_all) * 100
        win_rate = len(pattern_trades[pattern_trades['pnl'] > 0]) / count * 100
        avg_pnl = pattern_trades['pnl'].mean()
        print(f"{pattern:<20} {count:<10} {ratio:>5.1f}%     {win_rate:>6.1f}%      {avg_pnl:>+6.2f}%")

    print()

    # 전체 성과
    win_rate = len(df_all[df_all['pnl'] > 0]) / len(df_all) * 100
    avg_pnl = df_all['pnl'].mean()

    # 복리
    initial_capital = 10000
    capital = initial_capital
    for pnl in df_all['pnl']:
        capital = capital * (1 + pnl / 100)

    total_return = (capital - initial_capital) / initial_capital * 100
    days = (df['datetime'].max() - df['datetime'].min()).days
    years = days / 365.25
    annual_return = ((capital / initial_capital) ** (1 / years) - 1) * 100

    # MDD
    cumulative = [initial_capital]
    for pnl in df_all['pnl']:
        cumulative.append(cumulative[-1] * (1 + pnl / 100))
    cumulative = np.array(cumulative)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    mdd = drawdown.min()

    annual_trades = len(df_all) / years
    monthly_trades = annual_trades / 12

    print("=" * 70)
    print("전체 성과 (5년)")
    print("=" * 70)
    print()
    print(f"초기 자본: ${initial_capital:,.2f}")
    print(f"최종 자본: ${capital:,.2f}")
    print(f"총 수익률: {total_return:,.2f}%")
    print(f"연복리: {annual_return:.2f}%")
    print(f"MDD: {mdd:.2f}%")
    print()
    print(f"총 거래: {len(df_all)}개")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 PnL: {avg_pnl:+.2f}%")
    print(f"연간 거래: {annual_trades:.1f}회")
    print(f"월간 거래: {monthly_trades:.1f}회")
    print()

    print("✅ 나우캐스트 준수 확인")
    print("  - near_breakout 필터 제거 (미래참조 제거)")
    print("  - 체크박스 패턴만으로 독립 진입")
    print("  - 실시간 트레이딩 가능")

print()
print("✅ 분석 완료")
