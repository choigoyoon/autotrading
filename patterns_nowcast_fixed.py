"""
8-패턴 조기 진입 전략 (나우캐스트 준수)
======================================================================
【수정 사항】
- near_breakout 필터 완전 제거 (미래참조 문제)
- 체크박스 패턴만으로 독립 진입
- 추세선 돌파 여부와 무관하게 작동
- 실시간 트레이딩 가능

【8개 독립 패턴】
1. L값 강한 반등 (1%+ 반등)
2. RSI < 30 과매도
3. BB 하단 밴드
4. EMA50 -1% 이탈
5. CCI < -100 과매도
6. 해머 캔들
7. Stochastic < 20 과매도
8. Williams %R < -80 과매도

각 패턴은 독립적으로 작동 (OR 로직)
TP 2% / SL 2%
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("8-패턴 조기 진입 전략 (나우캐스트 준수)")
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

# ═══════════════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

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

# L값 수집
l_values = []
for i in range(1, len(df)):
    if df.iloc[i-1]['macd_hist'] < 0 and df.iloc[i]['macd_hist'] >= 0:
        l_values.append({'idx': i, 'price': df.iloc[i]['low']})

print(f"L값: {len(l_values):,}개\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴별 거래 수집 (나우캐스트 준수)
# ═══════════════════════════════════════════════════════════════════

all_trades = []

def is_duplicate(new_time, existing_trades):
    """15분 이내 중복 방지"""
    for trade in existing_trades:
        if abs((new_time - trade['datetime']).total_seconds()) < 900:
            return True
    return False

def backtest_entry(entry_idx):
    """진입 후 TP/SL 백테스트"""
    if entry_idx >= len(df):
        return None, None

    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price * 1.02
    sl_price = entry_price * 0.98

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        c = df.iloc[j]
        if c['low'] <= sl_price:
            return -2.0, 'SL'
        elif c['high'] >= tp_price:
            return 2.0, 'TP'

    # Timeout
    pnl = (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100
    return pnl, 'TIMEOUT'

print("패턴별 진입 신호 수집 중...\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 1: L값 강한 반등
# ═══════════════════════════════════════════════════════════════════

print("1. L값 강한 반등...")
count = 0
for l in l_values:
    l_idx = l['idx']
    l_price = l['price']

    if l_idx + 50 >= len(df):
        continue

    # L값 확정 후 5봉 이내 1%+ 반등 확인
    bounce_idx = None
    for i in range(1, 6):
        c = df.iloc[l_idx + i]
        bounce_pct = (c['close'] - l_price) / l_price * 100

        if c['close'] > c['open'] and bounce_pct >= 1.0:
            bounce_idx = l_idx + i
            break

    if bounce_idx is None:
        continue

    entry_idx = bounce_idx + 1
    entry_time = df.iloc[entry_idx]['datetime']

    if is_duplicate(entry_time, all_trades):
        continue

    pnl, exit_type = backtest_entry(entry_idx)
    if pnl is not None:
        all_trades.append({
            'datetime': entry_time,
            'pattern': 'L-bounce',
            'pnl': pnl,
            'exit': exit_type
        })
        count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 2: RSI < 30
# ═══════════════════════════════════════════════════════════════════

print("2. RSI < 30 과매도...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    # RSI < 30 + MACD < 0
    if row['rsi'] < 30 and row['macd_hist'] < 0:
        entry_time = row['datetime']

        if is_duplicate(entry_time, all_trades):
            continue

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            all_trades.append({
                'datetime': entry_time,
                'pattern': 'RSI<30',
                'pnl': pnl,
                'exit': exit_type
            })
            count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 3: BB 하단
# ═══════════════════════════════════════════════════════════════════

print("3. BB 하단 밴드...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    # 가격 <= BB 하단 + MACD < 0
    if (pd.notna(row['bb_lower']) and
        row['close'] <= row['bb_lower'] and
        row['macd_hist'] < 0):

        entry_time = row['datetime']

        if is_duplicate(entry_time, all_trades):
            continue

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            all_trades.append({
                'datetime': entry_time,
                'pattern': 'BB-lower',
                'pnl': pnl,
                'exit': exit_type
            })
            count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 4: EMA50 -1% 이탈
# ═══════════════════════════════════════════════════════════════════

print("4. EMA50 -1% 이탈...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    if pd.notna(row['ema50']):
        dev = (row['close'] - row['ema50']) / row['ema50'] * 100

        # -1% 이상 이탈 + MACD < 0
        if dev <= -1.0 and row['macd_hist'] < 0:
            entry_time = row['datetime']

            if is_duplicate(entry_time, all_trades):
                continue

            pnl, exit_type = backtest_entry(i + 1)
            if pnl is not None:
                all_trades.append({
                    'datetime': entry_time,
                    'pattern': 'EMA50-dev',
                    'pnl': pnl,
                    'exit': exit_type
                })
                count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 5: CCI < -100
# ═══════════════════════════════════════════════════════════════════

print("5. CCI < -100 과매도...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    # CCI < -100 + MACD < 0
    if (pd.notna(row['cci']) and
        row['cci'] < -100 and
        row['macd_hist'] < 0):

        entry_time = row['datetime']

        if is_duplicate(entry_time, all_trades):
            continue

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            all_trades.append({
                'datetime': entry_time,
                'pattern': 'CCI<-100',
                'pnl': pnl,
                'exit': exit_type
            })
            count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 6: 해머 캔들
# ═══════════════════════════════════════════════════════════════════

print("6. 해머 캔들...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    # 해머 조건
    body = abs(row['close'] - row['open'])
    lower_wick = min(row['open'], row['close']) - row['low']
    upper_wick = row['high'] - max(row['open'], row['close'])

    if (body > 0 and
        lower_wick >= 2 * body and
        upper_wick < 0.5 * body and
        row['macd_hist'] < 0):

        entry_time = row['datetime']

        if is_duplicate(entry_time, all_trades):
            continue

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            all_trades.append({
                'datetime': entry_time,
                'pattern': 'Hammer',
                'pnl': pnl,
                'exit': exit_type
            })
            count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 7: Stochastic < 20
# ═══════════════════════════════════════════════════════════════════

print("7. Stochastic < 20 과매도...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    # Stochastic < 20 + MACD < 0
    if (pd.notna(row['stoch_k']) and
        row['stoch_k'] < 20 and
        row['macd_hist'] < 0):

        entry_time = row['datetime']

        if is_duplicate(entry_time, all_trades):
            continue

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            all_trades.append({
                'datetime': entry_time,
                'pattern': 'Stoch<20',
                'pnl': pnl,
                'exit': exit_type
            })
            count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 8: Williams %R < -80
# ═══════════════════════════════════════════════════════════════════

print("8. Williams %R < -80 과매도...")
count = 0
for i in range(50, len(df) - 50):
    row = df.iloc[i]

    # Williams %R < -80 + MACD < 0
    if (pd.notna(row['williams_r']) and
        row['williams_r'] < -80 and
        row['macd_hist'] < 0):

        entry_time = row['datetime']

        if is_duplicate(entry_time, all_trades):
            continue

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            all_trades.append({
                'datetime': entry_time,
                'pattern': 'WillR<-80',
                'pnl': pnl,
                'exit': exit_type
            })
            count += 1

print(f"   → {count}개 진입\n")

# ═══════════════════════════════════════════════════════════════════
# 통합 결과
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("통합 전략 성과 (나우캐스트 준수)")
print("=" * 70)
print()

if len(all_trades) > 0:
    df_all = pd.DataFrame(all_trades)
    df_all = df_all.sort_values('datetime').reset_index(drop=True)

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

    # 복리 계산
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

    print("=" * 70)
    print("✅ 나우캐스트 준수 확인")
    print("=" * 70)
    print()
    print("✓ near_breakout 필터 제거 (미래참조 제거)")
    print("✓ 체크박스 패턴만으로 독립 진입")
    print("✓ 실시간 트레이딩 가능")
    print("✓ 모든 지표는 확정된 과거 데이터만 사용")

else:
    print("거래 없음")

print()
print("✅ 분석 완료")
