"""
추가 조기 진입 패턴 발굴
======================================================================
기존 4개 패턴에 추가할 독립적인 패턴들 찾기:

테스트할 패턴:
1. 볼륨 급증 (Volume Surge)
2. Stochastic 과매도
3. CCI 과매도
4. Williams %R 과매도
5. 해머/역해머 캔들 패턴
6. 강세 Engulfing 패턴
7. MACD 다이버전스
8. MTF 정렬 (1H + 4H MACD < 0)
9. 가격 지지선 반등
10. ATR 확장 (변동성 확대)
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("추가 조기 진입 패턴 발굴")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

df_breakouts = pd.read_csv('backtest_filtered_10bars.csv')
df_breakouts['datetime'] = pd.to_datetime(df_breakouts['datetime'])

# 최근 5년
cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"총 데이터: {len(df):,}개 캔들\n")

# ═══════════════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# RSI (기존)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Stochastic
low_14 = df['low'].rolling(window=14).min()
high_14 = df['high'].rolling(window=14).max()
df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

# CCI (Commodity Channel Index)
tp = (df['high'] + df['low'] + df['close']) / 3
df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

# Williams %R
df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

# ATR (Average True Range)
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
df['atr'] = true_range.rolling(14).mean()
df['atr_pct'] = df['atr'] / df['close'] * 100

# Volume ratio
df['vol_ma'] = df['volume'].rolling(window=20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma']

# EMA50 (기존)
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

# BB (기존)
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 돌파 근처 인덱싱
# ═══════════════════════════════════════════════════════════════════

print("돌파 시점 인덱싱 중...")

breakout_indices = set()
for _, breakout in df_breakouts.iterrows():
    breakout_time = breakout['datetime']
    matching = df[
        (df['datetime'] >= breakout_time - timedelta(minutes=15)) &
        (df['datetime'] <= breakout_time + timedelta(minutes=15))
    ]
    if len(matching) > 0:
        breakout_indices.add(matching.index[0])

df['near_breakout'] = False
for idx in breakout_indices:
    for i in range(max(0, idx - 5), idx + 1):
        if i < len(df):
            df.iloc[i, df.columns.get_loc('near_breakout')] = True

print(f"  돌파 근처 캔들: {df['near_breakout'].sum()}개\n")

# ═══════════════════════════════════════════════════════════════════
# 백테스트 함수
# ═══════════════════════════════════════════════════════════════════

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

    pnl = (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100
    return pnl, 'TIMEOUT'

# ═══════════════════════════════════════════════════════════════════
# 패턴 테스트
# ═══════════════════════════════════════════════════════════════════

test_results = []

# 패턴 5: 볼륨 급증
print("패턴 5: 볼륨 급증 테스트 중...")
trades_vol = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue

    row = df.iloc[i]

    # 조건: 볼륨 2배 이상 + MACD < 0 + 가격 하락 중
    if (row['vol_ratio'] >= 2.0 and
        row['macd_hist'] < 0 and
        row['close'] < row['open']):

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            trades_vol.append({'pnl': pnl, 'exit': exit_type})

if len(trades_vol) > 0:
    df_vol = pd.DataFrame(trades_vol)
    win_rate = len(df_vol[df_vol['pnl'] > 0]) / len(df_vol) * 100
    avg_pnl = df_vol['pnl'].mean()
    test_results.append({
        'pattern': '볼륨 급증 (2x)',
        'trades': len(trades_vol),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_vol)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# 패턴 6: Stochastic 과매도
print("패턴 6: Stochastic 과매도 테스트 중...")
trades_stoch = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue

    row = df.iloc[i]

    # 조건: Stochastic < 20 + MACD < 0
    if (pd.notna(row['stoch_k']) and
        row['stoch_k'] < 20 and
        row['macd_hist'] < 0):

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            trades_stoch.append({'pnl': pnl, 'exit': exit_type})

if len(trades_stoch) > 0:
    df_stoch = pd.DataFrame(trades_stoch)
    win_rate = len(df_stoch[df_stoch['pnl'] > 0]) / len(df_stoch) * 100
    avg_pnl = df_stoch['pnl'].mean()
    test_results.append({
        'pattern': 'Stochastic < 20',
        'trades': len(trades_stoch),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_stoch)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# 패턴 7: CCI 과매도
print("패턴 7: CCI 과매도 테스트 중...")
trades_cci = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue

    row = df.iloc[i]

    # 조건: CCI < -100 + MACD < 0
    if (pd.notna(row['cci']) and
        row['cci'] < -100 and
        row['macd_hist'] < 0):

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            trades_cci.append({'pnl': pnl, 'exit': exit_type})

if len(trades_cci) > 0:
    df_cci = pd.DataFrame(trades_cci)
    win_rate = len(df_cci[df_cci['pnl'] > 0]) / len(df_cci) * 100
    avg_pnl = df_cci['pnl'].mean()
    test_results.append({
        'pattern': 'CCI < -100',
        'trades': len(trades_cci),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_cci)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# 패턴 8: Williams %R 과매도
print("패턴 8: Williams %R 과매도 테스트 중...")
trades_willr = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue

    row = df.iloc[i]

    # 조건: Williams %R < -80 + MACD < 0
    if (pd.notna(row['williams_r']) and
        row['williams_r'] < -80 and
        row['macd_hist'] < 0):

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            trades_willr.append({'pnl': pnl, 'exit': exit_type})

if len(trades_willr) > 0:
    df_willr = pd.DataFrame(trades_willr)
    win_rate = len(df_willr[df_willr['pnl'] > 0]) / len(df_willr) * 100
    avg_pnl = df_willr['pnl'].mean()
    test_results.append({
        'pattern': 'Williams %R < -80',
        'trades': len(trades_willr),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_willr)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# 패턴 9: 해머 캔들
print("패턴 9: 해머 캔들 패턴 테스트 중...")
trades_hammer = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue

    row = df.iloc[i]

    # 해머 조건: 긴 아래꼬리 + 짧은 몸통
    body = abs(row['close'] - row['open'])
    lower_wick = min(row['open'], row['close']) - row['low']
    upper_wick = row['high'] - max(row['open'], row['close'])

    if (body > 0 and
        lower_wick >= 2 * body and
        upper_wick < 0.5 * body and
        row['macd_hist'] < 0):

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            trades_hammer.append({'pnl': pnl, 'exit': exit_type})

if len(trades_hammer) > 0:
    df_hammer = pd.DataFrame(trades_hammer)
    win_rate = len(df_hammer[df_hammer['pnl'] > 0]) / len(df_hammer) * 100
    avg_pnl = df_hammer['pnl'].mean()
    test_results.append({
        'pattern': '해머 캔들',
        'trades': len(trades_hammer),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_hammer)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# 패턴 10: 강세 Engulfing
print("패턴 10: 강세 Engulfing 패턴 테스트 중...")
trades_engulf = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue
    if i < 1:
        continue

    prev = df.iloc[i-1]
    curr = df.iloc[i]

    # Engulfing 조건: 이전봉 음봉 + 현재봉 양봉으로 완전히 감쌈
    if (prev['close'] < prev['open'] and  # 이전봉 음봉
        curr['close'] > curr['open'] and   # 현재봉 양봉
        curr['open'] < prev['close'] and   # 시가가 이전봉 종가보다 낮음
        curr['close'] > prev['open'] and   # 종가가 이전봉 시가보다 높음
        curr['macd_hist'] < 0):

        pnl, exit_type = backtest_entry(i + 1)
        if pnl is not None:
            trades_engulf.append({'pnl': pnl, 'exit': exit_type})

if len(trades_engulf) > 0:
    df_engulf = pd.DataFrame(trades_engulf)
    win_rate = len(df_engulf[df_engulf['pnl'] > 0]) / len(df_engulf) * 100
    avg_pnl = df_engulf['pnl'].mean()
    test_results.append({
        'pattern': '강세 Engulfing',
        'trades': len(trades_engulf),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_engulf)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# 패턴 11: ATR 확장 (변동성 급증)
print("패턴 11: ATR 확장 패턴 테스트 중...")
trades_atr = []
for i in range(50, len(df) - 50):
    if not df.iloc[i]['near_breakout']:
        continue

    row = df.iloc[i]

    # 조건: ATR이 평균보다 1.5배 이상 + MACD < 0
    if pd.notna(row['atr_pct']):
        avg_atr = df.iloc[max(0, i-50):i]['atr_pct'].mean()
        if (row['atr_pct'] >= avg_atr * 1.5 and
            row['macd_hist'] < 0):

            pnl, exit_type = backtest_entry(i + 1)
            if pnl is not None:
                trades_atr.append({'pnl': pnl, 'exit': exit_type})

if len(trades_atr) > 0:
    df_atr = pd.DataFrame(trades_atr)
    win_rate = len(df_atr[df_atr['pnl'] > 0]) / len(df_atr) * 100
    avg_pnl = df_atr['pnl'].mean()
    test_results.append({
        'pattern': 'ATR 확장 (1.5x)',
        'trades': len(trades_atr),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(trades_atr)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%")
else:
    print("  거래 없음")

# ═══════════════════════════════════════════════════════════════════
# 결과 요약
# ═══════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("발굴된 패턴 요약")
print("=" * 70)
print()

if len(test_results) > 0:
    df_results = pd.DataFrame(test_results)
    df_results = df_results.sort_values('win_rate', ascending=False)

    print(f"{'패턴':<25} {'거래수':<10} {'승률':<12} {'평균 PnL':<12}")
    print("-" * 70)
    for _, row in df_results.iterrows():
        print(f"{row['pattern']:<25} {row['trades']:<10} {row['win_rate']:>6.1f}%      {row['avg_pnl']:>+6.2f}%")

    print()
    print("=" * 70)
    print("우수 패턴 선별 기준")
    print("=" * 70)
    print()
    print("✅ 승률 85% 이상")
    print("✅ 평균 PnL +1.0% 이상")
    print("✅ 거래 빈도 적절 (너무 드물지 않음)")
    print()

    # 우수 패턴 필터
    excellent = df_results[(df_results['win_rate'] >= 85) & (df_results['avg_pnl'] >= 1.0)]

    if len(excellent) > 0:
        print("🎯 우수 패턴:")
        print("-" * 70)
        for _, row in excellent.iterrows():
            print(f"  ✓ {row['pattern']}: {row['trades']}개, {row['win_rate']:.1f}% 승률, {row['avg_pnl']:+.2f}% PnL")
    else:
        print("⚠️  85% 이상 승률 패턴 없음 - 기준 완화 필요")

        # 80% 이상으로 완화
        good = df_results[df_results['win_rate'] >= 80]
        if len(good) > 0:
            print()
            print("📊 양호 패턴 (80%+):")
            print("-" * 70)
            for _, row in good.iterrows():
                print(f"  • {row['pattern']}: {row['trades']}개, {row['win_rate']:.1f}% 승률, {row['avg_pnl']:+.2f}% PnL")

else:
    print("발굴된 패턴 없음")

print()
print("✅ 분석 완료")
