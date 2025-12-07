"""
변곡점 분석 - 추세선 돌파 시점의 다른 지표들

목표:
1. 돌파 시점에 다른 지표들 상태 파악
2. 선행 지표 발견 (몇 봉 전에 먼저 반응?)
3. 조합으로 승률 90%+ 가능한지?
4. 선행 진입 전략 가능성?
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("변곡점 분석 - 추세선 돌파 시점의 지표 상태")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

breakouts = pd.read_csv('output_phase4_breakouts.csv')
breakouts['datetime'] = breakouts['break_idx'].apply(lambda x: df.iloc[x]['datetime'] if x < len(df) else None)
breakouts = breakouts.dropna(subset=['datetime'])

# 10봉 필터 적용
filtered_breakouts = []
last_break_idx = -999

for idx, row in breakouts.iterrows():
    break_idx = row['break_idx']
    if break_idx - last_break_idx >= 10:
        filtered_breakouts.append(row)
        last_break_idx = break_idx

filtered_df = pd.DataFrame(filtered_breakouts)
trendline_up = filtered_df[filtered_df['type'] == 'trendline_up'].copy()

print(f"\n분석 대상: {len(trendline_up):,}개 돌파 시점")

# ═══════════════════════════════════════════════════════════
# 추가 지표 계산
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("지표 계산 중...")
print("=" * 70)

# 1. Bollinger Bands
def calculate_bb(df, period=20, std=2):
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std)

    # BB 위치 (0~1, 0=하단, 0.5=중간, 1=상단)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df

# 2. RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df

# 3. Williams %R
def calculate_williams_r(df, period=14):
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()

    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)

    return df

# 지표 계산
df = calculate_bb(df, period=20, std=2)
df = calculate_rsi(df, period=14)
df = calculate_williams_r(df, period=14)

print("✓ Bollinger Bands (20, 2)")
print("✓ RSI (14)")
print("✓ Williams %R (14)")
print("✓ MACD Histogram (이미 계산됨)")

# ═══════════════════════════════════════════════════════════
# 1. 돌파 시점 지표 상태 분석
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1. 돌파 시점의 지표 상태")
print("=" * 70)

breakout_states = []

for idx, signal in trendline_up.iterrows():
    break_idx = signal['break_idx']

    if break_idx >= len(df):
        continue

    # 돌파 시점의 지표들
    breakout_candle = df.iloc[break_idx]

    breakout_states.append({
        'break_idx': break_idx,
        'bb_position': breakout_candle['bb_position'],
        'rsi': breakout_candle['rsi'],
        'williams_r': breakout_candle['williams_r'],
        'macd_histogram': breakout_candle['macd_histogram'],
        'close': breakout_candle['close'],
    })

states_df = pd.DataFrame(breakout_states)

print(f"\n돌파 시점 지표 분포:")
print(f"\nBB Position:")
print(f"  평균: {states_df['bb_position'].mean():.3f} (0=하단, 1=상단)")
print(f"  중앙값: {states_df['bb_position'].median():.3f}")
print(f"  하단 근처 (<0.3): {(states_df['bb_position'] < 0.3).sum()}개 ({(states_df['bb_position'] < 0.3).sum() / len(states_df) * 100:.1f}%)")
print(f"  중간 (0.3~0.7): {((states_df['bb_position'] >= 0.3) & (states_df['bb_position'] <= 0.7)).sum()}개")
print(f"  상단 근처 (>0.7): {(states_df['bb_position'] > 0.7).sum()}개")

print(f"\nRSI:")
print(f"  평균: {states_df['rsi'].mean():.1f}")
print(f"  중앙값: {states_df['rsi'].median():.1f}")
print(f"  과매도 (<30): {(states_df['rsi'] < 30).sum()}개 ({(states_df['rsi'] < 30).sum() / len(states_df) * 100:.1f}%)")
print(f"  중립 (30~70): {((states_df['rsi'] >= 30) & (states_df['rsi'] <= 70)).sum()}개")
print(f"  과매수 (>70): {(states_df['rsi'] > 70).sum()}개")

print(f"\nWilliams %R:")
print(f"  평균: {states_df['williams_r'].mean():.1f}")
print(f"  과매도 (<-80): {(states_df['williams_r'] < -80).sum()}개 ({(states_df['williams_r'] < -80).sum() / len(states_df) * 100:.1f}%)")
print(f"  중립 (-80~-20): {((states_df['williams_r'] >= -80) & (states_df['williams_r'] <= -20)).sum()}개")
print(f"  과매수 (>-20): {(states_df['williams_r'] > -20).sum()}개")

print(f"\nMACD Histogram:")
print(f"  평균: {states_df['macd_histogram'].mean():.2f}")
print(f"  양수: {(states_df['macd_histogram'] > 0).sum()}개 ({(states_df['macd_histogram'] > 0).sum() / len(states_df) * 100:.1f}%)")
print(f"  음수: {(states_df['macd_histogram'] < 0).sum()}개")

# ═══════════════════════════════════════════════════════════
# 2. 선행 지표 분석 (돌파 전 1~5봉)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2. 선행 지표 분석 - 어떤 게 먼저 반응?")
print("=" * 70)

leading_analysis = []

for idx, signal in trendline_up.iterrows():
    break_idx = signal['break_idx']

    if break_idx < 5 or break_idx >= len(df):
        continue

    # 돌파 전 5봉 스캔
    lookback = df.iloc[break_idx-5:break_idx+1]

    # 각 지표가 언제 반응했는지 체크
    bb_touch_before = None
    rsi_oversold_before = None
    williams_oversold_before = None

    for i in range(len(lookback) - 1):  # 마지막(돌파 시점) 제외
        candle = lookback.iloc[i]
        bars_before = len(lookback) - 1 - i

        # BB 하단 터치 (position < 0.2)
        if bb_touch_before is None and candle['bb_position'] < 0.2:
            bb_touch_before = bars_before

        # RSI 과매도 (< 30)
        if rsi_oversold_before is None and candle['rsi'] < 30:
            rsi_oversold_before = bars_before

        # Williams 과매도 (< -80)
        if williams_oversold_before is None and candle['williams_r'] < -80:
            williams_oversold_before = bars_before

    leading_analysis.append({
        'break_idx': break_idx,
        'bb_touch_before': bb_touch_before if bb_touch_before else 0,
        'rsi_oversold_before': rsi_oversold_before if rsi_oversold_before else 0,
        'williams_oversold_before': williams_oversold_before if williams_oversold_before else 0,
    })

leading_df = pd.DataFrame(leading_analysis)

print(f"\n돌파 전 신호 발생 통계 (5봉 이내):")

bb_count = (leading_df['bb_touch_before'] > 0).sum()
rsi_count = (leading_df['rsi_oversold_before'] > 0).sum()
williams_count = (leading_df['williams_oversold_before'] > 0).sum()

print(f"\nBB 하단 터치: {bb_count}개 ({bb_count / len(leading_df) * 100:.1f}%)")
if bb_count > 0:
    avg_before = leading_df[leading_df['bb_touch_before'] > 0]['bb_touch_before'].mean()
    print(f"  평균 {avg_before:.1f}봉 전에 반응")

print(f"\nRSI 과매도(<30): {rsi_count}개 ({rsi_count / len(leading_df) * 100:.1f}%)")
if rsi_count > 0:
    avg_before = leading_df[leading_df['rsi_oversold_before'] > 0]['rsi_oversold_before'].mean()
    print(f"  평균 {avg_before:.1f}봉 전에 반응")

print(f"\nWilliams 과매도(<-80): {williams_count}개 ({williams_count / len(leading_df) * 100:.1f}%)")
if williams_count > 0:
    avg_before = leading_df[leading_df['williams_oversold_before'] > 0]['williams_oversold_before'].mean()
    print(f"  평균 {avg_before:.1f}봉 전에 반응")

# 반응 순서
print(f"\n가장 빨리 반응하는 지표:")
first_reactions = []
for _, row in leading_df.iterrows():
    reactions = {
        'BB': row['bb_touch_before'],
        'RSI': row['rsi_oversold_before'],
        'Williams': row['williams_oversold_before'],
    }

    # 0이 아닌 것 중 최대값 (가장 먼저 반응)
    valid = {k: v for k, v in reactions.items() if v > 0}
    if valid:
        first = max(valid, key=valid.get)
        first_reactions.append(first)

if first_reactions:
    from collections import Counter
    counts = Counter(first_reactions)
    print(f"  BB 먼저: {counts.get('BB', 0)}회 ({counts.get('BB', 0) / len(first_reactions) * 100:.1f}%)")
    print(f"  RSI 먼저: {counts.get('RSI', 0)}회 ({counts.get('RSI', 0) / len(first_reactions) * 100:.1f}%)")
    print(f"  Williams 먼저: {counts.get('Williams', 0)}회 ({counts.get('Williams', 0) / len(first_reactions) * 100:.1f}%)")

# ═══════════════════════════════════════════════════════════
# 3. 조합 효과 분석
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3. 지표 조합 효과 - 승률 개선")
print("=" * 70)

def backtest_with_filters(trendline_up, df, bb_filter=None, rsi_filter=None, williams_filter=None):
    """필터 조건으로 백테스트"""

    trades = []

    for idx, signal in trendline_up.iterrows():
        break_idx = signal['break_idx']

        if break_idx >= len(df) - 1:
            continue

        breakout_candle = df.iloc[break_idx]

        # 필터 적용
        if bb_filter and not bb_filter(breakout_candle['bb_position']):
            continue

        if rsi_filter and not rsi_filter(breakout_candle['rsi']):
            continue

        if williams_filter and not williams_filter(breakout_candle['williams_r']):
            continue

        # 백테스트
        entry_price = signal['break_price']
        tp_price = entry_price * 1.02
        sl_price = entry_price * 0.98

        max_idx = min(break_idx + 50, len(df) - 1)
        future = df.iloc[break_idx:max_idx+1]

        exit_price = None

        for i in range(1, len(future)):
            candle = future.iloc[i]

            if candle['high'] >= tp_price:
                exit_price = tp_price
                break

            if candle['low'] <= sl_price:
                exit_price = sl_price
                break

        if exit_price is None:
            exit_price = future.iloc[-1]['close']

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'pnl_pct': pnl_pct,
        })

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100
    avg_pnl = trades_df['pnl_pct'].mean() - 0.11

    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
    }

# 기본 (필터 없음)
base = backtest_with_filters(trendline_up, df)

print(f"\n[기본] 필터 없음:")
print(f"  거래: {base['trades']}개")
print(f"  승률: {base['win_rate']:.1f}%")
print(f"  평균 PnL: {base['avg_pnl']:.3f}%")

# 조합 1: BB 하단
bb_lower = backtest_with_filters(
    trendline_up, df,
    bb_filter=lambda x: x < 0.3
)

if bb_lower:
    print(f"\n[조합 1] BB 하단 (<0.3):")
    print(f"  거래: {bb_lower['trades']}개")
    print(f"  승률: {bb_lower['win_rate']:.1f}% ({bb_lower['win_rate'] - base['win_rate']:+.1f}%p)")
    print(f"  평균 PnL: {bb_lower['avg_pnl']:.3f}%")

# 조합 2: RSI 과매도
rsi_oversold = backtest_with_filters(
    trendline_up, df,
    rsi_filter=lambda x: x < 35
)

if rsi_oversold:
    print(f"\n[조합 2] RSI 과매도 (<35):")
    print(f"  거래: {rsi_oversold['trades']}개")
    print(f"  승률: {rsi_oversold['win_rate']:.1f}% ({rsi_oversold['win_rate'] - base['win_rate']:+.1f}%p)")
    print(f"  평균 PnL: {rsi_oversold['avg_pnl']:.3f}%")

# 조합 3: Williams 과매도
williams_oversold = backtest_with_filters(
    trendline_up, df,
    williams_filter=lambda x: x < -70
)

if williams_oversold:
    print(f"\n[조합 3] Williams 과매도 (<-70):")
    print(f"  거래: {williams_oversold['trades']}개")
    print(f"  승률: {williams_oversold['win_rate']:.1f}% ({williams_oversold['win_rate'] - base['win_rate']:+.1f}%p)")
    print(f"  평균 PnL: {williams_oversold['avg_pnl']:.3f}%")

# 조합 4: BB + RSI
bb_rsi = backtest_with_filters(
    trendline_up, df,
    bb_filter=lambda x: x < 0.3,
    rsi_filter=lambda x: x < 40
)

if bb_rsi:
    print(f"\n[조합 4] BB 하단 + RSI 과매도:")
    print(f"  거래: {bb_rsi['trades']}개")
    print(f"  승률: {bb_rsi['win_rate']:.1f}% ({bb_rsi['win_rate'] - base['win_rate']:+.1f}%p)")
    print(f"  평균 PnL: {bb_rsi['avg_pnl']:.3f}%")

# 조합 5: BB + Williams
bb_williams = backtest_with_filters(
    trendline_up, df,
    bb_filter=lambda x: x < 0.3,
    williams_filter=lambda x: x < -70
)

if bb_williams:
    print(f"\n[조합 5] BB 하단 + Williams 과매도:")
    print(f"  거래: {bb_williams['trades']}개")
    print(f"  승률: {bb_williams['win_rate']:.1f}% ({bb_williams['win_rate'] - base['win_rate']:+.1f}%p)")
    print(f"  평균 PnL: {bb_williams['avg_pnl']:.3f}%")

# 조합 6: 트리플 (BB + RSI + Williams)
triple = backtest_with_filters(
    trendline_up, df,
    bb_filter=lambda x: x < 0.4,
    rsi_filter=lambda x: x < 45,
    williams_filter=lambda x: x < -65
)

if triple:
    print(f"\n[조합 6] 트리플 필터 (BB<0.4 + RSI<45 + Williams<-65):")
    print(f"  거래: {triple['trades']}개")
    print(f"  승률: {triple['win_rate']:.1f}% ({triple['win_rate'] - base['win_rate']:+.1f}%p) ⭐")
    print(f"  평균 PnL: {triple['avg_pnl']:.3f}%")

# ═══════════════════════════════════════════════════════════
# 4. 선행 진입 전략
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4. 선행 진입 전략 - 돌파 전 진입 가능?")
print("=" * 70)

def backtest_early_entry(trendline_up, df, entry_bars_before=1):
    """돌파 N봉 전에 진입"""

    trades = []

    for idx, signal in trendline_up.iterrows():
        break_idx = signal['break_idx']

        # N봉 전에 진입
        entry_idx = break_idx - entry_bars_before

        if entry_idx < 0 or entry_idx >= len(df) - 1:
            continue

        entry_candle = df.iloc[entry_idx]

        # 조건: BB 하단 + RSI 과매도
        if entry_candle['bb_position'] >= 0.4:
            continue
        if entry_candle['rsi'] >= 40:
            continue

        entry_price = entry_candle['close']
        tp_price = entry_price * 1.02
        sl_price = entry_price * 0.98

        max_idx = min(entry_idx + 50, len(df) - 1)
        future = df.iloc[entry_idx:max_idx+1]

        exit_price = None

        for i in range(1, len(future)):
            candle = future.iloc[i]

            if candle['high'] >= tp_price:
                exit_price = tp_price
                break

            if candle['low'] <= sl_price:
                exit_price = sl_price
                break

        if exit_price is None:
            exit_price = future.iloc[-1]['close']

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'pnl_pct': pnl_pct,
        })

    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100
    avg_pnl = trades_df['pnl_pct'].mean() - 0.11

    return {
        'trades': len(trades_df),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
    }

# 1~3봉 전 진입 테스트
for bars_before in [1, 2, 3]:
    early = backtest_early_entry(trendline_up, df, entry_bars_before=bars_before)

    if early:
        print(f"\n돌파 {bars_before}봉 전 진입 (BB<0.4 + RSI<40):")
        print(f"  거래: {early['trades']}개")
        print(f"  승률: {early['win_rate']:.1f}%")
        print(f"  평균 PnL: {early['avg_pnl']:.3f}%")

print("\n" + "=" * 70)
print("분석 완료!")
print("=" * 70)
