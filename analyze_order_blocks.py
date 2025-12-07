"""
Order Block (OB) 기반 매매법 탐색

Order Block = 기관의 대량 주문이 발생한 영역
- 강한 지지/저항으로 작용
- 재방문 시 반등/하락 가능성 높음

목표:
1. Bullish OB 감지 (매수 영역)
2. Bearish OB 감지 (매도 영역)
3. OB + 추세선 돌파 조합
4. OB만으로 진입 가능한지 테스트
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("Order Block (OB) 기반 매매법 탐색")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\n데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"총 캔들: {len(df):,}개")

# ═══════════════════════════════════════════════════════════
# 1. Order Block 감지
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("1. Order Block 감지")
print("=" * 70)

def detect_order_blocks(df, lookback=20):
    """
    Order Block 감지

    Bullish OB:
    - 하락 후 강한 상승 캔들
    - 마지막 하락 캔들 = OB (기관 매수 영역)

    Bearish OB:
    - 상승 후 강한 하락 캔들
    - 마지막 상승 캔들 = OB (기관 매도 영역)
    """

    order_blocks = []

    for i in range(lookback, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 캔들 사이즈
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']

        # 평균 대비 큰 캔들인지 (2배 이상)
        avg_range = df.iloc[i-20:i]['close'].diff().abs().mean()
        is_strong = current_body > avg_range * 2

        if not is_strong:
            continue

        # Bullish OB: 강한 상승 캔들 직전의 하락 캔들
        if current['close'] > current['open']:  # 상승 캔들
            # 직전 캔들이 하락?
            if prev['close'] < prev['open']:
                # 최근 N봉 중 최저점?
                recent_lows = df.iloc[i-lookback:i]['low'].min()
                if prev['low'] <= recent_lows * 1.01:  # 최저점 근처
                    order_blocks.append({
                        'type': 'bullish',
                        'ob_idx': i - 1,  # 직전 하락 캔들
                        'ob_high': prev['high'],
                        'ob_low': prev['low'],
                        'ob_close': prev['close'],
                        'trigger_idx': i,  # 강한 상승 캔들
                        'strength': current_body / avg_range,
                        'datetime': prev['datetime'] if 'datetime' in df.columns else None,
                    })

        # Bearish OB: 강한 하락 캔들 직전의 상승 캔들
        elif current['close'] < current['open']:  # 하락 캔들
            # 직전 캔들이 상승?
            if prev['close'] > prev['open']:
                # 최근 N봉 중 최고점?
                recent_highs = df.iloc[i-lookback:i]['high'].max()
                if prev['high'] >= recent_highs * 0.99:  # 최고점 근처
                    order_blocks.append({
                        'type': 'bearish',
                        'ob_idx': i - 1,
                        'ob_high': prev['high'],
                        'ob_low': prev['low'],
                        'ob_close': prev['close'],
                        'trigger_idx': i,
                        'strength': current_body / avg_range,
                        'datetime': prev['datetime'] if 'datetime' in df.columns else None,
                    })

    return pd.DataFrame(order_blocks)

# OB 감지
order_blocks_df = detect_order_blocks(df, lookback=20)

print(f"\n감지된 Order Block: {len(order_blocks_df):,}개")
print(f"\nOB 타입 분포:")
print(order_blocks_df['type'].value_counts())

# 강도 분석
print(f"\nOB 강도 통계:")
print(f"  평균: {order_blocks_df['strength'].mean():.2f}배")
print(f"  중앙값: {order_blocks_df['strength'].median():.2f}배")
print(f"  최대: {order_blocks_df['strength'].max():.2f}배")

# ═══════════════════════════════════════════════════════════
# 2. OB 재방문 시 반응 분석
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2. OB 재방문 시 반응 분석")
print("=" * 70)

def test_ob_revisit(df, order_blocks_df, max_distance=50):
    """
    OB 재방문 시 가격 반응 테스트
    """

    revisits = []

    for idx, ob in order_blocks_df.iterrows():
        ob_type = ob['type']
        ob_idx = ob['ob_idx']
        ob_high = ob['ob_high']
        ob_low = ob['ob_low']

        # OB 이후 최대 N봉 스캔
        search_end = min(ob_idx + max_distance, len(df) - 1)

        for i in range(ob_idx + 2, search_end):  # 트리거 다음부터
            candle = df.iloc[i]

            # Bullish OB 재방문 (가격이 OB 영역으로 하락)
            if ob_type == 'bullish':
                if candle['low'] <= ob_high and candle['low'] >= ob_low:
                    # 재방문 확인! 이후 반등?

                    # 향후 10봉 체크
                    reaction_end = min(i + 10, len(df) - 1)
                    future = df.iloc[i:reaction_end+1]

                    max_rise = ((future['high'].max() - candle['close']) / candle['close'] * 100)
                    max_fall = ((future['low'].min() - candle['close']) / candle['close'] * 100)

                    revisits.append({
                        'ob_type': 'bullish',
                        'revisit_idx': i,
                        'bars_after_ob': i - ob_idx,
                        'max_rise': max_rise,
                        'max_fall': max_fall,
                        'bounced': max_rise > 1.0,  # 1% 이상 반등
                    })

                    break  # 첫 재방문만

            # Bearish OB 재방문 (가격이 OB 영역으로 상승)
            elif ob_type == 'bearish':
                if candle['high'] >= ob_low and candle['high'] <= ob_high:
                    # 재방문 확인! 이후 하락?

                    reaction_end = min(i + 10, len(df) - 1)
                    future = df.iloc[i:reaction_end+1]

                    max_rise = ((future['high'].max() - candle['close']) / candle['close'] * 100)
                    max_fall = ((future['low'].min() - candle['close']) / candle['close'] * 100)

                    revisits.append({
                        'ob_type': 'bearish',
                        'revisit_idx': i,
                        'bars_after_ob': i - ob_idx,
                        'max_rise': max_rise,
                        'max_fall': max_fall,
                        'rejected': max_fall < -1.0,  # 1% 이상 하락
                    })

                    break

    return pd.DataFrame(revisits)

revisits_df = test_ob_revisit(df, order_blocks_df, max_distance=50)

print(f"\nOB 재방문 케이스: {len(revisits_df):,}개")

if len(revisits_df) > 0:
    # Bullish OB 반등률
    bullish_revisits = revisits_df[revisits_df['ob_type'] == 'bullish']
    if len(bullish_revisits) > 0:
        bounce_rate = bullish_revisits['bounced'].sum() / len(bullish_revisits) * 100
        avg_rise = bullish_revisits['max_rise'].mean()

        print(f"\nBullish OB 재방문 ({len(bullish_revisits)}개):")
        print(f"  반등률: {bounce_rate:.1f}% (1%+ 상승)")
        print(f"  평균 상승: {avg_rise:.2f}%")
        print(f"  평균 하락: {bullish_revisits['max_fall'].mean():.2f}%")

    # Bearish OB 거부율
    bearish_revisits = revisits_df[revisits_df['ob_type'] == 'bearish']
    if len(bearish_revisits) > 0:
        reject_rate = bearish_revisits['rejected'].sum() / len(bearish_revisits) * 100
        avg_fall = bearish_revisits['max_fall'].mean()

        print(f"\nBearish OB 재방문 ({len(bearish_revisits)}개):")
        print(f"  거부율: {reject_rate:.1f}% (1%+ 하락)")
        print(f"  평균 하락: {avg_fall:.2f}%")
        print(f"  평균 상승: {bearish_revisits['max_rise'].mean():.2f}%")

# ═══════════════════════════════════════════════════════════
# 3. OB 기반 진입 전략 백테스트
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3. OB 기반 진입 전략")
print("=" * 70)

def backtest_ob_strategy(df, order_blocks_df, tp_pct=2.0, sl_pct=2.0):
    """
    OB 재방문 시 진입하는 전략
    """

    trades = []

    for idx, ob in order_blocks_df.iterrows():
        if ob['type'] != 'bullish':  # 롱만 테스트
            continue

        ob_idx = ob['ob_idx']
        ob_high = ob['ob_high']
        ob_low = ob['ob_low']

        # OB 이후 50봉 스캔
        search_end = min(ob_idx + 50, len(df) - 1)

        for i in range(ob_idx + 2, search_end):
            candle = df.iloc[i]

            # OB 재방문?
            if candle['low'] <= ob_high and candle['low'] >= ob_low:
                # 진입!
                entry_price = candle['close']
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)

                # 향후 50봉 체크
                exit_end = min(i + 50, len(df) - 1)
                future = df.iloc[i:exit_end+1]

                exit_price = None
                hit_tp = False

                for j in range(1, len(future)):
                    f_candle = future.iloc[j]

                    if f_candle['high'] >= tp_price:
                        exit_price = tp_price
                        hit_tp = True
                        break

                    if f_candle['low'] <= sl_price:
                        exit_price = sl_price
                        break

                if exit_price is None:
                    exit_price = future.iloc[-1]['close']

                pnl = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'hit_tp': hit_tp,
                })

                break  # 첫 재방문만

    return pd.DataFrame(trades)

ob_trades = backtest_ob_strategy(df, order_blocks_df, tp_pct=2.0, sl_pct=2.0)

if len(ob_trades) > 0:
    win_rate = (ob_trades['pnl'] > 0).sum() / len(ob_trades) * 100
    avg_pnl = ob_trades['pnl'].mean() - 0.11

    print(f"\nOB 재방문 진입 전략:")
    print(f"  총 거래: {len(ob_trades):,}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:.3f}% (수수료 후)")

    # 월간 수익
    months = (df['datetime'].max() - df['datetime'].min()).days / 30
    monthly_trades = len(ob_trades) / months
    monthly_return = monthly_trades * avg_pnl

    print(f"\n  월 거래: {monthly_trades:.1f}회")
    print(f"  월 수익: {monthly_return:.2f}%")
else:
    print(f"\n⚠️ OB 거래 없음")

# ═══════════════════════════════════════════════════════════
# 4. OB + 추세선 돌파 조합
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4. OB + 추세선 돌파 조합")
print("=" * 70)

# 기존 추세선 돌파 로드
breakouts = pd.read_csv('output_phase4_breakouts.csv')
trendline_up = breakouts[breakouts['type'] == 'trendline_up'].copy()

# 10봉 필터
filtered = []
last_idx = -999
for _, row in trendline_up.iterrows():
    if row['break_idx'] - last_idx >= 10:
        filtered.append(row)
        last_idx = row['break_idx']

trendline_filtered = pd.DataFrame(filtered)

print(f"\n추세선 돌파 (10봉 필터): {len(trendline_filtered):,}개")

# OB 근처에서 돌파한 케이스 찾기
def find_ob_confluence(breakouts_df, order_blocks_df, price_tolerance=0.02):
    """
    추세선 돌파가 OB 근처에서 발생했는지 확인
    """

    confluences = []

    for _, breakout in breakouts_df.iterrows():
        break_idx = breakout['break_idx']
        break_price = breakout['break_price']

        # 근처 OB 찾기 (±2% 이내)
        for _, ob in order_blocks_df.iterrows():
            if ob['type'] != 'bullish':
                continue

            ob_idx = ob['ob_idx']

            # OB가 돌파 전에 있어야 함
            if ob_idx >= break_idx:
                continue

            # 거리 체크 (시간)
            if break_idx - ob_idx > 50:  # 너무 멀면 스킵
                continue

            # 가격 체크
            ob_high = ob['ob_high']
            ob_low = ob['ob_low']

            price_diff = min(
                abs(break_price - ob_high) / break_price,
                abs(break_price - ob_low) / break_price
            )

            if price_diff <= price_tolerance:
                confluences.append({
                    'break_idx': break_idx,
                    'ob_idx': ob_idx,
                    'distance': break_idx - ob_idx,
                    'price_diff': price_diff * 100,
                })
                break

    return confluences

confluences = find_ob_confluence(trendline_filtered, order_blocks_df, price_tolerance=0.02)

print(f"\nOB + 추세선 돌파 합류:")
print(f"  합류 케이스: {len(confluences)}개")
print(f"  비율: {len(confluences) / len(trendline_filtered) * 100:.1f}%")

if len(confluences) > 0:
    confluences_df = pd.DataFrame(confluences)
    print(f"\n  평균 거리: {confluences_df['distance'].mean():.1f}봉")
    print(f"  평균 가격차: {confluences_df['price_diff'].mean():.2f}%")

print("\n" + "=" * 70)
print("분석 완료!")
print("=" * 70)

print(f"""
\n다음 단계:
1. OB 감지 로직 개선 (거래량 포함)
2. OB + FVG 조합 테스트
3. OB + 추세선 + MACD 트리플 필터
4. 최적 파라미터 찾기
""")
