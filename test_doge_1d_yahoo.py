"""
도지코인(DOGE) 1일봉 추세선 돌파 전략 테스트
Yahoo Finance CSV 다운로드 사용

목표:
1. DOGE-USD 1일봉 데이터 수집
2. 동일한 전략 적용
3. 최종 수익 계산
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

print("=" * 70)
print("도지코인 1일봉 추세선 돌파 전략")
print("=" * 70)

# ═══════════════════════════════════════════════════════════
# 1. 도지코인 데이터 수집
# ═══════════════════════════════════════════════════════════

print("\n1. 데이터 수집 중...")

def fetch_doge_yahoo():
    """
    Yahoo Finance에서 DOGE-USD 데이터 다운로드
    """

    # 5년 전부터 현재까지
    end_date = int(time.time())
    start_date = end_date - (1825 * 86400)  # 5년

    url = f"https://query1.finance.yahoo.com/v7/finance/download/DOGE-USD"

    params = {
        'period1': start_date,
        'period2': end_date,
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            print(f"  오류: HTTP {response.status_code}")
            return pd.DataFrame()

        # CSV 파싱
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        df['datetime'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        df = df.sort_values('datetime').reset_index(drop=True)

        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"  오류: {e}")
        return pd.DataFrame()

# DOGE 데이터 수집
df = fetch_doge_yahoo()

if len(df) == 0:
    print("\n⚠️ 데이터 수집 실패!")
    print("\n대안: 기존 BTC 1일봉 결과 참조")
    print("  - BTC 1D: 9거래, 77.8% 승률, 21.8% 총 수익")
    print("  - DOGE는 변동성이 더 크므로 유사하거나 더 높은 수익 예상")
    exit(1)

print(f"\n  수집 완료: {len(df)}개 캔들")
print(f"  기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"  가격 범위: ${df['low'].min():.6f} ~ ${df['high'].max():.6f}")

# ═══════════════════════════════════════════════════════════
# 2. MACD 계산 및 L/H 라벨링
# ═══════════════════════════════════════════════════════════

print("\n2. MACD 계산 및 L/H 라벨링...")

def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal

    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    return df

def label_hl(df):
    """L/H 라벨링"""
    df['label'] = None
    df['label_price'] = None

    for i in range(1, len(df)):
        prev_hist = df.iloc[i-1]['macd_hist']
        curr_hist = df.iloc[i]['macd_hist']

        # 음수 → 양수 = L
        if prev_hist < 0 and curr_hist >= 0:
            df.loc[df.index[i], 'label'] = 'L'
            df.loc[df.index[i], 'label_price'] = df.iloc[i]['low']

        # 양수 → 음수 = H
        elif prev_hist >= 0 and curr_hist < 0:
            df.loc[df.index[i], 'label'] = 'H'
            df.loc[df.index[i], 'label_price'] = df.iloc[i]['high']

    return df

df = calculate_macd(df)
df = label_hl(df)

l_count = (df['label'] == 'L').sum()
h_count = (df['label'] == 'H').sum()

print(f"  L 라벨: {l_count}개")
print(f"  H 라벨: {h_count}개")

# ═══════════════════════════════════════════════════════════
# 3. 추세선 생성
# ═══════════════════════════════════════════════════════════

print("\n3. 추세선 생성...")

def generate_trendlines(df, min_touches=2):
    """추세선 생성 (하락 추세선만 - 롱 진입용)"""

    labeled = df[df['label'] == 'H'].copy()  # H값 연결
    trendlines = []

    for i in range(len(labeled) - min_touches + 1):
        sequence = [i]
        current_price = labeled.iloc[i]['label_price']

        for j in range(i + 1, len(labeled)):
            next_price = labeled.iloc[j]['label_price']

            # 하락하면 추가
            if next_price < current_price:
                sequence.append(j)
                current_price = next_price
            # 상승하면 종료
            elif next_price > current_price * 1.02:
                break

        if len(sequence) >= min_touches:
            start_idx = labeled.index[sequence[0]]
            end_idx = labeled.index[sequence[-1]]

            start_price = labeled.iloc[sequence[0]]['label_price']
            end_price = labeled.iloc[sequence[-1]]['label_price']

            duration = end_idx - start_idx
            slope = (end_price - start_price) / duration if duration > 0 else 0

            trendlines.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_price': start_price,
                'end_price': end_price,
                'slope': slope,
                'touches': len(sequence),
            })

    return trendlines

trendlines = generate_trendlines(df, min_touches=2)

print(f"  하락 추세선: {len(trendlines)}개")

# ═══════════════════════════════════════════════════════════
# 4. 돌파 감지
# ═══════════════════════════════════════════════════════════

print("\n4. 돌파 감지...")

def detect_breakouts(df, trendlines):
    """추세선 상향 돌파 감지"""

    breakouts = []

    for tl in trendlines:
        start_idx = tl['start_idx']
        end_idx = tl['end_idx']

        # 추세선 이후 100일 스캔
        search_end = min(end_idx + 100, len(df) - 1)

        for i in range(end_idx, search_end):
            # 추세선 가격 계산
            tl_price = tl['start_price'] + tl['slope'] * (i - start_idx)

            current_close = df.iloc[i]['close']

            # 상향 돌파?
            if current_close > tl_price:
                if i > 0:
                    prev_close = df.iloc[i-1]['close']
                    prev_tl = tl['start_price'] + tl['slope'] * (i - 1 - start_idx)

                    # 이전은 아래, 현재는 위
                    if prev_close <= prev_tl:
                        breakouts.append({
                            'break_idx': i,
                            'break_price': current_close,
                            'tl_price': tl_price,
                            'datetime': df.iloc[i]['datetime'],
                        })
                        break

    return breakouts

breakouts = detect_breakouts(df, trendlines)

print(f"  돌파 신호: {len(breakouts)}개")

# ═══════════════════════════════════════════════════════════
# 5. 10일 간격 필터
# ═══════════════════════════════════════════════════════════

print("\n5. 간격 필터 적용 (10일)...")

filtered = []
last_idx = -999

for bp in breakouts:
    if bp['break_idx'] - last_idx >= 10:
        filtered.append(bp)
        last_idx = bp['break_idx']

print(f"  필터 후: {len(filtered)}개")

# ═══════════════════════════════════════════════════════════
# 6. 백테스트
# ═══════════════════════════════════════════════════════════

print("\n6. 백테스트 (TP 10%, SL 10% - 알트코인용)...")

trades = []

for bp in filtered:
    break_idx = bp['break_idx']
    entry_price = bp['break_price']

    # 알트코인 변동성 고려 TP/SL 10%
    tp_price = entry_price * 1.10  # 10%
    sl_price = entry_price * 0.90  # 10%

    # 향후 50일 스캔
    max_idx = min(break_idx + 50, len(df) - 1)
    future = df.iloc[break_idx:max_idx+1]

    exit_price = None
    hit_tp = False

    for i in range(1, len(future)):
        candle = future.iloc[i]

        if candle['high'] >= tp_price:
            exit_price = tp_price
            hit_tp = True
            break

        if candle['low'] <= sl_price:
            exit_price = sl_price
            break

    if exit_price is None:
        exit_price = future.iloc[-1]['close']

    pnl = (exit_price - entry_price) / entry_price * 100

    trades.append({
        'date': bp['datetime'],
        'entry': entry_price,
        'exit': exit_price,
        'pnl': pnl,
        'hit_tp': hit_tp,
    })

trades_df = pd.DataFrame(trades)

# ═══════════════════════════════════════════════════════════
# 7. 결과 분석
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("결과")
print("=" * 70)

if len(trades_df) == 0:
    print("\n⚠️ 거래 없음")
else:
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
    avg_pnl = trades_df['pnl'].mean()
    avg_pnl_after_fee = avg_pnl - 0.11

    print(f"\n기본 통계:")
    print(f"  총 거래: {len(trades_df)}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:.2f}%")
    print(f"  수수료 후: {avg_pnl_after_fee:.2f}%")

    # 연도별
    if len(trades_df['date'].dt.year.unique()) > 1:
        print(f"\n연도별 성과:")
        for year in sorted(trades_df['date'].dt.year.unique()):
            year_trades = trades_df[trades_df['date'].dt.year == year]
            yr_win = (year_trades['pnl'] > 0).sum() / len(year_trades) * 100
            yr_avg = year_trades['pnl'].mean() - 0.11

            print(f"  {year}: {len(year_trades)}개, 승률 {yr_win:.1f}%, 평균 {yr_avg:+.2f}%")

    # 최종 수익 계산
    print(f"\n최종 수익 계산:")

    # 기간
    total_years = (trades_df['date'].max() - trades_df['date'].min()).days / 365
    trades_per_year = len(trades_df) / total_years if total_years > 0 else 0

    print(f"\n[단리, 레버리지 없음]")
    print(f"  연간 거래: {trades_per_year:.1f}회")
    print(f"  거래당: {avg_pnl_after_fee:.2f}%")
    print(f"  연간 수익: {trades_per_year * avg_pnl_after_fee:.1f}%")

    # 복리
    balance = 1000
    for pnl in trades_df['pnl']:
        actual_pnl = (pnl - 0.11) / 100
        balance *= (1 + actual_pnl)

    total_return = (balance - 1000) / 1000 * 100

    print(f"\n[복리, 레버리지 없음]")
    print(f"  초기 자본: $1,000")
    print(f"  최종 잔고: ${balance:,.2f}")
    print(f"  총 수익: {total_return:,.1f}%")
    if total_years > 0:
        print(f"  연간 수익: {total_return / total_years:.1f}%")

    # 레버리지 3배, 30% 포지션
    balance_lev = 1000
    for pnl in trades_df['pnl']:
        actual_pnl = (pnl - 0.11) * 3 * 0.3 / 100  # 3배 레버, 30% 포지션
        balance_lev *= (1 + actual_pnl)

    total_return_lev = (balance_lev - 1000) / 1000 * 100

    print(f"\n[복리, 3배 레버리지, 30% 포지션]")
    print(f"  초기 자본: $1,000")
    print(f"  최종 잔고: ${balance_lev:,.2f}")
    print(f"  총 수익: {total_return_lev:,.1f}%")
    if total_years > 0:
        print(f"  연간 수익: {total_return_lev / total_years:.1f}%")

    # MDD
    cumulative = (1 + (trades_df['pnl'] - 0.11) / 100).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak * 100
    mdd = drawdown.min()

    print(f"\nMDD: {mdd:.2f}%")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

    # 결과 저장
    trades_df.to_csv('doge_1d_backtest_results.csv', index=False)
    print(f"\n결과 저장: doge_1d_backtest_results.csv")
