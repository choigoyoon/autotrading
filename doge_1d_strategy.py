"""
도지코인(DOGE) 1일봉 추세선 돌파 전략
BTC 15분봉과 동일한 로직 적용

사용법:
1. DOGE 1일봉 데이터 준비 (CSV 파일 또는 API)
2. 이 스크립트 실행
3. 매매 신호 확인

전략:
- MACD(12,26,9) 히스토그램으로 L/H 라벨링
- H값 연결 → 하락 추세선 생성
- 추세선 상향 돌파 → 롱 진입
- 10일 간격 필터 적용
- TP 10%, SL 10% (알트코인 변동성 고려)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

# ═══════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════

SYMBOL = 'DOGE/USDT'
TIMEFRAME = '1D'

# MACD 파라미터 (BTC와 동일)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 출구 전략 (추세 추종)
EXIT_MODE = 'TREND'  # 'TREND' = 추세 끝까지, 'FIXED' = 고정 TP/SL

# 추세 추종 모드 (TREND)
TRAIL_STOP_PCT = 15.0  # 15% 트레일링 스탑 (고점 대비)
INITIAL_SL_PCT = 10.0  # 10% 초기 손절

# 고정 TP/SL 모드 (FIXED) - 스윙용
FIXED_TP_PCT = 50.0   # 50% (큰 수익 목표)
FIXED_SL_PCT = 10.0   # 10%

# 필터
MIN_INTERVAL_DAYS = 10  # 최소 10일 간격
REQUIRE_FVG = True       # FVG 필수 (확실한 자리만!)

# FVG 설정
FVG_LOOKBACK = 3  # 3봉으로 FVG 감지

# 수수료
FEE_TOTAL = 0.11  # 0.11%

# ═══════════════════════════════════════════════════════════
# 1. 데이터 로드 함수
# ═══════════════════════════════════════════════════════════

def load_doge_data_from_csv(csv_path):
    """
    CSV 파일에서 DOGE 데이터 로드

    CSV 형식:
    Date,Open,High,Low,Close,Volume
    2020-01-01,0.002,0.0021,0.0019,0.002,1000000
    ...
    """
    df = pd.read_csv(csv_path)
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

def load_doge_data_from_binance(days=1825):
    """
    Binance API에서 DOGE 데이터 수집 (대체 방법)
    """
    url = "https://api.binance.com/api/v3/klines"

    all_data = []
    end_time = int(time.time() * 1000)

    # 1000개씩 수집
    for _ in range(2):  # 2000일
        params = {
            'symbol': 'DOGEUSDT',
            'interval': '1d',
            'limit': 1000,
            'endTime': end_time
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if not isinstance(data, list):
                break

            all_data.extend(data)

            if len(data) < 1000:
                break

            end_time = int(data[0][0]) - 1
            time.sleep(0.5)

        except Exception as e:
            print(f"API 오류: {e}")
            break

    # 데이터프레임 변환
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])

    df = df.sort_values('timestamp').reset_index(drop=True)

    return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

# ═══════════════════════════════════════════════════════════
# 2. MACD 계산
# ═══════════════════════════════════════════════════════════

def calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
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

# ═══════════════════════════════════════════════════════════
# 3. L/H 라벨링
# ═══════════════════════════════════════════════════════════

def label_hl(df):
    """L/H 라벨링 (MACD 히스토그램 크로스)"""
    df['label'] = None
    df['label_price'] = None

    for i in range(1, len(df)):
        prev_hist = df.iloc[i-1]['macd_hist']
        curr_hist = df.iloc[i]['macd_hist']

        # 음수 → 양수 = L (저점)
        if prev_hist < 0 and curr_hist >= 0:
            df.loc[df.index[i], 'label'] = 'L'
            df.loc[df.index[i], 'label_price'] = df.iloc[i]['low']

        # 양수 → 음수 = H (고점)
        elif prev_hist >= 0 and curr_hist < 0:
            df.loc[df.index[i], 'label'] = 'H'
            df.loc[df.index[i], 'label_price'] = df.iloc[i]['high']

    return df

# ═══════════════════════════════════════════════════════════
# 4. 추세선 생성
# ═══════════════════════════════════════════════════════════

def generate_trendlines(df, min_touches=2):
    """
    하락 추세선 생성 (H값 연결)

    하락 추세선 = 저항선
    → 상향 돌파 시 롱 진입
    """
    labeled = df[df['label'] == 'H'].copy()
    trendlines = []

    for i in range(len(labeled) - min_touches + 1):
        sequence = [i]
        current_price = labeled.iloc[i]['label_price']

        for j in range(i + 1, len(labeled)):
            next_price = labeled.iloc[j]['label_price']

            # 하락 추세 (가격이 낮아짐)
            if next_price < current_price:
                sequence.append(j)
                current_price = next_price
            # 상승하면 추세선 종료
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

# ═══════════════════════════════════════════════════════════
# 5. 돌파 감지
# ═══════════════════════════════════════════════════════════

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

                    # 이전 봉: 아래, 현재 봉: 위
                    if prev_close <= prev_tl:
                        breakouts.append({
                            'break_idx': i,
                            'break_price': current_close,
                            'tl_price': tl_price,
                            'datetime': df.iloc[i]['datetime'],
                        })
                        break

    return breakouts

# ═══════════════════════════════════════════════════════════
# 6. FVG (Fair Value Gap) 감지
# ═══════════════════════════════════════════════════════════

def detect_fvg(df, idx):
    """
    FVG (Fair Value Gap) 감지

    Bullish FVG:
    candle[i-2].high < candle[i].low
    → 중간에 갭 존재 = 강한 상승 압력
    """
    if idx < 2:
        return False

    candle_now = df.iloc[idx]
    candle_2ago = df.iloc[idx - 2]

    # Bullish FVG (롱 진입용)
    if candle_2ago['high'] < candle_now['low']:
        return True

    return False

def add_fvg_flags(df, breakouts):
    """
    각 돌파 신호에 FVG 여부 추가
    """
    for bp in breakouts:
        break_idx = bp['break_idx']
        bp['has_fvg'] = detect_fvg(df, break_idx)

    return breakouts

# ═══════════════════════════════════════════════════════════
# 7. 간격 필터 + FVG 필터
# ═══════════════════════════════════════════════════════════

def filter_signals(breakouts, df, min_interval=MIN_INTERVAL_DAYS, require_fvg=REQUIRE_FVG):
    """
    신호 필터링

    1. 최소 N일 간격 유지
    2. FVG 필수 (옵션)
    """
    # FVG 플래그 추가
    breakouts = add_fvg_flags(df, breakouts)

    filtered = []
    last_idx = -999

    for bp in breakouts:
        # 간격 체크
        if bp['break_idx'] - last_idx < min_interval:
            continue

        # FVG 체크 (필수인 경우)
        if require_fvg and not bp['has_fvg']:
            continue

        filtered.append(bp)
        last_idx = bp['break_idx']

    return filtered

# ═══════════════════════════════════════════════════════════
# 7. 백테스트
# ═══════════════════════════════════════════════════════════

def backtest(df, breakouts, mode=EXIT_MODE):
    """
    백테스트 실행

    mode='TREND': 추세 추종 (트레일링 스탑 + MACD 반전)
    mode='FIXED': 고정 TP/SL
    """
    trades = []

    for bp in breakouts:
        break_idx = bp['break_idx']
        entry_price = bp['break_price']

        if mode == 'TREND':
            # 추세 추종 모드
            exit_price, exit_type, exit_date = backtest_trend_following(
                df, break_idx, entry_price
            )
        else:
            # 고정 TP/SL 모드
            exit_price, exit_type, exit_date = backtest_fixed_tpsl(
                df, break_idx, entry_price
            )

        pnl = (exit_price - entry_price) / entry_price * 100

        trades.append({
            'entry_date': bp['datetime'],
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl,
            'pnl_after_fee': pnl - FEE_TOTAL,
        })

    return pd.DataFrame(trades)

def backtest_trend_following(df, break_idx, entry_price):
    """
    추세 추종 전략

    출구:
    1. 초기 SL (진입가 대비 -10%)
    2. 트레일링 스탑 (최고가 대비 -15%)
    3. MACD 히스토그램 음전환 (추세 종료)
    """
    initial_sl = entry_price * (1 - INITIAL_SL_PCT / 100)
    highest_price = entry_price
    trailing_stop = initial_sl

    # 향후 200일 스캔 (큰 추세 캡처)
    max_idx = min(break_idx + 200, len(df) - 1)
    future = df.iloc[break_idx:max_idx+1]

    exit_price = None
    exit_type = None
    exit_date = None

    for i in range(1, len(future)):
        candle = future.iloc[i]

        # 최고가 갱신
        if candle['high'] > highest_price:
            highest_price = candle['high']
            # 트레일링 스탑 업데이트
            trailing_stop = highest_price * (1 - TRAIL_STOP_PCT / 100)

        # 1. 트레일링 스탑 도달
        if candle['low'] <= trailing_stop:
            exit_price = trailing_stop
            exit_type = 'TRAIL_STOP'
            exit_date = candle['datetime']
            break

        # 2. MACD 반전 (양수 → 음수)
        if i < len(future) - 1:  # 다음 봉이 있어야 확인 가능
            curr_hist = candle['macd_hist']
            next_hist = future.iloc[i+1]['macd_hist']

            if curr_hist >= 0 and next_hist < 0:
                # 추세 종료! 다음봉 시가에 청산
                exit_price = future.iloc[i+1]['open']
                exit_type = 'MACD_REVERSAL'
                exit_date = future.iloc[i+1]['datetime']
                break

    # 200일 내 출구 없으면 마지막 종가
    if exit_price is None:
        exit_price = future.iloc[-1]['close']
        exit_type = 'TIMEOUT'
        exit_date = future.iloc[-1]['datetime']

    return exit_price, exit_type, exit_date

def backtest_fixed_tpsl(df, break_idx, entry_price):
    """
    고정 TP/SL 전략 (스윙 트레이딩)
    """
    tp_price = entry_price * (1 + FIXED_TP_PCT / 100)
    sl_price = entry_price * (1 - FIXED_SL_PCT / 100)

    # 향후 100일 스캔
    max_idx = min(break_idx + 100, len(df) - 1)
    future = df.iloc[break_idx:max_idx+1]

    exit_price = None
    exit_type = None
    exit_date = None

    for i in range(1, len(future)):
        candle = future.iloc[i]

        # TP 체크
        if candle['high'] >= tp_price:
            exit_price = tp_price
            exit_type = 'TP'
            exit_date = candle['datetime']
            break

        # SL 체크
        if candle['low'] <= sl_price:
            exit_price = sl_price
            exit_type = 'SL'
            exit_date = candle['datetime']
            break

    # 100일 내 TP/SL 미도달
    if exit_price is None:
        exit_price = future.iloc[-1]['close']
        exit_type = 'TIMEOUT'
        exit_date = future.iloc[-1]['datetime']

    return exit_price, exit_type, exit_date

# ═══════════════════════════════════════════════════════════
# 8. 성과 분석
# ═══════════════════════════════════════════════════════════

def analyze_performance(trades_df):
    """성과 분석 및 리포트 출력"""

    if len(trades_df) == 0:
        print("\n⚠️ 거래 없음")
        return

    print("\n" + "=" * 70)
    print("DOGE 1일봉 백테스트 결과")
    print("=" * 70)

    # 기본 통계
    total_trades = len(trades_df)
    win_rate = (trades_df['pnl_after_fee'] > 0).sum() / total_trades * 100
    avg_pnl = trades_df['pnl_after_fee'].mean()

    print(f"\n기본 통계:")
    print(f"  총 거래: {total_trades}개")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 PnL: {avg_pnl:.2f}% (수수료 후)")

    # 출구 타입 분포
    print(f"\n출구 타입:")
    for exit_type in ['TP', 'SL', 'TIMEOUT']:
        count = (trades_df['exit_type'] == exit_type).sum()
        pct = count / total_trades * 100
        print(f"  {exit_type}: {count}개 ({pct:.1f}%)")

    # 연도별 성과
    if 'entry_date' in trades_df.columns:
        trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
        years = sorted(trades_df['year'].unique())

        if len(years) > 1:
            print(f"\n연도별 성과:")
            for year in years:
                year_trades = trades_df[trades_df['year'] == year]
                yr_count = len(year_trades)
                yr_win = (year_trades['pnl_after_fee'] > 0).sum() / yr_count * 100
                yr_avg = year_trades['pnl_after_fee'].mean()

                print(f"  {year}: {yr_count}개, 승률 {yr_win:.1f}%, 평균 {yr_avg:+.2f}%")

    # 최종 수익
    print(f"\n최종 수익:")

    # 복리 계산
    balance = 1000
    for pnl in trades_df['pnl_after_fee']:
        balance *= (1 + pnl / 100)

    total_return = (balance - 1000) / 1000 * 100

    # 기간 계산
    if 'entry_date' in trades_df.columns:
        start_date = pd.to_datetime(trades_df['entry_date'].min())
        end_date = pd.to_datetime(trades_df['entry_date'].max())
        total_years = (end_date - start_date).days / 365

        if total_years > 0:
            annual_return = total_return / total_years
            print(f"  기간: {start_date.date()} ~ {end_date.date()} ({total_years:.1f}년)")
            print(f"  총 수익: {total_return:.1f}%")
            print(f"  연간 수익: {annual_return:.1f}%")

    print(f"  최종 잔고: ${balance:,.2f} (초기 $1,000)")

    # $100 투기 시나리오
    balance_100 = 100
    for pnl in trades_df['pnl_after_fee']:
        balance_100 *= (1 + pnl / 100)

    profit_100 = balance_100 - 100

    print(f"\n💰 투기 시나리오 ($100):")
    print(f"  초기: $100")
    print(f"  최종: ${balance_100:,.2f}")
    print(f"  수익: ${profit_100:+.2f}")

    # 레버리지 3배
    balance_100_lev = 100
    for pnl in trades_df['pnl_after_fee']:
        lev_pnl = pnl * 3  # 3배 레버
        balance_100_lev *= (1 + lev_pnl / 100)

    profit_100_lev = balance_100_lev - 100

    print(f"\n💰 투기 + 레버리지 3배 ($100):")
    print(f"  초기: $100")
    print(f"  최종: ${balance_100_lev:,.2f}")
    print(f"  수익: ${profit_100_lev:+.2f}")

    # MDD
    cumulative = (1 + trades_df['pnl_after_fee'] / 100).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak * 100
    mdd = drawdown.min()

    print(f"\nMDD: {mdd:.2f}%")

    # 최대/최소 수익 거래
    if len(trades_df) > 0:
        best_trade = trades_df.loc[trades_df['pnl_after_fee'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl_after_fee'].idxmin()]

        print(f"\n최고 수익 거래:")
        print(f"  날짜: {best_trade['entry_date']}")
        print(f"  수익: {best_trade['pnl_after_fee']:.2f}%")
        print(f"  출구: {best_trade['exit_type']}")

        print(f"\n최악 손실 거래:")
        print(f"  날짜: {worst_trade['entry_date']}")
        print(f"  손실: {worst_trade['pnl_after_fee']:.2f}%")
        print(f"  출구: {worst_trade['exit_type']}")

    print("\n" + "=" * 70)

# ═══════════════════════════════════════════════════════════
# 9. 메인 실행
# ═══════════════════════════════════════════════════════════

def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("DOGE 1일봉 추세선 돌파 전략")
    print("=" * 70)

    print(f"\n전략 설정:")
    print(f"  타입: 투기 전략 (확실한 자리만!)")
    print(f"  MACD: ({MACD_FAST}, {MACD_SLOW}, {MACD_SIGNAL})")
    print(f"\n진입 조건:")
    print(f"  1. 추세선 상향 돌파 (H값 연결)")
    print(f"  2. FVG 발생 필수: {'예' if REQUIRE_FVG else '아니오'}")
    print(f"  3. 최소 간격: {MIN_INTERVAL_DAYS}일")

    print(f"\n출구 전략: {EXIT_MODE}")
    if EXIT_MODE == 'TREND':
        print(f"  - 초기 손절: {INITIAL_SL_PCT}%")
        print(f"  - 트레일링 스탑: {TRAIL_STOP_PCT}% (최고가 대비)")
        print(f"  - MACD 반전 시 청산")
        print(f"  → 추세 끝까지 타기!")
    else:
        print(f"  - 고정 TP: {FIXED_TP_PCT}%")
        print(f"  - 고정 SL: {FIXED_SL_PCT}%")

    print(f"\n권장 포지션:")
    print(f"  투기 자금: $100")
    print(f"  레버리지: 3-5배 (선택)")
    print(f"  리스크: 전액 손실 가능 (투기)")

    # 1. 데이터 로드
    print(f"\n1. 데이터 로드...")

    try:
        # 방법 1: CSV 파일 (권장)
        # df = load_doge_data_from_csv('doge_1d_data.csv')

        # 방법 2: Binance API
        df = load_doge_data_from_binance(days=1825)

        print(f"  ✓ 수집 완료: {len(df)}개 캔들")
        print(f"  ✓ 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    except Exception as e:
        print(f"  ✗ 데이터 로드 실패: {e}")
        return

    # 2. MACD 계산
    print(f"\n2. MACD 계산...")
    df = calculate_macd(df)
    print(f"  ✓ 완료")

    # 3. L/H 라벨링
    print(f"\n3. L/H 라벨링...")
    df = label_hl(df)
    l_count = (df['label'] == 'L').sum()
    h_count = (df['label'] == 'H').sum()
    print(f"  ✓ L: {l_count}개, H: {h_count}개")

    # 4. 추세선 생성
    print(f"\n4. 추세선 생성...")
    trendlines = generate_trendlines(df, min_touches=2)
    print(f"  ✓ 하락 추세선: {len(trendlines)}개")

    # 5. 돌파 감지
    print(f"\n5. 돌파 감지...")
    breakouts = detect_breakouts(df, trendlines)
    print(f"  ✓ 돌파 신호: {len(breakouts)}개")

    # 6. 필터 적용 (간격 + FVG)
    print(f"\n6. 필터 적용...")
    print(f"  - 최소 간격: {MIN_INTERVAL_DAYS}일")
    print(f"  - FVG 필수: {'예' if REQUIRE_FVG else '아니오'}")
    filtered = filter_signals(breakouts, df, min_interval=MIN_INTERVAL_DAYS, require_fvg=REQUIRE_FVG)
    print(f"  ✓ 필터 후: {len(filtered)}개")

    if REQUIRE_FVG and len(filtered) > 0:
        fvg_count = sum(1 for bp in filtered if bp.get('has_fvg', False))
        print(f"  ✓ FVG 있는 신호: {fvg_count}개")

    # 7. 백테스트
    print(f"\n7. 백테스트 실행...")
    trades_df = backtest(df, filtered, mode=EXIT_MODE)
    print(f"  ✓ 완료")

    # 8. 성과 분석
    analyze_performance(trades_df)

    # 9. 결과 저장
    if len(trades_df) > 0:
        trades_df.to_csv('doge_1d_trades.csv', index=False)
        print(f"\n결과 저장: doge_1d_trades.csv")

if __name__ == '__main__':
    main()
