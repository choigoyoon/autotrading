"""
샘플 BTC 데이터 생성기
실제 시장과 유사한 패턴을 가진 샘플 15분 봉 데이터를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_realistic_btc_data(start_date, num_days=1825, interval_minutes=15):
    """
    실제 시장과 유사한 BTC 15분 봉 데이터 생성

    Args:
        start_date: 시작 날짜
        num_days: 생성할 일수 (기본 5년 = 1825일)
        interval_minutes: 봉 간격 (분)

    Returns:
        DataFrame with OHLCV data
    """
    # 시간 인덱스 생성
    num_candles = int(num_days * 24 * 60 / interval_minutes)
    timestamps = [start_date + timedelta(minutes=i*interval_minutes) for i in range(num_candles)]

    # 초기 가격
    initial_price = 10000

    # 가격 생성 (기하 브라운 운동 + 트렌드)
    np.random.seed(42)

    prices = [initial_price]

    # 장기 트렌드 패턴 (5년간)
    # 2020: 10k -> 30k (상승)
    # 2021: 30k -> 69k -> 30k (대상승 후 조정)
    # 2022: 30k -> 15k (하락)
    # 2023: 15k -> 30k (회복)
    # 2024-2025: 30k -> 70k (상승)

    trend_points = [
        (0, 1.0),           # 시작
        (365, 3.0),         # 1년: 3배
        (500, 6.9),         # 2021-04: ATH
        (550, 3.0),         # 2021-05: 반토막
        (670, 6.9),         # 2021-11: ATH2
        (900, 1.5),         # 2022-06: 루나
        (1000, 1.5),        # 2022-11: FTX
        (1200, 3.0),        # 2023: 회복
        (1600, 7.3),        # 2024-03: 신고점
        (1825, 6.5),        # 2025: 조정
    ]

    # 각 구간별 트렌드 계산
    def get_trend_multiplier(day_index):
        for i in range(len(trend_points) - 1):
            d1, m1 = trend_points[i]
            d2, m2 = trend_points[i + 1]
            if d1 <= day_index < d2:
                # 선형 보간
                progress = (day_index - d1) / (d2 - d1)
                return m1 + (m2 - m1) * progress
        return trend_points[-1][1]

    for i in range(1, num_candles):
        day_index = i * interval_minutes / (24 * 60)
        trend_mult = get_trend_multiplier(day_index)
        target_price = initial_price * trend_mult

        # 현재 가격에서 목표 가격으로 조금씩 이동
        drift = (target_price - prices[-1]) / (24 * 60 / interval_minutes * 30)  # 30일에 걸쳐 조정

        # 변동성
        volatility = 0.002  # 0.2% per 15min
        random_change = np.random.normal(0, volatility)

        # 가격 업데이트
        new_price = prices[-1] * (1 + random_change) + drift
        prices.append(max(new_price, 100))  # 최소 100달러

    # OHLCV 생성
    data = []

    for i in range(len(timestamps)):
        close = prices[i]

        # Open은 이전 Close
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1]

        # High, Low 생성
        high = close * (1 + abs(np.random.normal(0, 0.003)))
        low = close * (1 - abs(np.random.normal(0, 0.003)))

        # High는 open, close 보다 높아야 함
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume 생성 (가격 변동성에 비례)
        price_change = abs(close - open_price) / open_price
        base_volume = 1000000
        volume = base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 1.5)

        data.append({
            'timestamp': int(timestamps[i].timestamp() * 1000),
            'datetime': timestamps[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)
    return df


def main():
    print("=" * 60)
    print("샘플 BTC 데이터 생성")
    print("=" * 60)

    # 2020-01-01부터 5년치 데이터 생성
    start_date = datetime(2020, 1, 1)

    print(f"\n생성 시작: {start_date}")
    print(f"기간: 5년 (1825일)")
    print(f"간격: 15분")

    df = generate_realistic_btc_data(start_date, num_days=1825)

    print(f"\n생성 완료:")
    print(f"  총 캔들: {len(df):,}개")
    print(f"  시작: {df['datetime'].iloc[0]}")
    print(f"  종료: {df['datetime'].iloc[-1]}")
    print(f"  시작 가격: ${df['close'].iloc[0]:,.2f}")
    print(f"  종료 가격: ${df['close'].iloc[-1]:,.2f}")
    print(f"  최고가: ${df['high'].max():,.2f}")
    print(f"  최저가: ${df['low'].min():,.2f}")

    # 데이터 저장
    os.makedirs('data', exist_ok=True)
    filename = f"data/BTC_USDT_USDT_15m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

    print(f"\n저장 완료: {filename}")
    print(f"파일 크기: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")

    # 샘플 데이터 출력
    print("\n처음 5개 레코드:")
    print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())

    print("\n마지막 5개 레코드:")
    print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail())


if __name__ == "__main__":
    main()
