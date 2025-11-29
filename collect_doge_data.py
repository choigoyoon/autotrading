"""
DOGE 1일봉 데이터 수집
BTC와 동일한 방법 (ccxt Bybit)
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

print("=" * 60)
print("DOGE 1일봉 수집 (Bybit)")
print("=" * 60)

# Bybit 거래소 연결 (BTC 수집과 동일)
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear',  # USDT Perpetual
    }
})

symbol = 'DOGE/USDT:USDT'  # Perpetual futures
timeframe = '1d'

# 5년 데이터 수집
end_date = datetime.now()
start_date = end_date - timedelta(days=1825)  # 5년

print(f"\nSymbol: {symbol}")
print(f"Timeframe: {timeframe}")
print(f"기간: {start_date} ~ {end_date}")

all_data = []
since = int(start_date.timestamp() * 1000)
end_timestamp = int(end_date.timestamp() * 1000)

batch = 0

while since < end_timestamp:
    batch += 1
    print(f"\nBatch {batch}: {datetime.fromtimestamp(since/1000)}")

    try:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=1000
        )

        if not ohlcv or len(ohlcv) == 0:
            print("  데이터 없음, 종료")
            break

        all_data.extend(ohlcv)

        # 마지막 타임스탬프
        last_timestamp = ohlcv[-1][0]

        # 다음 시작점 (1일 후)
        since = last_timestamp + 24 * 60 * 60 * 1000

        print(f"  수집: {len(ohlcv)}개")
        print(f"  누적: {len(all_data)}개")

        # 종료 조건
        if since >= end_timestamp:
            print("  목표 기간 도달, 종료")
            break

        # Rate limit
        time.sleep(exchange.rateLimit / 1000)

    except Exception as e:
        print(f"  에러: {e}")
        break

# DataFrame 변환
if len(all_data) > 0:
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 기간 필터링
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    # 중복 제거
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    print(f"\n최종 데이터:")
    print(f"  총 개수: {len(df)}개")
    print(f"  기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  가격 범위: ${df['low'].min():.6f} ~ ${df['high'].max():.6f}")

    # CSV 형식으로 저장 (Yahoo Finance 형식과 동일)
    df_export = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_export['Date'] = df_export['datetime'].dt.strftime('%Y-%m-%d')
    df_export = df_export[['Date', 'open', 'high', 'low', 'close', 'volume']]
    df_export.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # 저장
    filename = 'doge_1d_real_data.csv'
    df_export.to_csv(filename, index=False)

    print(f"\n저장: {filename}")
    print(f"\n✅ 완료! 이제 전략 실행 가능:")
    print(f"  python doge_1d_strategy.py")

else:
    print("\n❌ 데이터 수집 실패")
