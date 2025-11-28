"""
MTF 데이터 수집
- 1시간 (60분)
- 4시간 (240분)
- 1일 (D)
- Bybit API 사용
- 15분 데이터와 동일 기간 (5년)
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

print("=" * 60)
print("MTF 데이터 수집 (Bybit)")
print("=" * 60)

# Bybit 거래소 연결 (Linear - 선물)
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear',  # USDT Perpetual
    }
})

symbol = 'BTC/USDT:USDT'  # Perpetual futures

# 15분 데이터와 동일 기간 설정
# 15분 데이터 확인
df_15m = pd.read_csv('output_phase1_labeled.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

start_date = df_15m['datetime'].min()
end_date = df_15m['datetime'].max()

print(f"\nSymbol: {symbol}")
print(f"기간: {start_date} ~ {end_date}")
print(f"일수: {(end_date - start_date).days}일")

# 타임프레임별 수집
timeframes = {
    '1h': '1시간',
    '4h': '4시간',
    '1d': '1일',
}

def download_ohlcv(timeframe, since, limit=1000):
    """OHLCV 데이터 다운로드"""
    try:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )
        return ohlcv
    except Exception as e:
        print(f"  에러: {e}")
        return None

def collect_timeframe_data(timeframe, name):
    """타임프레임별 데이터 수집"""

    print(f"\n{'-' * 60}")
    print(f"{name} ({timeframe}) 수집 시작")
    print(f"{'-' * 60}")

    all_data = []
    since = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    batch = 0

    while since < end_timestamp:
        batch += 1
        print(f"\nBatch {batch}: {datetime.fromtimestamp(since/1000)}")

        data = download_ohlcv(timeframe, since, limit=1000)

        if data is None or len(data) == 0:
            print("  데이터 없음, 종료")
            break

        all_data.extend(data)

        # 마지막 타임스탬프
        last_timestamp = data[-1][0]

        # 다음 시작점
        if timeframe == '1h':
            since = last_timestamp + 60 * 60 * 1000  # 1시간 후
        elif timeframe == '4h':
            since = last_timestamp + 4 * 60 * 60 * 1000  # 4시간 후
        elif timeframe == '1d':
            since = last_timestamp + 24 * 60 * 60 * 1000  # 1일 후

        print(f"  수집: {len(data)}개")
        print(f"  누적: {len(all_data)}개")

        # 종료 조건
        if since >= end_timestamp:
            print("  목표 기간 도달, 종료")
            break

        # Rate limit
        time.sleep(exchange.rateLimit / 1000)

    # DataFrame 변환
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 기간 필터링
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    print(f"\n최종 데이터:")
    print(f"  총 개수: {len(df)}개")
    print(f"  기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 저장
    filename = f'btcusdt_{timeframe}_raw.csv'
    df.to_csv(filename, index=False)
    print(f"  저장: {filename}")

    return df

# 각 타임프레임별 수집
results = {}

for tf, name in timeframes.items():
    try:
        df = collect_timeframe_data(tf, name)
        results[tf] = df
        print(f"\n✅ {name} 완료!")
    except Exception as e:
        print(f"\n❌ {name} 실패: {e}")

# 요약
print("\n" + "=" * 60)
print("수집 완료")
print("=" * 60)

print(f"\n기존 15분 데이터: {len(df_15m):,}개")
for tf, name in timeframes.items():
    if tf in results:
        print(f"{name}: {len(results[tf]):,}개")

# 타임프레임 비율 확인
print(f"\n타임프레임 비율 (15분 기준):")
if '1h' in results:
    ratio = len(df_15m) / len(results['1h'])
    print(f"  1시간: {ratio:.1f}배 (이론: 4배)")
if '4h' in results:
    ratio = len(df_15m) / len(results['4h'])
    print(f"  4시간: {ratio:.1f}배 (이론: 16배)")
if '1d' in results:
    ratio = len(df_15m) / len(results['1d'])
    print(f"  1일: {ratio:.1f}배 (이론: 96배)")

print("\n생성된 파일:")
print("  - btcusdt_1h_raw.csv")
print("  - btcusdt_4h_raw.csv")
print("  - btcusdt_1d_raw.csv")

print("\n다음 단계:")
print("  1. MTF L/H 라벨링")
print("  2. MTF Zone 추출")
print("  3. 상황 분류 (A~H)")
print("  4. 통합 분석")
