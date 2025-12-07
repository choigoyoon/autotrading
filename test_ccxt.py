"""
CCXT 버전 간단 테스트 - 최근 7일 데이터
"""

from bybit_collector_ccxt import BybitDataCollector
from datetime import datetime, timedelta

def main():
    collector = BybitDataCollector()

    print("=" * 60)
    print("CCXT Bybit 데이터 수집기 테스트 (최근 7일)")
    print("=" * 60)

    symbol = 'BTC/USDT:USDT'

    try:
        # 최근 7일 데이터만 수집
        since_date = datetime.now() - timedelta(days=7)

        df = collector.collect_historical_data(
            symbol=symbol,
            timeframe='15m',
            since_date=since_date
        )

        if df is not None and len(df) > 0:
            collector.save_to_csv(df, symbol, timeframe='15m')

            print(f"\n{symbol} 데이터 요약:")
            print(f"  - 레코드 수: {len(df)}")
            print(f"  - 시작 날짜: {df['datetime'].iloc[0]}")
            print(f"  - 종료 날짜: {df['datetime'].iloc[-1]}")
            print(f"  - 최고가: ${df['high'].max():,.2f}")
            print(f"  - 최저가: ${df['low'].min():,.2f}")

            print("\n처음 5개 레코드:")
            print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())

            print("\n테스트 성공! 전체 데이터를 수집하려면 bybit_collector_ccxt.py를 실행하세요.")
        else:
            print("데이터를 수집하지 못했습니다.")

    except Exception as e:
        print(f"에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
