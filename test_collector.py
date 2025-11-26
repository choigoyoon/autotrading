"""
Bybit 데이터 수집기 테스트 스크립트
최근 7일 데이터만 수집하여 테스트합니다.
"""

from bybit_data_collector import BybitDataCollector

def main():
    collector = BybitDataCollector()

    print("=" * 60)
    print("Bybit 데이터 수집기 테스트 (최근 7일)")
    print("=" * 60)

    # BTCUSDT 최근 7일 데이터 수집
    symbol = "BTCUSDT"

    try:
        df = collector.collect_historical_data(
            symbol=symbol,
            interval="15",
            days_back=7  # 최근 7일만 테스트
        )

        if df is not None:
            collector.save_to_csv(df, symbol, interval="15")

            print(f"\n{symbol} 데이터 요약:")
            print(f"  - 레코드 수: {len(df)}")
            print(f"  - 시작 날짜: {df['datetime'].iloc[0]}")
            print(f"  - 종료 날짜: {df['datetime'].iloc[-1]}")
            print(f"  - 최고가: {df['high'].max()}")
            print(f"  - 최저가: {df['low'].min()}")

            print("\n처음 5개 레코드:")
            print(df.head())

            print("\n테스트 성공! 전체 데이터를 수집하려면 bybit_data_collector.py를 실행하세요.")

    except Exception as e:
        print(f"에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
