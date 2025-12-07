"""
Bybit 15분 봉 데이터 수집 스크립트 (CCXT 사용)
상장 초기부터 현재까지 모든 데이터를 수집합니다.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os


class BybitDataCollector:
    def __init__(self):
        """Bybit 거래소 초기화"""
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,  # API 레이트 리밋 자동 처리
            'options': {
                'defaultType': 'linear',  # USDT 무기한 선물
            }
        })

    def collect_historical_data(self, symbol, timeframe='15m', since_date=None):
        """
        지정된 기간의 모든 히스토리 데이터를 수집합니다.

        Args:
            symbol: 거래 심볼 (예: BTC/USDT:USDT)
            timeframe: 시간 프레임 (15m = 15분)
            since_date: 시작 날짜 (datetime 객체, None이면 가능한 모든 데이터)

        Returns:
            DataFrame
        """
        all_ohlcv = []

        # 시작 시간 설정
        if since_date:
            since = int(since_date.timestamp() * 1000)
        else:
            # 2020년 1월 1일부터 시작
            since = int(datetime(2020, 1, 1).timestamp() * 1000)

        print(f"\n{symbol} 데이터 수집 시작...")
        print(f"시작 시간: {datetime.fromtimestamp(since/1000)}")
        print(f"시간 프레임: {timeframe}")

        iteration = 0
        total_records = 0

        while True:
            try:
                iteration += 1

                # OHLCV 데이터 가져오기
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000  # 한 번에 최대 1000개
                )

                if not ohlcv:
                    print(f"더 이상 데이터가 없습니다.")
                    break

                # 데이터 추가
                all_ohlcv.extend(ohlcv)
                total_records += len(ohlcv)

                # 마지막 캔들의 시간
                last_timestamp = ohlcv[-1][0]
                last_datetime = datetime.fromtimestamp(last_timestamp / 1000)

                print(f"반복 {iteration}: {len(ohlcv)}개 레코드 수집 (총: {total_records}개)")
                print(f"마지막 데이터 시간: {last_datetime}")

                # 현재 시간에 도달하면 중단
                current_time = int(datetime.now().timestamp() * 1000)
                if last_timestamp >= current_time - (15 * 60 * 1000):
                    print(f"현재 시간에 도달했습니다.")
                    break

                # 다음 배치를 위해 since 업데이트 (마지막 타임스탬프 + 1ms)
                since = last_timestamp + 1

                # API 레이트 리밋 준수
                time.sleep(self.exchange.rateLimit / 1000)

                # 안전장치: 최대 반복 횟수 제한
                if iteration >= 10000:
                    print(f"최대 반복 횟수에 도달했습니다.")
                    break

            except ccxt.NetworkError as e:
                print(f"네트워크 에러: {e}. 재시도 중...")
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                print(f"거래소 에러: {e}")
                break
            except Exception as e:
                print(f"예상치 못한 에러: {e}")
                break

        if not all_ohlcv:
            print(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return None

        # DataFrame으로 변환
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 타임스탬프를 datetime으로 변환
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 중복 제거 (같은 타임스탬프)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        # 시간순으로 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"\n총 {len(df)}개의 고유한 레코드를 수집했습니다.")
        if len(df) > 0:
            print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

        return df

    def save_to_csv(self, df, symbol, timeframe='15m'):
        """
        DataFrame을 CSV 파일로 저장합니다.

        Args:
            df: 저장할 DataFrame
            symbol: 심볼 이름
            timeframe: 시간 프레임
        """
        if df is None or len(df) == 0:
            print("저장할 데이터가 없습니다.")
            return None

        # data 디렉토리 생성
        os.makedirs("data", exist_ok=True)

        # 파일명에서 특수문자 제거
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"data/{clean_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # CSV 저장
        df.to_csv(filename, index=False)
        print(f"\n데이터가 저장되었습니다: {filename}")
        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"파일 크기: {file_size_mb:.2f} MB")

        return filename

    def get_available_symbols(self, quote_currency='USDT'):
        """
        거래 가능한 심볼 목록을 가져옵니다.

        Args:
            quote_currency: 견적 통화 (기본값: USDT)

        Returns:
            심볼 리스트
        """
        try:
            markets = self.exchange.load_markets()
            symbols = [
                symbol for symbol, market in markets.items()
                if market.get('quote') == quote_currency and
                market.get('linear') and
                market.get('active')
            ]
            return sorted(symbols)
        except Exception as e:
            print(f"심볼 목록 가져오기 에러: {e}")
            return []


def main():
    """
    메인 실행 함수
    """
    # 수집할 심볼 리스트
    # CCXT 형식: 'BTC/USDT:USDT' (base/quote:settle)
    symbols = [
        'BTC/USDT:USDT',   # 비트코인
        'ETH/USDT:USDT',   # 이더리움
        # 추가 심볼을 여기에 넣으세요
        # 'SOL/USDT:USDT',   # 솔라나
        # 'XRP/USDT:USDT',   # 리플
        # 'DOGE/USDT:USDT',  # 도지코인
    ]

    collector = BybitDataCollector()

    print("=" * 60)
    print("Bybit 15분 봉 데이터 수집기 (CCXT)")
    print("=" * 60)

    # 사용 가능한 심볼 출력 (선택사항)
    # print("\n사용 가능한 USDT 선물 심볼:")
    # available_symbols = collector.get_available_symbols('USDT')
    # for i, sym in enumerate(available_symbols[:20], 1):
    #     print(f"  {i}. {sym}")
    # print(f"  ... 총 {len(available_symbols)}개\n")

    for symbol in symbols:
        try:
            # 데이터 수집
            df = collector.collect_historical_data(
                symbol=symbol,
                timeframe='15m',
                since_date=None  # 모든 가능한 데이터 수집
                # since_date=datetime.now() - timedelta(days=30)  # 최근 30일
            )

            # CSV 저장
            if df is not None and len(df) > 0:
                collector.save_to_csv(df, symbol, timeframe='15m')

                # 기본 통계 출력
                print(f"\n{symbol} 데이터 요약:")
                print(f"  - 레코드 수: {len(df)}")
                print(f"  - 시작 날짜: {df['datetime'].iloc[0]}")
                print(f"  - 종료 날짜: {df['datetime'].iloc[-1]}")
                print(f"  - 최고가: ${df['high'].max():,.2f}")
                print(f"  - 최저가: ${df['low'].min():,.2f}")
                print(f"  - 평균 거래량: {df['volume'].mean():.2f}")
                print(f"  - 총 거래량: {df['volume'].sum():.2f}")

                # 처음과 마지막 몇 개 레코드 출력
                print(f"\n처음 3개 레코드:")
                print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head(3).to_string(index=False))
                print(f"\n마지막 3개 레코드:")
                print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail(3).to_string(index=False))

        except Exception as e:
            print(f"\n{symbol} 처리 중 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            continue

        print("\n" + "=" * 60 + "\n")

    print("모든 데이터 수집이 완료되었습니다!")


if __name__ == "__main__":
    main()
