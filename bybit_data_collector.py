"""
Bybit 15분 봉 데이터 수집 스크립트
상장 초기부터 현재까지 모든 데이터를 수집합니다.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys


class BybitDataCollector:
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()

    def get_kline_data(self, symbol, interval="15", start_time=None, end_time=None, limit=1000):
        """
        Bybit API를 사용하여 K-line 데이터를 가져옵니다.

        Args:
            symbol: 거래 심볼 (예: BTCUSDT)
            interval: 시간 간격 (15 = 15분)
            start_time: 시작 시간 (밀리초 타임스탬프)
            end_time: 종료 시간 (밀리초 타임스탬프)
            limit: 가져올 데이터 개수 (최대 1000)

        Returns:
            데이터 리스트 또는 None
        """
        endpoint = "/v5/market/kline"
        url = f"{self.base_url}{endpoint}"

        params = {
            "category": "linear",  # USDT 무기한 선물
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        if start_time:
            params["start"] = int(start_time)
        if end_time:
            params["end"] = int(end_time)

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] == 0:
                return data["result"]["list"]
            else:
                print(f"API 에러: {data['retMsg']}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"요청 에러: {e}")
            return None
        except Exception as e:
            print(f"예상치 못한 에러: {e}")
            return None

    def collect_historical_data(self, symbol, interval="15", days_back=None):
        """
        지정된 기간의 모든 히스토리 데이터를 수집합니다.

        Args:
            symbol: 거래 심볼
            interval: 시간 간격 (15 = 15분)
            days_back: 수집할 과거 일수 (None이면 가능한 모든 데이터)

        Returns:
            DataFrame
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)

        # 시작 시간 설정 (days_back이 None이면 충분히 오래된 시간으로 설정)
        if days_back:
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        else:
            # 2020년 1월 1일부터 시작 (대부분의 코인이 이후에 상장됨)
            start_time = int(datetime(2020, 1, 1).timestamp() * 1000)

        print(f"\n{symbol} 데이터 수집 시작...")
        print(f"시작 시간: {datetime.fromtimestamp(start_time/1000)}")
        print(f"종료 시간: {datetime.fromtimestamp(end_time/1000)}")

        iteration = 0
        total_records = 0

        while True:
            iteration += 1

            # API 호출
            kline_data = self.get_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )

            if not kline_data or len(kline_data) == 0:
                print(f"더 이상 데이터가 없습니다.")
                break

            # 데이터 추가
            all_data.extend(kline_data)
            total_records += len(kline_data)

            # 가장 오래된 데이터의 시간을 다음 end_time으로 설정
            oldest_time = int(kline_data[-1][0])

            print(f"반복 {iteration}: {len(kline_data)}개 레코드 수집 (총: {total_records}개)")
            print(f"현재 데이터 시간: {datetime.fromtimestamp(oldest_time/1000)}")

            # 시작 시간에 도달하면 중단
            if oldest_time <= start_time:
                print(f"시작 시간에 도달했습니다.")
                break

            # 다음 배치를 위해 end_time 업데이트
            end_time = oldest_time - 1

            # API 레이트 리밋 방지를 위한 딜레이
            time.sleep(0.1)

            # 안전장치: 최대 반복 횟수 제한
            if iteration >= 10000:
                print(f"최대 반복 횟수에 도달했습니다.")
                break

        if not all_data:
            print(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return None

        # DataFrame으로 변환
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # 데이터 타입 변환
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['turnover'] = pd.to_numeric(df['turnover'])

        # 타임스탬프를 datetime으로 변환
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 시간순으로 정렬 (오래된 것부터)
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"\n총 {len(df)}개의 레코드를 수집했습니다.")
        print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

        return df

    def save_to_csv(self, df, symbol, interval="15"):
        """
        DataFrame을 CSV 파일로 저장합니다.

        Args:
            df: 저장할 DataFrame
            symbol: 심볼 이름
            interval: 시간 간격
        """
        if df is None or len(df) == 0:
            print("저장할 데이터가 없습니다.")
            return

        # data 디렉토리 생성
        os.makedirs("data", exist_ok=True)

        # 파일명 생성
        filename = f"data/{symbol}_{interval}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # CSV 저장
        df.to_csv(filename, index=False)
        print(f"\n데이터가 저장되었습니다: {filename}")
        print(f"파일 크기: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")

        return filename


def main():
    """
    메인 실행 함수
    """
    # 수집할 심볼 리스트 (필요에 따라 수정)
    symbols = [
        "BTCUSDT",   # 비트코인
        "ETHUSDT",   # 이더리움
        # 추가 심볼을 여기에 넣으세요
    ]

    collector = BybitDataCollector()

    print("=" * 60)
    print("Bybit 15분 봉 데이터 수집기")
    print("=" * 60)

    for symbol in symbols:
        try:
            # 데이터 수집
            df = collector.collect_historical_data(
                symbol=symbol,
                interval="15",
                days_back=None  # 모든 가능한 데이터 수집
            )

            # CSV 저장
            if df is not None:
                collector.save_to_csv(df, symbol, interval="15")

                # 기본 통계 출력
                print(f"\n{symbol} 데이터 요약:")
                print(f"  - 레코드 수: {len(df)}")
                print(f"  - 시작 날짜: {df['datetime'].iloc[0]}")
                print(f"  - 종료 날짜: {df['datetime'].iloc[-1]}")
                print(f"  - 최고가: {df['high'].max()}")
                print(f"  - 최저가: {df['low'].min()}")
                print(f"  - 평균 거래량: {df['volume'].mean():.2f}")

        except Exception as e:
            print(f"\n{symbol} 처리 중 에러 발생: {e}")
            continue

        print("\n" + "=" * 60 + "\n")

    print("모든 데이터 수집이 완료되었습니다!")


if __name__ == "__main__":
    main()
