"""
실시간 BTC 나우캐스트 시스템
실시간으로 데이터를 가져와 다음 15분 가격을 예측합니다.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from technical_indicators import calculate_indicators
from btc_nowcast_model import BTCNowcastModel


class RealtimeNowcast:
    """실시간 나우캐스트 시스템"""

    def __init__(self, model_path='models/btc_nowcast', symbol='BTC/USDT:USDT'):
        """
        Args:
            model_path: 훈련된 모델 경로
            symbol: 거래 심볼
        """
        self.symbol = symbol
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        # 모델 로드
        print(f"모델 로드 중: {model_path}")
        self.model = BTCNowcastModel()
        self.model.load(model_path)

        print(f"심볼: {self.symbol}")
        print(f"시퀀스 길이: {self.model.sequence_length}")
        print(f"특징 수: {len(self.model.feature_columns)}")

    def fetch_recent_data(self, lookback_hours=24):
        """
        최근 데이터 가져오기

        Args:
            lookback_hours: 조회할 과거 시간 (시간)

        Returns:
            DataFrame
        """
        # 필요한 캔들 수 계산 (15분 단위)
        num_candles = (lookback_hours * 60) // 15 + self.model.sequence_length + 50

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe='15m',
                limit=min(num_candles, 1000)
            )

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df

        except Exception as e:
            print(f"데이터 가져오기 에러: {e}")
            return None

    def predict_once(self):
        """
        한 번 예측 수행

        Returns:
            예측 결과 딕셔너리
        """
        # 최근 데이터 가져오기
        df = self.fetch_recent_data(lookback_hours=48)

        if df is None or len(df) < self.model.sequence_length + 50:
            print("데이터가 충분하지 않습니다.")
            return None

        # 기술적 지표 추가
        df_with_indicators = calculate_indicators(df)

        # NaN 제거
        df_clean = df_with_indicators.dropna()

        if len(df_clean) < self.model.sequence_length:
            print("지표 계산 후 데이터가 충분하지 않습니다.")
            return None

        # 예측
        try:
            current_price = df_clean['close'].iloc[-1]
            predicted_price, price_change = self.model.predict_price(
                df_clean,
                self.model.feature_columns
            )

            current_time = df_clean['datetime'].iloc[-1]
            prediction_time = current_time + timedelta(minutes=15)

            result = {
                'current_time': current_time,
                'prediction_time': prediction_time,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change * 100,
                'direction': 'UP' if price_change > 0 else 'DOWN'
            }

            return result

        except Exception as e:
            print(f"예측 에러: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_continuous(self, interval_seconds=60):
        """
        연속 예측 실행

        Args:
            interval_seconds: 예측 간격 (초)
        """
        print("\n" + "=" * 60)
        print("실시간 BTC 나우캐스트 시작")
        print("=" * 60)
        print(f"업데이트 간격: {interval_seconds}초")
        print("종료하려면 Ctrl+C를 누르세요.\n")

        prediction_count = 0

        try:
            while True:
                prediction_count += 1
                print(f"\n[예측 #{prediction_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 60)

                result = self.predict_once()

                if result:
                    print(f"현재 시간:     {result['current_time']}")
                    print(f"예측 시간:     {result['prediction_time']}")
                    print(f"현재 가격:     ${result['current_price']:,.2f}")
                    print(f"예측 가격:     ${result['predicted_price']:,.2f}")
                    print(f"예상 변화:     {result['price_change_pct']:+.2f}% ({result['direction']})")

                    # 거래 신호
                    if abs(result['price_change_pct']) > 0.5:
                        signal = "🔴 STRONG SELL" if result['price_change_pct'] < -0.5 else "🟢 STRONG BUY"
                        print(f"거래 신호:     {signal}")
                    elif abs(result['price_change_pct']) > 0.2:
                        signal = "🟠 SELL" if result['price_change_pct'] < -0.2 else "🟢 BUY"
                        print(f"거래 신호:     {signal}")
                    else:
                        print(f"거래 신호:     ⚪ HOLD")

                else:
                    print("예측 실패")

                # 대기
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n나우캐스트 종료")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='BTC 실시간 나우캐스트')
    parser.add_argument(
        '--model-path',
        default='models/btc_nowcast',
        help='모델 파일 경로'
    )
    parser.add_argument(
        '--symbol',
        default='BTC/USDT:USDT',
        help='거래 심볼'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='업데이트 간격 (초)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='한 번만 예측하고 종료'
    )

    args = parser.parse_args()

    # 모델 존재 확인
    if not os.path.exists(f"{args.model_path}_model.keras"):
        print(f"에러: 모델을 찾을 수 없습니다: {args.model_path}")
        print("먼저 train_nowcast.py를 실행하여 모델을 훈련시키세요.")
        return

    # 나우캐스트 시스템 초기화
    nowcast = RealtimeNowcast(
        model_path=args.model_path,
        symbol=args.symbol
    )

    if args.once:
        # 한 번만 예측
        print("\n" + "=" * 60)
        print("BTC 나우캐스트 (단일 예측)")
        print("=" * 60)

        result = nowcast.predict_once()

        if result:
            print(f"\n현재 시간:     {result['current_time']}")
            print(f"예측 시간:     {result['prediction_time']}")
            print(f"현재 가격:     ${result['current_price']:,.2f}")
            print(f"예측 가격:     ${result['predicted_price']:,.2f}")
            print(f"예상 변화:     {result['price_change_pct']:+.2f}% ({result['direction']})")
        else:
            print("\n예측 실패")
    else:
        # 연속 예측
        nowcast.run_continuous(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
