"""
신호 엔진 (데이터 연동 + 매매 신호 생성)
- Bybit 웹소켓으로 15분봉 실시간 수신
- MACD 계산 및 L/H 라벨링
- 추세선 돌파 감지
- 나우캐스트 준수 (완성된 캔들만 사용)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from collections import deque

import config

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SignalEngine:
    """신호 생성 엔진"""

    def __init__(self):
        self.candles = deque(maxlen=config.BUFFER_SIZE)  # 최근 200개 캔들
        self.last_trade_idx = -999  # 마지막 거래 인덱스
        self.trade_count_today = 0
        self.last_reset_date = datetime.now().date()

        logger.info("SignalEngine 초기화 완료")
        logger.info(f"파라미터: TP={config.TP_PERCENT}%, SL={config.SL_PERCENT}%, MIN_BARS={config.MIN_BARS_BETWEEN_TRADES}")

    def add_candle(self, candle):
        """
        새 캔들 추가

        candle = {
            'timestamp': 1234567890000,
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 123.45
        }
        """
        self.candles.append(candle)
        logger.debug(f"캔들 추가: {datetime.fromtimestamp(candle['timestamp']/1000)} C={candle['close']}")

    def get_df(self):
        """캔들 버퍼를 DataFrame으로 변환"""
        if len(self.candles) < 50:
            return None

        df = pd.DataFrame(list(self.candles))
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def calculate_macd(self, df):
        """MACD 계산"""
        close = df['close']

        ema_fast = close.ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=config.MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        macd_hist = macd_line - signal_line

        df['macd_hist'] = macd_hist
        return df

    def label_lh(self, df):
        """
        L/H 라벨링 (나우캐스트)

        L: 히스토그램이 음수에서 양수로 전환 (i-1 < 0, i >= 0)
        H: 히스토그램이 양수에서 음수로 전환 (i-1 >= 0, i < 0)
        """
        df['label'] = None

        for i in range(1, len(df)):
            prev_hist = df.iloc[i-1]['macd_hist']
            curr_hist = df.iloc[i]['macd_hist']

            if prev_hist < 0 and curr_hist >= 0:
                df.iloc[i, df.columns.get_loc('label')] = 'L'
            elif prev_hist >= 0 and curr_hist < 0:
                df.iloc[i, df.columns.get_loc('label')] = 'H'

        return df

    def generate_trendlines(self, df):
        """
        하락 추세선 생성 (H점 연결)

        Returns: [(idx1, idx2, slope, intercept), ...]
        """
        h_points = df[df['label'] == 'H'].copy()

        if len(h_points) < config.TRENDLINE_MIN_TOUCHES:
            return []

        # 최근 50개 H점만 사용
        h_points = h_points.tail(config.TRENDLINE_LOOKBACK)

        trendlines = []

        for i in range(len(h_points) - 1):
            for j in range(i + 1, len(h_points)):
                p1_idx = h_points.index[i]
                p2_idx = h_points.index[j]

                p1_high = df.loc[p1_idx, 'high']
                p2_high = df.loc[p2_idx, 'high']

                # 하락 추세선만 (기울기 < 0)
                slope = (p2_high - p1_high) / (p2_idx - p1_idx)

                if slope >= 0:
                    continue

                intercept = p1_high - slope * p1_idx

                trendlines.append({
                    'idx1': p1_idx,
                    'idx2': p2_idx,
                    'slope': slope,
                    'intercept': intercept
                })

        return trendlines

    def detect_breakout(self, df, trendlines):
        """
        추세선 돌파 감지 (나우캐스트)

        조건:
        1. 이전 봉 종가가 추세선 아래
        2. 현재 봉 종가가 추세선 위
        3. 최소 간격 유지 (10봉)

        Returns: breakout signal or None
        """
        if len(df) < 2:
            return None

        current_idx = len(df) - 1

        # 간격 체크
        bars_since_last = current_idx - self.last_trade_idx
        if bars_since_last < config.MIN_BARS_BETWEEN_TRADES:
            logger.debug(f"간격 부족: {bars_since_last}봉 < {config.MIN_BARS_BETWEEN_TRADES}봉")
            return None

        # 일일 거래 횟수 체크
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.trade_count_today = 0
            self.last_reset_date = today

        if self.trade_count_today >= config.MAX_TRADES_PER_DAY:
            logger.warning(f"일일 최대 거래 횟수 도달: {self.trade_count_today}")
            return None

        prev_idx = current_idx - 1
        prev_candle = df.iloc[prev_idx]
        curr_candle = df.iloc[current_idx]

        for tl in trendlines:
            # 추세선이 현재 구간에 유효한지 확인
            if tl['idx2'] < prev_idx:
                continue

            # 추세선 가격 계산
            prev_tl_price = tl['slope'] * prev_idx + tl['intercept']
            curr_tl_price = tl['slope'] * current_idx + tl['intercept']

            # 돌파 조건
            if prev_candle['close'] < prev_tl_price and curr_candle['close'] > curr_tl_price:

                # 돌파가 발생!
                signal = {
                    'timestamp': curr_candle['timestamp'],
                    'datetime': curr_candle['datetime'],
                    'type': 'LONG',
                    'entry_price': curr_candle['close'],  # 현재 종가 (나우캐스트)
                    'trendline_price': curr_tl_price,
                    'tl_slope': tl['slope'],
                    'bars_since_last': bars_since_last
                }

                logger.info(f"🚀 돌파 신호 발생!")
                logger.info(f"  시간: {signal['datetime']}")
                logger.info(f"  진입가: ${signal['entry_price']:,.2f}")
                logger.info(f"  추세선: ${curr_tl_price:,.2f}")
                logger.info(f"  간격: {bars_since_last}봉")

                self.last_trade_idx = current_idx
                self.trade_count_today += 1

                return signal

        return None

    def check_signal(self):
        """
        신호 체크 (메인 루프에서 호출)

        Returns: signal dict or None
        """
        df = self.get_df()

        if df is None or len(df) < config.MACD_SLOW + config.MACD_SIGNAL:
            logger.debug(f"데이터 부족: {len(self.candles) if self.candles else 0}개")
            return None

        # MACD 계산
        df = self.calculate_macd(df)

        # L/H 라벨링
        df = self.label_lh(df)

        # 추세선 생성
        trendlines = self.generate_trendlines(df)

        if len(trendlines) == 0:
            logger.debug("추세선 없음")
            return None

        logger.debug(f"추세선 {len(trendlines)}개 생성")

        # 돌파 감지
        signal = self.detect_breakout(df, trendlines)

        return signal


# ═══════════════════════════════════════════════════════════
# 웹소켓 클라이언트 (Bybit)
# ═══════════════════════════════════════════════════════════

"""
웹소켓 연결은 pybit 또는 websocket-client 라이브러리 사용
실제 구현 시:

from pybit.unified_trading import WebSocket

ws = WebSocket(
    testnet=config.USE_TESTNET,
    channel_type="linear"
)

def handle_kline(message):
    # 15분봉 완성 시 호출
    candle = {
        'timestamp': message['data']['timestamp'],
        'open': float(message['data']['open']),
        'high': float(message['data']['high']),
        'low': float(message['data']['low']),
        'close': float(message['data']['close']),
        'volume': float(message['data']['volume'])
    }

    signal_engine.add_candle(candle)
    signal = signal_engine.check_signal()

    if signal:
        # 주문 실행!
        order_manager.execute_trade(signal)

ws.kline_stream(
    interval=15,
    symbol=config.SYMBOL,
    callback=handle_kline
)
"""


if __name__ == "__main__":
    # 테스트
    engine = SignalEngine()

    # 샘플 데이터로 테스트
    df_test = pd.read_csv('output_phase1_labeled.csv')
    df_test['datetime'] = pd.to_datetime(df_test['datetime'])

    print("=== 신호 엔진 테스트 ===")
    print(f"테스트 데이터: {len(df_test)}개 캔들")

    # 처음 100개 캔들로 버퍼 초기화
    for i in range(100):
        candle = {
            'timestamp': int(df_test.iloc[i]['datetime'].timestamp() * 1000),
            'open': df_test.iloc[i]['open'],
            'high': df_test.iloc[i]['high'],
            'low': df_test.iloc[i]['low'],
            'close': df_test.iloc[i]['close'],
            'volume': df_test.iloc[i]['volume']
        }
        engine.add_candle(candle)

    # 나머지 캔들로 신호 테스트
    signal_count = 0

    for i in range(100, min(1000, len(df_test))):
        candle = {
            'timestamp': int(df_test.iloc[i]['datetime'].timestamp() * 1000),
            'open': df_test.iloc[i]['open'],
            'high': df_test.iloc[i]['high'],
            'low': df_test.iloc[i]['low'],
            'close': df_test.iloc[i]['close'],
            'volume': df_test.iloc[i]['volume']
        }
        engine.add_candle(candle)

        signal = engine.check_signal()
        if signal:
            signal_count += 1
            print(f"\n신호 #{signal_count}")
            print(f"  시간: {signal['datetime']}")
            print(f"  진입가: ${signal['entry_price']:,.2f}")

    print(f"\n총 신호: {signal_count}개")
