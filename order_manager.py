"""
주문 관리자 (Order Manager)
- Bybit API로 주문 실행
- 포지션 관리 (진입/청산)
- TP/SL 설정 및 모니터링
"""

from pybit.unified_trading import HTTP
from datetime import datetime
import logging
import time

import config

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class OrderManager:
    """주문 실행 및 포지션 관리"""

    def __init__(self):
        # Bybit API 클라이언트
        self.session = HTTP(
            testnet=config.USE_TESTNET,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET
        )

        self.active_positions = {}  # {symbol: position_info}
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()

        # 레버리지 설정
        self._set_leverage()

        logger.info("OrderManager 초기화 완료")
        logger.info(f"테스트넷: {config.USE_TESTNET}")
        logger.info(f"레버리지: {config.LEVERAGE}x")

    def _set_leverage(self):
        """레버리지 설정"""
        try:
            result = self.session.set_leverage(
                category="linear",
                symbol=config.SYMBOL,
                buyLeverage=str(config.LEVERAGE),
                sellLeverage=str(config.LEVERAGE)
            )
            logger.info(f"레버리지 설정 완료: {config.LEVERAGE}x")
            return result
        except Exception as e:
            logger.error(f"레버리지 설정 실패: {e}")
            return None

    def get_current_price(self):
        """현재가 조회"""
        try:
            ticker = self.session.get_tickers(
                category="linear",
                symbol=config.SYMBOL
            )
            price = float(ticker['result']['list'][0]['lastPrice'])
            return price
        except Exception as e:
            logger.error(f"현재가 조회 실패: {e}")
            return None

    def calculate_quantity(self, price):
        """
        주문 수량 계산

        포지션 크기: $100
        레버리지: 3x
        → 실제 주문 금액: $100 / 3 = $33.33
        """
        notional = config.POSITION_SIZE_USD / config.LEVERAGE
        qty = notional / price

        # Bybit 최소 주문 단위 (0.001 BTC)
        qty = round(qty, 3)

        return qty

    def execute_trade(self, signal):
        """
        거래 실행 (신호 발생 시)

        signal = {
            'timestamp': 1234567890000,
            'datetime': datetime,
            'type': 'LONG',
            'entry_price': 50000.0,
            ...
        }
        """
        if config.SYMBOL in self.active_positions:
            logger.warning("이미 활성 포지션 존재, 진입 거부")
            return None

        # 일일 손실 체크
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today

        if self.daily_pnl <= -config.MAX_DAILY_LOSS_PERCENT:
            logger.error(f"일일 최대 손실 도달: {self.daily_pnl:.2f}%")
            return None

        # 현재가 조회
        current_price = self.get_current_price()
        if current_price is None:
            logger.error("현재가 조회 실패")
            return None

        # 수량 계산
        qty = self.calculate_quantity(current_price)

        logger.info(f"🔥 주문 실행 준비")
        logger.info(f"  타입: {signal['type']}")
        logger.info(f"  진입가: ${current_price:,.2f}")
        logger.info(f"  수량: {qty} BTC")
        logger.info(f"  포지션 크기: ${config.POSITION_SIZE_USD}")

        # TP/SL 계산
        if signal['type'] == 'LONG':
            tp_price = current_price * (1 + config.TP_PERCENT / 100)
            sl_price = current_price * (1 - config.SL_PERCENT / 100)
            side = "Buy"
        else:
            # 숏은 현재 전략에서 미사용
            tp_price = current_price * (1 - config.TP_PERCENT / 100)
            sl_price = current_price * (1 + config.SL_PERCENT / 100)
            side = "Sell"

        # TP/SL 반올림 (Bybit는 0.5 단위)
        tp_price = round(tp_price * 2) / 2
        sl_price = round(sl_price * 2) / 2

        logger.info(f"  TP: ${tp_price:,.2f} (+{config.TP_PERCENT}%)")
        logger.info(f"  SL: ${sl_price:,.2f} (-{config.SL_PERCENT}%)")

        try:
            # 시장가 주문 (TP/SL 포함)
            order = self.session.place_order(
                category="linear",
                symbol=config.SYMBOL,
                side=side,
                orderType="Market",
                qty=str(qty),
                takeProfit=str(tp_price),
                stopLoss=str(sl_price),
                tpTriggerBy="LastPrice",
                slTriggerBy="LastPrice",
                positionIdx=0  # One-way mode
            )

            if order['retCode'] == 0:
                logger.info(f"✅ 주문 성공!")
                logger.info(f"  주문 ID: {order['result']['orderId']}")

                # 포지션 저장
                self.active_positions[config.SYMBOL] = {
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'qty': qty,
                    'side': side,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'order_id': order['result']['orderId']
                }

                # 텔레그램 알림 (선택)
                if config.ENABLE_TELEGRAM:
                    self._send_telegram(f"🔥 진입: {side} {qty} BTC @ ${current_price:,.2f}")

                return order

            else:
                logger.error(f"주문 실패: {order['retMsg']}")
                return None

        except Exception as e:
            logger.error(f"주문 실행 중 에러: {e}")
            return None

    def check_positions(self):
        """
        활성 포지션 체크 (TP/SL 도달 확인)

        Bybit는 TP/SL 자동 실행하므로 주로 로깅 용도
        """
        if not self.active_positions:
            return

        try:
            positions = self.session.get_positions(
                category="linear",
                symbol=config.SYMBOL
            )

            position_list = positions['result']['list']

            if len(position_list) == 0:
                # 포지션 없음 = TP/SL 청산됨
                if config.SYMBOL in self.active_positions:
                    logger.info("✅ 포지션 청산됨 (TP 또는 SL)")

                    # PnL 계산
                    entry_info = self.active_positions[config.SYMBOL]
                    entry_price = entry_info['entry_price']
                    current_price = self.get_current_price()

                    if current_price:
                        if entry_info['side'] == 'Buy':
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price * 100

                        pnl_pct *= config.LEVERAGE  # 레버리지 반영

                        logger.info(f"  진입가: ${entry_price:,.2f}")
                        logger.info(f"  청산가: ${current_price:,.2f}")
                        logger.info(f"  수익: {pnl_pct:+.2f}%")

                        self.daily_pnl += pnl_pct

                    # 텔레그램 알림
                    if config.ENABLE_TELEGRAM:
                        self._send_telegram(f"✅ 청산: PnL {pnl_pct:+.2f}%")

                    # 포지션 삭제
                    del self.active_positions[config.SYMBOL]

            else:
                # 포지션 여전히 활성
                pos = position_list[0]
                unrealized_pnl = float(pos['unrealisedPnl'])
                logger.debug(f"활성 포지션: 미실현 손익 ${unrealized_pnl:,.2f}")

        except Exception as e:
            logger.error(f"포지션 조회 중 에러: {e}")

    def close_all_positions(self):
        """모든 포지션 강제 청산 (긴급 종료)"""
        try:
            positions = self.session.get_positions(
                category="linear",
                symbol=config.SYMBOL
            )

            for pos in positions['result']['list']:
                if float(pos['size']) > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"

                    self.session.place_order(
                        category="linear",
                        symbol=config.SYMBOL,
                        side=side,
                        orderType="Market",
                        qty=pos['size'],
                        reduceOnly=True,
                        positionIdx=0
                    )

                    logger.warning(f"⚠️ 긴급 청산: {pos['side']} {pos['size']} BTC")

            self.active_positions.clear()

        except Exception as e:
            logger.error(f"긴급 청산 중 에러: {e}")

    def _send_telegram(self, message):
        """텔레그램 알림"""
        if not config.ENABLE_TELEGRAM:
            return

        try:
            import requests
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message
            }
            requests.post(url, data=data)
        except Exception as e:
            logger.error(f"텔레그램 알림 실패: {e}")

    def get_status(self):
        """현재 상태 조회"""
        return {
            'active_positions': len(self.active_positions),
            'daily_pnl': self.daily_pnl,
            'positions': self.active_positions
        }


if __name__ == "__main__":
    # 테스트 (주의: 실제 API 호출!)
    print("=== 주문 관리자 테스트 ===")
    print(f"테스트넷: {config.USE_TESTNET}")

    if not config.USE_TESTNET:
        print("⚠️ 경고: 실전 모드입니다!")
        response = input("계속하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("취소됨")
            exit()

    manager = OrderManager()

    # 현재가 조회 테스트
    price = manager.get_current_price()
    print(f"\n현재가: ${price:,.2f}")

    # 수량 계산 테스트
    qty = manager.calculate_quantity(price)
    print(f"주문 수량: {qty} BTC")
    print(f"주문 금액: ${qty * price:,.2f}")

    # 포지션 조회 테스트
    print("\n포지션 조회...")
    manager.check_positions()

    print("\n상태:")
    status = manager.get_status()
    print(f"  활성 포지션: {status['active_positions']}개")
    print(f"  오늘 PnL: {status['daily_pnl']:+.2f}%")
