# 🚀 실전 배포 완전 가이드

## 📋 목차
1. [현재 상태 점검](#현재-상태-점검)
2. [실전 배포 로드맵](#실전-배포-로드맵)
3. [1H 전략 구축](#1h-전략-구축)
4. [리스크 관리 시스템](#리스크-관리-시스템)
5. [Bybit API 연동](#bybit-api-연동)
6. [자동 거래 시스템](#자동-거래-시스템)
7. [모니터링 & 알림](#모니터링--알림)
8. [체크리스트](#최종-체크리스트)

---

## 현재 상태 점검

### ✅ 완료된 작업
- [x] 15분 MACD 파이프라인 (Phase 1-5)
- [x] MTF 데이터 수집 (1H/4H/1D)
- [x] 8가지 상황 분류 시스템
- [x] 상황별 파라미터 최적화
- [x] MTF Zone 추출
- [x] 나우캐스트 검증 (0건 위반)
- [x] 통합 백테스트 (승률 90.2%)

### ⏳ 실전 배포 전 필요 작업
- [ ] 1H 전략 백테스트
- [ ] 수수료/슬리피지 반영
- [ ] 복리 시뮬레이션
- [ ] API 연동 및 테스트
- [ ] 페이퍼 트레이딩 (1-2주)
- [ ] 소액 실전 테스트
- [ ] 모니터링 시스템 구축

---

## 실전 배포 로드맵

### 🎯 Phase 1: 백테스트 완성 (2-3일)

#### Day 1: 1H 전략 구축
```bash
# 1. 1H 데이터 생성
python create_1h_strategy.py

# 2. MACD L/H 라벨링
python label_1h_macd.py

# 3. 상황 분류
python classify_1h_situations.py

# 4. 파라미터 최적화
python optimize_1h_parameters.py

# 예상 결과:
# - 승률: 93-95%
# - 거래 수: ~2,500개 (5년)
# - 평균 TP: 4.0%, SL: 6.5%
```

#### Day 2: 현실적 백테스트
```bash
# 수수료/슬리피지 반영
python backtest_with_costs.py

# Bybit 수수료 구조:
# - Maker: 0.02%
# - Taker: 0.055%
# - 슬리피지: ~0.02-0.05%
# - 총 비용: ~0.1% per 거래

# 복리 백테스트
python compound_backtest.py

# Kelly Criterion 포지션 크기
python calculate_kelly_position.py
```

#### Day 3: 리스크 분석
```bash
# MDD (Maximum Drawdown) 분석
# Drawdown Duration
# 연속 손실 시나리오
# VaR (Value at Risk)
python risk_analysis.py
```

---

### 🎯 Phase 2: API 연동 (2-3일)

#### Bybit API 설정

1. **API 키 발급**
```python
# https://www.bybit.com/app/user/api-management
#
# 권한 설정:
# ✅ Read: 계좌 정보 조회
# ✅ Trade: 주문 생성/취소
# ❌ Withdraw: 출금 (보안상 비활성화)
#
# IP 화이트리스트: 반드시 설정!
```

2. **연결 테스트**
```python
# test_bybit_connection.py

import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_API_SECRET'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear',  # USDT Perpetual
    }
})

# 1. 연결 테스트
try:
    balance = exchange.fetch_balance()
    print(f"✅ API 연결 성공")
    print(f"USDT 잔고: {balance['USDT']['free']:.2f}")
except Exception as e:
    print(f"❌ API 연결 실패: {e}")

# 2. 시장 데이터 테스트
ticker = exchange.fetch_ticker('BTC/USDT:USDT')
print(f"현재 BTC 가격: {ticker['last']:.2f}")

# 3. 테스트 주문 (dry-run)
exchange.set_sandbox_mode(True)  # 테스트넷 사용
order = exchange.create_limit_buy_order(
    symbol='BTC/USDT:USDT',
    amount=0.001,  # 최소 수량
    price=ticker['last'] * 0.95  # 현재가 -5%
)
print(f"✅ 테스트 주문 성공: {order['id']}")
```

3. **환경 변수 설정**
```bash
# .env 파일 생성
cat > .env << EOF
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_secret_here
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

# gitignore에 추가
echo ".env" >> .gitignore
```

---

### 🎯 Phase 3: 페이퍼 트레이딩 (1-2주)

#### 실시간 신호 생성기
```python
# live_signal_generator.py

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

class LiveSignalGenerator:
    def __init__(self, exchange, symbol='BTC/USDT:USDT'):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = '1h'

        # 최적 파라미터 로드
        self.optimal_params = self.load_optimal_params()

        # MTF Zone 로드
        self.zones_4h = pd.read_csv('mtf_zones_4h_1h.csv')
        self.zones_1d = pd.read_csv('mtf_zones_1d_1h.csv')
        self.zones_1w = pd.read_csv('mtf_zones_1w_1h.csv')

    def fetch_latest_data(self, limit=500):
        """최신 데이터 가져오기"""
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol,
            self.timeframe,
            limit=limit
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def detect_trendline_breakout(self, df):
        """추세선 돌파 감지"""
        # Phase 2 로직 재사용
        # 나우캐스트 준수: 완성된 캔들만 사용

        latest_completed = df.iloc[-2]  # 마지막에서 2번째 (완성봉)

        # 추세선 계산
        # ... (기존 로직)

        return breakout_signal

    def classify_situation(self, df):
        """현재 상황 분류"""
        # 4H/1D/1W MACD 상태 확인
        # 나우캐스트: 완성된 캔들만 사용

        current_time = df.iloc[-1]['datetime']

        # MTF 데이터 로드
        df_4h = self.fetch_mtf_data('4h', current_time)
        df_1d = self.fetch_mtf_data('1d', current_time)
        df_1w = self.fetch_mtf_data('1w', current_time)

        # MACD 계산
        macd_4h = self.calculate_macd(df_4h)
        macd_1d = self.calculate_macd(df_1d)
        macd_1w = self.calculate_macd(df_1w)

        # 상황 분류
        situation = self.classify_from_macd(macd_1w, macd_1d, macd_4h)

        return situation

    def generate_signal(self):
        """실시간 신호 생성"""
        df = self.fetch_latest_data()

        # 1. 추세선 돌파 확인
        breakout = self.detect_trendline_breakout(df)

        if not breakout:
            return None

        # 2. 상황 분류
        situation = self.classify_situation(df)

        # 3. 파라미터 선택
        params = self.optimal_params[situation]

        # 4. 신호 생성
        signal = {
            'datetime': datetime.now(),
            'symbol': self.symbol,
            'direction': breakout['direction'],
            'entry_price': df.iloc[-1]['close'],
            'situation': situation,
            'tp_pct': params['tp'],
            'sl_pct': params['sl'],
            'tp_price': self.calculate_tp_price(
                df.iloc[-1]['close'],
                breakout['direction'],
                params['tp']
            ),
            'sl_price': self.calculate_sl_price(
                df.iloc[-1]['close'],
                breakout['direction'],
                params['sl']
            ),
        }

        return signal

    def run(self, interval_minutes=15):
        """실시간 모니터링 루프"""
        print("실시간 신호 생성기 시작...")

        while True:
            try:
                signal = self.generate_signal()

                if signal:
                    print(f"\n{'='*60}")
                    print(f"🚨 신호 발생!")
                    print(f"시간: {signal['datetime']}")
                    print(f"방향: {signal['direction']}")
                    print(f"진입: {signal['entry_price']:.2f}")
                    print(f"TP: {signal['tp_price']:.2f} (+{signal['tp_pct']:.1f}%)")
                    print(f"SL: {signal['sl_price']:.2f} (-{signal['sl_pct']:.1f}%)")
                    print(f"상황: {signal['situation']}")
                    print(f"{'='*60}\n")

                    # 페이퍼 트레이딩: 기록만
                    self.log_paper_trade(signal)

                    # 텔레그램 알림
                    self.send_telegram_alert(signal)

                # 대기
                time.sleep(interval_minutes * 60)

            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(60)

# 실행
if __name__ == '__main__':
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET'),
        'enableRateLimit': True,
    })

    generator = LiveSignalGenerator(exchange)
    generator.run(interval_minutes=15)
```

#### 페이퍼 트레이딩 결과 추적
```python
# paper_trading_tracker.py

class PaperTradingTracker:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []

    def execute_trade(self, signal):
        """페이퍼 트레이드 실행 (기록만)"""

        trade = {
            'entry_time': signal['datetime'],
            'entry_price': signal['entry_price'],
            'direction': signal['direction'],
            'tp_price': signal['tp_price'],
            'sl_price': signal['sl_price'],
            'situation': signal['situation'],
            'status': 'open',
        }

        self.trades.append(trade)

        print(f"페이퍼 트레이드 실행: {trade}")

    def check_exits(self, current_price):
        """청산 조건 확인"""

        for trade in self.trades:
            if trade['status'] != 'open':
                continue

            # TP 도달
            if trade['direction'] == 'long' and current_price >= trade['tp_price']:
                self.close_trade(trade, current_price, 'TP')

            elif trade['direction'] == 'short' and current_price <= trade['tp_price']:
                self.close_trade(trade, current_price, 'TP')

            # SL 도달
            elif trade['direction'] == 'long' and current_price <= trade['sl_price']:
                self.close_trade(trade, current_price, 'SL')

            elif trade['direction'] == 'short' and current_price >= trade['sl_price']:
                self.close_trade(trade, current_price, 'SL')

    def close_trade(self, trade, exit_price, reason):
        """트레이드 종료"""

        trade['exit_time'] = datetime.now()
        trade['exit_price'] = exit_price
        trade['exit_reason'] = reason
        trade['status'] = 'closed'

        # PnL 계산
        if trade['direction'] == 'long':
            pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price'] * 100
        else:
            pnl_pct = (trade['entry_price'] - exit_price) / trade['entry_price'] * 100

        trade['pnl_pct'] = pnl_pct

        # 자본 업데이트 (30% 포지션 가정)
        position_size = self.current_capital * 0.30
        pnl_amount = position_size * pnl_pct / 100

        self.current_capital += pnl_amount
        trade['pnl_amount'] = pnl_amount

        print(f"\n트레이드 종료:")
        print(f"  사유: {reason}")
        print(f"  수익: {pnl_pct:+.2f}% (${pnl_amount:+.2f})")
        print(f"  자본: ${self.current_capital:.2f}")

    def get_stats(self):
        """통계"""
        closed = [t for t in self.trades if t['status'] == 'closed']

        if len(closed) == 0:
            return None

        wins = [t for t in closed if t['pnl_pct'] > 0]

        return {
            'total_trades': len(closed),
            'wins': len(wins),
            'win_rate': len(wins) / len(closed) * 100,
            'total_pnl': sum(t['pnl_pct'] for t in closed),
            'avg_pnl': sum(t['pnl_pct'] for t in closed) / len(closed),
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
        }
```

---

### 🎯 Phase 4: 소액 실전 (2-4주)

#### 자동 거래 봇
```python
# auto_trading_bot.py

class AutoTradingBot:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config

        # 리스크 관리
        self.max_position_size = config['max_position_size']  # 30%
        self.max_daily_loss = config['max_daily_loss']  # -5%
        self.max_concurrent_trades = config['max_concurrent_trades']  # 3

        # 상태
        self.active_positions = []
        self.daily_pnl = 0

    def check_risk_limits(self):
        """리스크 한도 확인"""

        # 1. 일일 손실 한도
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"⚠️ 일일 손실 한도 도달: {self.daily_pnl:.2f}%")
            return False

        # 2. 동시 포지션 한도
        if len(self.active_positions) >= self.max_concurrent_trades:
            print(f"⚠️ 동시 포지션 한도 도달: {len(self.active_positions)}")
            return False

        return True

    def execute_trade(self, signal):
        """실제 거래 실행"""

        if not self.check_risk_limits():
            return None

        try:
            # 1. 잔고 확인
            balance = self.exchange.fetch_balance()
            available = balance['USDT']['free']

            # 2. 포지션 크기 계산
            position_size_usd = available * self.max_position_size

            # 3. BTC 수량 계산
            btc_amount = position_size_usd / signal['entry_price']

            # 4. 레버리지 설정
            self.exchange.set_leverage(3, signal['symbol'])

            # 5. 주문 생성
            if signal['direction'] == 'long':
                order = self.exchange.create_market_buy_order(
                    signal['symbol'],
                    btc_amount
                )
            else:
                order = self.exchange.create_market_sell_order(
                    signal['symbol'],
                    btc_amount
                )

            # 6. TP/SL 설정
            self.set_tp_sl(order['id'], signal)

            # 7. 포지션 기록
            position = {
                'order_id': order['id'],
                'signal': signal,
                'entry_time': datetime.now(),
                'amount': btc_amount,
            }
            self.active_positions.append(position)

            print(f"✅ 주문 실행: {order['id']}")
            self.send_telegram_alert(f"주문 실행: {signal['direction']} @ {signal['entry_price']}")

            return order

        except Exception as e:
            print(f"❌ 주문 실패: {e}")
            self.send_telegram_alert(f"주문 실패: {e}")
            return None

    def set_tp_sl(self, order_id, signal):
        """TP/SL 설정"""

        try:
            # Bybit TP/SL 설정
            self.exchange.private_post_position_trading_stop({
                'symbol': signal['symbol'].replace('/', '').replace(':USDT', ''),
                'take_profit': signal['tp_price'],
                'stop_loss': signal['sl_price'],
            })

            print(f"✅ TP/SL 설정 완료")

        except Exception as e:
            print(f"⚠️ TP/SL 설정 실패: {e}")

    def monitor_positions(self):
        """포지션 모니터링"""

        for position in self.active_positions:
            try:
                # 포지션 상태 확인
                orders = self.exchange.fetch_orders(
                    position['signal']['symbol'],
                    limit=10
                )

                # 청산 여부 확인
                # ... (로직)

            except Exception as e:
                print(f"모니터링 오류: {e}")

    def run(self):
        """메인 루프"""

        signal_generator = LiveSignalGenerator(self.exchange)

        print("🤖 자동 거래 봇 시작")
        print(f"최대 포지션 크기: {self.max_position_size * 100}%")
        print(f"일일 손실 한도: {self.max_daily_loss * 100}%")
        print(f"최대 동시 포지션: {self.max_concurrent_trades}")

        while True:
            try:
                # 1. 신호 확인
                signal = signal_generator.generate_signal()

                if signal:
                    self.execute_trade(signal)

                # 2. 기존 포지션 모니터링
                self.monitor_positions()

                # 3. 대기
                time.sleep(15 * 60)  # 15분

            except Exception as e:
                print(f"오류: {e}")
                self.send_telegram_alert(f"봇 오류: {e}")
                time.sleep(60)

# 설정
config = {
    'max_position_size': 0.30,  # 30%
    'max_daily_loss': 0.05,  # -5%
    'max_concurrent_trades': 3,
    'leverage': 3,
}

bot = AutoTradingBot(exchange, config)
bot.run()
```

---

## 리스크 관리 시스템

### 1. 포지션 크기 관리

```python
# position_sizer.py

class PositionSizer:
    def __init__(self, total_capital, risk_per_trade=0.02):
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade  # 2%

    def calculate_position_size(self, entry_price, sl_price, direction):
        """
        Kelly Criterion + 고정 리스크 혼합
        """

        # 1. 리스크 금액
        risk_amount = self.total_capital * self.risk_per_trade

        # 2. SL 거리
        if direction == 'long':
            sl_distance_pct = (entry_price - sl_price) / entry_price
        else:
            sl_distance_pct = (sl_price - entry_price) / entry_price

        # 3. 포지션 크기
        position_size = risk_amount / sl_distance_pct

        # 4. 한도 적용 (최대 40%)
        max_position = self.total_capital * 0.40
        position_size = min(position_size, max_position)

        return position_size

# 예시
sizer = PositionSizer(total_capital=10000, risk_per_trade=0.02)

entry = 50000
sl = 46750  # -6.5%
direction = 'long'

position = sizer.calculate_position_size(entry, sl, direction)
print(f"포지션 크기: ${position:.2f}")
# 출력: 포지션 크기: $3076.92
# (리스크 $200 / SL 6.5% = $3076)
```

### 2. 일일/주간 한도

```python
# risk_limits.py

class RiskLimits:
    def __init__(self):
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.consecutive_losses = 0

        # 한도
        self.max_daily_loss = -0.05  # -5%
        self.max_weekly_loss = -0.10  # -10%
        self.max_consecutive_losses = 3

    def update_pnl(self, pnl_pct):
        """PnL 업데이트"""
        self.daily_pnl += pnl_pct
        self.weekly_pnl += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def can_trade(self):
        """거래 가능 여부"""

        # 1. 일일 손실 한도
        if self.daily_pnl <= self.max_daily_loss:
            print(f"⛔ 일일 거래 중단 (손실 {self.daily_pnl:.2f}%)")
            return False

        # 2. 주간 손실 한도
        if self.weekly_pnl <= self.max_weekly_loss:
            print(f"⛔ 주간 거래 중단 (손실 {self.weekly_pnl:.2f}%)")
            return False

        # 3. 연속 손실 한도
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"⛔ 연속 손실 {self.consecutive_losses}회 - 거래 중단")
            return False

        return True

    def reset_daily(self):
        """일일 초기화"""
        self.daily_pnl = 0

    def reset_weekly(self):
        """주간 초기화"""
        self.weekly_pnl = 0
```

### 3. 비상 정지 조건

```python
# circuit_breaker.py

class CircuitBreaker:
    def __init__(self):
        self.is_active = False
        self.activation_time = None

    def check_market_conditions(self, df):
        """시장 이상 감지"""

        # 1. 급격한 가격 변동 (1시간에 ±10%)
        latest_range = (df.iloc[-1]['high'] - df.iloc[-1]['low']) / df.iloc[-1]['close']
        if latest_range > 0.10:
            self.activate("급격한 가격 변동")
            return True

        # 2. 거래량 급증 (평균의 5배)
        avg_volume = df['volume'].rolling(24).mean().iloc[-1]
        current_volume = df.iloc[-1]['volume']
        if current_volume > avg_volume * 5:
            self.activate("거래량 급증")
            return True

        # 3. API 오류 빈번
        # ... (로직)

        return False

    def activate(self, reason):
        """비상 정지 활성화"""
        self.is_active = True
        self.activation_time = datetime.now()

        print(f"🚨 비상 정지 활성화: {reason}")

        # 모든 포지션 청산
        self.close_all_positions()

        # 텔레그램 긴급 알림
        send_telegram_alert(f"🚨 비상 정지: {reason}")

    def close_all_positions(self):
        """모든 포지션 청산"""
        # ... (로직)
        pass
```

---

## 모니터링 & 알림

### 텔레그램 봇 설정

```python
# telegram_bot.py

import requests

class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send_message(self, text):
        """메시지 전송"""
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'HTML'
        }

        try:
            response = requests.post(url, json=payload)
            return response.json()
        except Exception as e:
            print(f"텔레그램 전송 실패: {e}")

    def send_trade_alert(self, signal, order_id):
        """거래 알림"""

        message = f"""
🚨 <b>거래 신호 발생</b>

📊 종목: {signal['symbol']}
📈 방향: {signal['direction'].upper()}
💰 진입: ${signal['entry_price']:.2f}

🎯 TP: ${signal['tp_price']:.2f} (+{signal['tp_pct']:.1f}%)
🛡 SL: ${signal['sl_price']:.2f} (-{signal['sl_pct']:.1f}%)

📌 상황: {signal['situation']}
🆔 주문ID: {order_id}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        self.send_message(message)

    def send_exit_alert(self, trade, reason):
        """청산 알림"""

        message = f"""
✅ <b>포지션 청산</b>

사유: {reason}
수익: {trade['pnl_pct']:+.2f}%
금액: ${trade['pnl_amount']:+.2f}

진입: ${trade['entry_price']:.2f}
청산: ${trade['exit_price']:.2f}

보유 시간: {trade['holding_time']}
        """

        self.send_message(message)

    def send_daily_report(self, stats):
        """일일 보고"""

        message = f"""
📊 <b>일일 트레이딩 리포트</b>

거래 수: {stats['trades']}
승률: {stats['win_rate']:.1f}%
수익: {stats['daily_pnl']:+.2f}%

자본: ${stats['capital']:.2f}
누적 수익률: {stats['total_return']:+.1f}%

⏰ {datetime.now().strftime('%Y-%m-%d')}
        """

        self.send_message(message)
```

---

## 최종 체크리스트

### 백테스트 단계
- [ ] 1H 데이터 생성 및 검증
- [ ] 파라미터 최적화 완료
- [ ] 수수료/슬리피지 반영 백테스트
- [ ] 복리 시뮬레이션
- [ ] 리스크 분석 (MDD, VaR)
- [ ] 승률 93%+ 달성 확인

### API 연동 단계
- [ ] Bybit API 키 발급
- [ ] IP 화이트리스트 설정
- [ ] 연결 테스트 성공
- [ ] 테스트넷에서 주문 테스트
- [ ] 환경 변수 보안 설정

### 페이퍼 트레이딩 단계
- [ ] 실시간 신호 생성기 구동
- [ ] 페이퍼 거래 추적 시스템
- [ ] 1-2주 실행
- [ ] 승률 90%+ 유지 확인
- [ ] 모든 엣지 케이스 테스트

### 실전 배포 단계
- [ ] 소액 자본 시작 (전체의 10%)
- [ ] 자동 거래 봇 활성화
- [ ] 리스크 한도 설정
- [ ] 텔레그램 알림 설정
- [ ] 비상 정지 시스템 준비
- [ ] 일일 모니터링 루틴 확립

### 리스크 관리 단계
- [ ] 포지션 크기 계산기 적용
- [ ] 일일/주간 손실 한도 설정
- [ ] 연속 손실 제한
- [ ] 비상 정지 조건 구현
- [ ] 모든 포지션에 TP/SL 필수

---

## 📊 예상 성과 (1H 전략, 3배 레버리지, 30% 포지션)

```
초기 자본: $10,000
월 거래 수: 30회
승률: 93%
평균 TP: 4.0%
평균 SL: 6.5%

월 수익 계산:
- 승리 거래: 30 × 0.93 = 27.9회
- 패배 거래: 30 × 0.07 = 2.1회
- 승리 수익: 27.9 × 4.0% × 30% = +33.5%
- 패배 손실: 2.1 × 6.5% × 30% = -4.1%
- 순수익: +29.4%/월

연간 수익 (복리):
10,000 × (1.294)^12 = $329,254 (3,192% 수익률)

⚠️ 주의: 실제 성과는 다를 수 있습니다!
```

---

## 🎯 최종 요약

### 실전 배포까지 남은 작업

**Week 1-2: 백테스트 완성**
- 1H 전략 구축
- 현실적 비용 반영
- 리스크 분석

**Week 3-4: API 연동**
- Bybit 설정
- 테스트 주문
- 보안 강화

**Week 5-6: 페이퍼 트레이딩**
- 실시간 신호 테스트
- 성과 검증

**Week 7+: 소액 실전**
- 10% 자본으로 시작
- 점진적 확대

---

**지금 바로 시작하시겠습니까?**

1. `python create_1h_strategy.py` 실행
2. 1H 파이프라인 구축
3. 백테스트 완성

선택해주세요!
