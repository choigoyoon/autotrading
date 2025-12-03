# 전략 사용 가이드 및 코드 구조

## 📊 5년 백테스트 결과 요약

### 🏆 추천 전략: RSI<30 AND MACD<0

| 지표 | 값 |
|------|-----|
| **총 거래** | 654회 (연 131회) |
| **승률** | **87.77%** |
| **총 수익률** | **+826%** (5년) |
| **평균 손익** | **+1.263%** |
| **MDD** | **-4.04%** |
| **Sharpe Ratio** | **26.07** |
| **Profit Factor** | **13.46** |
| **최대 연속 승** | 31회 |
| **최대 연속 패** | 3회 |

### 📈 연도별 성과

| 연도 | 거래 | 승률 | 수익률 | 평균 손익 |
|------|------|------|--------|----------|
| 2020 | 139회 | 89.9% | +172.48% | +1.241% |
| 2021 | 123회 | 87.8% | +162.95% | +1.325% |
| 2022 | 137회 | 83.9% | +154.61% | +1.129% |
| 2023 | 132회 | 89.4% | +175.21% | +1.327% |
| 2024 | 123회 | 87.8% | +160.76% | +1.307% |

**일관성:** 5년 내내 83.9%~89.9% 승률 유지 ✅

---

## 🎯 전략 사용 방법

### 진입 조건

1. **L값 확정** (1H MACD Dead Cross ~ Golden Cross 사이 최저점)
2. **L+1 시가 진입** (L값 확정 후 다음 봉 시가)
3. **필터 조건** (둘 다 충족):
   - RSI < 30 (과매도)
   - MACD Hist < 0 (하락 모멘텀)

### 청산 조건

- **TP**: +2.0% (비용 후 +1.79%)
- **SL**: -1.5% (비용 후 -1.71%)
- **최대 보유**: 50봉 (12.5시간)

### 비용

- **슬리피지**: 0.05% x 2 = 0.1%
- **수수료**: 0.055% x 2 = 0.11% (Bybit 테이커)
- **총 비용**: 0.21%

---

## 🏗️ 코드 구조 설계

### 1. 전체 아키텍처

```
autotrading/
├── core/
│   ├── __init__.py
│   ├── strategy.py          # 전략 로직 (공통)
│   ├── indicators.py        # 지표 계산
│   └── risk_manager.py      # 리스크 관리
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py            # 백테스트 엔진
│   ├── data_loader.py       # 과거 데이터 로드
│   └── reporter.py          # 성과 리포트
│
├── live/
│   ├── __init__.py
│   ├── trader.py            # 실시간 거래
│   ├── exchange.py          # 거래소 연결
│   ├── monitor.py           # 모니터링
│   └── logger.py            # 로깅
│
├── config/
│   ├── backtest_config.py   # 백테스트 설정
│   └── live_config.py       # 실거래 설정
│
└── utils/
    ├── __init__.py
    └── helpers.py           # 공통 유틸리티
```

### 2. 핵심 원칙

**✅ 해야 할 것:**
1. **전략 로직 공통화** - `core/strategy.py`에 단일 구현
2. **데이터 소스 분리** - 백테스트는 CSV, 실거래는 WebSocket
3. **주문 실행 추상화** - 백테스트는 시뮬레이션, 실거래는 API
4. **설정 파일 분리** - 백테스트/실거래 별도 설정

**❌ 하지 말아야 할 것:**
1. 백테스트와 실거래 코드 중복
2. 전략 로직에 데이터 소스 하드코딩
3. 환경별 코드 분기 (if/else)

---

## 📝 코드 구현 예시

### core/strategy.py (공통 전략 로직)

```python
class LValueStrategy:
    """
    L값 기반 전략 (백테스트/실거래 공통)
    """

    def __init__(self, tp_pct=2.0, sl_pct=-1.5,
                 rsi_threshold=30, macd_threshold=0):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.rsi_threshold = rsi_threshold
        self.macd_threshold = macd_threshold

    def check_entry_signal(self, l_candle):
        """
        진입 신호 확인 (L값 시점 데이터)

        Args:
            l_candle: L값 봉 데이터 (dict)
                - rsi: float
                - macd_hist: float

        Returns:
            bool: 진입 신호 여부
        """
        if l_candle['rsi'] < self.rsi_threshold and \
           l_candle['macd_hist'] < self.macd_threshold:
            return True
        return False

    def calculate_exit_levels(self, entry_price):
        """
        청산 레벨 계산

        Args:
            entry_price: float

        Returns:
            dict: {'tp': float, 'sl': float}
        """
        return {
            'tp': entry_price * (1 + self.tp_pct / 100),
            'sl': entry_price * (1 + self.sl_pct / 100)
        }

    def check_exit(self, bar, tp_level, sl_level):
        """
        청산 조건 확인

        Args:
            bar: 현재 봉 데이터 (dict)
                - open, high, low, close
            tp_level: TP 레벨
            sl_level: SL 레벨

        Returns:
            str or None: 'TP', 'SL', None
        """
        hit_tp = bar['high'] >= tp_level
        hit_sl = bar['low'] <= sl_level

        # 둘 다 터치
        if hit_tp and hit_sl:
            if bar['open'] <= sl_level:
                return 'SL'
            elif bar['open'] >= tp_level:
                return 'TP'
            else:
                return 'SL'  # 보수적

        if hit_sl:
            return 'SL'
        if hit_tp:
            return 'TP'

        return None
```

### backtest/engine.py (백테스트 전용)

```python
from core.strategy import LValueStrategy
import pandas as pd

class BacktestEngine:
    """백테스트 엔진"""

    def __init__(self, strategy, slippage=0.05, fee=0.055):
        self.strategy = strategy
        self.slippage = slippage
        self.fee = fee
        self.total_cost = (slippage + fee) * 2

    def run(self, df, df_l):
        """
        백테스트 실행

        Args:
            df: 15분 OHLCV DataFrame
            df_l: L값 DataFrame

        Returns:
            DataFrame: 거래 내역
        """
        trades = []

        for _, l_row in df_l.iterrows():
            l_idx = int(l_row['l_idx'])
            l_candle = df.iloc[l_idx].to_dict()

            # 진입 신호 확인
            if not self.strategy.check_entry_signal(l_candle):
                continue

            # 진입
            entry_idx = l_idx + 1
            entry_price = df.iloc[entry_idx]['open']

            # 청산 레벨
            levels = self.strategy.calculate_exit_levels(entry_price)

            # 청산 시뮬레이션
            exit_result = self._simulate_exit(
                df, entry_idx, levels['tp'], levels['sl']
            )

            # 비용 적용
            net_pnl = exit_result['gross_pnl'] - self.total_cost

            trades.append({
                'entry_time': df.iloc[entry_idx]['datetime'],
                'entry_price': entry_price,
                'exit_time': exit_result['exit_time'],
                'exit_price': exit_result['exit_price'],
                'exit_reason': exit_result['reason'],
                'gross_pnl': exit_result['gross_pnl'],
                'net_pnl': net_pnl
            })

        return pd.DataFrame(trades)

    def _simulate_exit(self, df, entry_idx, tp_level, sl_level, max_hold=50):
        """청산 시뮬레이션"""
        for i in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            bar = df.iloc[i].to_dict()
            exit_reason = self.strategy.check_exit(bar, tp_level, sl_level)

            if exit_reason == 'TP':
                return {
                    'exit_time': bar['datetime'],
                    'exit_price': tp_level,
                    'reason': 'TP',
                    'gross_pnl': (tp_level - df.iloc[entry_idx]['open']) /
                                 df.iloc[entry_idx]['open'] * 100
                }
            elif exit_reason == 'SL':
                return {
                    'exit_time': bar['datetime'],
                    'exit_price': sl_level,
                    'reason': 'SL',
                    'gross_pnl': (sl_level - df.iloc[entry_idx]['open']) /
                                 df.iloc[entry_idx]['open'] * 100
                }

        # 타임아웃
        final_bar = df.iloc[min(entry_idx + max_hold, len(df) - 1)]
        return {
            'exit_time': final_bar['datetime'],
            'exit_price': final_bar['close'],
            'reason': 'TIMEOUT',
            'gross_pnl': (final_bar['close'] - df.iloc[entry_idx]['open']) /
                         df.iloc[entry_idx]['open'] * 100
        }
```

### live/trader.py (실거래 전용)

```python
from core.strategy import LValueStrategy
import ccxt

class LiveTrader:
    """실시간 거래 실행"""

    def __init__(self, strategy, exchange, symbol='BTC/USDT:USDT'):
        self.strategy = strategy
        self.exchange = exchange
        self.symbol = symbol
        self.position = None  # 현재 포지션

    def on_l_value_confirmed(self, l_candle):
        """
        L값 확정 시 호출

        Args:
            l_candle: L값 봉 데이터
        """
        # 이미 포지션 있으면 무시
        if self.position is not None:
            return

        # 진입 신호 확인
        if not self.strategy.check_entry_signal(l_candle):
            print("진입 조건 미충족")
            return

        # 진입 주문
        self._enter_position()

    def _enter_position(self):
        """포지션 진입"""
        try:
            # 현재가 조회
            ticker = self.exchange.fetch_ticker(self.symbol)
            entry_price = ticker['ask']  # 매수 호가

            # 청산 레벨 계산
            levels = self.strategy.calculate_exit_levels(entry_price)

            # 주문 실행 (예: 시장가)
            order = self.exchange.create_market_buy_order(
                self.symbol,
                amount=0.01  # BTC 수량
            )

            # TP/SL 주문
            self._place_exit_orders(entry_price, levels['tp'], levels['sl'])

            # 포지션 기록
            self.position = {
                'entry_time': order['datetime'],
                'entry_price': entry_price,
                'tp_level': levels['tp'],
                'sl_level': levels['sl'],
                'amount': 0.01
            }

            print(f"✅ 진입: {entry_price} | TP: {levels['tp']} | SL: {levels['sl']}")

        except Exception as e:
            print(f"❌ 진입 실패: {e}")

    def _place_exit_orders(self, entry_price, tp_level, sl_level):
        """TP/SL 주문 설정"""
        # TP 주문
        self.exchange.create_limit_sell_order(
            self.symbol,
            amount=0.01,
            price=tp_level
        )

        # SL 주문 (Stop Market)
        self.exchange.create_order(
            self.symbol,
            type='stop_market',
            side='sell',
            amount=0.01,
            params={'stopPrice': sl_level}
        )

    def on_bar_close(self, bar):
        """
        봉 마감 시 호출 (모니터링)

        Args:
            bar: 현재 봉 데이터
        """
        if self.position is None:
            return

        # 포지션 상태 확인
        current_pnl = (bar['close'] - self.position['entry_price']) / \
                      self.position['entry_price'] * 100

        print(f"포지션 현황: {current_pnl:+.2f}%")
```

### 실행 스크립트

**backtest_run.py (백테스트)**
```python
from core.strategy import LValueStrategy
from backtest.engine import BacktestEngine
from backtest.data_loader import load_data
from backtest.reporter import generate_report

# 전략 초기화
strategy = LValueStrategy(
    tp_pct=2.0,
    sl_pct=-1.5,
    rsi_threshold=30,
    macd_threshold=0
)

# 데이터 로드
df, df_l = load_data('output_phase1_labeled.csv', 'l_labels_1h_cross.csv')

# 백테스트 실행
engine = BacktestEngine(strategy, slippage=0.05, fee=0.055)
trades = engine.run(df, df_l)

# 리포트 생성
generate_report(trades)
```

**live_run.py (실거래)**
```python
from core.strategy import LValueStrategy
from live.trader import LiveTrader
from live.monitor import DataMonitor
import ccxt

# 거래소 연결
exchange = ccxt.bybit({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'enableRateLimit': True
})

# 전략 초기화 (동일한 파라미터)
strategy = LValueStrategy(
    tp_pct=2.0,
    sl_pct=-1.5,
    rsi_threshold=30,
    macd_threshold=0
)

# 트레이더 초기화
trader = LiveTrader(strategy, exchange)

# 데이터 모니터 (WebSocket)
monitor = DataMonitor(
    on_l_value=trader.on_l_value_confirmed,
    on_bar_close=trader.on_bar_close
)

# 실행
monitor.start()
```

---

## 🔐 안전 장치

### 1. 실거래 전 체크리스트

- [ ] 백테스트 결과 재확인
- [ ] API 키 권한 확인 (거래 전용, 출금 불가)
- [ ] 테스트넷에서 먼저 실행
- [ ] 소액으로 시작 (전체 자본의 5% 이하)
- [ ] 알림 설정 (텔레그램, 이메일 등)
- [ ] 로깅 활성화
- [ ] 수동 중지 버튼 준비

### 2. 리스크 관리

```python
class RiskManager:
    """리스크 관리"""

    def __init__(self, max_daily_loss=-5.0, max_position_size=0.1):
        self.max_daily_loss = max_daily_loss  # 일일 최대 손실 %
        self.max_position_size = max_position_size  # 최대 포지션 크기
        self.daily_pnl = 0.0

    def can_trade(self):
        """거래 가능 여부"""
        if self.daily_pnl <= self.max_daily_loss:
            print("⛔ 일일 손실 한도 도달")
            return False
        return True

    def calculate_position_size(self, account_balance, entry_price, sl_price):
        """포지션 크기 계산 (켈리 기준)"""
        risk_per_trade = abs(entry_price - sl_price) / entry_price * 100
        position_size = min(
            account_balance * 0.02 / (risk_per_trade / 100),  # 2% 리스크
            account_balance * self.max_position_size  # 최대 10%
        )
        return position_size / entry_price  # BTC 수량
```

---

## 📊 모니터링 대시보드

### 필수 지표

1. **실시간 PnL** (미실현 + 실현)
2. **일일/주간/월간 수익률**
3. **현재 MDD**
4. **승률** (최근 10/50/100 거래)
5. **포지션 현황**
6. **다음 L값 예상 시간**

### 알림 조건

- 포지션 진입/청산
- 일일 손실 -3% 도달
- MDD 신고점 갱신
- 연속 3회 손실
- API 연결 끊김

---

## 🚀 다음 단계

1. **백테스트 재확인** ✅
2. **코드 리팩토링** - 위 구조로 재작성
3. **테스트넷 테스트** - Bybit Testnet에서 2주 실행
4. **소액 실거래** - $100로 1개월 운영
5. **점진적 확대** - 성과 확인 후 자본 증액

---

## ⚠️ 주의사항

1. **백테스트 ≠ 실거래**: 슬리피지, 리퀘스트 등 예상 외 변수 존재
2. **과최적화 경계**: 파라미터 변경 금지
3. **감정 배제**: 규칙대로만 거래
4. **기록 유지**: 모든 거래 로그 저장
5. **시장 변화 대응**: 3개월마다 성과 재평가

---

**작성일**: 2024-12-29
**전략**: L값 기반 (RSI<30 AND MACD<0)
**5년 백테스트 승률**: 87.77%
**5년 총 수익률**: +826%
**MDD**: -4.04%
