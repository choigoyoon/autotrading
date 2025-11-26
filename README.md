# BTC 자동매매 시스템

Bybit 거래소의 15분 봉 데이터 수집 + LSTM 기반 BTC 가격 나우캐스트 시스템입니다.

## 시스템 구성

1. **데이터 수집**: Bybit 거래소에서 15분 봉 데이터를 상장 초기부터 수집
2. **기술적 지표**: RSI, MACD, 볼린저 밴드 등 40+ 기술적 지표 자동 계산
3. **LSTM 모델**: 딥러닝 기반 시계열 예측 모델로 다음 15분 가격 예측
4. **실시간 나우캐스트**: 실시간 데이터로 지속적인 가격 예측 및 거래 신호 생성

## 기능

- CCXT 라이브러리를 사용한 안정적인 데이터 수집
- Bybit USDT 무기한 선물 15분 봉 데이터 수집
- 상장 초기부터 현재까지 모든 히스토리 데이터 수집
- 여러 심볼(코인) 지원
- CSV 파일로 자동 저장
- API 레이트 리밋 자동 처리
- 중복 데이터 자동 제거

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 실행

CCXT 버전 (권장):
```bash
python bybit_collector_ccxt.py
```

기본값으로 BTC/USDT와 ETH/USDT 데이터를 수집합니다.

### 수집할 심볼 변경

`bybit_collector_ccxt.py` 파일을 열고 `main()` 함수의 `symbols` 리스트를 수정하세요:

```python
symbols = [
    'BTC/USDT:USDT',   # 비트코인
    'ETH/USDT:USDT',   # 이더리움
    'SOL/USDT:USDT',   # 솔라나
    'XRP/USDT:USDT',   # 리플
    'DOGE/USDT:USDT',  # 도지코인
    # 원하는 심볼 추가...
]
```

**심볼 형식**: `BASE/QUOTE:SETTLE` (예: `BTC/USDT:USDT`)
- BASE: 기초 자산 (예: BTC)
- QUOTE: 견적 통화 (예: USDT)
- SETTLE: 결제 통화 (예: USDT)

### 수집 기간 변경

스크립트 내에서 `since_date` 파라미터를 수정:

```python
# 모든 가능한 데이터 수집 (2020년부터)
df = collector.collect_historical_data(
    symbol=symbol,
    timeframe='15m',
    since_date=None
)

# 최근 30일만 수집
from datetime import datetime, timedelta
df = collector.collect_historical_data(
    symbol=symbol,
    timeframe='15m',
    since_date=datetime.now() - timedelta(days=30)
)
```

## 출력 데이터

수집된 데이터는 `data/` 디렉토리에 CSV 파일로 저장됩니다.

파일명 형식: `{SYMBOL}_{TIMEFRAME}_{YYYYMMDD_HHMMSS}.csv`

예: `BTC_USDT_USDT_15m_20251126_143000.csv`

### CSV 컬럼:
- `timestamp`: 유닉스 타임스탬프 (밀리초)
- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가
- `volume`: 거래량
- `datetime`: 날짜시간 (읽기 쉬운 형식)

## 예상 실행 시간

데이터 양에 따라 다르지만, 대략적인 예상 시간:
- BTC/USDT (2020년~현재): 약 3-5분
- 여러 심볼 동시 수집: 심볼당 3-5분

## 주의사항

- Bybit Public API를 사용하므로 API 키가 필요하지 않습니다
- CCXT가 API 레이트 리밋을 자동으로 처리합니다
- 대량의 데이터를 수집하는 경우 시간이 오래 걸릴 수 있습니다
- 네트워크 에러 발생 시 자동으로 재시도합니다
- 중복 타임스탬프는 자동으로 제거됩니다

## 전체 워크플로우

### 1단계: 데이터 수집
```bash
python bybit_collector_ccxt.py
```
- BTC/USDT, ETH/USDT 등의 15분 봉 데이터를 2020년부터 수집
- `data/` 폴더에 CSV 파일로 저장

### 2단계: 모델 훈련
```bash
python train_nowcast.py
```
- 수집된 데이터에 기술적 지표 추가
- LSTM 모델 훈련 (약 50-100 에폭)
- 백테스트 수행 및 성능 평가
- 훈련된 모델을 `models/` 폴더에 저장

### 3단계: 실시간 예측
```bash
# 연속 예측 (60초마다)
python realtime_nowcast.py

# 한 번만 예측
python realtime_nowcast.py --once
```
- 실시간으로 Bybit에서 데이터를 가져와 예측
- 다음 15분 가격 예측 및 거래 신호 생성

## 나우캐스트 모델 상세

### 모델 구조
- **입력**: 최근 6시간(24개 15분 봉) + 40+ 기술적 지표
- **구조**: 3층 LSTM (128 → 64 → 32 units) + Dense 레이어
- **출력**: 다음 15분 가격 변화율 예측

### 기술적 지표
- **추세**: SMA, EMA (7, 14, 21, 50, 100, 200), MACD
- **모멘텀**: RSI (6, 12, 24), Stochastic, ROC
- **변동성**: Bollinger Bands, ATR, Historical Volatility
- **거래량**: OBV, CMF, Volume MA, Volume Ratio

### 거래 신호
- 🟢 **STRONG BUY**: 예상 변화 > +0.5%
- 🟢 **BUY**: 예상 변화 > +0.2%
- ⚪ **HOLD**: 예상 변화 -0.2% ~ +0.2%
- 🟠 **SELL**: 예상 변화 < -0.2%
- 🔴 **STRONG SELL**: 예상 변화 < -0.5%

## 파일 구조

```
autotrading/
├── bybit_collector_ccxt.py    # 데이터 수집 (CCXT 버전)
├── bybit_data_collector.py    # 데이터 수집 (직접 API 버전)
├── technical_indicators.py    # 기술적 지표 계산 모듈
├── btc_nowcast_model.py       # LSTM 나우캐스트 모델
├── train_nowcast.py           # 모델 훈련 스크립트
├── realtime_nowcast.py        # 실시간 예측 시스템
├── test_collector.py          # 데이터 수집 테스트
├── test_ccxt.py              # CCXT 테스트
├── requirements.txt           # 필요한 패키지 목록
├── .gitignore                # Git 무시 파일 목록
├── data/                     # 수집된 CSV 파일 저장
├── models/                   # 훈련된 모델 저장
└── README.md                 # 이 파일
```

## 문제 해결

### ImportError: No module named 'ccxt'
```bash
pip install ccxt
```

### API 에러 발생 시
- 인터넷 연결을 확인하세요
- 스크립트를 다시 실행하면 자동으로 재시도합니다

### 데이터가 수집되지 않는 경우
- 심볼 이름이 정확한지 확인하세요 (CCXT 형식: `BTC/USDT:USDT`)
- Bybit에서 해당 심볼이 거래 가능한지 확인하세요

## 라이센스

MIT