# L값 근접도 분석 시스템 명세서

## 📋 개요

**목표**: 15분 확정된 L값에 최대한 근접해서 진입 (꼬리 잡기)

**핵심 원리**:
- L값 = MACD 히스토그램 0-cross로 확정된 "저점"
- 확정 전 진입 = 가장 좋은 가격
- 체크박스 = 필터 아님, **L값 근접도 측정 도구**
- 많이 켜질수록 = L값에 가까워지는 중

---

## 🎯 매매법 이론 기반

### 1. Dow Theory
- 상승 추세 = Higher Low + Higher High
- 하락 → 상승 전환 = LL(Lower Low) 종료 + HL(Higher Low) 시작
- **L값 = 다우의 "저점" (추세 전환점)**

### 2. Wyckoff 방법론
- **Spring** = 저점 이탈 후 즉시 반등
- = 유동성 스윕 (Liquidity Sweep, 스탑헌팅)
- = 최고의 매수 타이밍 (약세 함정)

### 3. ICT/Smart Money Concepts
- **FVG (Fair Value Gap)** = 고래의 시장가 주문 흔적
- 급하락 + 지지 없음 + FVG = 깔끔한 바닥 (진입 좋음)
- 급하락 + 지지 많음 = 꼬리 말림 (진입 애매)
- **Order Block** = 기관 주문 영역
- **Liquidity Sweep** = 저점 돌파 후 즉시 반등

### 4. Bollinger Bands
- 하단 터치 = 과매도 영역
- 스퀴즈 (수축) = 변동성 폭발 임박

### 5. Elliott Wave
- 파동 카운팅
- 3파 = 가장 긴 상승파
- 5파 끝 = 조정 시작 (L값 형성)

---

## 📊 데이터 분류

### 즉발성 지표 (지연 0봉, 실시간 진입 신호용)
- 캔들 OHLCV
- 몸통/꼬리 크기
- FVG 발생 여부
- 거래량 급증
- 캔들 패턴 (해머, 엔걸핑)

### 후행성 지표 (지연 N봉, 확인용)
- MACD, RSI, BB, CCI, Stochastic
- 이동평균선
- 다이버전스

### 범위성 지표 (예측/목표 설정용)
- 피보나치 되돌림
- 지지/저항선
- 과거 L/H 기반 가격 영역

---

## ✅ 체크박스 목록 (범위 기반 근접도)

### MTF (Multi-Timeframe) 관련
- [ ] 1H MACD 방향 (양수/음수)
- [ ] 4H MACD 방향 (양수/음수)
- [ ] 1D MACD 방향 (양수/음수)
- [ ] MTF 방향 일치도 (0~3개 일치)

### 모멘텀 근접도 (과매도 영역)
- [ ] **RSI**: 20~40 범위 내 위치 (%)
  - 계산: `(40 - RSI) / (40 - 20) * 100`
  - 예: RSI 25 → 75% 근접

- [ ] **CCI**: -200~-100 범위 내 위치 (%)
  - 계산: `(-100 - CCI) / (-100 - (-200)) * 100`

- [ ] **Williams %R**: -100~-80 범위 내 위치 (%)

- [ ] **Stochastic**: 0~20 범위 내 위치 (%)

### 밴드 근접도
- [ ] **Bollinger Bands**: 하단~중단 범위 내 위치 (%)
  - 계산: `(price - lower) / (middle - lower) * 100`

- [ ] **나다라야 엔벨로프**: 하단~중단 범위 내 위치 (%)

### MACD 상세 분석
- [ ] **MACD 히스토그램 0 접근도** (%)
  - 계산: `1 - abs(hist) / max(abs(hist_recent))`

- [ ] **MACD 선 기울기/각도**
  - 음수 → 0 방향 전환 중

- [ ] **MACD 굴곡 횟수**
  - 최근 20봉 내 방향 전환 횟수

- [ ] **MACD 굴곡 간격** (수렴 여부)
  - 간격 좁아짐 = 0-cross 임박

### 가격 구조
- [ ] **HL (Higher Low) 형성 여부**
  - 이전 저점보다 높은 저점

- [ ] **HL 값 변화** (수렴 여부)
  - 저점이 점점 높아지는가?

- [ ] **HL 간격** (봉 수)
  - 저점 간격 좁아짐 = 추세 전환 임박

- [ ] **쌍바닥/역헤드앤숄더 패턴**

### 다이버전스
- [ ] **RSI 다이버전스 개수** (0/1/2/3)
  - 가격은 하락, RSI는 상승

- [ ] **MACD 다이버전스 개수**

- [ ] **히든 다이버전스 여부**
  - 추세 지속 신호

### ICT/Smart Money 구조
- [ ] **FVG 발생 여부**
  - `candle[i-2].high < candle[i].low`

- [ ] **FVG 크기** (소/중/대)
  - 소: < 0.5% / 중: 0.5~1% / 대: > 1%

- [ ] **급하락 여부** (각도)
  - 최근 5봉 하락 각도 > 45도

- [ ] **중간 지지 개수** (0이면 좋음)
  - 하락 중 반등 횟수 (적을수록 깔끔)

- [ ] **Order Block 영역 여부**
  - 큰 거래량 + 급반등 캔들

- [ ] **Liquidity Sweep 여부**
  - 이전 저점 돌파 후 즉시 반등

### 캔들 패턴
- [ ] **일반 캔들 패턴**
  - 해머, 역해머, 강세 엔걸핑

- [ ] **하이킨아시 패턴**
  - 핀바, 도지

- [ ] **몸통 크기 변화** (가속도)
  - 하락 캔들 몸통 점점 작아짐 = 매도 약화

- [ ] **꼬리 비율**
  - 아래 꼬리 > 몸통 * 2 = 강한 지지

### 거래량
- [ ] **거래량 급증** (평균 대비 2배 이상)
- [ ] **거래량 다이버전스**
  - 가격 하락, 거래량 감소 = 매도 약화

---

## 🔬 추론 로직 (핵심!)

### 1단계: 전체 캔들 모니터링
```python
# 모든 캔들에서 체크박스 상태 계산 (필터링 없음!)
for i in range(len(df)):
    checkboxes[i] = calculate_all_checkboxes(df, i)
```

### 2단계: 각 체크박스 근접도 계산
```python
# 예시: RSI 근접도
rsi = 25
target_range = (20, 40)  # 과매도 영역

if rsi < target_range[0]:
    proximity = 0  # 범위 밖
elif rsi > target_range[1]:
    proximity = 0  # 범위 밖
else:
    # 범위 내 위치 계산 (40에 가까울수록 100%)
    proximity = (target_range[1] - rsi) / (target_range[1] - target_range[0]) * 100

# RSI 25 → (40-25)/(40-20) = 75% 근접
```

### 3단계: 평균 근접도 계산
```python
# 모든 체크박스의 근접도 평균
avg_proximity = sum(all_proximities) / len(all_proximities)

# 예: [75%, 80%, 60%, 90%, 70%] → 평균 75%
```

### 4단계: 근접도 변화 추적
```python
# 시간에 따른 근접도 변화
proximities = [60, 65, 70, 75, 80, 85, 90]  # 7봉 동안

# 패턴 인식:
# - 상승 추세 (60→90) = L값 접근 중
# - 하락 추세 (90→60) = L값 멀어짐
# - 80% 이상 유지 = L값 임박
```

### 5단계: 진입 조건
```python
# 조건 1: 평균 근접도 80% 이상
if avg_proximity >= 80:

    # 조건 2: 확인 신호 (즉발성 지표)
    if (fvg_exists or bullish_candle or liquidity_sweep):

        # 진입!
        entry_price = next_candle_open
        entry_reason = "L값 근접도 80%+ & 확인 신호"
```

---

## 📈 구현 요청사항

### 1. L값 수집
```python
def collect_l_values(df):
    """
    확정된 15분 L값 수집 (MACD 0-cross 기준)

    Returns:
        l_values = [
            {'idx': 1234, 'datetime': '2024-01-01 10:00', 'price': 50000},
            ...
        ]
    """
    l_values = []

    for i in range(1, len(df)):
        prev_hist = df.iloc[i-1]['macd_hist']
        curr_hist = df.iloc[i]['macd_hist']

        # L값 확정 조건 (음수 → 양수)
        if prev_hist < 0 and curr_hist >= 0:
            l_values.append({
                'idx': i,
                'datetime': df.iloc[i]['datetime'],
                'price': df.iloc[i]['low']
            })

    return l_values
```

### 2. 체크박스 상태 기록
```python
def calculate_checkboxes(df, idx):
    """
    특정 시점의 모든 체크박스 근접도 계산

    Returns:
        {
            'rsi_proximity': 75.0,
            'cci_proximity': 80.0,
            'bb_proximity': 60.0,
            'macd_0_proximity': 90.0,
            'fvg_exists': True,
            'fvg_size': 'large',
            'liquidity_sweep': False,
            ...
        }
    """
    # 각 지표 계산 및 근접도 변환
    # ...
```

### 3. L값 전후 분석
```python
def analyze_l_value_context(df, l_idx, window=20):
    """
    L값 기준 전후 N봉의 체크박스 패턴 분석

    Args:
        l_idx: L값 확정 인덱스
        window: 분석 범위 (전후 20봉)

    Returns:
        {
            'before': [체크박스 상태 20개],
            'at_l': 체크박스 상태,
            'after': [체크박스 상태 20개],
            'proximity_trend': [60, 65, 70, ...],  # 근접도 추세
            'success': True/False  # L값 이후 상승 성공 여부
        }
    """
    # 전: l_idx - 20 ~ l_idx - 1
    # 중: l_idx
    # 후: l_idx + 1 ~ l_idx + 20
```

### 4. 패턴 학습
```python
def find_effective_patterns(all_l_contexts):
    """
    어떤 체크박스 조합이 L값 예측에 효과적인지 분석

    교집합 분석:
    - 성공한 L값들의 공통 체크박스
    - 실패한 L값들의 공통 체크박스
    - 차이점 추출

    Returns:
        {
            'high_win_patterns': [
                {'checkboxes': ['rsi_proximity > 80', 'fvg_large'], 'win_rate': 85%},
                ...
            ],
            'low_win_patterns': [...],
            'critical_checkboxes': ['fvg_exists', 'liquidity_sweep', ...]
        }
    """
```

### 5. 백테스트
```python
def backtest_l_proximity(df):
    """
    L값 근접도 기반 백테스트

    전략:
    1. 매 봉마다 근접도 계산
    2. 근접도 80% 이상 + 확인 신호 → 진입
    3. TP 2% / SL 2%
    4. L값 확정 시점과 비교

    Returns:
        {
            'total_trades': 500,
            'win_rate': 85%,
            'avg_entry_distance_from_l': -0.3%,  # L값보다 0.3% 위에서 진입
            'avg_pnl': 1.2%,
            'best_checkboxes': [...]
        }
    """
```

---

## 📊 분석 결과 보고 형식

### 요약
```
L값 근접도 분석 결과
===================

총 L값 개수: 1,234개
분석 기간: 2020-01-01 ~ 2024-12-29

평균 진입 타이밍:
- L값 확정 -3봉 (45분 전)
- 평균 진입가: L값 대비 +0.5%

근접도 임계값:
- 80% 이상: 진입 신호
- 평균 85% 도달 시점: L값 -5봉

효과적인 체크박스 Top 5:
1. FVG 발생 (승률 +15%)
2. Liquidity Sweep (승률 +12%)
3. RSI 근접도 80%+ (승률 +10%)
4. MACD 히스토 0 근접도 90%+ (승률 +8%)
5. MTF 방향 일치 (승률 +7%)

백테스트 성과:
- 거래: 500회
- 승률: 82%
- 평균 PnL: 1.5%
- MDD: -3.2%
```

### 상세 분석
```
성공 L값 특징 (승률 > 80%):
- FVG 발생: 95%
- Liquidity Sweep: 87%
- 중간 지지 0개: 78%
- RSI 근접도 평균: 85%
- 근접도 상승 추세: 90%

실패 L값 특징 (승률 < 50%):
- FVG 없음: 80%
- 중간 지지 3개 이상: 65%
- RSI 근접도 평균: 60%
- 근접도 하락 추세: 70%

차이점 (결정적 요소):
→ FVG 발생 여부
→ 중간 지지 개수 (0개가 좋음)
→ 근접도 추세 방향
```

---

## 🎯 최종 목표

**진입 시스템**:
```
IF 평균_근접도 >= 80%
   AND (FVG_발생 OR Liquidity_Sweep OR 중간지지_0개)
   AND 근접도_추세 == 상승
   AND MTF_방향_일치 >= 2개
THEN
   다음_봉_시가_진입
```

**기대 효과**:
- L값 확정 전 진입 (평균 -3봉)
- 진입가 개선 (L값 대비 +0.5% 이내)
- 승률 향상 (82% 이상)
- 리스크 감소 (L값 = 자연 손절선)

---

## 📝 추가 고려사항

1. **동적 임계값**
   - 시장 변동성에 따라 근접도 임계값 조정
   - 예: 변동성 높을 때 75%, 낮을 때 85%

2. **체크박스 가중치**
   - 모든 체크박스 동일 가중치 아님
   - FVG, Liquidity Sweep = 높은 가중치
   - 일반 캔들 패턴 = 낮은 가중치

3. **거짓 L값 필터**
   - L값 확정 후 즉시 하락 → 거짓 L값
   - 이후 분석에서 제외

4. **실시간 적용**
   - 웹소켓으로 15분봉 실시간 수신
   - 매 봉마다 근접도 계산
   - 80% 도달 시 알림

---

## ✅ 체크리스트

- [ ] L값 수집 로직 구현
- [ ] 모든 체크박스 근접도 계산 함수 구현
- [ ] L값 전후 패턴 분석 구현
- [ ] 교집합 분석 (성공 vs 실패 특징)
- [ ] 백테스트 실행
- [ ] 결과 보고서 작성
- [ ] 효과적인 체크박스 조합 도출
- [ ] 실시간 적용 가능 여부 검토

---

**문의사항 있으면 언제든지 연락주세요!**
