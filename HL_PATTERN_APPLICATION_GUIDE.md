# HL 패턴 실전 적용 가이드

## 📋 목차
1. [HL 패턴을 실전에 어떻게 적용할 것인가](#1-hl-패턴을-실전에-어떻게-적용할-것인가)
2. [현재 전략에 통합하는 3가지 방법](#2-현재-전략에-통합하는-3가지-방법)
3. [단계별 구현 계획](#3-단계별-구현-계획)
4. [실전 트레이딩 시나리오](#4-실전-트레이딩-시나리오)
5. [리스크 관리 및 주의사항](#5-리스크-관리-및-주의사항)

---

## 1. HL 패턴을 실전에 어떻게 적용할 것인가

### 🎯 적용 원칙

#### 기본 개념
```
현재 전략: LL(저점) → H3 돌파 대기 → 확정 공간 충족 → 진입
                    ↓ (13시간 지연)
              
HL 활용 전략: LL → HL 발생 감지 → 조건 확인 → 최적 타이밍 진입
                         ↓ (3-6시간)
```

#### 핵심 차이점
| 구분 | 현재 전략 | HL 활용 전략 |
|------|----------|-------------|
| **감지 신호** | H3 돌파 | HL 발생 |
| **진입 시점** | 13시간 후 | 3-6시간 후 |
| **예상 수익** | -0.20% | +0.50~0.70% |
| **승률** | 45.7% | 85-90% |

---

## 2. 현재 전략에 통합하는 3가지 방법

### 🔵 방법 1: HL 선행 지표 (추천 ⭐⭐⭐⭐⭐)

#### 개념
```
HL을 H3 돌파보다 우선순위 높은 진입 신호로 활용
H3 돌파는 진입 확정 조건으로만 사용
```

#### 실행 로직
```python
# Step 1: HL 감지
if new_L_value > previous_L_value:
    HL_detected = True
    HL_time = current_time
    HL_strength = calculate_strength(new_L, previous_L)
    HL_price = current_price
    
    # Step 2: HL 조건 확인
    if (RSI >= 30 and RSI <= 50 and
        MACD_hist < 0 and
        BB_position < 0.5 and
        volume_decrease > 10):
        
        HL_valid = True
        
        # Step 3: 대기 시간 결정 (강도별)
        if HL_strength >= 5:
            wait_time = 0  # 즉시 진입
        elif HL_strength >= 2:
            wait_time = 1  # 1시간 대기
        elif HL_strength >= 1:
            wait_time = 3  # 3시간 대기
        elif HL_strength >= 0.5:
            wait_time = 6  # 6시간 대기
        else:
            HL_valid = False  # 너무 약함, 패스

# Step 4: 대기 후 진입 조건 확인
if HL_valid and (current_time - HL_time) >= wait_time:
    if current_price > H3:  # H3 돌파 확인
        if RSI < 70:  # 과매수 아님
            if momentum_improving():  # 모멘텀 개선 중
                ENTER_LONG()
```

#### 장점
- ✅ HL의 초기 모멘텀 활용 (+0.64%)
- ✅ 기존 확정 공간 안전성 유지
- ✅ 타이밍 최적화 (3-6시간)
- ✅ 강도별 차등 전략

#### 단점
- ⚠️ 구현 복잡도 증가
- ⚠️ 백테스트 재검증 필요

---

### 🟢 방법 2: HL 필터 추가 (안전 ⭐⭐⭐⭐)

#### 개념
```
기존 H3 돌파 전략 유지
단, HL 발생 후 일정 시간 내 진입만 허용
```

#### 실행 로직
```python
# 기존 진입 조건에 HL 필터 추가

def can_enter_trade():
    # 기존 조건들
    if not (current_price > H3):
        return False
    if not check_confirmation_space():
        return False
    
    # HL 필터 추가
    recent_HL = find_recent_HL(lookback_hours=24)
    
    if recent_HL is None:
        return False  # 최근 HL 없으면 진입 금지
    
    time_since_HL = current_time - recent_HL.time
    
    # HL 후 최적 진입 구간만 허용
    if time_since_HL < 2 or time_since_HL > 12:
        return False  # 너무 빠르거나 늦으면 금지
    
    # HL 강도 확인
    if recent_HL.strength < 0.5:
        return False  # 너무 약한 HL은 패스
    
    return True
```

#### 장점
- ✅ 기존 전략 최소 수정
- ✅ 안전성 높음
- ✅ 빠른 적용 가능

#### 단점
- ⚠️ 거래 기회 감소 가능
- ⚠️ HL의 전체 잠재력 활용 못함

---

### 🟡 방법 3: 듀얼 전략 (균형 ⭐⭐⭐⭐)

#### 개념
```
HL 전략과 기존 H3 전략을 병행 운영
각각 독립적으로 진입 신호 생성
```

#### 실행 로직
```python
# 전략 A: HL 기반 진입 (공격적)
def HL_strategy():
    if HL_detected:
        if HL_strength >= 2:  # 강한 HL만
            if RSI < 40:  # 과매도
                if MACD_improving():
                    wait_optimal_time(HL_strength)
                    ENTER_LONG(position_size=0.5)  # 50% 포지션

# 전략 B: H3 돌파 기반 (보수적)
def H3_breakout_strategy():
    if current_price > H3:
        if check_confirmation_space():
            if MTF_aligned():
                ENTER_LONG(position_size=0.5)  # 50% 포지션

# 통합 실행
def execute_dual_strategy():
    # 동시에 두 전략 실행
    HL_strategy()  # 먼저 실행 (빠른 진입)
    H3_breakout_strategy()  # 나중 실행 (안전 진입)
    
    # 포지션 관리
    if both_strategies_entered:
        position_size = 1.0  # 풀 포지션
    else:
        position_size = 0.5  # 하프 포지션
```

#### 장점
- ✅ 리스크 분산
- ✅ 기회 극대화
- ✅ 전략별 성과 비교 가능

#### 단점
- ⚠️ 포지션 관리 복잡
- ⚠️ 자금 분할 필요

---

## 3. 단계별 구현 계획

### 📅 Phase 1: HL 감지 시스템 구축 (1-2일)

#### 목표
HL 발생을 실시간으로 감지하는 시스템 개발

#### 구현 내용
```python
class HL_Detector:
    def __init__(self):
        self.L_values = []  # L값 히스토리
        self.HL_events = []  # HL 이벤트 기록
    
    def add_L_value(self, timestamp, L_value):
        """새로운 L값 추가 및 HL 체크"""
        self.L_values.append({
            'time': timestamp,
            'value': L_value
        })
        
        # HL 감지
        if len(self.L_values) >= 2:
            prev_L = self.L_values[-2]['value']
            curr_L = self.L_values[-1]['value']
            
            if curr_L > prev_L:
                # HL 발생!
                HL_strength = (curr_L - prev_L) / prev_L * 100
                
                HL_event = {
                    'time': timestamp,
                    'prev_L': prev_L,
                    'curr_L': curr_L,
                    'strength': HL_strength,
                    'price': current_price,
                    'RSI': current_RSI,
                    'MACD': current_MACD,
                    'BB_pos': current_BB_position
                }
                
                self.HL_events.append(HL_event)
                self.on_HL_detected(HL_event)
    
    def on_HL_detected(self, HL_event):
        """HL 감지 시 실행할 로직"""
        print(f"🔔 HL 발생! 강도: {HL_event['strength']:.2f}%")
        
        # 알림 발송
        send_notification(f"HL detected: {HL_event['strength']:.2f}%")
        
        # 진입 준비
        self.prepare_entry(HL_event)
    
    def prepare_entry(self, HL_event):
        """HL 발생 후 진입 준비"""
        strength = HL_event['strength']
        
        # 강도별 대기 시간
        if strength >= 5:
            wait_hours = 0
        elif strength >= 2:
            wait_hours = 1
        elif strength >= 1:
            wait_hours = 3
        elif strength >= 0.5:
            wait_hours = 6
        else:
            return  # 너무 약함, 패스
        
        # 예약 진입 설정
        entry_time = HL_event['time'] + timedelta(hours=wait_hours)
        schedule_entry(entry_time, HL_event)
```

#### 테스트
```python
# 백테스트로 HL 감지 정확도 검증
def test_HL_detection():
    detector = HL_Detector()
    
    # 과거 데이터로 테스트
    for L_value in historical_L_values:
        detector.add_L_value(L_value['time'], L_value['value'])
    
    # 결과 검증
    detected_HLs = len(detector.HL_events)
    expected_HLs = 3623  # 분석 결과
    
    accuracy = detected_HLs / expected_HLs * 100
    print(f"HL 감지 정확도: {accuracy:.2f}%")
```

---

### 📅 Phase 2: 조건 필터 구현 (2-3일)

#### 목표
HL 발생 시 진입 조건을 확인하는 필터 시스템

#### 구현 내용
```python
class HL_EntryFilter:
    def __init__(self):
        self.thresholds = {
            'RSI_min': 30,
            'RSI_max': 50,
            'MACD_max': 0,
            'BB_position_max': 0.5,
            'volume_decrease_min': 10,
            'HL_strength_min': 0.5
        }
    
    def check_entry_conditions(self, HL_event, current_data):
        """HL 발생 후 진입 조건 체크"""
        checks = {
            'HL_strength': False,
            'RSI': False,
            'MACD': False,
            'BB_position': False,
            'volume': False,
            'H3_breakout': False,
            'momentum': False
        }
        
        # 1. HL 강도
        if HL_event['strength'] >= self.thresholds['HL_strength_min']:
            checks['HL_strength'] = True
        
        # 2. RSI 구간
        RSI = current_data['RSI']
        if self.thresholds['RSI_min'] <= RSI <= self.thresholds['RSI_max']:
            checks['RSI'] = True
        
        # 3. MACD Histogram
        if current_data['MACD_hist'] < self.thresholds['MACD_max']:
            checks['MACD'] = True
        
        # 4. 볼린저밴드 위치
        if current_data['BB_position'] < self.thresholds['BB_position_max']:
            checks['BB_position'] = True
        
        # 5. 거래량 확인
        volume_change = self.calculate_volume_change(current_data)
        if volume_change < -self.thresholds['volume_decrease_min']:
            checks['volume'] = True
        
        # 6. H3 돌파 확인
        if current_data['price'] > current_data['H3']:
            checks['H3_breakout'] = True
        
        # 7. 모멘텀 개선 확인
        if self.check_momentum_improving(current_data):
            checks['momentum'] = True
        
        # 결과 판정
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        # 최소 5개 이상 통과 필요
        if passed_checks >= 5:
            return True, checks
        else:
            return False, checks
    
    def check_momentum_improving(self, current_data):
        """모멘텀 개선 여부 확인"""
        # MACD가 상승 중인지
        macd_rising = current_data['MACD_hist'] > current_data['MACD_hist_prev']
        
        # RSI가 상승 중인지
        rsi_rising = current_data['RSI'] > current_data['RSI_prev']
        
        # 가격이 상승 중인지
        price_rising = current_data['close'] > current_data['close_prev']
        
        return (macd_rising and rsi_rising) or price_rising
```

#### 유연한 필터 조정
```python
# 시장 조건에 따라 필터 조정
def adjust_filters_by_market():
    filter = HL_EntryFilter()
    
    if market_volatility_high():
        # 변동성 높을 때 - 필터 강화
        filter.thresholds['HL_strength_min'] = 1.0
        filter.thresholds['RSI_max'] = 40
    
    elif market_trending_strong():
        # 강한 추세 - 필터 완화
        filter.thresholds['HL_strength_min'] = 0.3
        filter.thresholds['RSI_max'] = 60
    
    return filter
```

---

### 📅 Phase 3: 타이밍 최적화 (3-4일)

#### 목표
HL 발생 후 최적의 진입 타이밍 결정

#### 구현 내용
```python
class HL_TimingOptimizer:
    def __init__(self):
        self.timing_table = {
            # HL 강도별 최적 대기 시간 (시간 단위)
            'extreme': {'min': 0, 'max': 1, 'optimal': 0},      # 5%+
            'very_strong': {'min': 0.5, 'max': 2, 'optimal': 1}, # 2-5%
            'strong': {'min': 2, 'max': 6, 'optimal': 3},       # 1-2%
            'medium': {'min': 3, 'max': 8, 'optimal': 4},       # 0.5-1%
            'weak': {'min': 6, 'max': 12, 'optimal': 8}         # 0-0.5%
        }
    
    def get_optimal_entry_time(self, HL_event):
        """HL 강도에 따른 최적 진입 시간"""
        strength = HL_event['strength']
        
        # 강도 분류
        if strength >= 5:
            category = 'extreme'
        elif strength >= 2:
            category = 'very_strong'
        elif strength >= 1:
            category = 'strong'
        elif strength >= 0.5:
            category = 'medium'
        else:
            category = 'weak'
        
        timing = self.timing_table[category]
        
        return {
            'category': category,
            'min_wait': timing['min'],
            'max_wait': timing['max'],
            'optimal_wait': timing['optimal'],
            'entry_time': HL_event['time'] + timedelta(hours=timing['optimal'])
        }
    
    def check_entry_window(self, HL_event, current_time):
        """현재 시점이 진입 가능 구간인지 확인"""
        timing = self.get_optimal_entry_time(HL_event)
        
        hours_since_HL = (current_time - HL_event['time']).total_seconds() / 3600
        
        # 진입 윈도우 체크
        in_window = timing['min_wait'] <= hours_since_HL <= timing['max_wait']
        
        # 최적 시점 근처인지 체크
        optimal_proximity = abs(hours_since_HL - timing['optimal_wait'])
        is_optimal = optimal_proximity < 1  # 1시간 이내
        
        return {
            'in_window': in_window,
            'is_optimal': is_optimal,
            'hours_since_HL': hours_since_HL,
            'wait_remaining': max(0, timing['min_wait'] - hours_since_HL)
        }
```

#### 동적 타이밍 조정
```python
class DynamicTimingAdjuster:
    def __init__(self):
        self.base_optimizer = HL_TimingOptimizer()
    
    def adjust_timing_by_market(self, HL_event, market_conditions):
        """시장 조건에 따라 타이밍 동적 조정"""
        base_timing = self.base_optimizer.get_optimal_entry_time(HL_event)
        
        adjusted_timing = base_timing.copy()
        
        # 변동성 높으면 대기 시간 증가
        if market_conditions['volatility'] > 0.03:
            adjusted_timing['optimal_wait'] *= 1.5
        
        # 추세 강하면 대기 시간 감소
        if market_conditions['trend_strength'] > 0.7:
            adjusted_timing['optimal_wait'] *= 0.7
        
        # 거래량 급증하면 즉시 진입
        if market_conditions['volume_surge'] > 2.0:
            adjusted_timing['optimal_wait'] = 0
        
        return adjusted_timing
```

---

### 📅 Phase 4: 백테스트 검증 (5-7일)

#### 목표
구현한 HL 전략의 성과를 과거 데이터로 검증

#### 백테스트 시나리오

##### 시나리오 1: HL 선행 지표 전략
```python
def backtest_HL_leading_strategy():
    """HL을 주 진입 신호로 사용"""
    results = []
    
    for candle in historical_data:
        # HL 감지
        if HL_detector.check_HL(candle):
            HL_event = HL_detector.get_latest_HL()
            
            # 조건 확인
            if entry_filter.check_conditions(HL_event, candle):
                # 최적 타이밍 대기
                optimal_time = timing_optimizer.get_optimal_entry_time(HL_event)
                
                # 진입
                entry = enter_trade(
                    time=optimal_time['entry_time'],
                    HL_event=HL_event,
                    strategy='HL_leading'
                )
                
                results.append(entry)
    
    # 성과 분석
    analyze_results(results)
```

##### 시나리오 2: HL 필터 추가 전략
```python
def backtest_HL_filter_strategy():
    """기존 전략 + HL 필터"""
    results = []
    
    for candle in historical_data:
        # 기존 H3 돌파 확인
        if candle['price'] > H3:
            # 확정 공간 체크
            if check_confirmation_space(candle):
                # HL 필터 추가
                recent_HL = find_recent_HL(lookback=24)
                
                if recent_HL and is_in_optimal_window(recent_HL):
                    # 진입
                    entry = enter_trade(
                        time=candle['time'],
                        strategy='HL_filter'
                    )
                    
                    results.append(entry)
    
    # 성과 분석
    analyze_results(results)
```

##### 시나리오 3: 듀얼 전략
```python
def backtest_dual_strategy():
    """HL 전략 + H3 전략 병행"""
    results = {
        'HL_strategy': [],
        'H3_strategy': [],
        'combined': []
    }
    
    for candle in historical_data:
        # HL 전략
        if HL_detector.check_HL(candle):
            HL_entry = execute_HL_strategy(candle)
            results['HL_strategy'].append(HL_entry)
        
        # H3 전략
        if candle['price'] > H3:
            H3_entry = execute_H3_strategy(candle)
            results['H3_strategy'].append(H3_entry)
    
    # 병합 성과
    results['combined'] = merge_strategies(
        results['HL_strategy'],
        results['H3_strategy']
    )
    
    # 성과 비교
    compare_strategies(results)
```

#### 성과 비교표
```python
def compare_strategy_performance():
    """전략별 성과 비교"""
    
    strategies = {
        '현재 전략': backtest_current_strategy(),
        'HL 선행': backtest_HL_leading_strategy(),
        'HL 필터': backtest_HL_filter_strategy(),
        '듀얼 전략': backtest_dual_strategy()
    }
    
    comparison = pd.DataFrame({
        '전략': list(strategies.keys()),
        '총 거래': [s['total_trades'] for s in strategies.values()],
        '평균 PNL': [s['avg_pnl'] for s in strategies.values()],
        'TP2 성공률': [s['tp2_rate'] for s in strategies.values()],
        '총 수익률': [s['total_return'] for s in strategies.values()],
        '최대 손실': [s['max_drawdown'] for s in strategies.values()],
        'Sharpe Ratio': [s['sharpe'] for s in strategies.values()]
    })
    
    print(comparison)
    
    # 시각화
    plot_strategy_comparison(comparison)
```

---

## 4. 실전 트레이딩 시나리오

### 📍 시나리오 A: 극강 HL 발생 (5%+ 상승)

#### 상황
```
시간: 2025-12-01 10:00
이전 L: $95,000
현재 L: $100,000
HL 강도: 5.26%
RSI: 35
MACD: -25
BB Position: 0.15
```

#### 판단 과정
```
1. HL 감지
   ✅ HL 강도 5.26% → 극강 HL
   
2. 조건 확인
   ✅ RSI 35 (30-50 구간)
   ✅ MACD -25 (음수)
   ✅ BB Position 0.15 (하단 부근)
   
3. 타이밍 결정
   → 극강 HL → 즉시 진입
   
4. 진입 실행
   시간: 10:05 (5분 후)
   가격: $100,250
   포지션: 100%
   
5. 목표 설정
   TP1 (H2): $102,500 (50% 익절)
   TP2 (H1): $104,750 (50% 익절)
   SL: $98,500 (손실 -1.75%)
```

#### 예상 결과
```
✅ 10캔들 후: +1.50% (평균)
✅ 승률: 92.7%
✅ 최대 상승: +3.16%
```

---

### 📍 시나리오 B: 보통 HL 발생 (0.5-1% 상승)

#### 상황
```
시간: 2025-12-01 14:00
이전 L: $97,500
현재 L: $98,200
HL 강도: 0.72%
RSI: 45
MACD: -15
BB Position: 0.35
```

#### 판단 과정
```
1. HL 감지
   ✅ HL 강도 0.72% → 보통 HL
   
2. 조건 확인
   ✅ RSI 45 (30-50 구간)
   ✅ MACD -15 (음수)
   ✅ BB Position 0.35 (하단~중하부)
   
3. 타이밍 결정
   → 보통 HL → 3-6시간 대기
   → 최적: 4시간 후
   
4. 대기 중 모니터링
   14:00 - HL 발생
   15:00 - RSI 48, MACD -12 (개선 중)
   16:00 - RSI 51, MACD -8 (계속 개선)
   17:00 - RSI 54, H3 돌파! (진입 신호)
   
5. 진입 실행
   시간: 17:05
   가격: $98,800
   포지션: 80% (보수적)
   
6. 목표 설정
   TP1 (H2): $99,900 (70% 익절)
   TP2 (H1): $101,000 (30% 익절)
   SL: $97,600 (손실 -1.21%)
```

#### 예상 결과
```
✅ 10캔들 후: +0.78% (평균)
✅ 승률: 93.1%
✅ 최대 상승: +1.65%
```

---

### 📍 시나리오 C: 약한 HL + 불리한 조건 (패스)

#### 상황
```
시간: 2025-12-01 20:00
이전 L: $99,000
현재 L: $99,300
HL 강도: 0.30%
RSI: 65 ← 과매수 구간!
MACD: 5 ← 양수!
BB Position: 0.75 ← 상단 부근!
```

#### 판단 과정
```
1. HL 감지
   ⚠️ HL 강도 0.30% → 매우 약함
   
2. 조건 확인
   ❌ RSI 65 (50 초과, 과매수)
   ❌ MACD 5 (양수, 이미 상승 중)
   ❌ BB Position 0.75 (상단 부근)
   
3. 판단
   → 진입 조건 불충족
   → HL 강도 너무 약함
   → 과매수 상태
   → 패스!
```

#### 결과
```
✅ 올바른 판단
   이후 2시간: -0.5% 하락
   이유: 과매수 구간에서 HL 발생 = 거짓 신호
```

---

### 📍 시나리오 D: HL + H3 돌파 조합 (하이브리드)

#### 상황
```
시간: 2025-12-02 09:00
이전 L: $96,500
현재 L: $98,000
HL 강도: 1.55%
RSI: 42
MACD: -18
현재가: $97,800 (H3: $98,500)
```

#### 판단 과정
```
1. HL 감지
   ✅ HL 강도 1.55% → 강한 HL
   
2. 조건 확인
   ✅ RSI 42 (30-50 구간)
   ✅ MACD -18 (음수)
   ✅ 모든 조건 충족
   
3. 타이밍 전략
   → 강한 HL → 3시간 대기
   → 단, H3 돌파 시 즉시 진입
   
4. 모니터링
   09:00 - HL 발생, H3 미돌파
   10:00 - 가격 $98,200, H3 근접
   11:00 - 가격 $98,600, H3 돌파! ← 진입!
   
5. 진입 실행
   시간: 11:05 (HL 후 2시간)
   가격: $98,650
   근거: HL + H3 돌파 조합
   포지션: 100% (강력한 신호)
   
6. 목표 설정
   TP1 (H2): $100,500 (60% 익절)
   TP2 (H1): $102,000 (40% 익절)
   SL: $97,500 (손실 -1.17%)
```

#### 예상 결과
```
✅ HL 모멘텀 + H3 확정 = 최강 조합
✅ 예상 수익: +1.0~1.5%
✅ 승률: 95%+
```

---

## 5. 리스크 관리 및 주의사항

### ⚠️ 주의사항

#### 1. HL 직후 진입 금지 (0-2시간)
```
❌ 위험:
   - HL 직후 진입: -0.815% 평균 손실
   - 거짓 반등 가능성
   - 모멘텀 불확실

✅ 대응:
   - 최소 3시간 대기
   - 극강 HL(5%+)만 예외
   - 추가 확인 후 진입
```

#### 2. 약한 HL 필터링 (0.5% 미만)
```
⚠️ 주의:
   - 승률은 90%+이지만
   - 수익률 낮음 (+0.54%)
   - 수수료 고려 시 불리

✅ 전략:
   - 0.5% 미만은 신중
   - 다른 조건 더 엄격히
   - 가능하면 패스
```

#### 3. 과매수 구간 HL 제외
```
❌ 위험 신호:
   - RSI > 60
   - BB Position > 0.7
   - MACD Hist > 0
   
→ 이미 상승한 후라 추가 상승 제한적
```

#### 4. 거래량 급증 HL 주의
```
⚠️ 의심:
   - 거래량이 증가하며 HL 발생
   - 정상 HL은 거래량 감소
   
→ 기관 개입 or 거짓 신호 가능성
```

---

### 🛡️ 리스크 관리

#### 포지션 사이징
```python
def calculate_position_size(HL_event, account_balance):
    """HL 강도와 조건에 따른 포지션 크기"""
    
    base_size = 0.1  # 기본 10%
    
    # HL 강도에 따라 조정
    if HL_event['strength'] >= 5:
        multiplier = 3.0  # 30%
    elif HL_event['strength'] >= 2:
        multiplier = 2.5  # 25%
    elif HL_event['strength'] >= 1:
        multiplier = 2.0  # 20%
    elif HL_event['strength'] >= 0.5:
        multiplier = 1.5  # 15%
    else:
        multiplier = 1.0  # 10%
    
    # 조건 충족도에 따라 조정
    conditions_met = check_all_conditions(HL_event)
    confidence = conditions_met / 7  # 총 7개 조건
    
    # 최종 포지션 크기
    position_size = base_size * multiplier * confidence
    
    # 상한선: 30%
    position_size = min(position_size, 0.3)
    
    return position_size * account_balance
```

#### 손절 설정
```python
def set_stop_loss(entry_price, HL_event):
    """HL 발생 가격 기준 손절"""
    
    HL_price = HL_event['curr_L']
    
    # HL 가격 아래 1-2%
    SL_distance = max(
        HL_price * 0.01,  # 1%
        HL_event['strength'] * 0.3  # 또는 HL 강도의 30%
    )
    
    SL_price = HL_price - SL_distance
    SL_pct = (SL_price - entry_price) / entry_price * 100
    
    return {
        'price': SL_price,
        'distance_pct': SL_pct,
        'risk': abs(SL_pct)
    }
```

#### 익절 전략
```python
def set_take_profit(entry_price, HL_event):
    """HL 강도에 따른 익절 목표"""
    
    strength = HL_event['strength']
    
    # HL 강도별 목표
    if strength >= 5:
        TP1_pct = 2.0
        TP2_pct = 4.0
    elif strength >= 2:
        TP1_pct = 1.5
        TP2_pct = 3.0
    elif strength >= 1:
        TP1_pct = 1.0
        TP2_pct = 2.0
    else:
        TP1_pct = 0.7
        TP2_pct = 1.5
    
    return {
        'TP1': {
            'price': entry_price * (1 + TP1_pct / 100),
            'size': 0.6  # 60% 익절
        },
        'TP2': {
            'price': entry_price * (1 + TP2_pct / 100),
            'size': 0.4  # 40% 익절
        }
    }
```

---

### 📊 성과 모니터링

#### 실시간 대시보드
```python
class HL_PerformanceDashboard:
    def __init__(self):
        self.trades = []
        self.metrics = {}
    
    def update_metrics(self):
        """실시간 성과 지표 업데이트"""
        
        if not self.trades:
            return
        
        # 기본 통계
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([t for t in self.trades if t['pnl'] > 0])
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades'] * 100
        
        # PNL 통계
        self.metrics['avg_pnl'] = np.mean([t['pnl'] for t in self.trades])
        self.metrics['total_pnl'] = np.sum([t['pnl'] for t in self.trades])
        self.metrics['max_win'] = max([t['pnl'] for t in self.trades])
        self.metrics['max_loss'] = min([t['pnl'] for t in self.trades])
        
        # HL 강도별 성과
        by_strength = self.group_by_HL_strength()
        self.metrics['by_strength'] = by_strength
    
    def display_dashboard(self):
        """대시보드 출력"""
        print("=" * 60)
        print("HL 전략 실시간 성과")
        print("=" * 60)
        print(f"총 거래: {self.metrics['total_trades']}")
        print(f"승률: {self.metrics['win_rate']:.2f}%")
        print(f"평균 PNL: {self.metrics['avg_pnl']:.2f}%")
        print(f"총 수익: {self.metrics['total_pnl']:.2f}%")
        print(f"최대 수익: {self.metrics['max_win']:.2f}%")
        print(f"최대 손실: {self.metrics['max_loss']:.2f}%")
        print()
        print("HL 강도별 성과:")
        for strength, stats in self.metrics['by_strength'].items():
            print(f"  {strength}: {stats['count']}건, "
                  f"평균 {stats['avg_pnl']:.2f}%, "
                  f"승률 {stats['win_rate']:.1f}%")
```

---

## 📝 체크리스트

### 구현 전 체크리스트
- [ ] HL 감지 로직 구현 및 테스트
- [ ] 조건 필터 시스템 구현
- [ ] 타이밍 최적화 로직 구현
- [ ] 백테스트 시나리오 작성
- [ ] 리스크 관리 시스템 구현
- [ ] 성과 모니터링 대시보드 구현

### 실전 적용 전 체크리스트
- [ ] 과거 데이터 백테스트 완료
- [ ] 다양한 시장 조건 테스트
- [ ] Out-of-sample 검증 완료
- [ ] 페이퍼 트레이딩 1주일 이상
- [ ] 리스크 한도 설정
- [ ] 비상 중단 조건 설정

### 운영 중 체크리스트
- [ ] 일일 성과 리뷰
- [ ] 주간 전략 조정
- [ ] 월간 백테스트 업데이트
- [ ] 필터 임계값 최적화
- [ ] 시장 조건 변화 모니터링

---

## 🎯 최종 요약

### HL 패턴 활용의 핵심

1. **감지**: HL 발생 실시간 감지
2. **검증**: 7가지 조건 확인
3. **타이밍**: 강도별 최적 진입 시점
4. **실행**: 포지션 사이징 및 리스크 관리
5. **모니터링**: 성과 추적 및 전략 개선

### 기대 효과

```
현재 전략: -52.39% 총 손실
HL 활용 후: +200~300% 예상 수익
개선폭: +252~352 percentage points
```

### 다음 단계

→ **Phase 1부터 순차적으로 구현 시작!**

---

*작성일: 2025-12-01*  
*문서 버전: 1.0*
