# 향후 개선 사항

## 🎯 Phase 6: 선행 체크박스 (Leading Indicators)

### 현재 문제
- L/H 확정 시점 (T)의 값만 사용
- 어떻게 그 값에 도달했는지 모름
- 방향성/흐름 파악 불가

### 개선 방안: 선행 시점 체크박스 추가

#### 1. 시점별 체크박스

```python
# 현재 (T 시점만)
checkboxes_T = {
    'rsi': 22,
    'macd': -0.1,
    'volume_ratio': 1.2
}

# 개선 (T, T-1, T-2, T-3...)
checkboxes_extended = {
    # 현재 시점
    'rsi_t0': 22,
    'macd_t0': -0.1,

    # 1봉 전
    'rsi_t1': 18,
    'macd_t1': -0.3,

    # 2봉 전
    'rsi_t2': 15,
    'macd_t2': -0.5,

    # 3봉 전
    'rsi_t3': 12,
    'macd_t3': -0.7
}
```

#### 2. 변화량 체크박스

```python
# 변화량 (Delta)
checkboxes_delta = {
    'rsi_change_1': 22 - 18,      # +4 (상승 중)
    'rsi_change_3': 22 - 12,      # +10 (강한 상승)
    'macd_change_1': -0.1 - (-0.3), # +0.2 (수렴 중)
}
```

#### 3. 패턴 체크박스

```python
# 패턴 인식
checkboxes_pattern = {
    # 방향성
    'rsi_rising_3bars': True,     # 3봉 연속 상승
    'rsi_rising_5bars': False,

    # 기울기
    'rsi_slope_3': +2.33,         # (22 + 18 + 15) / 3 기울기
    'rsi_acceleration': +0.5,     # 기울기 가속도

    # 수렴/발산
    'macd_converging': True,      # 절대값 감소
    'macd_divergence': False,     # 가격 vs MACD 다이버전스

    # 극값
    'rsi_local_min': True,        # N봉 내 최저
    'rsi_oversold': True,         # 30 이하
}
```

### 예시: L 예측 개선

#### Before (현재)
```
L 확정 시점만 체크:
✓ RSI = 22
✓ MACD = -0.1
→ 승률: 60%
```

#### After (선행 체크박스)
```
L 확정 전 패턴 체크:
✓ RSI = 22 (현재)
✓ RSI_T1 = 18 (1봉 전)
✓ RSI_T2 = 15 (2봉 전)
✓ RSI_rising_3bars = True  ← 추가!
✓ MACD_converging = True   ← 추가!
✓ Volume_increasing = True ← 추가!
→ 승률: 75% (개선!)
```

### 구현 계획

#### Phase 6-1: 데이터 준비
```python
def add_leading_indicators(df, lookback=5):
    """
    선행 체크박스 추가

    Args:
        lookback: 몇 봉 전까지 추가할지
    """
    for i in range(1, lookback + 1):
        # 시점별 값
        df[f'rsi_t{i}'] = df['rsi'].shift(i)
        df[f'macd_t{i}'] = df['macd'].shift(i)
        df[f'volume_t{i}'] = df['volume'].shift(i)

    # 변화량
    df['rsi_change_1'] = df['rsi'] - df['rsi_t1']
    df['rsi_change_3'] = df['rsi'] - df['rsi_t3']

    # 패턴
    df['rsi_rising_3'] = (
        (df['rsi'] > df['rsi_t1']) &
        (df['rsi_t1'] > df['rsi_t2']) &
        (df['rsi_t2'] > df['rsi_t3'])
    )

    # 기울기
    df['rsi_slope_3'] = (df['rsi'] - df['rsi_t3']) / 3

    # 수렴
    df['macd_converging'] = (
        df['macd'].abs() < df['macd_t1'].abs()
    )

    return df
```

#### Phase 6-2: 체크박스 확장
```python
def generate_extended_checkboxes(df, idx):
    """L/H 시점의 확장 체크박스"""

    checkboxes = {
        # 현재 값
        'rsi': df.iloc[idx]['rsi'],
        'macd': df.iloc[idx]['macd'],

        # 선행 값 (T-1, T-2, T-3)
        'rsi_t1': df.iloc[idx]['rsi_t1'],
        'rsi_t2': df.iloc[idx]['rsi_t2'],
        'rsi_t3': df.iloc[idx]['rsi_t3'],

        # 변화량
        'rsi_change_1': df.iloc[idx]['rsi_change_1'],
        'rsi_change_3': df.iloc[idx]['rsi_change_3'],

        # 패턴
        'rsi_rising_3': df.iloc[idx]['rsi_rising_3'],
        'macd_converging': df.iloc[idx]['macd_converging'],
    }

    return checkboxes
```

#### Phase 6-3: 매매법 탐색
```python
def find_best_checkbox_combinations_extended(labeled_df):
    """
    확장 체크박스로 매매법 탐색

    목표:
    - 선행 패턴 포함
    - 승률 > 70%
    - 평균 수익 > 1.0%
    """

    # 모든 체크박스 조합 테스트
    # ...

    # 최적 조합 찾기
    # 예: "RSI 상승 중 + MACD 수렴 + Volume 증가"
```

### 기대 효과

1. **승률 향상**
   - 현재: 60-65%
   - 목표: 70-75%
   - 방향성 확인으로 가짜 신호 제거

2. **진입 타이밍 개선**
   - L이 형성되는 중간에 진입 가능
   - "RSI가 상승 시작" 시점 포착

3. **리스크 감소**
   - 역추세 진입 방지
   - 강한 신호만 선별

### 추가 아이디어

#### 다중 타임프레임 선행 체크박스
```python
# 15분봉 (현재)
'rsi_15m': 22,
'rsi_15m_rising': True,

# 1시간봉 (상위 TF)
'rsi_1h': 35,
'rsi_1h_rising': True,  ← 상위 TF도 상승!

# 4시간봉
'rsi_4h': 40,
'rsi_4h_trend': 'up',

→ 모든 TF 정렬 시 신뢰도 ↑
```

#### 다이버전스 자동 감지
```python
'price_macd_divergence': {
    'type': 'bullish',  # 강세 다이버전스
    'strength': 0.8,    # 강도
    'bars': 15,         # 지속 기간
}
```

#### ML 피처로 활용
```python
# 나중에 ML 모델 사용 시
features = [
    'rsi_t0', 'rsi_t1', 'rsi_t2', 'rsi_t3',
    'rsi_change_1', 'rsi_slope_3',
    'macd_converging', 'volume_increasing',
    # ... 총 100+ 피처
]
→ XGBoost / Random Forest 학습
```

---

## 🎯 우선순위

1. **Phase 5 완료** ← 현재
   - 파라미터 최적화
   - 총 수익률 극대화

2. **Phase 6: 선행 체크박스** ← 다음
   - 데이터 준비
   - 패턴 인식
   - 체크박스 확장

3. **Phase 7: ML 통합** ← 미래
   - 피처 엔지니어링
   - 모델 학습
   - 실시간 예측

---

## 📌 메모

- 선행 체크박스는 **단순 값보다 패턴**이 중요
- **3-5봉 lookback**이 적당 (너무 길면 노이즈)
- **변화량 + 방향성**이 핵심
- 나중에 **자동 패턴 탐색** 가능

---

*작성일: 2025-11-28*
*작성자: AI*
*상태: 계획 단계*
