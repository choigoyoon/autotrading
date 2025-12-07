"""
상황 분류 시스템 (A~H)
- 1D/4H/1H MACD 조합으로 8가지 시장 상황 분류
- 나우캐스트: 15분 기준, 완성된 MTF 캔들만 사용
- 각 15분봉마다 현재 시장 상황 판별
"""

import pandas as pd
import numpy as np

print("="*60)
print("상황 분류 시스템 (A~H)")
print("="*60)

# 데이터 로드
print("\n데이터 로드 중...")
df_15m = pd.read_csv('output_phase1_labeled.csv')
df_1h = pd.read_csv('btcusdt_1h_labeled.csv')
df_4h = pd.read_csv('btcusdt_4h_labeled.csv')
df_1d = pd.read_csv('btcusdt_1d_labeled.csv')

# datetime 변환 및 인덱스 설정
for df in [df_15m, df_1h, df_4h, df_1d]:
    df['datetime'] = pd.to_datetime(df['datetime'])

print(f"15분: {len(df_15m):,}개")
print(f"1시간: {len(df_1h):,}개")
print(f"4시간: {len(df_4h):,}개")
print(f"1일: {len(df_1d):,}개")

# 상황 정의
SITUATIONS = {
    'A': {'1d': '+', '4h': '+', '1h': '+', 'desc': '상승 강세'},
    'B': {'1d': '+', '4h': '+', '1h': '-', 'desc': '상승 중 눌림 (반등값 큼)'},
    'C': {'1d': '+', '4h': '-', '1h': '-', 'desc': '상승 초기'},
    'D': {'1d': '+', '4h': '-', '1h': '+', 'desc': '혼조 상승'},
    'E': {'1d': '-', '4h': '-', '1h': '+', 'desc': '하락 중 반등'},
    'F': {'1d': '-', '4h': '-', '1h': '-', 'desc': '바닥 잡기'},
    'G': {'1d': '-', '4h': '+', '1h': '+', 'desc': '하락 초기'},
    'H': {'1d': '-', '4h': '+', '1h': '-', 'desc': '혼조 하락'},
}

def classify_situation(macd_1d, macd_4h, macd_1h):
    """MACD 부호로 상황 분류"""

    sign_1d = '+' if macd_1d >= 0 else '-'
    sign_4h = '+' if macd_4h >= 0 else '-'
    sign_1h = '+' if macd_1h >= 0 else '-'

    for sit, spec in SITUATIONS.items():
        if spec['1d'] == sign_1d and spec['4h'] == sign_4h and spec['1h'] == sign_1h:
            return sit

    return None

def get_mtf_macd_at_time(dt, df_1d, df_4h, df_1h):
    """
    특정 15분 시점의 MTF MACD 값 가져오기
    나우캐스트: 완성된 캔들만 사용
    """

    # 1일봉: 해당 시점 이전의 마지막 완성봉
    df_1d_before = df_1d[df_1d['datetime'] < dt]
    if len(df_1d_before) == 0:
        return None, None, None
    macd_1d = df_1d_before.iloc[-1]['macd_hist']

    # 4시간봉: 해당 시점 이전의 마지막 완성봉
    df_4h_before = df_4h[df_4h['datetime'] < dt]
    if len(df_4h_before) == 0:
        return None, None, None
    macd_4h = df_4h_before.iloc[-1]['macd_hist']

    # 1시간봉: 해당 시점 이전의 마지막 완성봉
    df_1h_before = df_1h[df_1h['datetime'] < dt]
    if len(df_1h_before) == 0:
        return None, None, None
    macd_1h = df_1h_before.iloc[-1]['macd_hist']

    return macd_1d, macd_4h, macd_1h

print("\n"+"="*60)
print("15분봉별 상황 분류 시작")
print("="*60)

# 각 15분봉에 상황 추가
situations = []
situation_details = []

print("\n진행 중...")
total = len(df_15m)
for i, row in df_15m.iterrows():
    if i % 10000 == 0:
        print(f"  {i:,} / {total:,} ({i/total*100:.1f}%)")

    dt = row['datetime']

    # MTF MACD 가져오기
    macd_1d, macd_4h, macd_1h = get_mtf_macd_at_time(dt, df_1d, df_4h, df_1h)

    if macd_1d is None:
        situations.append(None)
        situation_details.append({
            'macd_1d': None,
            'macd_4h': None,
            'macd_1h': None
        })
        continue

    # 상황 분류
    situation = classify_situation(macd_1d, macd_4h, macd_1h)
    situations.append(situation)
    situation_details.append({
        'macd_1d': macd_1d,
        'macd_4h': macd_4h,
        'macd_1h': macd_1h
    })

# 15분 데이터에 추가
df_15m['situation'] = situations
df_15m['mtf_macd_1d'] = [d['macd_1d'] for d in situation_details]
df_15m['mtf_macd_4h'] = [d['macd_4h'] for d in situation_details]
df_15m['mtf_macd_1h'] = [d['macd_1h'] for d in situation_details]

# 통계
print("\n"+"="*60)
print("상황별 통계")
print("="*60)

situation_counts = df_15m['situation'].value_counts().sort_index()

total_classified = situation_counts.sum()
print(f"\n총 분류된 캔들: {total_classified:,}개 / {len(df_15m):,}개 ({total_classified/len(df_15m)*100:.1f}%)")

print(f"\n상황별 분포:")
for sit in sorted(SITUATIONS.keys()):
    count = situation_counts.get(sit, 0)
    pct = count / total_classified * 100 if total_classified > 0 else 0
    desc = SITUATIONS[sit]['desc']
    spec = f"1D{SITUATIONS[sit]['1d']} 4H{SITUATIONS[sit]['4h']} 1H{SITUATIONS[sit]['1h']}"
    print(f"  {sit}: {count:,}개 ({pct:5.2f}%) - {desc} ({spec})")

# 샘플 출력
print(f"\n상황별 샘플:")
for sit in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
    samples = df_15m[df_15m['situation'] == sit].head(1)
    if len(samples) > 0:
        row = samples.iloc[0]
        print(f"\n상황{sit} ({SITUATIONS[sit]['desc']}):")
        print(f"  시간: {row['datetime']}")
        print(f"  1D MACD: {row['mtf_macd_1d']:.2f}")
        print(f"  4H MACD: {row['mtf_macd_4h']:.2f}")
        print(f"  1H MACD: {row['mtf_macd_1h']:.2f}")
        print(f"  15분 종가: {row['close']:.2f}")

# 저장
output_file = 'output_mtf_situation_classified.csv'
df_15m.to_csv(output_file, index=False)
print(f"\n저장: {output_file}")

# 시계열 분석
print(f"\n"+"="*60)
print("시계열 전환 분석")
print("="*60)

# 상황 전환 감지
df_15m['situation_change'] = df_15m['situation'] != df_15m['situation'].shift(1)
transitions = df_15m[df_15m['situation_change'] == True]

print(f"\n총 상황 전환: {len(transitions):,}회")
print(f"평균 지속: {len(df_15m) / len(transitions):.1f}봉 ({len(df_15m) / len(transitions) * 15 / 60:.1f}시간)")

# 전환 패턴
print(f"\n상황 전환 샘플 (최근 10건):")
for idx, row in transitions.tail(10).iterrows():
    prev_sit = df_15m.iloc[idx-1]['situation'] if idx > 0 else None
    curr_sit = row['situation']
    print(f"  {row['datetime']}: {prev_sit} → {curr_sit}")

print("\n"+"="*60)
print("완료!")
print("="*60)

print(f"\n생성 파일: {output_file}")
print(f"컬럼 추가:")
print(f"  - situation: A~H 상황 분류")
print(f"  - mtf_macd_1d: 1일봉 MACD")
print(f"  - mtf_macd_4h: 4시간봉 MACD")
print(f"  - mtf_macd_1h: 1시간봉 MACD")
print(f"  - situation_change: 상황 전환 여부")

print(f"\n다음 단계:")
print(f"  1. MTF Zone 추출 (L/H 꼬리 범위)")
print(f"  2. 상황별 매매법 분석")
print(f"  3. 블러 처리 (진입 불가 제거)")
print(f"  4. 15분 진입 예측")
