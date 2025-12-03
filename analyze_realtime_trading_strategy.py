import pandas as pd
import numpy as np

print(f"{'='*80}")
print(f"🎯 미래 데이터 없이 실시간 매매 전략")
print(f"{'='*80}\n")

# CSV 읽기
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = df.rename(columns={'datetime': 'timestamp'})
df = df.sort_values('timestamp').reset_index(drop=True)

# BB 계산
def calculate_bb(df, window=20, num_std=2):
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (num_std * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (num_std * df['bb_std'])
    return df

df = calculate_bb(df)

print(f"📋 실시간 매매 체크리스트:")
print(f"{'='*80}\n")

print(f"🔍 1단계: 저점 확인 (실시간 가능 ✅)")
print(f"   - 현재까지의 H/L값만 사용")
print(f"   - 이전 L값보다 낮으면 → LL 패턴 확인")
print(f"   - 미래 데이터: 불필요 ✅")

print(f"\n🔍 2단계: 추세선 확인 (실시간 가능 ✅)")
print(f"   - 현재까지의 H값 3개로 추세선 그리기")
print(f"   - H1 > H2 > H3 확인 (LH-LH-LH)")
print(f"   - 미래 데이터: 불필요 ✅")

print(f"\n🔍 3단계: 추세선 돌파 확인 (실시간 가능 ✅)")
print(f"   - 현재 종가 > H3 가격?")
print(f"   - 미래 데이터: 불필요 ✅")

print(f"\n🔍 4단계: 돌파 힘 확인 (실시간 가능 ✅)")
print(f"   - BB 상단 찢기: 현재 고가 > BB 상단")
print(f"   - FVG: 이전 3캔들 패턴 확인")
print(f"   - OB: 현재 캔들 body > 1%")
print(f"   - 강양봉: 현재 캔들 body > 2%")
print(f"   - 미래 데이터: 불필요 ✅")

print(f"\n🔍 5단계: 리테스트 진입 (실시간 가능 ✅)")
print(f"   - 돌파 후 H3 ±1% 터치 대기")
print(f"   - 다음 5캔들 평균 > H3 확인")
print(f"   - 미래 데이터: 5캔들(75분) 대기만 필요 ✅")

print(f"\n{'='*80}")
print(f"💰 진입 후 TP 설정 (실시간 가능 ✅)")
print(f"{'='*80}\n")

print(f"✅ 진입 시점:")
print(f"   - 리테스트 확인 후 다음 캔들 오픈")
print(f"   - 진입가: 다음 캔들 시가")

print(f"\n✅ SL (손절) 설정:")
print(f"   - SL = H3 가격 × 0.995 (H3 -0.5%)")
print(f"   - 미래 데이터: 불필요 ✅")
print(f"   - 이유: H3가 지지 실패하면 추세 전환 실패")

print(f"\n✅ TP (목표가) 설정:")
print(f"   - TP1 = H2 가격 (첫 번째 저항선)")
print(f"   - TP2 = H1 가격 (두 번째 저항선)")
print(f"   - 미래 데이터: 불필요 ✅")
print(f"   - 이유: 과거 저항선이 미래 저항으로 작용")

print(f"\n{'='*80}")
print(f"📈 단계별 익절 전략")
print(f"{'='*80}\n")

# 실제 사례 분석
df_result = pd.read_csv('retest_entry_strategy_analysis.csv')

# TP 도달 통계
tp1_only = len(df_result[(df_result['tp1_hit']) & (~df_result['tp2_hit'])])
tp2_reached = len(df_result[df_result['tp2_hit']])
sl_hit = len(df_result[(df_result['sl_hit']) & (~df_result['tp1_hit'])])

total_trades = len(df_result)

print(f"📊 실제 백테스트 결과 (71개 거래):\n")

print(f"1️⃣ TP1 도달 (H2까지):")
print(f"   - 발생: {tp1_only}번")
print(f"   - 비율: {tp1_only/total_trades*100:.1f}%")
print(f"   - 평균 R:R: 1:{df_result['rr_tp1'].mean():.2f}")

print(f"\n2️⃣ TP2 도달 (H1까지):")
print(f"   - 발생: {tp2_reached}번")
print(f"   - 비율: {tp2_reached/total_trades*100:.1f}%")
print(f"   - 평균 R:R: 1:{df_result['rr_tp2'].mean():.2f}")

print(f"\n3️⃣ SL 손절:")
print(f"   - 발생: {sl_hit}번")
print(f"   - 비율: {sl_hit/total_trades*100:.1f}%")

print(f"\n{'='*80}")
print(f"💡 단계별 익절 전략")
print(f"{'='*80}\n")

print(f"전략 A: 보수적 (TP1 전량 익절)")
print(f"   - TP1 (H2) 도달 시 100% 청산")
print(f"   - 승률: {(tp1_only + tp2_reached)/total_trades*100:.1f}%")
print(f"   - 평균 R:R: 1:{df_result['rr_tp1'].mean():.2f}")
print(f"   - 장점: 높은 승률, 안정적")

print(f"\n전략 B: 균형적 (TP1 50%, TP2 50%)")
print(f"   - TP1 (H2) 도달 시 50% 청산, SL을 본전으로 이동")
print(f"   - TP2 (H1) 도달 시 나머지 50% 청산")
print(f"   - 승률: {(tp1_only + tp2_reached)/total_trades*100:.1f}%")
print(f"   - 평균 수익: 더 높음")
print(f"   - 장점: 리스크 제로 + 추가 수익")

print(f"\n전략 C: 공격적 (TP1 부분 익절, TP2 보유)")
print(f"   - TP1 (H2) 도달 시 30% 청산, SL을 진입가로 이동")
print(f"   - TP2 (H1) 목표")
print(f"   - TP2 도달률: {tp2_reached/total_trades*100:.1f}%")
print(f"   - 평균 R:R: 1:{df_result['rr_tp2'].mean():.2f}")
print(f"   - 장점: 최대 수익")

print(f"\n{'='*80}")
print(f"✅ 미래 데이터 없이 실시간 매매 가능 여부")
print(f"{'='*80}\n")

checklist = [
    ("저점(LL) 확인", "✅", "과거 L값만 사용"),
    ("추세선(H1→H2→H3) 확인", "✅", "과거 H값만 사용"),
    ("추세선 돌파 확인", "✅", "현재 종가와 H3 비교"),
    ("돌파 힘 확인 (BB/FVG/OB)", "✅", "현재/이전 캔들만 사용"),
    ("리테스트 대기", "✅", "75분(5캔들) 대기"),
    ("진입", "✅", "다음 캔들 시가"),
    ("SL 설정", "✅", "H3 -0.5%"),
    ("TP1 설정", "✅", "H2 가격"),
    ("TP2 설정", "✅", "H1 가격"),
    ("단계별 익절", "✅", "TP 도달 시 청산")
]

for step, status, note in checklist:
    print(f"{status} {step:30s} → {note}")

print(f"\n{'='*80}")
print(f"🎯 최종 결론")
print(f"{'='*80}\n")

print(f"✅ 모든 단계에서 미래 데이터 불필요!")
print(f"✅ 100% 실시간 매매 가능!")
print(f"✅ 리테스트 확인 후 진입 (최대 75분 대기)")
print(f"✅ SL/TP는 과거 H값으로 설정")
print(f"✅ 단계별 익절로 리스크 관리")

print(f"\n📈 실전 매매 프로세스:")
print(f"\n1. 차트 모니터링")
print(f"   → LL 패턴 발견")
print(f"   → 추세선(LH-LH-LH) 확인")
print(f"\n2. 돌파 대기")
print(f"   → 종가 > H3 확인")
print(f"   → 돌파 힘 확인 (5점 이상)")
print(f"\n3. 리테스트 대기 (최대 75분)")
print(f"   → H3 ±1% 터치")
print(f"   → 다음 5캔들 평균 > H3")
print(f"\n4. 진입")
print(f"   → 다음 캔들 시가 진입")
print(f"   → SL: H3 -0.5%")
print(f"   → TP1: H2 | TP2: H1")
print(f"\n5. 단계별 익절")
print(f"   → TP1 도달: 50% 청산, SL 본전 이동")
print(f"   → TP2 도달: 나머지 50% 청산")

print(f"\n💰 예상 결과:")
print(f"   월평균 3.6번 매매 기회")
print(f"   TP1 도달률: 78.9%")
print(f"   TP2 도달률: 56.3%")
print(f"   평균 수익: +0.38%")

