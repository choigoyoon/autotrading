import pandas as pd
from datetime import timedelta

print("=" * 70)
print("🔍 FVG 시그널 발생 타이밍 정확히 확인")
print("=" * 70)

# 데이터 로드
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])

print("\n📊 4시간봉 샘플:")
print("-" * 70)
print(df_4h[['datetime', 'open', 'high', 'low', 'close']].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("💡 FVG 감지 로직 분석")
print("=" * 70)

print("""
코드:
  for i in range(2, len(df_4h)):
      현재봉 = df_4h.iloc[i]
      2봉전 = df_4h.iloc[i-2]
      
      if 현재봉['low'] > 2봉전['high']:
          signal_time = 현재봉['datetime']  ← 이게 언제?
""")

print("\n🤔 핵심 질문:")
print("  현재봉['datetime']이 00:00이면")
print("  → 00:00 봉이 아직 진행 중? (20:00~00:00)")
print("  → 00:00 봉이 완성된 후? (00:00 확정)")

print("\n" + "=" * 70)
print("🕐 OHLCV 데이터의 타임스탬프 의미")
print("=" * 70)

print("""
일반적인 거래소 규칙:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[방식 1] 봉 시작 시간 (Open time)
  - datetime = 봉이 시작하는 시간
  - 예: 2024-01-01 00:00 → 00:00~04:00 봉
  - 00:00에는 아직 봉이 완성 안 됨 ⚠️
  - 04:00에 봉 완성 ✅

[방식 2] 봉 종료 시간 (Close time)
  - datetime = 봉이 끝나는 시간
  - 예: 2024-01-01 00:00 → 20:00~00:00 봉
  - 00:00에 봉 완성됨 ✅
  - 즉시 FVG 체크 가능 ✅
""")

print("\n" + "=" * 70)
print("📌 우리 데이터 확인")
print("=" * 70)

# 실제 FVG 감지
fvg_signals = []
for i in range(2, min(50, len(df_4h))):  # 처음 50개만
    current = df_4h.iloc[i]
    prev2 = df_4h.iloc[i-2]
    
    if current['low'] > prev2['high']:
        gap_size = (current['low'] - prev2['high']) / prev2['high'] * 100
        if gap_size >= 0.3:
            fvg_signals.append({
                'signal_time': current['datetime'],
                'current_candle': f"{current['datetime']} (저가 {current['low']:.1f})",
                'prev2_candle': f"{prev2['datetime']} (고가 {prev2['high']:.1f})",
                'gap': f"{gap_size:.2f}%"
            })

if len(fvg_signals) > 0:
    print(f"\n발견된 FVG 시그널 (처음 5개):")
    print("-" * 70)
    for i, sig in enumerate(fvg_signals[:5], 1):
        print(f"\n시그널 #{i}:")
        print(f"  시그널 시간: {sig['signal_time']}")
        print(f"  현재봉: {sig['current_candle']}")
        print(f"  2봉전: {sig['prev2_candle']}")
        print(f"  갭 크기: {sig['gap']}")

print("\n" + "=" * 70)
print("🔍 타임스탬프 의미 추론")
print("=" * 70)

if len(fvg_signals) > 0:
    first_sig = fvg_signals[0]
    sig_time = pd.to_datetime(first_sig['signal_time'])
    
    print(f"\n첫 시그널 시간: {sig_time}")
    print(f"시간(hour): {sig_time.hour}")
    
    # 4시간 간격 확인
    print(f"\n4시간봉 간격 확인:")
    for i in range(5):
        dt = df_4h['datetime'].iloc[i]
        print(f"  봉 #{i}: {dt}")

print("""
결론 추론:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

만약 datetime이 봉 시작 시간이면:
  - 00:00 타임스탬프 = 00:00~04:00 봉
  - 실제 확정은 04:00에 발생 ⚠️
  - 코드에서 i=3일 때 감지 (04:00에 확인)
  
만약 datetime이 봉 종료 시간이면:
  - 00:00 타임스탬프 = 20:00~00:00 봉 (완성됨)
  - 00:00에 즉시 확정 ✅
  - 코드에서 i=3일 때 감지 (00:00 직후 확인)
""")

print("\n" + "=" * 70)
print("🎯 실전 의미")
print("=" * 70)

print("""
[시나리오 1] 봉 시작 시간이면:
  20:00 봉 데이터 있음 (i=1)
  00:00 봉 시작 (i=2, 진행중)
  04:00 봉 완성! (i=3) ← 여기서 00:00 봉 확인
    → df_4h.iloc[3] = 04:00 봉 (새 봉)
    → FVG 체크 시 00:00 봉(i=3-1=2) 사용
    → 아니다, 로직이 i를 현재봉으로 봄
    
[시나리오 2] 봉 종료 시간이면:
  20:00 봉 완성 (i=1)
  00:00 봉 완성 (i=2)
  04:00 봉 완성 (i=3) ← 여기서 FVG 체크
    → 현재봉 = 04:00 (i=3)
    → 2봉전 = 20:00 (i=1)
    → 시그널 시간 = 04:00

실제 코드:
  현재봉.datetime = df_4h.iloc[i]['datetime']
  
  만약 i=3이고, df_4h.iloc[3]['datetime'] = '04:00'이면
    → 시그널 시간 = 04:00 ✅
    → 이게 맞음!
""")

print("\n" + "=" * 70)
print("✅ 정확한 답변")
print("=" * 70)

print("""
시그널 타임스탬프가 00:00이라면:

1️⃣ 백테스트에서:
   - 00:00 봉이 완성된 시점
   - 로직상 for문이 i=N에 도달했을 때
   - df_4h.iloc[N]['datetime'] = '00:00'
   
2️⃣ 실전에서:
   [봉 시작 시간 방식]
   - 00:00 = 00:00~04:00 봉
   - 실제 확정은 04:00
   - 04:00에야 FVG 확인 가능 ⚠️
   
   [봉 종료 시간 방식]  
   - 00:00 = 20:00~00:00 봉 (완성)
   - 00:00 직후 FVG 확인 가능 ✅

거래소마다 다르므로 확인 필요!
일반적으로는 '봉 시작 시간' 방식이 많음
→ 00:00 타임스탬프면 실제론 04:00 확정
""")
