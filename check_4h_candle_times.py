import pandas as pd

print("=" * 70)
print("🕐 4시간봉 확정 시간 확인")
print("=" * 70)

# 4시간봉 데이터 로드
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])

print("\n📊 4시간봉 데이터 샘플 (첫 20개):")
print("-" * 70)
print(df_4h[['datetime', 'open', 'high', 'low', 'close']].head(20).to_string(index=False))

print("\n" + "=" * 70)
print("⏰ 4시간봉 확정 시간 패턴")
print("=" * 70)

# 시간대 추출
df_4h['hour'] = df_4h['datetime'].dt.hour
hour_counts = df_4h['hour'].value_counts().sort_index()

print("\n시간대별 4시간봉 개수:")
for hour, count in hour_counts.items():
    print(f"  {hour:02d}:00 - {count}개")

print("\n" + "=" * 70)
print("💡 분석")
print("=" * 70)

unique_hours = sorted(df_4h['hour'].unique())
print(f"\n4시간봉이 확정되는 시간: {', '.join([f'{h:02d}:00' for h in unique_hours])}")

if set(unique_hours) == {0, 4, 8, 12, 16, 20}:
    print("\n✅ UTC 기준 4시간봉 확정:")
    print("  00:00, 04:00, 08:00, 12:00, 16:00, 20:00")
    print("\n한국 시간 (UTC+9):")
    print("  09:00, 13:00, 17:00, 21:00, 01:00(+1일), 05:00(+1일)")
elif set(unique_hours) == {8, 12, 16, 20, 0, 4}:
    print("\n✅ 거래소 시간 기준 4시간봉 확정:")
    print("  확정 시간:", ', '.join([f'{h:02d}:00' for h in unique_hours]))

print("\n" + "=" * 70)
print("📌 FVG 시그널 발생 시점")
print("=" * 70)

print("""
FVG 감지 로직:
  for i in range(2, len(df_4h)):
      현재봉: df_4h[i]
      2봉전: df_4h[i-2]
      
      if 현재봉.low > 2봉전.high:
          → Bullish FVG 발생!
          
시그널 시간 = df_4h[i]['datetime']
→ 현재봉이 확정되는 시점에 시그널 발생
""")

print("\n예시:")
print("  00:00 - 3번째 4H봉 확정")
print("  → 갭 체크: 현재봉(00:00) vs 2봉전(16:00 전날)")
print("  → 갭 발견 시 시그널 발생!")
print("  → 이후 15M 차트에서 진입 대기")

print("\n" + "=" * 70)
print("🎯 결론")
print("=" * 70)

if 0 in unique_hours:
    print("\n✅ 00:00은 4시간봉 확정 시간 중 하나입니다!")
    print(f"\n4시간봉 확정 시간: {', '.join([f'{h:02d}:00' for h in unique_hours])}")
    print("\n각 확정 시점마다:")
    print("  1. 새로운 4H봉 생성")
    print("  2. FVG/OB 패턴 체크")
    print("  3. 시그널 발생 시 15M 차트에서 진입 대기")
else:
    print("\n⚠️ 00:00은 4시간봉 확정 시간이 아닙니다!")
    print(f"\n실제 확정 시간: {', '.join([f'{h:02d}:00' for h in unique_hours])}")

print("\n" + "=" * 70)
print("📊 실제 시그널 발생 시간 분석")
print("=" * 70)

# FVG 시그널 샘플 확인
print("\n15M 데이터로 실제 진입 시간 확인...")
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])

print(f"\n15M 데이터:")
print(f"  시작: {df_15m['datetime'].min()}")
print(f"  종료: {df_15m['datetime'].max()}")
print(f"  총 개수: {len(df_15m):,}개")

print(f"\n4H 데이터:")
print(f"  시작: {df_4h['datetime'].min()}")
print(f"  종료: {df_4h['datetime'].max()}")
print(f"  총 개수: {len(df_4h):,}개")

print("\n✅ 15M 데이터는 4H 시그널 발생 직후부터")
print("   즉시 진입 타이밍을 포착합니다!")
